use super::Integrator;
use super::output::{write_historic_snapshot, write_recovery_snapshot};
use crate::Axes;
use crate::constants::{
    INTEGRATOR_EPSILON, INTEGRATOR_EPSILON_GLOBAL, INTEGRATOR_FORCE_IS_VELOCITYDEPENDENT,
    INTEGRATOR_MAX_DT, INTEGRATOR_MIN_DT, MAX_PARTICLES, SAFETY_FACTOR,
};
use crate::effects::{EvolutionType, GeneralRelativityImplementation};
use crate::particles::{IgnoreGravityTerms, Universe};
use itertools::izip;
use serde::{Deserialize, Serialize};
//use serde_big_array::BigArray;
use std::any::Any;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use time::{OffsetDateTime, format_description};

const ZERO: f64 = 0.0;
const EVOLUTION_ENABLED: bool = true;
const DANGULAR_MOMENTUM_DT_ENABLED: bool = true;
const ACCELERATION_ENABLED: bool = true;
const IGNORED_GRAVITY_TERMS: IgnoreGravityTerms = IgnoreGravityTerms::None;

////// Constants to compute g values
//
//let mut n = 0;
//let r : [f64, ..28];
//for j in 1..8 {
//for k in 0..j {
//r[n] = 1. / (h[j] - h[k]);
//n += 1;
//}
//}
//
const r: [f64; 28] = [
    0.056_262_560_536_922_15,
    0.180_240_691_736_892_36,
    0.123_978_131_199_970_21,
    0.352_624_717_113_169_6,
    0.296_362_156_576_247_5,
    0.172_384_025_376_277_28,
    0.547_153_626_330_555_4,
    0.490_891_065_793_633_23,
    0.366_912_934_593_663_03,
    0.194_528_909_217_385_75,
    0.734_210_177_215_410_5,
    0.677_947_616_678_488_4,
    0.553_969_485_478_518_2,
    0.381_585_460_102_240_87,
    0.187_056_550_884_855_15,
    0.885_320_946_839_095_8,
    0.829_058_386_302_173_7,
    0.705_080_255_102_203_4,
    0.532_696_229_725_926_1,
    0.338_167_320_508_540_37,
    0.151_110_769_623_685_25,
    0.977_520_613_561_287_5,
    0.921_258_053_024_365_4,
    0.797_279_921_824_395_1,
    0.624_895_896_448_117_8,
    0.430_366_987_230_732_13,
    0.243_310_436_345_876_96,
    0.092_199_666_722_191_74,
];

////// Constants to convert between b and g arrays (c = c21, c31, c32, c41, c42...)
//
//let mut c : [f64, ..21];
//let mut d : [f64, ..21];
//c[0] = - h[1];
//d[0] =   h[1];
//let mut n = 0;
//for j in 2..7 {
//n += 1;
//c[n] = -h[j] * c[n-j+1]
//d[n] =  h[1] * d[n-j+1]
//for k in 2..j {
//n += 1;
//c[n] = c[n-j]  -  h[j] * c[n-j+1];
//d[n] = d[n-j]  +  h[k] * d[n-j+1];
//}
//n += 1;
//c[n] = c[n-j] - h[j];
//d[n] = d[n-j] + h[j];
//}
//
const c: [f64; 21] = [
    -0.056_262_560_536_922_15,
    0.010_140_802_830_063_63,
    -0.236_503_252_273_814_52,
    -0.003_575_897_729_251_617_6,
    0.093_537_695_259_462_07,
    -0.589_127_969_386_984_2,
    0.001_956_565_409_947_221,
    -0.054_755_386_889_068_69,
    0.415_881_200_082_306_83,
    -1.136_281_595_717_539_6,
    -0.001_436_530_236_370_891_5,
    0.042_158_527_721_268_706,
    -0.360_099_596_502_056_8,
    1.250_150_711_840_691,
    -1.870_491_772_932_95,
    0.001_271_790_309_026_867_8,
    -0.038_760_357_915_906_77,
    0.360_962_243_452_846,
    -1.466_884_208_400_427,
    2.906_136_259_308_429_4,
    -2.755_812_719_772_045_7,
];

///<https://arxiv.org/abs/1409.4779>
///IAS15: A fast, adaptive, high-order integrator for gravitational dynamics, accurate to machine
///precision over a billion orbits
///
/// <https://arxiv.org/pdf/1110.4876v2.pdf>
/// variable time-steps also
/// break the symplectic nature of an integrator.

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Ias15 {
    time_step: f64,
    pub universe: Universe,
    pub current_time: f64,
    current_iteration: u32,
    pub recovery_snapshot_period: f64,
    pub historic_snapshot_period: f64,
    last_recovery_snapshot_time: f64,
    last_historic_snapshot_time: f64,
    pub n_historic_snapshots: usize,
    //// Integrator IAS15 data:
    n_particles: usize,
    integrator_iterations_max_exceeded: i32, // Count how many times the iteration did not converge
    time_step_last_success: f64,             // Last accepted timestep (corresponding to br and er)

    #[serde(skip)]
    integrate_spin: bool,

    #[serde(skip)]
    b: [[f64; 3 * MAX_PARTICLES]; 7], // Coefficient b: acceleration dimension
    #[serde(skip)]
    br: [[f64; 3 * MAX_PARTICLES]; 7], // Previous b
    #[serde(skip)]
    g: [[f64; 3 * MAX_PARTICLES]; 7], // Coefficient g (it can also be expressed in terms of b)
    #[serde(skip)]
    e: [[f64; 3 * MAX_PARTICLES]; 7],
    #[serde(skip)]
    er: [[f64; 3 * MAX_PARTICLES]; 7], // Previous g

    #[serde(skip)]
    sb: [[f64; 3 * MAX_PARTICLES]; 7], // Coefficient b: acceleration dimension
    #[serde(skip)]
    sbr: [[f64; 3 * MAX_PARTICLES]; 7], // Previous b
    #[serde(skip)]
    sg: [[f64; 3 * MAX_PARTICLES]; 7], // Coefficient g (it can also be expressed in terms of b)
    #[serde(skip)]
    se: [[f64; 3 * MAX_PARTICLES]; 7],
    #[serde(skip)]
    ser: [[f64; 3 * MAX_PARTICLES]; 7], // Previous g

    #[serde(skip)]
    accelerations: [Axes; MAX_PARTICLES], // Temporary buffer for acceleration
    #[serde(skip)]
    positions0: [Axes; MAX_PARTICLES], // Temporary buffer for position (used for initial values at h=0)
    #[serde(skip)]
    velocities0: [Axes; MAX_PARTICLES], // Temporary buffer for velocity (used for initial values at h=0)
    #[serde(skip)]
    inertial_accelerations0: [Axes; MAX_PARTICLES], // Temporary buffer for acceleration (used for initial values at h=0)
    #[serde(skip)]
    dangular_momentum_dtt: [Axes; MAX_PARTICLES], // Temporary buffer for dangular_momentum_dt
    #[serde(skip)]
    dangular_momentum_dt0: [Axes; MAX_PARTICLES], // Temporary buffer for dangular_momentum_dt (used for initial values at h=0)
    #[serde(skip)]
    angular_momentum0: [Axes; MAX_PARTICLES], // Temporary buffer for angular_momentum (used for initial values at h=0)

    // Compensated summation coefficients
    #[serde(skip)]
    cs_positions: [Axes; MAX_PARTICLES], // position
    #[serde(skip)]
    cs_velocities: [Axes; MAX_PARTICLES], // velocity
    #[serde(skip)]
    css: [Axes; MAX_PARTICLES], // spin

    #[serde(skip)]
    csb: [[Axes; MAX_PARTICLES]; 7], // spin
    #[serde(skip)]
    cssb: [[Axes; MAX_PARTICLES]; 7], // spin

    s: [f64; 9], // Summation coefficients
}

impl Ias15 {
    pub fn new(
        time_step: f64,
        recovery_snapshot_period: f64,
        historic_snapshot_period: f64,
        universe: Universe,
    ) -> Ias15 {
        let n_particles = universe.n_particles;

        Ias15 {
            time_step,
            recovery_snapshot_period,
            historic_snapshot_period,
            last_recovery_snapshot_time: -1.,
            last_historic_snapshot_time: -1.,
            n_historic_snapshots: 0,
            universe,
            current_time: 0.,
            current_iteration: 0,
            n_particles,
            integrator_iterations_max_exceeded: 0,
            time_step_last_success: 0.,
            integrate_spin: false,

            b: [[0.; 3 * MAX_PARTICLES]; 7],
            g: [[0.; 3 * MAX_PARTICLES]; 7],
            e: [[0.; 3 * MAX_PARTICLES]; 7],
            br: [[0.; 3 * MAX_PARTICLES]; 7],
            er: [[0.; 3 * MAX_PARTICLES]; 7],

            sb: [[0.; 3 * MAX_PARTICLES]; 7],
            sg: [[0.; 3 * MAX_PARTICLES]; 7],
            se: [[0.; 3 * MAX_PARTICLES]; 7],
            sbr: [[0.; 3 * MAX_PARTICLES]; 7],
            ser: [[0.; 3 * MAX_PARTICLES]; 7],

            accelerations: [Axes::new(); MAX_PARTICLES],
            positions0: [Axes::new(); MAX_PARTICLES],
            velocities0: [Axes::new(); MAX_PARTICLES],
            inertial_accelerations0: [Axes::new(); MAX_PARTICLES],

            dangular_momentum_dtt: [Axes::new(); MAX_PARTICLES],
            dangular_momentum_dt0: [Axes::new(); MAX_PARTICLES],
            angular_momentum0: [Axes::new(); MAX_PARTICLES],

            cs_positions: [Axes::new(); MAX_PARTICLES],
            cs_velocities: [Axes::new(); MAX_PARTICLES],
            css: [Axes::new(); MAX_PARTICLES],
            csb: [[Axes::new(); MAX_PARTICLES]; 7],
            cssb: [[Axes::new(); MAX_PARTICLES]; 7],

            s: [0.; 9],
        }
    }
}

impl Integrator for Ias15 {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn get_n_historic_snapshots(&self) -> usize {
        self.n_historic_snapshots
    }

    fn get_n_particles(&self) -> usize {
        self.universe.n_particles
    }

    fn get_current_time(&self) -> f64 {
        self.current_time
    }

    fn set_time_limit(&mut self, time_limit: f64) {
        if time_limit > 0. && self.universe.time_limit != time_limit {
            if time_limit > self.universe.time_limit && self.universe.consider_effects.evolution {
                // Check if the new time is in the range of the evolutionary model
                for (i, evolver) in self.universe.particles_evolvers.iter().enumerate() {
                    // If it is an evolving body.
                    if !matches!(evolver.evolution, EvolutionType::NonEvolving)
                        && evolver.time[evolver.time.len() - 1] < time_limit
                    {
                        panic!(
                            "Your new time limit ({} days) is greater than the maximum allowed age of the evolving body #{} ({} days)",
                            time_limit,
                            i + 1,
                            evolver.time[evolver.time.len() - 1]
                        );
                    }
                }
            } else if time_limit < self.universe.time_limit && time_limit < self.current_time {
                panic!(
                    "Your new time limit ({} days) is smaller than the current time ({} days)",
                    time_limit, self.current_time
                );
            }
            println!(
                "[INFO {} UTC] The time limit changed from {} to {} days",
                OffsetDateTime::now_utc()
                    .format(
                        &format_description::parse("[year].[month].[day] [hour]:[minute]:[second]")
                            .unwrap()
                    )
                    .unwrap(),
                self.universe.time_limit,
                time_limit
            );
            self.universe.time_limit = time_limit;
        }
    }

    fn set_snapshot_periods(
        &mut self,
        historic_snapshot_period: f64,
        recovery_snapshot_period: f64,
    ) {
        if historic_snapshot_period > 0.
            && self.historic_snapshot_period != historic_snapshot_period
        {
            println!(
                "[INFO {} UTC] The historic snapshot period changed from {} to {} days",
                OffsetDateTime::now_utc()
                    .format(
                        &format_description::parse("[year].[month].[day] [hour]:[minute]:[second]")
                            .unwrap()
                    )
                    .unwrap(),
                self.historic_snapshot_period,
                historic_snapshot_period
            );
            self.historic_snapshot_period = historic_snapshot_period;
        } else {
            println!(
                "[INFO {} UTC] A historic snapshot will be saved every {} days",
                OffsetDateTime::now_utc()
                    .format(
                        &format_description::parse("[year].[month].[day] [hour]:[minute]:[second]")
                            .unwrap()
                    )
                    .unwrap(),
                self.historic_snapshot_period
            );
        }

        if recovery_snapshot_period > 0.
            && self.recovery_snapshot_period != recovery_snapshot_period
        {
            println!(
                "[INFO {} UTC] The recovery snapshot period changed from {} to {} days",
                OffsetDateTime::now_utc()
                    .format(
                        &format_description::parse("[year].[month].[day] [hour]:[minute]:[second]")
                            .unwrap()
                    )
                    .unwrap(),
                self.recovery_snapshot_period,
                recovery_snapshot_period
            );
            self.recovery_snapshot_period = recovery_snapshot_period;
        } else {
            println!(
                "[INFO {} UTC] A recovery snapshot will be saved every {} days",
                OffsetDateTime::now_utc()
                    .format(
                        &format_description::parse("[year].[month].[day] [hour]:[minute]:[second]")
                            .unwrap()
                    )
                    .unwrap(),
                self.recovery_snapshot_period
            );
        }
    }

    fn initialize_physical_values(&mut self) {
        assert!(
            !(self.current_time != 0.),
            "Physical values cannot be initialized on a resumed simulation"
        );
        let evolution = true;
        self.universe
            .calculate_spin_and_evolving_quantities(self.current_time, evolution); // Make sure we start with the good initial values
        self.universe.calculate_roche_radiuses(); // Needed for collision detection
    }

    fn iterate(
        &mut self,
        universe_history_writer: &mut BufWriter<File>,
        silent_mode: bool,
    ) -> Result<bool, String> {
        // Output
        let first_snapshot_trigger = self.last_historic_snapshot_time < 0.;
        let historic_snapshot_time_trigger =
            self.last_historic_snapshot_time + self.historic_snapshot_period <= self.current_time;
        let recovery_snapshot_time_trigger =
            self.last_recovery_snapshot_time + self.recovery_snapshot_period <= self.current_time;
        if first_snapshot_trigger || historic_snapshot_time_trigger {
            self.universe.inertial_to_heliocentric();
            let evolution = true;
            self.universe
                .calculate_spin_and_evolving_quantities(self.current_time, evolution);
            if self.universe.consider_effects.tides {
                self.universe.calculate_denergy_dt();
            }
            write_historic_snapshot(
                universe_history_writer,
                &self.universe,
                self.current_time,
                self.time_step,
            );
            self.last_historic_snapshot_time = self.current_time;
            self.n_historic_snapshots += 1;
            let current_time_years = self.current_time / 365.25;
            if !silent_mode {
                print!(
                    "Year: {:0.0} ({:0.1e}) | Time step: {:0.3} days                    \r",
                    current_time_years, current_time_years, self.time_step
                );
                let _ = std::io::stdout().flush();
            }
        }

        // Calculate accelerations.
        self.universe
            .gravity_calculate_acceleration(IGNORED_GRAVITY_TERMS);
        self.universe.inertial_to_heliocentric();
        // Calculate non-gravity accelerations.
        self.universe.calculate_additional_effects(
            self.current_time,
            EVOLUTION_ENABLED,
            DANGULAR_MOMENTUM_DT_ENABLED,
            ACCELERATION_ENABLED,
            IGNORED_GRAVITY_TERMS,
        );
        self.universe.apply_acceleration_corrections();

        self.integrator();
        self.current_iteration += 1;

        // Return
        if self.current_time + self.time_step > self.universe.time_limit {
            Err("Simulation completed".to_string())
        } else {
            Ok(first_snapshot_trigger || recovery_snapshot_time_trigger)
        }
    }

    fn write_recovery_snapshot(
        &mut self,
        snapshot_path: &Path,
        universe_history_writer: &mut BufWriter<File>,
    ) {
        self.last_recovery_snapshot_time = self.current_time;
        universe_history_writer.flush().unwrap();
        write_recovery_snapshot(snapshot_path, &self);
    }
}

impl Ias15 {
    #[allow(dead_code)]
    fn integrator(&mut self) {
        // Gauss-Radau spacings for substeps within a sequence, for the 15th order
        // integrator. The sum of the h values should be 3.733333333333333
        let h: [f64; 8] = [
            0.0,
            0.056_262_560_536_922_15,
            0.180_240_691_736_892_36,
            0.352_624_717_113_169_6,
            0.547_153_626_330_555_4,
            0.734_210_177_215_410_5,
            0.885_320_946_839_095_8,
            0.977_520_613_561_287_5,
        ];

        let d: [f64; 21] = [
            0.056_262_560_536_922_15,
            0.003_165_475_718_170_829_3,
            0.236_503_252_273_814_52,
            0.000_178_097_769_221_743_38,
            0.045_792_985_506_027_92,
            0.589_127_969_386_984_2,
            0.000_010_020_236_522_329_128,
            0.008_431_857_153_525_702,
            0.253_534_069_054_569_27,
            1.136_281_595_717_539_6,
            0.000_000_563_764_163_931_820_8,
            0.001_529_784_002_500_465_7,
            0.097_834_236_532_444_01,
            0.875_254_664_684_091_1,
            1.870_491_772_932_95,
            0.000_000_031_718_815_401_761_364,
            0.000_276_293_090_982_647_7,
            0.036_028_553_983_736_46,
            0.576_733_000_277_078_7,
            2.248_588_760_769_16,
            2.755_812_719_772_045_7,
        ];

        let mut consider_dangular_momentum_dt_from_general_relativity = false;
        if self.universe.consider_effects.general_relativity
            && self.universe.general_relativity_implementation
                == GeneralRelativityImplementation::Kidder1995
        {
            consider_dangular_momentum_dt_from_general_relativity = true;
        }
        self.integrate_spin = self.universe.consider_effects.tides
            || self.universe.consider_effects.rotational_flattening
            || self.universe.consider_effects.evolution
            || consider_dangular_momentum_dt_from_general_relativity;

        loop {
            for (
                particle,
                position,
                velocity,
                angular_momentum,
                inertial_acceleration,
                dangular_momentum_dt0,
            ) in izip!(
                &self.universe.particles,
                &mut self.positions0,
                &mut self.velocities0,
                &mut self.angular_momentum0,
                &mut self.inertial_accelerations0,
                &mut self.dangular_momentum_dt0,
            )
            .take(self.universe.n_particles)
            {
                *position = particle.inertial_position;
                *velocity = particle.inertial_velocity;
                *inertial_acceleration = particle.inertial_acceleration;

                if self.integrate_spin {
                    *dangular_momentum_dt0 = particle.dangular_momentum_dt;
                    *angular_momentum = particle.angular_momentum;
                }
            }

            // Find g values from b values predicted at the last call (Eqs. 7 of Everhart)
            let [g0, g1, g2, g3, g4, g5, g6] = &mut self.g;
            let [b0, b1, b2, b3, b4, b5, b6] = &self.b;
            for (
                g0_i,
                g1_i,
                g2_i,
                g3_i,
                g4_i,
                g5_i,
                g6_i,
                b0_i,
                b1_i,
                b2_i,
                b3_i,
                b4_i,
                b5_i,
                b6_i,
            ) in izip!(
                &mut *g0, &mut *g1, &mut *g2, &mut *g3, &mut *g4, &mut *g5, &mut *g6, b0, b1, b2,
                b3, b4, b5, b6,
            )
            .take(self.n_particles)
            {
                *g0_i = b6_i * d[15]
                    + b5_i * d[10]
                    + b4_i * d[6]
                    + b3_i * d[3]
                    + b2_i * d[1]
                    + b1_i * d[0]
                    + b0_i;
                *g1_i =
                    b6_i * d[16] + b5_i * d[11] + b4_i * d[7] + b3_i * d[4] + b2_i * d[2] + b1_i;
                *g2_i = b6_i * d[17] + b5_i * d[12] + b4_i * d[8] + b3_i * d[5] + b2_i;
                *g3_i = b6_i * d[18] + b5_i * d[13] + b4_i * d[9] + b3_i;
                *g4_i = b6_i * d[19] + b5_i * d[14] + b4_i;
                *g5_i = b6_i * d[20] + b5_i;
                *g6_i = *b6_i;
            }
            if self.integrate_spin {
                let [sg0, sg1, sg2, sg3, sg4, sg5, sg6] = &mut self.sg;
                let [sb0, sb1, sb2, sb3, sb4, sb5, sb6] = &self.sb;
                for (
                    sg0_i,
                    sg1_i,
                    sg2_i,
                    sg3_i,
                    sg4_i,
                    sg5_i,
                    sg6_i,
                    sb0_i,
                    sb1_i,
                    sb2_i,
                    sb3_i,
                    sb4_i,
                    sb5_i,
                    sb6_i,
                ) in izip!(
                    &mut *sg0, &mut *sg1, &mut *sg2, &mut *sg3, &mut *sg4, &mut *sg5, &mut *sg6,
                    sb0, sb1, sb2, sb3, sb4, sb5, sb6,
                )
                .take(self.n_particles)
                {
                    *sg0_i = sb6_i * d[15]
                        + sb5_i * d[10]
                        + sb4_i * d[6]
                        + sb3_i * d[3]
                        + sb2_i * d[1]
                        + sb1_i * d[0]
                        + sb0_i;
                    *sg1_i = sb6_i * d[16]
                        + sb5_i * d[11]
                        + sb4_i * d[7]
                        + sb3_i * d[4]
                        + sb2_i * d[2]
                        + sb1_i;
                    *sg2_i = sb6_i * d[17] + sb5_i * d[12] + sb4_i * d[8] + sb3_i * d[5] + sb2_i;
                    *sg3_i = sb6_i * d[18] + sb5_i * d[13] + sb4_i * d[9] + sb3_i;
                    *sg4_i = sb6_i * d[19] + sb5_i * d[14] + sb4_i;
                    *sg5_i = sb6_i * d[20] + sb5_i;
                    *sg6_i = *sb6_i;
                }
            }

            let t_beginning = self.current_time;
            //let mut predictor_corrector_error = 1e10;
            let mut predictor_corrector_error = 1e300;
            let mut predictor_corrector_error_last = 2.;
            let mut iterations: i32 = 0;

            // Predictor corrector loop
            // Stops if one of the following conditions is satisfied:
            //   1) predictor_corrector_error better than 1e-16
            //   2) predictor_corrector_error starts to oscillate
            //   3) more than 12 iterations
            //
            //  NOTE:
            //  In the original Everhart algorithm, there were six iterations
            //  for first call to the subroutine and two for the rest of calls
            'predictor_corrector: loop {
                if predictor_corrector_error < 1e-16 {
                    break 'predictor_corrector;
                }
                if iterations > 2 && predictor_corrector_error_last <= predictor_corrector_error {
                    break 'predictor_corrector;
                }
                if iterations >= 12 {
                    self.integrator_iterations_max_exceeded += 1;
                    let integrator_iterations_warning = 10;
                    if self.integrator_iterations_max_exceeded == integrator_iterations_warning {
                        println!(
                            "[WARNING {} UTC] At least {} predictor corrector loops in integrator IAS15 did not converge. This is typically an indication of the timestep being too large.",
                            OffsetDateTime::now_utc()
                                .format(
                                    &format_description::parse(
                                        "[year].[month].[day] [hour]:[minute]:[second]"
                                    )
                                    .unwrap()
                                )
                                .unwrap(),
                            integrator_iterations_warning
                        );
                    }
                    break 'predictor_corrector;
                }
                predictor_corrector_error_last = predictor_corrector_error;
                predictor_corrector_error = 0.;
                iterations += 1;

                for (n, h_n) in h.iter().enumerate().skip(1) {
                    // Loop over interval using Gauss-Radau spacings

                    // Calculate position predictors using Eqn. 9 of Everhart
                    self.s[0] = self.time_step * h_n;
                    self.s[1] = self.s[0].powi(2) / 2.;
                    self.s[2] = self.s[1] * h_n / 3.;
                    self.s[3] = self.s[2] * h_n / 2.;
                    self.s[4] = 3. * self.s[3] * h_n / 5.;
                    self.s[5] = 2. * self.s[4] * h_n / 3.;
                    self.s[6] = 5. * self.s[5] * h_n / 7.;
                    self.s[7] = 3. * self.s[6] * h_n / 4.;
                    self.s[8] = 7. * self.s[7] * h_n / 9.;

                    self.current_time = t_beginning + self.s[0];

                    //// Prepare particles arrays for force calculation
                    // Predict positions at interval n using self.b values
                    self.universe.particles[..self.universe.n_particles]
                        .iter_mut()
                        .enumerate()
                        .for_each(|(i, particle)| {
                            let k0: usize = 3 * i;
                            let k1: usize = 3 * i + 1;
                            let k2: usize = 3 * i + 2;
                            let inertial_acceleration = &self.inertial_accelerations0[i];
                            // Equation 7 in paper 2015MNRAS.446.1424R
                            particle.inertial_position = Axes::from(
                                -self.cs_positions[i].x
                                    + (self.s[8] * self.b[6][k0]
                                        + self.s[7] * self.b[5][k0]
                                        + self.s[6] * self.b[4][k0]
                                        + self.s[5] * self.b[3][k0]
                                        + self.s[4] * self.b[2][k0]
                                        + self.s[3] * self.b[1][k0]
                                        + self.s[2] * self.b[0][k0]
                                        + self.s[1] * inertial_acceleration.x
                                        + self.s[0] * self.velocities0[i].x)
                                    + self.positions0[i].x,
                                -self.cs_positions[i].y
                                    + (self.s[8] * self.b[6][k1]
                                        + self.s[7] * self.b[5][k1]
                                        + self.s[6] * self.b[4][k1]
                                        + self.s[5] * self.b[3][k1]
                                        + self.s[4] * self.b[2][k1]
                                        + self.s[3] * self.b[1][k1]
                                        + self.s[2] * self.b[0][k1]
                                        + self.s[1] * inertial_acceleration.y
                                        + self.s[0] * self.velocities0[i].y)
                                    + self.positions0[i].y,
                                -self.cs_positions[i].z
                                    + (self.s[8] * self.b[6][k2]
                                        + self.s[7] * self.b[5][k2]
                                        + self.s[6] * self.b[4][k2]
                                        + self.s[5] * self.b[3][k2]
                                        + self.s[4] * self.b[2][k2]
                                        + self.s[3] * self.b[1][k2]
                                        + self.s[2] * self.b[0][k2]
                                        + self.s[1] * inertial_acceleration.z
                                        + self.s[0] * self.velocities0[i].z)
                                    + self.positions0[i].z,
                            );
                        });

                    if INTEGRATOR_FORCE_IS_VELOCITYDEPENDENT {
                        // If necessary, calculate velocity predictors too, from Eqn. 10 of Everhart
                        self.s[0] = self.time_step * h_n;
                        self.s[1] = self.s[0] * h_n / 2.;
                        self.s[2] = 2. * self.s[1] * h_n / 3.;
                        self.s[3] = 3. * self.s[2] * h_n / 4.;
                        self.s[4] = 4. * self.s[3] * h_n / 5.;
                        self.s[5] = 5. * self.s[4] * h_n / 6.;
                        self.s[6] = 6. * self.s[5] * h_n / 7.;
                        self.s[7] = 7. * self.s[6] * h_n / 8.;

                        // Predict velocities at interval n using self.b values
                        self.universe.particles[..self.universe.n_particles]
                            .iter_mut()
                            .enumerate()
                            .for_each(|(i, particle)| {
                                let k0 = 3 * i;
                                let k1 = 3 * i + 1;
                                let k2 = 3 * i + 2;
                                let inertial_acceleration = &self.inertial_accelerations0[i];

                                // Equation 6 in paper 2015MNRAS.446.1424R
                                particle.inertial_velocity = Axes::from(
                                    -self.cs_velocities[i].x
                                        + self.s[7] * self.b[6][k0]
                                        + self.s[6] * self.b[5][k0]
                                        + self.s[5] * self.b[4][k0]
                                        + self.s[4] * self.b[3][k0]
                                        + self.s[3] * self.b[2][k0]
                                        + self.s[2] * self.b[1][k0]
                                        + self.s[1] * self.b[0][k0]
                                        + self.s[0] * inertial_acceleration.x
                                        + self.velocities0[i].x,
                                    -self.cs_velocities[i].y
                                        + self.s[7] * self.b[6][k1]
                                        + self.s[6] * self.b[5][k1]
                                        + self.s[5] * self.b[4][k1]
                                        + self.s[4] * self.b[3][k1]
                                        + self.s[3] * self.b[2][k1]
                                        + self.s[2] * self.b[1][k1]
                                        + self.s[1] * self.b[0][k1]
                                        + self.s[0] * inertial_acceleration.y
                                        + self.velocities0[i].y,
                                    -self.cs_velocities[i].z
                                        + self.s[7] * self.b[6][k2]
                                        + self.s[6] * self.b[5][k2]
                                        + self.s[5] * self.b[4][k2]
                                        + self.s[4] * self.b[3][k2]
                                        + self.s[3] * self.b[2][k2]
                                        + self.s[2] * self.b[1][k2]
                                        + self.s[1] * self.b[0][k2]
                                        + self.s[0] * inertial_acceleration.z
                                        + self.velocities0[i].z,
                                );
                            });

                        if self.integrate_spin {
                            // Spin integration
                            // Predict velocities at interval n using self.b values
                            self.universe.particles[..self.universe.n_particles]
                                .iter_mut()
                                .enumerate()
                                .for_each(|(i, particle)| {
                                    let k0 = 3 * i;
                                    let k1 = 3 * i + 1;
                                    let k2 = 3 * i + 2;
                                    // Equation 6 in paper 2015MNRAS.446.1424R
                                    particle.angular_momentum = Axes::from(
                                        -self.css[i].x
                                            + self.s[7] * self.sb[6][k0]
                                            + self.s[6] * self.sb[5][k0]
                                            + self.s[5] * self.sb[4][k0]
                                            + self.s[4] * self.sb[3][k0]
                                            + self.s[3] * self.sb[2][k0]
                                            + self.s[2] * self.sb[1][k0]
                                            + self.s[1] * self.sb[0][k0]
                                            + self.s[0] * self.dangular_momentum_dt0[i].x
                                            + self.angular_momentum0[i].x,
                                        -self.css[i].y
                                            + self.s[7] * self.sb[6][k1]
                                            + self.s[6] * self.sb[5][k1]
                                            + self.s[5] * self.sb[4][k1]
                                            + self.s[4] * self.sb[3][k1]
                                            + self.s[3] * self.sb[2][k1]
                                            + self.s[2] * self.sb[1][k1]
                                            + self.s[1] * self.sb[0][k1]
                                            + self.s[0] * self.dangular_momentum_dt0[i].y
                                            + self.angular_momentum0[i].y,
                                        -self.css[i].z
                                            + self.s[7] * self.sb[6][k2]
                                            + self.s[6] * self.sb[5][k2]
                                            + self.s[5] * self.sb[4][k2]
                                            + self.s[4] * self.sb[3][k2]
                                            + self.s[3] * self.sb[2][k2]
                                            + self.s[2] * self.sb[1][k2]
                                            + self.s[1] * self.sb[0][k2]
                                            + self.s[0] * self.dangular_momentum_dt0[i].z
                                            + self.angular_momentum0[i].z,
                                    );
                                });
                        }
                    }
                    // Calculate accelerations.
                    self.universe
                        .gravity_calculate_acceleration(IGNORED_GRAVITY_TERMS);
                    self.universe.inertial_to_heliocentric();
                    // Calculate non-gravity accelerations.
                    self.universe.calculate_additional_effects(
                        self.current_time,
                        EVOLUTION_ENABLED,
                        DANGULAR_MOMENTUM_DT_ENABLED,
                        ACCELERATION_ENABLED,
                        IGNORED_GRAVITY_TERMS,
                    );
                    self.universe.apply_acceleration_corrections();

                    for (particle, inertial_acceleration, dangular_momentum_dtt) in izip!(
                        &self.universe.particles[..self.universe.n_particles],
                        &mut self.accelerations,
                        &mut self.dangular_momentum_dtt
                    ) {
                        *inertial_acceleration = particle.inertial_acceleration;

                        if self.integrate_spin {
                            // angular_momentum_dt = dmoment_of_inertia_spin_dt
                            *dangular_momentum_dtt = particle.dangular_momentum_dt;
                        }
                    }

                    // Update G values using Eqs. 4 of Everhart, and update B values using Eqs. 5
                    predictor_corrector_error = self.update_b_and_g(n, predictor_corrector_error);
                } // end loop over interval using Gauss-Radau spacings
            } // end predictor corrector loop

            // Set time back to initial value (will be updated below)
            self.current_time = t_beginning;

            // Find new timestep
            let dt_done = match self.calculate_new_timestep() {
                Some(d) => d,
                None => continue, // Step rejected
            };

            // Find new position and velocity values at end of the sequence
            // (Eqs. 11, 12 of Everhart)
            self.find_new_position_and_velocity(dt_done);
            self.current_time += dt_done;
            self.time_step_last_success = dt_done;

            // Swap particle buffers
            for (particle, position, velocity, angular_momentum) in izip!(
                &mut self.universe.particles[..self.universe.n_particles],
                &self.positions0,
                &self.velocities0,
                &self.angular_momentum0,
            ) {
                particle.inertial_position = *position; // Set final position
                particle.inertial_velocity = *velocity; // Set final velocity
                if self.integrate_spin {
                    particle.angular_momentum = *angular_momentum; // Set final angular momentum
                }
            }

            // TODO check if a mem::swap is possible here
            self.er.clone_from(&self.e);
            self.br.clone_from(&self.b);
            self.ser.clone_from(&self.se);
            self.sbr.clone_from(&self.sb);
            let ratio = self.time_step / dt_done;
            self.predict_next_step(ratio);
            break; // Success.
        } // end main loop
    }

    fn update_b_and_g(&mut self, n: usize, mut predictor_corrector_error: f64) -> f64 {
        let mut maxak = 0.0;
        let mut maxb6ktmp = 0.0;
        let mut max_dangular_momentum_dtk = 0.0;
        let mut maxb6kstmp = 0.0;

        match n {
            1 => {
                for k in 0..3 * self.n_particles {
                    let tmp = self.g[0][k];
                    let (gk, gk_cs) = add_cs(
                        self.accelerations[k / 3].get_by_index(k % 3),
                        ZERO,
                        -self.inertial_accelerations0[k / 3].get_by_index(k % 3),
                    );
                    let (gk, _) = add_cs(gk, gk_cs, ZERO);
                    self.g[0][k] = gk / r[0];
                    let (tmp_b, tmp_csb) = add_cs(
                        self.b[0][k],
                        self.csb[0][k / 3].get_by_index(k % 3),
                        self.g[0][k] - tmp,
                    );
                    self.b[0][k] = tmp_b;
                    self.csb[0][k / 3].set_by_index(k % 3, tmp_csb);
                }
            }
            2 => {
                for k in 0..3 * self.n_particles {
                    let mut tmp = self.g[1][k];
                    let (gk, gk_cs) = add_cs(
                        self.accelerations[k / 3].get_by_index(k % 3),
                        ZERO,
                        -self.inertial_accelerations0[k / 3].get_by_index(k % 3),
                    );
                    let (gk, _) = add_cs(gk, gk_cs, ZERO);
                    self.g[1][k] = (gk / r[1] - self.g[0][k]) / r[2];
                    tmp = self.g[1][k] - tmp;
                    let (tmp_b, tmp_csb) = add_cs(
                        self.b[0][k],
                        self.csb[0][k / 3].get_by_index(k % 3),
                        tmp * c[0],
                    );
                    self.b[0][k] = tmp_b;
                    self.csb[0][k / 3].set_by_index(k % 3, tmp_csb);

                    let (tmp_b, tmp_csb) =
                        add_cs(self.b[1][k], self.csb[1][k / 3].get_by_index(k % 3), tmp);
                    self.b[1][k] = tmp_b;
                    self.csb[1][k / 3].set_by_index(k % 3, tmp_csb);
                }
            }
            3 => {
                for k in 0..3 * self.n_particles {
                    let mut tmp = self.g[2][k];
                    let (gk, gk_cs) = add_cs(
                        self.accelerations[k / 3].get_by_index(k % 3),
                        ZERO,
                        -self.inertial_accelerations0[k / 3].get_by_index(k % 3),
                    );
                    let (gk, _) = add_cs(gk, gk_cs, ZERO);
                    self.g[2][k] = ((gk / r[3] - self.g[0][k]) / r[4] - self.g[1][k]) / r[5];
                    tmp = self.g[2][k] - tmp;
                    let (tmp_b, tmp_csb) = add_cs(
                        self.b[0][k],
                        self.csb[0][k / 3].get_by_index(k % 3),
                        tmp * c[1],
                    );
                    self.b[0][k] = tmp_b;
                    self.csb[0][k / 3].set_by_index(k % 3, tmp_csb);
                    let (tmp_b, tmp_csb) = add_cs(
                        self.b[1][k],
                        self.csb[1][k / 3].get_by_index(k % 3),
                        tmp * c[2],
                    );
                    self.b[1][k] = tmp_b;
                    self.csb[1][k / 3].set_by_index(k % 3, tmp_csb);
                    let (tmp_b, tmp_csb) =
                        add_cs(self.b[2][k], self.csb[2][k / 3].get_by_index(k % 3), tmp);
                    self.b[2][k] = tmp_b;
                    self.csb[2][k / 3].set_by_index(k % 3, tmp_csb);
                }
            }
            4 => {
                for k in 0..3 * self.n_particles {
                    let mut tmp = self.g[3][k];
                    let (gk, gk_cs) = add_cs(
                        self.accelerations[k / 3].get_by_index(k % 3),
                        ZERO,
                        -self.inertial_accelerations0[k / 3].get_by_index(k % 3),
                    );
                    let (gk, _) = add_cs(gk, gk_cs, ZERO);
                    self.g[3][k] = (((gk / r[6] - self.g[0][k]) / r[7] - self.g[1][k]) / r[8]
                        - self.g[2][k])
                        / r[9];
                    tmp = self.g[3][k] - tmp;
                    let (tmp_b, tmp_csb) = add_cs(
                        self.b[0][k],
                        self.csb[0][k / 3].get_by_index(k % 3),
                        tmp * c[3],
                    );
                    self.b[0][k] = tmp_b;
                    self.csb[0][k / 3].set_by_index(k % 3, tmp_csb);
                    let (tmp_b, tmp_csb) = add_cs(
                        self.b[1][k],
                        self.csb[1][k / 3].get_by_index(k % 3),
                        tmp * c[4],
                    );
                    self.b[1][k] = tmp_b;
                    self.csb[1][k / 3].set_by_index(k % 3, tmp_csb);
                    let (tmp_b, tmp_csb) = add_cs(
                        self.b[2][k],
                        self.csb[2][k / 3].get_by_index(k % 3),
                        tmp * c[5],
                    );
                    self.b[2][k] = tmp_b;
                    self.csb[2][k / 3].set_by_index(k % 3, tmp_csb);
                    let (tmp_b, tmp_csb) =
                        add_cs(self.b[3][k], self.csb[3][k / 3].get_by_index(k % 3), tmp);
                    self.b[3][k] = tmp_b;
                    self.csb[3][k / 3].set_by_index(k % 3, tmp_csb);
                }
            }
            5 => {
                for k in 0..3 * self.n_particles {
                    let mut tmp = self.g[4][k];
                    let (gk, gk_cs) = add_cs(
                        self.accelerations[k / 3].get_by_index(k % 3),
                        ZERO,
                        -self.inertial_accelerations0[k / 3].get_by_index(k % 3),
                    );
                    let (gk, _) = add_cs(gk, gk_cs, ZERO);
                    self.g[4][k] = ((((gk / r[10] - self.g[0][k]) / r[11] - self.g[1][k]) / r[12]
                        - self.g[2][k])
                        / r[13]
                        - self.g[3][k])
                        / r[14];
                    tmp = self.g[4][k] - tmp;
                    let (tmp_b, tmp_csb) = add_cs(
                        self.b[0][k],
                        self.csb[0][k / 3].get_by_index(k % 3),
                        tmp * c[6],
                    );
                    self.b[0][k] = tmp_b;
                    self.csb[0][k / 3].set_by_index(k % 3, tmp_csb);
                    let (tmp_b, tmp_csb) = add_cs(
                        self.b[1][k],
                        self.csb[1][k / 3].get_by_index(k % 3),
                        tmp * c[7],
                    );
                    self.b[1][k] = tmp_b;
                    self.csb[1][k / 3].set_by_index(k % 3, tmp_csb);
                    let (tmp_b, tmp_csb) = add_cs(
                        self.b[2][k],
                        self.csb[2][k / 3].get_by_index(k % 3),
                        tmp * c[8],
                    );
                    self.b[2][k] = tmp_b;
                    self.csb[2][k / 3].set_by_index(k % 3, tmp_csb);
                    let (tmp_b, tmp_csb) = add_cs(
                        self.b[3][k],
                        self.csb[3][k / 3].get_by_index(k % 3),
                        tmp * c[9],
                    );
                    self.b[3][k] = tmp_b;
                    self.csb[3][k / 3].set_by_index(k % 3, tmp_csb);
                    let (tmp_b, tmp_csb) =
                        add_cs(self.b[4][k], self.csb[4][k / 3].get_by_index(k % 3), tmp);
                    self.b[4][k] = tmp_b;
                    self.csb[4][k / 3].set_by_index(k % 3, tmp_csb);
                }
            }
            6 => {
                for k in 0..3 * self.n_particles {
                    let mut tmp = self.g[5][k];
                    let (gk, gk_cs) = add_cs(
                        self.accelerations[k / 3].get_by_index(k % 3),
                        ZERO,
                        -self.inertial_accelerations0[k / 3].get_by_index(k % 3),
                    );
                    let (gk, _) = add_cs(gk, gk_cs, ZERO);
                    self.g[5][k] = (((((gk / r[15] - self.g[0][k]) / r[16] - self.g[1][k])
                        / r[17]
                        - self.g[2][k])
                        / r[18]
                        - self.g[3][k])
                        / r[19]
                        - self.g[4][k])
                        / r[20];
                    tmp = self.g[5][k] - tmp;
                    let (tmp_b, tmp_csb) = add_cs(
                        self.b[0][k],
                        self.csb[0][k / 3].get_by_index(k % 3),
                        tmp * c[10],
                    );
                    self.b[0][k] = tmp_b;
                    self.csb[0][k / 3].set_by_index(k % 3, tmp_csb);
                    let (tmp_b, tmp_csb) = add_cs(
                        self.b[1][k],
                        self.csb[1][k / 3].get_by_index(k % 3),
                        tmp * c[11],
                    );
                    self.b[1][k] = tmp_b;
                    self.csb[1][k / 3].set_by_index(k % 3, tmp_csb);
                    let (tmp_b, tmp_csb) = add_cs(
                        self.b[2][k],
                        self.csb[2][k / 3].get_by_index(k % 3),
                        tmp * c[12],
                    );
                    self.b[2][k] = tmp_b;
                    self.csb[2][k / 3].set_by_index(k % 3, tmp_csb);
                    let (tmp_b, tmp_csb) = add_cs(
                        self.b[3][k],
                        self.csb[3][k / 3].get_by_index(k % 3),
                        tmp * c[13],
                    );
                    self.b[3][k] = tmp_b;
                    self.csb[3][k / 3].set_by_index(k % 3, tmp_csb);
                    let (tmp_b, tmp_csb) = add_cs(
                        self.b[4][k],
                        self.csb[4][k / 3].get_by_index(k % 3),
                        tmp * c[14],
                    );
                    self.b[4][k] = tmp_b;
                    self.csb[4][k / 3].set_by_index(k % 3, tmp_csb);
                    let (tmp_b, tmp_csb) =
                        add_cs(self.b[5][k], self.csb[5][k / 3].get_by_index(k % 3), tmp);
                    self.b[5][k] = tmp_b;
                    self.csb[5][k / 3].set_by_index(k % 3, tmp_csb);
                }
            }
            7 => {
                for k in 0..3 * self.n_particles {
                    let mut tmp = self.g[6][k];
                    let (gk, gk_cs) = add_cs(
                        self.accelerations[k / 3].get_by_index(k % 3),
                        ZERO,
                        -self.inertial_accelerations0[k / 3].get_by_index(k % 3),
                    );
                    let (gk, _) = add_cs(gk, gk_cs, ZERO);
                    self.g[6][k] = ((((((gk / r[21] - self.g[0][k]) / r[22] - self.g[1][k])
                        / r[23]
                        - self.g[2][k])
                        / r[24]
                        - self.g[3][k])
                        / r[25]
                        - self.g[4][k])
                        / r[26]
                        - self.g[5][k])
                        / r[27];
                    tmp = self.g[6][k] - tmp;
                    let (tmp_b, tmp_csb) = add_cs(
                        self.b[0][k],
                        self.csb[0][k / 3].get_by_index(k % 3),
                        tmp * c[15],
                    );
                    self.b[0][k] = tmp_b;
                    self.csb[0][k / 3].set_by_index(k % 3, tmp_csb);
                    let (tmp_b, tmp_csb) = add_cs(
                        self.b[1][k],
                        self.csb[1][k / 3].get_by_index(k % 3),
                        tmp * c[16],
                    );
                    self.b[1][k] = tmp_b;
                    self.csb[1][k / 3].set_by_index(k % 3, tmp_csb);
                    let (tmp_b, tmp_csb) = add_cs(
                        self.b[2][k],
                        self.csb[2][k / 3].get_by_index(k % 3),
                        tmp * c[17],
                    );
                    self.b[2][k] = tmp_b;
                    self.csb[2][k / 3].set_by_index(k % 3, tmp_csb);
                    let (tmp_b, tmp_csb) = add_cs(
                        self.b[3][k],
                        self.csb[3][k / 3].get_by_index(k % 3),
                        tmp * c[18],
                    );
                    self.b[3][k] = tmp_b;
                    self.csb[3][k / 3].set_by_index(k % 3, tmp_csb);
                    let (tmp_b, tmp_csb) = add_cs(
                        self.b[4][k],
                        self.csb[4][k / 3].get_by_index(k % 3),
                        tmp * c[19],
                    );
                    self.b[4][k] = tmp_b;
                    self.csb[4][k / 3].set_by_index(k % 3, tmp_csb);
                    let (tmp_b, tmp_csb) = add_cs(
                        self.b[5][k],
                        self.csb[5][k / 3].get_by_index(k % 3),
                        tmp * c[20],
                    );
                    self.b[5][k] = tmp_b;
                    self.csb[5][k / 3].set_by_index(k % 3, tmp_csb);
                    let (tmp_b, tmp_csb) =
                        add_cs(self.b[6][k], self.csb[6][k / 3].get_by_index(k % 3), tmp);
                    self.b[6][k] = tmp_b;
                    self.csb[6][k / 3].set_by_index(k % 3, tmp_csb);

                    // Monitor change in self.b[6][k] relative to self.accelerations[k]. The predictor corrector scheme is converged if it is close to 0.
                    if INTEGRATOR_EPSILON_GLOBAL {
                        // global error estimate (equation 9)
                        let ak = abs!(self.accelerations[k / 3].get_by_index(k % 3));
                        if ak.is_normal() && ak > maxak {
                            maxak = ak;
                        }
                        let b6ktmp = abs!(tmp); // change of b6ktmp coefficient
                        if b6ktmp.is_normal() && b6ktmp > maxb6ktmp {
                            maxb6ktmp = b6ktmp;
                        }
                    } else {
                        // local error estimate (equation 9)
                        let ak = self.accelerations[k / 3].get_by_index(k % 3);
                        let b6ktmp = tmp;
                        let errork = abs!(b6ktmp / ak);
                        predictor_corrector_error =
                            if errork.is_normal() && errork > predictor_corrector_error {
                                errork
                            } else {
                                0.0
                            }
                    }
                }
            }
            _ => unreachable!(),
        } // end match
        if self.integrate_spin {
            match n {
                1 => {
                    for k in 0..3 * self.n_particles {
                        let stmp = self.sg[0][k];
                        let (sgk, sgk_cs) = add_cs(
                            self.dangular_momentum_dtt[k / 3].get_by_index(k % 3),
                            ZERO,
                            -self.dangular_momentum_dt0[k / 3].get_by_index(k % 3),
                        );
                        let (sgk, _sgk_cs) = add_cs(sgk, sgk_cs, ZERO);
                        self.sg[0][k] = sgk / r[0];
                        let (tmp_sb, tmp_cssb) = add_cs(
                            self.sb[0][k],
                            self.cssb[0][k / 3].get_by_index(k % 3),
                            self.sg[0][k] - stmp,
                        );
                        self.sb[0][k] = tmp_sb;
                        self.cssb[0][k / 3].set_by_index(k % 3, tmp_cssb);
                    }
                }
                2 => {
                    for k in 0..3 * self.n_particles {
                        let mut stmp = self.sg[1][k];
                        let (sgk, sgk_cs) = add_cs(
                            self.dangular_momentum_dtt[k / 3].get_by_index(k % 3),
                            ZERO,
                            -self.dangular_momentum_dt0[k / 3].get_by_index(k % 3),
                        );
                        let (sgk, _sgk_cs) = add_cs(sgk, sgk_cs, ZERO);
                        self.sg[1][k] = (sgk / r[1] - self.sg[0][k]) / r[2];
                        stmp = self.sg[1][k] - stmp;
                        let (tmp_sb, tmp_cssb) = add_cs(
                            self.sb[0][k],
                            self.cssb[0][k / 3].get_by_index(k % 3),
                            stmp * c[0],
                        );
                        self.sb[0][k] = tmp_sb;
                        self.cssb[0][k / 3].set_by_index(k % 3, tmp_cssb);
                        let (tmp_sb, tmp_cssb) =
                            add_cs(self.sb[1][k], self.cssb[1][k / 3].get_by_index(k % 3), stmp);
                        self.sb[1][k] = tmp_sb;
                        self.cssb[1][k / 3].set_by_index(k % 3, tmp_cssb);
                    }
                }
                3 => {
                    for k in 0..3 * self.n_particles {
                        let mut stmp = self.sg[2][k];
                        let (sgk, sgk_cs) = add_cs(
                            self.dangular_momentum_dtt[k / 3].get_by_index(k % 3),
                            ZERO,
                            -self.dangular_momentum_dt0[k / 3].get_by_index(k % 3),
                        );
                        let (sgk, _sgk_cs) = add_cs(sgk, sgk_cs, ZERO);
                        self.sg[2][k] =
                            ((sgk / r[3] - self.sg[0][k]) / r[4] - self.sg[1][k]) / r[5];
                        stmp = self.sg[2][k] - stmp;
                        let (tmp_sb, tmp_cssb) = add_cs(
                            self.sb[0][k],
                            self.cssb[0][k / 3].get_by_index(k % 3),
                            stmp * c[1],
                        );
                        self.sb[0][k] = tmp_sb;
                        self.cssb[0][k / 3].set_by_index(k % 3, tmp_cssb);
                        let (tmp_sb, tmp_cssb) = add_cs(
                            self.sb[1][k],
                            self.cssb[1][k / 3].get_by_index(k % 3),
                            stmp * c[2],
                        );
                        self.sb[1][k] = tmp_sb;
                        self.cssb[1][k / 3].set_by_index(k % 3, tmp_cssb);
                        let (tmp_sb, tmp_cssb) =
                            add_cs(self.sb[2][k], self.cssb[2][k / 3].get_by_index(k % 3), stmp);
                        self.sb[2][k] = tmp_sb;
                        self.cssb[2][k / 3].set_by_index(k % 3, tmp_cssb);
                    }
                }
                4 => {
                    for k in 0..3 * self.n_particles {
                        let mut stmp = self.sg[3][k];
                        let (sgk, sgk_cs) = add_cs(
                            self.dangular_momentum_dtt[k / 3].get_by_index(k % 3),
                            ZERO,
                            -self.dangular_momentum_dt0[k / 3].get_by_index(k % 3),
                        );
                        let (sgk, _sgk_cs) = add_cs(sgk, sgk_cs, ZERO);
                        self.sg[3][k] = (((sgk / r[6] - self.sg[0][k]) / r[7] - self.sg[1][k])
                            / r[8]
                            - self.sg[2][k])
                            / r[9];
                        stmp = self.sg[3][k] - stmp;
                        let (tmp_sb, tmp_cssb) = add_cs(
                            self.sb[0][k],
                            self.cssb[0][k / 3].get_by_index(k % 3),
                            stmp * c[3],
                        );
                        self.sb[0][k] = tmp_sb;
                        self.cssb[0][k / 3].set_by_index(k % 3, tmp_cssb);
                        let (tmp_sb, tmp_cssb) = add_cs(
                            self.sb[1][k],
                            self.cssb[1][k / 3].get_by_index(k % 3),
                            stmp * c[4],
                        );
                        self.sb[1][k] = tmp_sb;
                        self.cssb[1][k / 3].set_by_index(k % 3, tmp_cssb);
                        let (tmp_sb, tmp_cssb) = add_cs(
                            self.sb[2][k],
                            self.cssb[2][k / 3].get_by_index(k % 3),
                            stmp * c[5],
                        );
                        self.sb[2][k] = tmp_sb;
                        self.cssb[2][k / 3].set_by_index(k % 3, tmp_cssb);
                        let (tmp_sb, tmp_cssb) =
                            add_cs(self.sb[3][k], self.cssb[3][k / 3].get_by_index(k % 3), stmp);
                        self.sb[3][k] = tmp_sb;
                        self.cssb[3][k / 3].set_by_index(k % 3, tmp_cssb);
                    }
                }
                5 => {
                    for k in 0..3 * self.n_particles {
                        let mut stmp = self.sg[4][k];
                        let (sgk, sgk_cs) = add_cs(
                            self.dangular_momentum_dtt[k / 3].get_by_index(k % 3),
                            ZERO,
                            -self.dangular_momentum_dt0[k / 3].get_by_index(k % 3),
                        );
                        let (sgk, _sgk_cs) = add_cs(sgk, sgk_cs, ZERO);
                        self.sg[4][k] = ((((sgk / r[10] - self.sg[0][k]) / r[11] - self.sg[1][k])
                            / r[12]
                            - self.sg[2][k])
                            / r[13]
                            - self.sg[3][k])
                            / r[14];
                        stmp = self.sg[4][k] - stmp;
                        let (tmp_sb, tmp_cssb) = add_cs(
                            self.sb[0][k],
                            self.cssb[0][k / 3].get_by_index(k % 3),
                            stmp * c[6],
                        );
                        self.sb[0][k] = tmp_sb;
                        self.cssb[0][k / 3].set_by_index(k % 3, tmp_cssb);
                        let (tmp_sb, tmp_cssb) = add_cs(
                            self.sb[1][k],
                            self.cssb[1][k / 3].get_by_index(k % 3),
                            stmp * c[7],
                        );
                        self.sb[1][k] = tmp_sb;
                        self.cssb[1][k / 3].set_by_index(k % 3, tmp_cssb);
                        let (tmp_sb, tmp_cssb) = add_cs(
                            self.sb[2][k],
                            self.cssb[2][k / 3].get_by_index(k % 3),
                            stmp * c[8],
                        );
                        self.sb[2][k] = tmp_sb;
                        self.cssb[2][k / 3].set_by_index(k % 3, tmp_cssb);
                        let (tmp_sb, tmp_cssb) = add_cs(
                            self.sb[3][k],
                            self.cssb[3][k / 3].get_by_index(k % 3),
                            stmp * c[9],
                        );
                        self.sb[3][k] = tmp_sb;
                        self.cssb[3][k / 3].set_by_index(k % 3, tmp_cssb);
                        let (tmp_sb, tmp_cssb) =
                            add_cs(self.sb[4][k], self.cssb[4][k / 3].get_by_index(k % 3), stmp);
                        self.sb[4][k] = tmp_sb;
                        self.cssb[4][k / 3].set_by_index(k % 3, tmp_cssb);
                    }
                }
                6 => {
                    for k in 0..3 * self.n_particles {
                        let mut stmp = self.sg[5][k];
                        let (sgk, sgk_cs) = add_cs(
                            self.dangular_momentum_dtt[k / 3].get_by_index(k % 3),
                            ZERO,
                            -self.dangular_momentum_dt0[k / 3].get_by_index(k % 3),
                        );
                        let (sgk, _sgk_cs) = add_cs(sgk, sgk_cs, ZERO);
                        self.sg[5][k] =
                            (((((sgk / r[15] - self.sg[0][k]) / r[16] - self.sg[1][k]) / r[17]
                                - self.sg[2][k])
                                / r[18]
                                - self.sg[3][k])
                                / r[19]
                                - self.sg[4][k])
                                / r[20];
                        stmp = self.sg[5][k] - stmp;
                        let (tmp_sb, tmp_cssb) = add_cs(
                            self.sb[0][k],
                            self.cssb[0][k / 3].get_by_index(k % 3),
                            stmp * c[10],
                        );
                        self.sb[0][k] = tmp_sb;
                        self.cssb[0][k / 3].set_by_index(k % 3, tmp_cssb);
                        let (tmp_sb, tmp_cssb) = add_cs(
                            self.sb[1][k],
                            self.cssb[1][k / 3].get_by_index(k % 3),
                            stmp * c[11],
                        );
                        self.sb[1][k] = tmp_sb;
                        self.cssb[1][k / 3].set_by_index(k % 3, tmp_cssb);
                        let (tmp_sb, tmp_cssb) = add_cs(
                            self.sb[2][k],
                            self.cssb[2][k / 3].get_by_index(k % 3),
                            stmp * c[12],
                        );
                        self.sb[2][k] = tmp_sb;
                        self.cssb[2][k / 3].set_by_index(k % 3, tmp_cssb);
                        let (tmp_sb, tmp_cssb) = add_cs(
                            self.sb[3][k],
                            self.cssb[3][k / 3].get_by_index(k % 3),
                            stmp * c[13],
                        );
                        self.sb[3][k] = tmp_sb;
                        self.cssb[3][k / 3].set_by_index(k % 3, tmp_cssb);
                        let (tmp_sb, tmp_cssb) = add_cs(
                            self.sb[4][k],
                            self.cssb[4][k / 3].get_by_index(k % 3),
                            stmp * c[14],
                        );
                        self.sb[4][k] = tmp_sb;
                        self.cssb[4][k / 3].set_by_index(k % 3, tmp_cssb);
                        let (tmp_sb, tmp_cssb) =
                            add_cs(self.sb[5][k], self.cssb[5][k / 3].get_by_index(k % 3), stmp);
                        self.sb[5][k] = tmp_sb;
                        self.cssb[5][k / 3].set_by_index(k % 3, tmp_cssb);
                    }
                }
                7 => {
                    for k in 0..3 * self.n_particles {
                        let mut stmp = self.sg[6][k];
                        let (sgk, sgk_cs) = add_cs(
                            self.dangular_momentum_dtt[k / 3].get_by_index(k % 3),
                            ZERO,
                            -self.dangular_momentum_dt0[k / 3].get_by_index(k % 3),
                        );
                        let (sgk, _sgk_cs) = add_cs(sgk, sgk_cs, ZERO);
                        self.sg[6][k] =
                            ((((((sgk / r[21] - self.sg[0][k]) / r[22] - self.sg[1][k]) / r[23]
                                - self.sg[2][k])
                                / r[24]
                                - self.sg[3][k])
                                / r[25]
                                - self.sg[4][k])
                                / r[26]
                                - self.sg[5][k])
                                / r[27];
                        stmp = self.sg[6][k] - stmp;
                        let (tmp_sb, tmp_cssb) = add_cs(
                            self.sb[0][k],
                            self.cssb[0][k / 3].get_by_index(k % 3),
                            stmp * c[15],
                        );
                        self.sb[0][k] = tmp_sb;
                        self.cssb[0][k / 3].set_by_index(k % 3, tmp_cssb);
                        let (tmp_sb, tmp_cssb) = add_cs(
                            self.sb[1][k],
                            self.cssb[1][k / 3].get_by_index(k % 3),
                            stmp * c[16],
                        );
                        self.sb[1][k] = tmp_sb;
                        self.cssb[1][k / 3].set_by_index(k % 3, tmp_cssb);
                        let (tmp_sb, tmp_cssb) = add_cs(
                            self.sb[2][k],
                            self.cssb[2][k / 3].get_by_index(k % 3),
                            stmp * c[17],
                        );
                        self.sb[2][k] = tmp_sb;
                        self.cssb[2][k / 3].set_by_index(k % 3, tmp_cssb);
                        let (tmp_sb, tmp_cssb) = add_cs(
                            self.sb[3][k],
                            self.cssb[3][k / 3].get_by_index(k % 3),
                            stmp * c[18],
                        );
                        self.sb[3][k] = tmp_sb;
                        self.cssb[3][k / 3].set_by_index(k % 3, tmp_cssb);
                        let (tmp_sb, tmp_cssb) = add_cs(
                            self.sb[4][k],
                            self.cssb[4][k / 3].get_by_index(k % 3),
                            stmp * c[19],
                        );
                        self.sb[4][k] = tmp_sb;
                        self.cssb[4][k / 3].set_by_index(k % 3, tmp_cssb);
                        let (tmp_sb, tmp_cssb) = add_cs(
                            self.sb[5][k],
                            self.cssb[5][k / 3].get_by_index(k % 3),
                            stmp * c[20],
                        );
                        self.sb[5][k] = tmp_sb;
                        self.cssb[5][k / 3].set_by_index(k % 3, tmp_cssb);
                        let (tmp_sb, tmp_cssb) =
                            add_cs(self.sb[6][k], self.cssb[6][k / 3].get_by_index(k % 3), stmp);
                        self.sb[6][k] = tmp_sb;
                        self.cssb[6][k / 3].set_by_index(k % 3, tmp_cssb);

                        // Monitor change in self.b[6][k] relative to self.accelerations[k]. The predictor corrector scheme is converged if it is close to 0.
                        if INTEGRATOR_EPSILON_GLOBAL {
                            // global error estimate (equation 9)
                            let dangular_momentum_dttk =
                                abs!(self.dangular_momentum_dtt[k / 3].get_by_index(k % 3));
                            if dangular_momentum_dttk.is_normal()
                                && dangular_momentum_dttk > max_dangular_momentum_dtk
                            {
                                max_dangular_momentum_dtk = dangular_momentum_dttk;
                            }
                            let b6kstmp = abs!(stmp); // change of b6ktmp coefficient
                            if b6kstmp.is_normal() && b6kstmp > maxb6kstmp {
                                maxb6kstmp = b6kstmp;
                            }
                        } else {
                            // local error estimate (equation 9)
                            let dangular_momentum_dttk =
                                self.dangular_momentum_dtt[k / 3].get_by_index(k % 3);
                            let b6kstmp = stmp;
                            let errorks = abs!(b6kstmp / dangular_momentum_dttk);
                            predictor_corrector_error =
                                if errorks.is_normal() && errorks > predictor_corrector_error {
                                    errorks
                                } else {
                                    0.0
                                };
                        }
                    }
                }
                _ => unreachable!(),
            } // end match
        } // end if integrate_spin

        if INTEGRATOR_EPSILON_GLOBAL && n == 7 {
            predictor_corrector_error =
                max!(maxb6ktmp, maxb6kstmp) / max!(maxak, max_dangular_momentum_dtk);
        }

        predictor_corrector_error
    }

    // Find new position and velocity values at end of the sequence
    // (Eqs. 11, 12 of Everhart)
    fn find_new_position_and_velocity(&mut self, dt_done: f64) {
        let dt_done2 = dt_done * dt_done;
        for i in 0..self.n_particles {
            {
                let k0 = 3 * i;
                let k1 = 3 * i + 1;
                let k2 = 3 * i + 2;
                let mut tmp = Axes::new();
                let mut tmp_cs = Axes::new();

                // position x
                (tmp.x, tmp_cs.x) = add_cs(
                    self.positions0[i].x,
                    self.cs_positions[i].x,
                    self.b[6][k0] / 72. * dt_done2,
                );

                // position y
                (tmp.y, tmp_cs.y) = add_cs(
                    self.positions0[i].y,
                    self.cs_positions[i].y,
                    self.b[6][k1] / 72. * dt_done2,
                );

                // position z
                (tmp.z, tmp_cs.z) = add_cs(
                    self.positions0[i].z,
                    self.cs_positions[i].z,
                    self.b[6][k2] / 72. * dt_done2,
                );

                for (i, j, f) in izip!((0..=5).rev(), (2..=7).rev(), (3..=8).rev()) {
                    (tmp.x, tmp_cs.x) =
                        add_cs(tmp.x, tmp_cs.x, self.b[i][k0] / (j * f) as f64 * dt_done2);

                    (tmp.y, tmp_cs.y) =
                        add_cs(tmp.y, tmp_cs.y, self.b[i][k1] / (j * f) as f64 * dt_done2);

                    (tmp.z, tmp_cs.z) =
                        add_cs(tmp.z, tmp_cs.z, self.b[i][k2] / (j * f) as f64 * dt_done2);
                }

                (tmp.x, tmp_cs.x) = add_cs(
                    tmp.x,
                    tmp_cs.x,
                    self.inertial_accelerations0[i].x / 2. * dt_done2,
                );
                (tmp.x, tmp_cs.x) = add_cs(tmp.x, tmp_cs.x, self.velocities0[i].x * dt_done);
                self.cs_positions[i].x = tmp_cs.x;
                self.positions0[i].x = tmp.x;

                (tmp.y, tmp_cs.y) = add_cs(
                    tmp.y,
                    tmp_cs.y,
                    self.inertial_accelerations0[i].y / 2. * dt_done2,
                );
                (tmp.y, tmp_cs.y) = add_cs(tmp.y, tmp_cs.y, self.velocities0[i].y * dt_done);
                self.cs_positions[i].y = tmp_cs.y;
                self.positions0[i].y = tmp.y;

                (tmp.z, tmp_cs.z) = add_cs(
                    tmp.z,
                    tmp_cs.z,
                    self.inertial_accelerations0[i].z / 2. * dt_done2,
                );
                (tmp.z, tmp_cs.z) = add_cs(tmp.z, tmp_cs.z, self.velocities0[i].z * dt_done);
                self.cs_positions[i].z = tmp_cs.z;
                self.positions0[i].z = tmp.z;

                // velocity x
                (tmp.x, tmp_cs.x) = add_cs(
                    self.velocities0[i].x,
                    self.cs_velocities[i].x,
                    self.b[6][k0] / 8. * dt_done,
                );
                // velocity y
                (tmp.y, tmp_cs.y) = add_cs(
                    self.velocities0[i].y,
                    self.cs_velocities[i].y,
                    self.b[6][k1] / 8. * dt_done,
                );

                // velocity z
                (tmp.z, tmp_cs.z) = add_cs(
                    self.velocities0[i].z,
                    self.cs_velocities[i].z,
                    self.b[6][k2] / 8. * dt_done,
                );

                for (i, j) in izip!((0..=5).rev(), (2..=7).rev()) {
                    (tmp.x, tmp_cs.x) =
                        add_cs(tmp.x, tmp_cs.x, self.b[i][k0] / (j as f64) * dt_done);
                    (tmp.y, tmp_cs.y) =
                        add_cs(tmp.y, tmp_cs.y, self.b[i][k1] / (j as f64) * dt_done);
                    (tmp.z, tmp_cs.z) =
                        add_cs(tmp.z, tmp_cs.z, self.b[i][k2] / (j as f64) * dt_done);
                }

                (tmp.x, tmp_cs.x) =
                    add_cs(tmp.x, tmp_cs.x, self.inertial_accelerations0[i].x * dt_done);
                self.cs_velocities[i].x = tmp_cs.x;
                self.velocities0[i].x = tmp.x;

                (tmp.y, tmp_cs.y) =
                    add_cs(tmp.y, tmp_cs.y, self.inertial_accelerations0[i].y * dt_done);
                self.cs_velocities[i].y = tmp_cs.y;
                self.velocities0[i].y = tmp.y;

                (tmp.z, tmp_cs.z) =
                    add_cs(tmp.z, tmp_cs.z, self.inertial_accelerations0[i].z * dt_done);
                self.cs_velocities[i].z = tmp_cs.z;
                self.velocities0[i].z = tmp.z;

                if self.integrate_spin {
                    // spin x
                    (tmp.x, tmp_cs.x) = add_cs(
                        self.angular_momentum0[i].x,
                        self.css[i].x,
                        self.sb[6][k0] / 8. * dt_done,
                    );

                    // spin y
                    (tmp.y, tmp_cs.y) = add_cs(
                        self.angular_momentum0[i].y,
                        self.css[i].y,
                        self.sb[6][k1] / 8. * dt_done,
                    );

                    // spin z
                    (tmp.z, tmp_cs.z) = add_cs(
                        self.angular_momentum0[i].z,
                        self.css[i].z,
                        self.sb[6][k2] / 8. * dt_done,
                    );

                    for (i, j) in izip!((0..=5).rev(), (2..=7).rev()) {
                        (tmp.x, tmp_cs.x) =
                            add_cs(tmp.x, tmp_cs.x, self.sb[i][k0] / (j as f64) * dt_done);
                        (tmp.y, tmp_cs.y) =
                            add_cs(tmp.y, tmp_cs.y, self.sb[i][k1] / (j as f64) * dt_done);
                        (tmp.z, tmp_cs.z) =
                            add_cs(tmp.z, tmp_cs.z, self.sb[i][k2] / (j as f64) * dt_done);
                    }

                    (tmp.x, tmp_cs.x) =
                        add_cs(tmp.x, tmp_cs.x, self.dangular_momentum_dt0[i].x * dt_done);
                    self.css[i].x = tmp_cs.x;
                    self.angular_momentum0[i].x = tmp.x;

                    (tmp.y, tmp_cs.y) =
                        add_cs(tmp.y, tmp_cs.y, self.dangular_momentum_dt0[i].y * dt_done);
                    self.css[i].y = tmp_cs.y;
                    self.angular_momentum0[i].y = tmp.y;

                    (tmp.z, tmp_cs.z) =
                        add_cs(tmp.z, tmp_cs.z, self.dangular_momentum_dt0[i].z * dt_done);
                    self.css[i].z = tmp_cs.z;
                    self.angular_momentum0[i].z = tmp.z;
                }
            }
        }
    }

    fn calculate_new_timestep(&mut self) -> Option<f64> {
        let dt_done = self.time_step;
        if INTEGRATOR_EPSILON > 0. {
            // Estimate error (given by last term in series expansion)
            // There are two options:
            // INTEGRATOR_EPSILON_GLOBAL==true  (default)
            //   First, we determine the maximum acceleration and the maximum of the last term in the series.
            //   Then, the two are divided.
            // INTEGRATOR_EPSILON_GLOBAL==false
            //   Here, the fractional error is calculated for each particle individually and we use the maximum of the fractional error.
            //   This might fail in cases where a particle does not experience any (physical) acceleration besides roundoff errors.
            let mut integrator_error = 0.0;
            if INTEGRATOR_EPSILON_GLOBAL {
                let mut maxak = 0.0;
                let mut maxb6k = 0.0;
                let mut max_dangular_momentum_dtk = 0.0;
                let mut maxb6ks = 0.0;
                // Looping over all particles and all 3 components of the acceleration.
                for (i, particle) in self.universe.particles[..self.universe.n_particles]
                    .iter()
                    .enumerate()
                {
                    let v2 = particle.inertial_velocity.dot(&particle.inertial_velocity);
                    let x2 = particle.inertial_position.dot(&particle.inertial_position);
                    if self.integrate_spin {
                        let dangular_momentum_dt2 = particle
                            .dangular_momentum_dt
                            .dot(&particle.dangular_momentum_dt);
                        let angular_momentum2 =
                            particle.angular_momentum.dot(&particle.angular_momentum);

                        // Skip slowly varying accelerations and angular momentums (spin)
                        if abs!(v2 * self.time_step * self.time_step / x2) < 1e-16
                            && abs!(
                                dangular_momentum_dt2 * self.time_step * self.time_step
                                    / angular_momentum2
                            ) < 1e-16
                        {
                            continue;
                        }
                    } else {
                        // Skip slowly varying accelerations
                        if abs!(v2 * self.time_step * self.time_step / x2) < 1e-16 {
                            continue;
                        }
                    }

                    for k in 3 * i..3 * (i + 1) {
                        let ak = abs!(self.accelerations[k / 3].get_by_index(k % 3));
                        if ak.is_normal() && ak > maxak {
                            maxak = ak;
                        }
                        let b6k = abs!(self.b[6][k]);
                        if b6k.is_normal() && b6k > maxb6k {
                            maxb6k = b6k;
                        }
                        //
                        if self.integrate_spin {
                            let dangular_momentum_dttk =
                                abs!(self.dangular_momentum_dtt[k / 3].get_by_index(k % 3));
                            if dangular_momentum_dttk.is_normal() && dangular_momentum_dttk > maxak
                            {
                                max_dangular_momentum_dtk = dangular_momentum_dttk;
                            }
                            let b6ks = abs!(self.sb[6][k]);
                            if b6ks.is_normal() && b6ks > maxb6ks {
                                maxb6ks = b6ks;
                            }
                        }
                    }
                }
                integrator_error = max!(maxb6k, maxb6ks) / max!(maxak, max_dangular_momentum_dtk);
            } else {
                for k in 0..3 * self.n_particles {
                    let mut integrator_error_a = 0.;
                    let mut integrator_error_s = 0.;
                    let ak = self.accelerations[k / 3].get_by_index(k % 3);
                    let b6k = self.b[6][k];
                    let errork = abs!(b6k / ak);
                    if errork.is_normal() && errork > integrator_error {
                        integrator_error_a = errork;
                    }
                    //
                    if self.integrate_spin {
                        let dangular_momentum_dttk =
                            self.dangular_momentum_dtt[k / 3].get_by_index(k % 3);
                        let b6ks = self.sb[6][k];
                        let errorks = abs!(b6ks / dangular_momentum_dttk);
                        if errorks.is_normal() && errorks > integrator_error {
                            integrator_error_s = errorks;
                        }
                    }
                    integrator_error = max!(integrator_error_a, integrator_error_s);
                }
            }

            let mut dt_new;
            if integrator_error.is_normal() {
                // if error estimate is available, then increase by more educated guess
                //dt_new = (INTEGRATOR_EPSILON/integrator_error).powf(1./7.) * dt_done;
                dt_new = sqrt7(INTEGRATOR_EPSILON / integrator_error) * dt_done;
            } else {
                // In the rare case that the error estimate doesn't give a finite number (e.g. when all forces accidentally cancel up to machine precission).
                dt_new = dt_done / SAFETY_FACTOR; // by default, increase timestep a little
            }

            if abs!(dt_new) < INTEGRATOR_MIN_DT {
                dt_new = INTEGRATOR_MIN_DT.copysign(dt_new);
            }

            if INTEGRATOR_MAX_DT > 0. && abs!(dt_new) > INTEGRATOR_MAX_DT {
                dt_new = abs!(INTEGRATOR_MAX_DT);
            }

            if abs!(dt_new / dt_done) < SAFETY_FACTOR {
                // New timestep is significantly smaller.
                // Reset particles
                for (
                    particle,
                    position,
                    velocity,
                    angular_momentum,
                    inertial_acceleration,
                    dangular_momentum_dt0,
                ) in izip!(
                    &mut self.universe.particles[..self.universe.n_particles],
                    &self.positions0,
                    &self.velocities0,
                    &self.angular_momentum0,
                    &self.inertial_accelerations0,
                    &self.dangular_momentum_dt0,
                ) {
                    // Set initial values
                    particle.inertial_position = *position;
                    particle.inertial_velocity = *velocity;
                    particle.inertial_acceleration = *inertial_acceleration;

                    if self.integrate_spin {
                        particle.dangular_momentum_dt = *dangular_momentum_dt0;
                        particle.angular_momentum = *angular_momentum;
                    }
                }
                self.time_step = dt_new;
                if self.time_step_last_success != 0. {
                    // Do not predict next self.e/self.b values if this is the first time step.
                    let ratio = self.time_step / self.time_step_last_success;
                    self.predict_next_step(ratio);
                }

                return None; // Step rejected. Do again.
            }
            if abs!(dt_new / dt_done) > 1.0 {
                // New timestep is larger.
                if dt_new / dt_done > 1. / SAFETY_FACTOR {
                    // Don't increase the timestep by too much compared to the last one.
                    dt_new = dt_done / SAFETY_FACTOR;
                }
            }
            self.time_step = dt_new;
        }
        Some(dt_done)
    }

    fn predict_next_step(&mut self, ratio: f64) {
        let mut consider_dangular_momentum_dt_from_general_relativity = false;
        if self.universe.consider_effects.general_relativity
            && self.universe.general_relativity_implementation
                == GeneralRelativityImplementation::Kidder1995
        {
            consider_dangular_momentum_dt_from_general_relativity = true;
        }
        self.integrate_spin = self.universe.consider_effects.tides
            || self.universe.consider_effects.rotational_flattening
            || self.universe.consider_effects.evolution
            || consider_dangular_momentum_dt_from_general_relativity;
        if ratio > 20. {
            // Do not predict if stepsize increase is very large.
            self.e = [[0.; 3 * MAX_PARTICLES]; 7];
            self.b = [[0.; 3 * MAX_PARTICLES]; 7];
            if self.integrate_spin {
                self.se = [[0.; 3 * MAX_PARTICLES]; 7];
                self.sb = [[0.; 3 * MAX_PARTICLES]; 7];
            }
        } else {
            // Predict new B values to use at the start of the next sequence. The predicted
            // values from the last call are saved as E. The correction, BD, between the
            // actual and predicted values of B is applied in advance as a correction.
            //
            let q1 = ratio;
            let q2 = ratio.powi(2);
            let q3 = ratio.powi(3);
            let q4 = ratio.powi(4);
            let q5 = ratio.powi(5);
            let q6 = ratio.powi(6);
            let q7 = ratio.powi(7);

            let [e0, e1, e2, e3, e4, e5, e6] = &mut self.e;
            let [br0, br1, br2, br3, br4, br5, br6] = &self.br;

            for (
                e0_i,
                e1_i,
                e2_i,
                e3_i,
                e4_i,
                e5_i,
                e6_i,
                br0_i,
                br1_i,
                br2_i,
                br3_i,
                br4_i,
                br5_i,
                br6_i,
            ) in izip!(
                &mut *e0, &mut *e1, &mut *e2, &mut *e3, &mut *e4, &mut *e5, &mut *e6, br0, br1,
                br2, br3, br4, br5, br6,
            )
            .take(self.n_particles)
            {
                *e0_i = q1
                    * (br6_i * 7.0
                        + br5_i * 6.0
                        + br4_i * 5.0
                        + br3_i * 4.0
                        + br2_i * 3.0
                        + br1_i * 2.0
                        + br0_i);
                *e1_i = q2
                    * (br6_i * 21.0
                        + br5_i * 15.0
                        + br4_i * 10.0
                        + br3_i * 6.0
                        + br2_i * 3.0
                        + br1_i);
                *e2_i = q3 * (br6_i * 35.0 + br5_i * 20.0 + br4_i * 10.0 + br3_i * 4.0 + br2_i);
                *e3_i = q4 * (br6_i * 35.0 + br5_i * 15.0 + br4_i * 5.0 + br3_i);
                *e4_i = q5 * (br6_i * 21.0 + br5_i * 5.0 + br4_i);
                *e5_i = q6 * (br6_i * 7.0 + br5_i);
                *e6_i = q7 * br6_i;
            }

            if self.integrate_spin {
                let [se0, se1, se2, se3, se4, se5, se6] = &mut self.se;
                let [sbr0, sbr1, sbr2, sbr3, sbr4, sbr5, sbr6] = &self.sbr;
                // Estimate B values for the next sequence (Eqs. 13 of Everhart).
                for (
                    se0_i,
                    se1_i,
                    se2_i,
                    se3_i,
                    se4_i,
                    se5_i,
                    se6_i,
                    sbr0_i,
                    sbr1_i,
                    sbr2_i,
                    sbr3_i,
                    sbr4_i,
                    sbr5_i,
                    sbr6_i,
                ) in izip!(
                    &mut *se0, &mut *se1, &mut *se2, &mut *se3, &mut *se4, &mut *se5, se6, sbr0,
                    sbr1, sbr2, sbr3, sbr4, sbr5, sbr6,
                )
                .take(self.n_particles)
                {
                    *se0_i = q1
                        * (sbr6_i * 7.0
                            + sbr5_i * 6.0
                            + sbr4_i * 5.0
                            + sbr3_i * 4.0
                            + sbr2_i * 3.0
                            + sbr1_i * 2.0
                            + sbr0_i);
                    *se1_i = q2
                        * (sbr6_i * 21.0
                            + sbr5_i * 15.0
                            + sbr4_i * 10.0
                            + sbr3_i * 6.0
                            + sbr2_i * 3.0
                            + sbr1_i);
                    *se2_i = q3
                        * (sbr6_i * 35.0 + sbr5_i * 20.0 + sbr4_i * 10.0 + sbr3_i * 4.0 + sbr2_i);
                    *se3_i = q4 * (sbr6_i * 35.0 + sbr5_i * 15.0 + sbr4_i * 5.0 + sbr3_i);
                    *se4_i = q5 * (sbr6_i * 21.0 + sbr5_i * 5.0 + sbr4_i);
                    *se5_i = q6 * (sbr6_i * 7.0 + sbr5_i);
                    *se6_i = q7 * sbr6_i;
                }
            }

            for i in 0..7 {
                for (b_i, e_i, br_i, er_i) in
                    izip!(&mut self.b[i], &self.e[i], &self.br[i], &self.er[i])
                        .take(self.n_particles)
                {
                    *b_i = e_i + br_i - er_i;
                }
            }

            if self.integrate_spin {
                for i in 0..7 {
                    for (sb_i, se_i, sbr_i, ser_i) in
                        izip!(&mut self.sb[i], &self.se[i], &self.sbr[i], &self.ser[i])
                            .take(self.n_particles)
                    {
                        *sb_i = se_i + sbr_i - ser_i;
                    }
                }
            }
        }
    }
}

fn sqrt7(a: f64) -> f64 {
    // Machine independent implementation of powf(1./7.)
    let mut x = 1.;
    for _k in 0..20 {
        // A smaller number should be ok too.
        let x6 = x * x * x * x * x * x;
        x += (a / x6 - x) / 7.;
    }
    x
}

fn add_cs(p: f64, csp: f64, inp: f64) -> (f64, f64) {
    let y = inp - csp;
    let new_p = p + y;
    let new_csp = (new_p - p) - y;
    (new_p, new_csp)
}
