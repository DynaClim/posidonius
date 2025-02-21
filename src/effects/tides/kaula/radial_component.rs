use super::super::super::super::Particle;
use super::super::super::super::constants::{DAY, G, TWO_PI};
use super::{
    alpha_pqkj, calculate_kaula_numbers, calculate_tidal_excitation_frequency_mode_sigma_2mpq,
    select_eccentricty_order_q,
};
use crate::{TidalModel, TidesEffect};
use itertools::izip;

pub fn calculate_radial_component_of_the_tidal_force_2d(
    tidal_host_particle: &Particle,
    particle: &mut Particle,
    keplerian_elements: (f64, f64, f64, f64, f64, f64, f64, f64),
    central_body: bool,
) -> (f64, f64) {
    let &mut (TidesEffect::CentralBody(TidalModel::Kaula(ref mut kaula_param))
    | TidesEffect::OrbitingBody(TidalModel::Kaula(ref mut kaula_param))) =
        &mut particle.tides.effect
    else {
        unreachable!()
    };

    // keplerian elements
    let (
        semi_major_axis,
        _perihelion_distance,
        eccentricity,
        _inclination,
        longitude_perihelion,
        longitude_of_ascending_node,
        mean_anomaly,
        orbital_period,
    ) = keplerian_elements;

    let argument_perihelion = longitude_perihelion - longitude_of_ascending_node;
    // Orbital mean motion in [rad.s^-1]
    let orbital_frequency = TWO_PI / (orbital_period * DAY);
    // Stellar/planetary spin in [rad.s^-1]
    let spin = sqrt!(particle.norm_spin_vector_2) / DAY;

    let tem_radius = if central_body {
        -tidal_host_particle.heliocentric_distance
    } else {
        particle.heliocentric_distance
    };

    let cste = -(G * tidal_host_particle.mass.powi(2) * particle.radius.powi(5))
        / (semi_major_axis.powi(6) * tem_radius);

    let radial_force;
    let radial_force_secular;
    // if the inclination btw the equatorial plane and the orbital plane is negligible
    if eccentricity == 0.0 {
        // If circular coplanar orbit
        let freq_2010 = calculate_tidal_excitation_frequency_mode_sigma_2mpq(
            0.,
            1.,
            0.,
            spin,
            orbital_frequency,
        );
        let freq_2200 = calculate_tidal_excitation_frequency_mode_sigma_2mpq(
            2.,
            0.,
            0.,
            spin,
            orbital_frequency,
        );
        let (rek2_2010, _imk2_2010) =
            calculate_kaula_numbers(freq_2010, kaula_param, central_body, spin);
        let (_rek2_2200, imk2_2200) =
            calculate_kaula_numbers(freq_2200, kaula_param, central_body, spin);
        radial_force = cste * ((3. / 4.) * rek2_2010 + (9. / 4.) * imk2_2200);
        radial_force_secular = radial_force;
    } else {
        // If eccentric orbit
        let mut sum_over_q = 0.;
        let mut sum_over_q_secular = 0.;

        let order = select_eccentricty_order_q(eccentricity);
        let q_min = 7 - order;
        let q_max = 2 * order + 1;

        for (q, g_201q) in kaula_param
            .polynomials
            .eccentricity_function_g_20q
            .iter()
            .zip(kaula_param.polynomials.eccentricity_function_g_21q.iter())
            .skip(q_min)
            .take(q_max)
            .enumerate()
        {
            let freq_201q = calculate_tidal_excitation_frequency_mode_sigma_2mpq(
                0.,
                1.,
                q as f64 - order as f64,
                spin,
                orbital_frequency,
            );
            let freq_220q = calculate_tidal_excitation_frequency_mode_sigma_2mpq(
                2.,
                0.,
                q as f64 - order as f64,
                spin,
                orbital_frequency,
            );
            let (rek2_201q, imk2_201q) =
                calculate_kaula_numbers(freq_201q, kaula_param, central_body, spin);
            let (rek2_220q, imk2_220q) =
                calculate_kaula_numbers(freq_220q, kaula_param, central_body, spin);

            let mut sum_over_j_1 = 0.;
            let mut sum_over_j_3 = 0.;

            let mut sum_over_j_1_secular = 0.;
            let mut sum_over_j_3_secular = 0.;

            for (j, g_201j) in kaula_param
                .polynomials
                .eccentricity_function_g_20q
                .iter()
                .zip(kaula_param.polynomials.eccentricity_function_g_21q.iter())
                .skip(q_min)
                .take(q_max)
                .enumerate()
            {
                let alpha_qj = alpha_pqkj(
                    0.,
                    q as f64 - order as f64,
                    0.,
                    j as f64 - order as f64,
                    mean_anomaly,
                    argument_perihelion,
                );
                let (g_20j, g_21j) = g_201j;

                sum_over_j_1 += g_21j * (cos!(alpha_qj) * rek2_201q - sin!(alpha_qj) * imk2_201q);
                sum_over_j_3 += g_20j * (cos!(alpha_qj) * rek2_220q - sin!(alpha_qj) * imk2_220q);
                if q == j {
                    sum_over_j_1_secular += g_21j * rek2_201q;
                    sum_over_j_3_secular += g_20j * rek2_220q;
                }
            }

            let (g_20q, g_21q) = g_201q;
            sum_over_q += (3. / 4.) * g_21q * sum_over_j_1 + (9. / 4.) * g_20q * sum_over_j_3;
            sum_over_q_secular +=
                (3. / 4.) * g_21q * sum_over_j_1_secular + (9. / 4.) * g_20q * sum_over_j_3_secular;
        }
        radial_force = cste * sum_over_q;
        radial_force_secular = cste * sum_over_q_secular;
    }
    (radial_force, radial_force_secular)
}

// radial (e_{r}) component of tidal force
pub fn calculate_radial_component_of_the_tidal_force_3d(
    tidal_host_particle: &Particle,
    particle: &mut Particle,
    keplerian_elements: (f64, f64, f64, f64, f64, f64, f64, f64),
    central_body: bool,
) -> (f64, f64) {
    let &mut (TidesEffect::CentralBody(TidalModel::Kaula(ref mut kaula_param))
    | TidesEffect::OrbitingBody(TidalModel::Kaula(ref mut kaula_param))) =
        &mut particle.tides.effect
    else {
        unreachable!()
    };

    // keplerian elements
    let (
        semi_major_axis,
        _perihelion_distance,
        eccentricity,
        _inclination,
        longitude_perihelion,
        longitude_of_ascending_node,
        mean_anomaly,
        orbital_period,
    ) = keplerian_elements;

    let argument_perihelion = longitude_perihelion - longitude_of_ascending_node;
    // Orbital mean motion in [rad.s^-1]
    let orbital_frequency = TWO_PI / (orbital_period * DAY);
    // Stellar/planetary spin in [rad.s^-1]
    let spin = sqrt!(particle.norm_spin_vector_2) / DAY;

    // if there is a non-negligible inclination (3D case)
    // Modification has been made but no testing has been done for 3D case (26/2)
    // let eccentricity_function_g_2pq = calculate_eccentricity_function_g_2pq(eccentricity);
    // let inclination_function_f_2mp = calculate_inclination_function_f_2mp(obliquity);
    let heliocentric_r;
    let particle_star_2;
    if central_body {
        // inclination = tools::calculate_inclination_orbital_equatorial_plane(tidal_host_particle.heliocentric_position, tidal_host_particle.heliocentric_velocity, tidal_host_particle.spin);
        heliocentric_r = tidal_host_particle.heliocentric_distance;
        particle_star_2 = particle.mass.powi(2);
    } else {
        // inclination = tools::calculate_inclination_orbital_equatorial_plane(particle.heliocentric_position, particle.heliocentric_velocity, particle.spin);
        heliocentric_r = particle.heliocentric_distance;
        particle_star_2 = tidal_host_particle.mass.powi(2);
    }

    let particle_radius_5 = particle.radius.powi(5);
    let semi_major_axis_6 = semi_major_axis.powi(6);

    let order = select_eccentricty_order_q(eccentricity);
    let q_min = 7 - order;
    let q_max = 2 * order + 1;

    let mut sum_over_p_m0 = 0.;
    let mut sum_over_p_m1 = 0.;
    let mut sum_over_p_m2 = 0.;
    let mut sum_over_p_m0_s = 0.;
    let mut sum_over_p_m1_s = 0.;
    let mut sum_over_p_m2_s = 0.;
    for (p, (f_20p, f_21p, f_22p)) in izip!(
        kaula_param.polynomials.inclination_function_f_20p,
        kaula_param.polynomials.inclination_function_f_21p,
        kaula_param.polynomials.inclination_function_f_22p
    )
    .enumerate()
    {
        let f_20p_2 = f_20p.powi(2);
        let f_21p_2 = f_21p.powi(2);
        let f_22p_2 = f_22p.powi(2);

        let mut sum_over_q_m0 = 0.;
        let mut sum_over_q_m1 = 0.;
        let mut sum_over_q_m2 = 0.;
        let mut sum_over_q_m0_s = 0.;
        let mut sum_over_q_m1_s = 0.;
        let mut sum_over_q_m2_s = 0.;

        let sum_g_2pq = kaula_param.polynomials.eccentricity_function_g_2pq[p];

        for (q, g_2pq) in sum_g_2pq.iter().skip(q_min).take(q_max).enumerate() {
            let g_2pq_2 = g_2pq.powi(2);

            let freq_20pq = calculate_tidal_excitation_frequency_mode_sigma_2mpq(
                0.,
                p as f64,
                q as f64 - order as f64,
                spin,
                orbital_frequency,
            );
            let freq_21pq = calculate_tidal_excitation_frequency_mode_sigma_2mpq(
                1.,
                p as f64,
                q as f64 - order as f64,
                spin,
                orbital_frequency,
            );
            let freq_22pq = calculate_tidal_excitation_frequency_mode_sigma_2mpq(
                2.,
                p as f64,
                q as f64 - order as f64,
                spin,
                orbital_frequency,
            );
            let (rek2_20pq, imk2_20pq) =
                calculate_kaula_numbers(freq_20pq, kaula_param, central_body, spin);
            let (rek2_21pq, imk2_21pq) =
                calculate_kaula_numbers(freq_21pq, kaula_param, central_body, spin);
            let (rek2_22pq, imk2_22pq) =
                calculate_kaula_numbers(freq_22pq, kaula_param, central_body, spin);

            let mut sum_over_k_m0 = 0.;
            let mut sum_over_k_m1 = 0.;
            let mut sum_over_k_m2 = 0.;

            for (k, f_2mk) in kaula_param
                .polynomials
                .inclination_function_f_20p
                .iter()
                .zip(
                    kaula_param
                        .polynomials
                        .inclination_function_f_21p
                        .iter()
                        .zip(kaula_param.polynomials.inclination_function_f_22p.iter()),
                )
                .enumerate()
            {
                let (f_20k, (f_21k, f_22k)) = f_2mk;

                let mut sum_over_j_m0 = 0.;
                let mut sum_over_j_m1 = 0.;
                let mut sum_over_j_m2 = 0.;
                let sum_g_2kj = kaula_param.polynomials.eccentricity_function_g_2pq[k];

                for (j, g_2kj) in sum_g_2kj.iter().skip(q_min).take(q_max).enumerate() {
                    let phase_alpha = alpha_pqkj(
                        p as f64,
                        q as f64 - order as f64,
                        k as f64,
                        j as f64 - order as f64,
                        mean_anomaly,
                        argument_perihelion,
                    );

                    let cos_alpha = cos!(phase_alpha);
                    let sin_alpha = sin!(phase_alpha);

                    sum_over_j_m0 += g_2kj * (cos_alpha * rek2_20pq - sin_alpha * imk2_20pq);
                    sum_over_j_m1 += g_2kj * (cos_alpha * rek2_21pq - sin_alpha * imk2_21pq);
                    sum_over_j_m2 += g_2kj * (cos_alpha * rek2_22pq - sin_alpha * imk2_22pq);
                }
                sum_over_k_m0 += f_20k * sum_over_j_m0;
                sum_over_k_m1 += f_21k * sum_over_j_m1;
                sum_over_k_m2 += f_22k * sum_over_j_m2;
            }
            sum_over_q_m0 += g_2pq * sum_over_k_m0;
            sum_over_q_m1 += g_2pq * sum_over_k_m1;
            sum_over_q_m2 += g_2pq * sum_over_k_m2;
            sum_over_q_m0_s += g_2pq_2 * rek2_20pq;
            sum_over_q_m1_s += g_2pq_2 * rek2_21pq;
            sum_over_q_m2_s += g_2pq_2 * rek2_22pq;
        }
        sum_over_p_m0 += f_20p * sum_over_q_m0;
        sum_over_p_m1 += f_21p * sum_over_q_m1;
        sum_over_p_m2 += f_22p * sum_over_q_m2;
        sum_over_p_m0_s += f_20p_2 * sum_over_q_m0_s;
        sum_over_p_m1_s += f_21p_2 * sum_over_q_m1_s;
        sum_over_p_m2_s += f_22p_2 * sum_over_q_m2_s;
    }

    let cste =
        (G * particle_star_2 * particle_radius_5 * 3.0) / (semi_major_axis_6 * heliocentric_r);
    let radial_force =
        (sum_over_p_m0 + (1. / 3.) * sum_over_p_m1 + (1. / 12.) * sum_over_p_m2) * cste;
    let radial_force_secular =
        (sum_over_p_m0_s + (1. / 3.) * sum_over_p_m1_s + (1. / 12.) * sum_over_p_m2_s) * cste;
    (radial_force, radial_force_secular)
}
