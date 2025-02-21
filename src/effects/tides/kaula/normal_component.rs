use super::super::super::super::Particle;
use super::super::super::super::constants::{DAY, G, TWO_PI};
use super::{
    calculate_kaula_numbers, calculate_tidal_excitation_frequency_mode_sigma_2mpq,
    compute_phase_alpha_1, compute_phase_alpha_2, compute_phase_beta, select_eccentricty_order_q,
};
use crate::{TidalModel, TidesEffect};
use itertools::izip;

pub fn calculate_normal_component_of_the_tidal_force_3d(
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

    // Keplerian elements
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

    // Planetary spin in [rad.s^-1]
    let spin = sqrt!(particle.norm_spin_vector_2) / DAY;

    // 3D case
    let heliocentric_r;
    let particle_star_2;
    let particle_radius_5;

    if central_body {
        heliocentric_r = tidal_host_particle.heliocentric_distance;
        particle_star_2 = particle.mass.powi(2);
        particle_radius_5 = tidal_host_particle.radius.powi(5);
    } else {
        // inclination = tools::calculate_inclination_orbital_equatorial_plane(particle.heliocentric_position, particle.heliocentric_velocity, particle.spin);
        heliocentric_r = particle.heliocentric_distance;
        particle_star_2 = tidal_host_particle.mass.powi(2);
        particle_radius_5 = particle.radius.powi(5);
    }

    let heliocentric_varphi = 0.;
    let spin_angle = 0.;
    let semi_major_axis_6 = semi_major_axis.powi(6);

    let order = select_eccentricty_order_q(eccentricity);
    let q_min = 7 - order;
    let q_max = 2 * order + 1;

    let mut sum_over_p = 0.;
    let mut sum_over_p_s = 0.;
    for (p, f_20p) in kaula_param
        .polynomials
        .inclination_function_f_20p
        .iter()
        .enumerate()
    {
        let mut sum_over_q = 0.;
        let mut sum_over_q_s = 0.;
        let sum_g_2pq = kaula_param.polynomials.eccentricity_function_g_2pq[p];

        for (q, g_2pq) in sum_g_2pq.iter().skip(q_min).take(q_max).enumerate() {
            let freq_20pq = calculate_tidal_excitation_frequency_mode_sigma_2mpq(
                0.,
                p as f64,
                q as f64 - order as f64,
                spin,
                orbital_frequency,
            );
            let mut sum_over_k = 0.;
            let mut sum_over_k_s = 0.;

            for (k, f_21k) in kaula_param
                .polynomials
                .inclination_function_f_21p
                .iter()
                .enumerate()
            {
                let mut sum_over_j = 0.;
                let mut sum_over_j_s = 0.;
                let sum_g_2kj = kaula_param.polynomials.eccentricity_function_g_2pq[k];

                for (j, g_2kj) in sum_g_2kj.iter().skip(q_min).take(q_max).enumerate() {
                    let phase_beta = compute_phase_beta(
                        p as f64,
                        q as f64 - order as f64,
                        k as f64,
                        j as f64 - order as f64,
                        mean_anomaly,
                        spin_angle,
                        argument_perihelion,
                        longitude_of_ascending_node,
                        heliocentric_varphi,
                    );
                    let phase_alpha_1 = compute_phase_alpha_1(
                        p as f64,
                        q as f64 - order as f64,
                        k as f64,
                        j as f64 - order as f64,
                        mean_anomaly,
                        spin_angle,
                        argument_perihelion,
                        longitude_of_ascending_node,
                        heliocentric_varphi,
                    );

                    let (rek2_20pq, imk2_20pq) =
                        calculate_kaula_numbers(freq_20pq, kaula_param, central_body, spin);

                    let cos_alpha_1 = cos!(phase_alpha_1);
                    let sin_alpha_1 = sin!(phase_alpha_1);
                    let cos_beta = cos!(phase_beta);
                    let sin_beta = sin!(phase_beta);

                    sum_over_j += g_2kj
                        * (0.5 * (cos_alpha_1 * rek2_20pq - sin_alpha_1 * imk2_20pq)
                            + 1.5 * (cos_beta * rek2_20pq - sin_beta * imk2_20pq));
                    sum_over_j_s += g_2kj * (0.5 * rek2_20pq + 1.5 * rek2_20pq);
                }
                sum_over_k += f_21k * sum_over_j;
                sum_over_k_s += f_21k * sum_over_j_s;
            }
            sum_over_q += g_2pq * sum_over_k;
            sum_over_q_s += g_2pq * sum_over_k_s;
        }
        sum_over_p += f_20p * sum_over_q;
        sum_over_p_s += f_20p * sum_over_q_s;
    }
    let term_m0 = sum_over_p;
    let term_m0_s = sum_over_p_s;

    sum_over_p = 0.;
    sum_over_p_s = 0.;
    for (p, f_21p) in kaula_param
        .polynomials
        .inclination_function_f_21p
        .iter()
        .enumerate()
    {
        let mut sum_over_q = 0.;
        let mut sum_over_q_s = 0.;
        let sum_g_2pq = kaula_param.polynomials.eccentricity_function_g_2pq[p];

        for (q, g_2pq) in sum_g_2pq.iter().skip(q_min).take(q_max).enumerate() {
            let freq_21pq = calculate_tidal_excitation_frequency_mode_sigma_2mpq(
                1.,
                p as f64,
                q as f64 - order as f64,
                spin,
                orbital_frequency,
            );

            let mut sum_over_k_m2 = 0.;
            let mut sum_over_k_m0 = 0.;
            let mut sum_over_k_m2_s = 0.;
            let mut sum_over_k_m0_s = 0.;

            for (k, (f_22k, f_20k)) in izip!(
                kaula_param.polynomials.inclination_function_f_22p,
                kaula_param.polynomials.inclination_function_f_20p
            )
            .enumerate()
            {
                // sum over p for each m
                let mut sum_over_j_p2 = 0.;
                let mut sum_over_j_p0 = 0.;
                let mut sum_over_j_p2_s = 0.;
                let mut sum_over_j_p0_s = 0.;
                let sum_g_2kj = kaula_param.polynomials.eccentricity_function_g_2pq[k];

                for (j, g_2kj) in sum_g_2kj.iter().skip(q_min).take(q_max).enumerate() {
                    let phase_alpha_1 = compute_phase_alpha_1(
                        p as f64,
                        q as f64 - order as f64,
                        k as f64,
                        j as f64 - order as f64,
                        mean_anomaly,
                        spin_angle,
                        argument_perihelion,
                        longitude_of_ascending_node,
                        heliocentric_varphi,
                    );
                    let phase_alpha_2 = compute_phase_alpha_2(
                        p as f64,
                        q as f64 - order as f64,
                        k as f64,
                        j as f64 - order as f64,
                        mean_anomaly,
                        spin_angle,
                        argument_perihelion,
                        longitude_of_ascending_node,
                        heliocentric_varphi,
                    );

                    let (rek2_21pq, imk2_21pq) =
                        calculate_kaula_numbers(freq_21pq, kaula_param, central_body, spin);

                    let cos_alpha_1 = cos!(phase_alpha_1);
                    let sin_alpha_1 = sin!(phase_alpha_1);
                    let cos_alpha_2 = cos!(phase_alpha_2);
                    let sin_alpha_2 = sin!(phase_alpha_2);

                    sum_over_j_p2 += g_2kj * (cos_alpha_1 * rek2_21pq - sin_alpha_1 * imk2_21pq);
                    sum_over_j_p0 += g_2kj * (cos_alpha_2 * rek2_21pq - sin_alpha_2 * imk2_21pq);
                    sum_over_j_p2_s += g_2kj * rek2_21pq;
                    sum_over_j_p0_s += g_2kj * rek2_21pq;
                }
                sum_over_k_m2 += f_22k * sum_over_j_p2;
                sum_over_k_m0 += f_20k * sum_over_j_p0;
                sum_over_k_m2_s += f_22k * sum_over_j_p2_s;
                sum_over_k_m0_s += f_20k * sum_over_j_p0_s;
            }
            sum_over_q += g_2pq * (0.5 * sum_over_k_m2 - 3. * sum_over_k_m0);
            sum_over_q_s += g_2pq * (0.5 * sum_over_k_m2_s - 3. * sum_over_k_m0_s);
        }
        sum_over_p += f_21p * sum_over_q;
        sum_over_p_s += f_21p * sum_over_q_s;
    }
    let term_m1 = (1. / 3.) * sum_over_p;
    let term_m1_s = (1. / 3.) * sum_over_p_s;

    sum_over_p = 0.;
    sum_over_p_s = 0.;
    for (p, f_22p) in kaula_param
        .polynomials
        .inclination_function_f_22p
        .iter()
        .enumerate()
    {
        let mut sum_over_q = 0.;
        let mut sum_over_q_s = 0.;
        let sum_g_2pq = kaula_param.polynomials.eccentricity_function_g_2pq[p];

        for (q, g_2pq) in sum_g_2pq.iter().skip(q_min).take(q_max).enumerate() {
            let freq_22pq = calculate_tidal_excitation_frequency_mode_sigma_2mpq(
                2.,
                p as f64,
                q as f64 - order as f64,
                spin,
                orbital_frequency,
            );

            let mut sum_over_k = 0.;
            let mut sum_over_k_s = 0.;

            for (k, f_21k) in kaula_param
                .polynomials
                .inclination_function_f_21p
                .iter()
                .enumerate()
            {
                let mut sum_over_j = 0.;
                let mut sum_over_j_s = 0.;

                let sum_g_2kj = kaula_param.polynomials.eccentricity_function_g_2pq[k];

                for (j, g_2kj) in sum_g_2kj.iter().skip(q_min).take(q_max).enumerate() {
                    let phase_alpha_2 = compute_phase_alpha_2(
                        p as f64,
                        q as f64 - order as f64,
                        k as f64,
                        j as f64 - order as f64,
                        mean_anomaly,
                        spin_angle,
                        argument_perihelion,
                        longitude_of_ascending_node,
                        heliocentric_varphi,
                    );

                    let (rek2_22pq, imk2_22pq) =
                        calculate_kaula_numbers(freq_22pq, kaula_param, central_body, spin);

                    let cos_alpha_2 = cos!(phase_alpha_2);
                    let sin_alpha_2 = sin!(phase_alpha_2);

                    sum_over_j += g_2kj * (cos_alpha_2 * rek2_22pq - sin_alpha_2 * imk2_22pq);
                    sum_over_j_s += g_2kj * rek2_22pq;
                }
                sum_over_k += f_21k * sum_over_j;
                sum_over_k_s += f_21k * sum_over_j_s;
            }
            sum_over_q += g_2pq * sum_over_k;
            sum_over_q_s += g_2pq * sum_over_k_s;
        }
        sum_over_p += f_22p * sum_over_q;
        sum_over_p_s += f_22p * sum_over_q_s;
    }
    let term_m2 = -(1. / 6.) * sum_over_p;
    let term_m2_s = -(1. / 6.) * sum_over_p_s;

    let cste = (G * particle_star_2 * particle_radius_5) / (semi_major_axis_6 * heliocentric_r);

    let normal_force = cste * (term_m0 + term_m1 + term_m2);
    let normal_force_secular = cste * (term_m0_s + term_m1_s + term_m2_s);

    (normal_force, normal_force_secular)
}
