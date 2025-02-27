use super::{
    alpha_pqkj, calculate_2d_constant, calculate_base_constant, compute_phase_alpha_3,
    compute_phase_alpha_4, select_eccentricty_order_q,
};
use crate::Particle;
use crate::tools::KeplerianElements;

use itertools::izip;

// Ortho-radial (the e_{\varphi}) component of tidal force
pub fn calculate_orthogonal_component_of_the_tidal_force_2d(
    tidal_host_particle: &Particle,
    particle: &Particle,
    keplerian_elements: &KeplerianElements,
    central_body: bool,
) -> (f64, f64) {
    let kaula = particle.tides.get_kaula();

    // Keplerian elements
    let (
        semi_major_axis,
        _perihelion_distance,
        eccentricity,
        _inclination,
        longitude_perihelion,
        longitude_of_ascending_node,
        mean_anomaly,
        _orbital_period,
    ) = keplerian_elements.unpack();

    let cste_2d = if central_body {
        calculate_2d_constant(
            tidal_host_particle,
            particle,
            -tidal_host_particle.heliocentric_distance,
            semi_major_axis,
        )
    } else {
        calculate_2d_constant(
            particle,
            tidal_host_particle,
            particle.heliocentric_distance,
            semi_major_axis,
        )
    };

    let argument_perihelion = longitude_perihelion - longitude_of_ascending_node;

    // Cartesian angles
    // summation
    let orthogonal_force;
    let orthogonal_force_secular;

    if eccentricity == 0. {
        let imk2_2200 = kaula.love_numbers.imaginary(2, 0, 0);

        orthogonal_force = cste_2d * (3. / 2.) * imk2_2200;
        orthogonal_force_secular = orthogonal_force;
    } else {
        let mut sum_over_q = 0.;
        let mut sum_over_q_secular = 0.;
        let (q_min, q_max) = select_eccentricty_order_q(eccentricity);

        // for q in -7..=7{
        for (q, g_20q) in kaula.polynomials.eccentricity_function_g_2pq[0]
            .iter()
            .take(q_max)
            .skip(q_min)
            .enumerate()
        {
            let tmp_q = q_min + q;
            let rek2_220q = kaula.love_numbers.real(2, 0, tmp_q);
            let imk2_220q = kaula.love_numbers.imaginary(2, 0, tmp_q);

            let mut sum_over_j_2 = 0.;
            let mut sum_over_j_2_secular = 0.;
            for (j, g_20j) in kaula.polynomials.eccentricity_function_g_2pq[0]
                .iter()
                .take(q_max)
                .skip(q_min)
                .enumerate()
            {
                let alpha_qj = alpha_pqkj(
                    0,
                    q_min + q,
                    0,
                    q_min + j,
                    mean_anomaly,
                    argument_perihelion,
                );

                sum_over_j_2 += g_20j * (sin!(alpha_qj) * rek2_220q + cos!(alpha_qj) * imk2_220q);
                if q == j {
                    // secular part of the force
                    sum_over_j_2_secular += g_20j * imk2_220q;
                }
            }

            sum_over_q += g_20q * (3. / 2.) * sum_over_j_2;
            sum_over_q_secular += g_20q * (3. / 2.) * sum_over_j_2_secular;
        }
        orthogonal_force = cste_2d * sum_over_q;
        orthogonal_force_secular = cste_2d * sum_over_q_secular;
    }
    (orthogonal_force, orthogonal_force_secular)
}

// Ortho-radial (the e_{\varphi}) component of tidal force
pub fn calculate_orthogonal_component_of_the_tidal_force_3d(
    tidal_host_particle: &Particle,
    particle: &Particle,
    keplerian_elements: &KeplerianElements,
    central_body: bool,
) -> (f64, f64) {
    let kaula = particle.tides.get_kaula();
    // Keplerian elements
    let (
        semi_major_axis,
        _perihelion_distance,
        eccentricity,
        _inclination,
        longitude_perihelion,
        longitude_of_ascending_node,
        mean_anomaly,
        _orbital_period,
    ) = keplerian_elements.unpack();

    let argument_perihelion = longitude_perihelion - longitude_of_ascending_node;
    // Cartesian angles
    // summation
    let heliocentric_varphi = 0.;
    let spin_angle = 0.;

    let (q_min, q_max) = select_eccentricty_order_q(eccentricity);
    let mut sum_over_p;
    let mut sum_over_p_s;

    sum_over_p = 0.;
    sum_over_p_s = 0.;
    let c1 = 1. / 6.;
    for (p, f_21p) in kaula.polynomials.inclination_function_f_2mp[1]
        .iter()
        .enumerate()
    {
        let mut sum_over_q = 0.;
        let mut sum_over_q_s = 0.;
        let sum_g_2pq = kaula.polynomials.eccentricity_function_g_2pq[p];

        for (q, g_2pq) in sum_g_2pq.iter().take(q_max).skip(q_min).enumerate() {
            let tmp_q = q_min + q;
            let rek2_21pq = kaula.love_numbers.real(1, p, tmp_q);
            let imk2_21pq = kaula.love_numbers.imaginary(1, p, tmp_q);

            let mut sum_over_kterm1 = 0.;
            let mut sum_over_kterm2 = 0.;
            let mut sum_over_kterm1_s = 0.;
            let mut sum_over_kterm2_s = 0.;

            for (k, (f_30k, f_32k)) in izip!(
                kaula.polynomials.inclination_function_f_3mp[0],
                kaula.polynomials.inclination_function_f_3mp[2],
            )
            .enumerate()
            {
                // Sum over p for each m
                let mut sum_over_jterm1 = 0.;
                let mut sum_over_jterm2 = 0.;
                let mut sum_over_jterm1_s = 0.;
                let mut sum_over_jterm2_s = 0.;
                let sum_g_3kj = kaula.polynomials.eccentricity_function_g_3pq[k];

                for (j, g_3kj) in sum_g_3kj.iter().take(q_max).skip(q_min).enumerate() {
                    let phase_alpha_3 = compute_phase_alpha_3(
                        p,
                        q_min + q,
                        k,
                        q_min + j,
                        mean_anomaly,
                        spin_angle,
                        argument_perihelion,
                        longitude_of_ascending_node,
                        heliocentric_varphi,
                    );
                    let phase_alpha_4 = compute_phase_alpha_4(
                        p,
                        q_min + q,
                        k,
                        q_min + j,
                        mean_anomaly,
                        spin_angle,
                        argument_perihelion,
                        longitude_of_ascending_node,
                        heliocentric_varphi,
                    );

                    let cos_alpha_1 = cos!(phase_alpha_3);
                    let sin_alpha_1 = sin!(phase_alpha_3);
                    let cos_alpha_2 = cos!(phase_alpha_4);
                    let sin_alpha_2 = sin!(phase_alpha_4);

                    sum_over_jterm1 += g_3kj * (sin_alpha_1 * rek2_21pq - cos_alpha_1 * imk2_21pq);
                    sum_over_jterm2 += g_3kj * (sin_alpha_2 * rek2_21pq - cos_alpha_2 * imk2_21pq);
                    sum_over_jterm1_s -= g_3kj * imk2_21pq;
                    sum_over_jterm2_s -= g_3kj * imk2_21pq;
                }
                sum_over_kterm1 += f_32k * sum_over_jterm1;
                sum_over_kterm2 += f_30k * sum_over_jterm2;
                sum_over_kterm1_s += f_32k * sum_over_jterm1_s;
                sum_over_kterm2_s += f_30k * sum_over_jterm2_s;
            }
            sum_over_q += g_2pq * (c1 * sum_over_kterm1 + sum_over_kterm2);
            sum_over_q_s += g_2pq * (c1 * sum_over_kterm1_s + sum_over_kterm2_s);
        }
        sum_over_p += f_21p * sum_over_q;
        sum_over_p_s += f_21p * sum_over_q_s;
    }
    let term_m1 = -4. / sqrt!(15_f64) * sum_over_p;
    let term_m1_s = -4. / sqrt!(15_f64) * sum_over_p_s;

    sum_over_p = 0.;
    sum_over_p_s = 0.;
    for (p, f_22p) in kaula.polynomials.inclination_function_f_2mp[2]
        .iter()
        .enumerate()
    {
        let mut sum_over_q = 0.;
        let mut sum_over_q_s = 0.;
        let sum_g_2pq = kaula.polynomials.eccentricity_function_g_2pq[p];

        for (q, g_2pq) in sum_g_2pq.iter().take(q_max).skip(q_min).enumerate() {
            let tmp_q = q_min + q;
            let rek2_22pq = kaula.love_numbers.real(2, p, tmp_q);
            let imk2_22pq = kaula.love_numbers.imaginary(2, p, tmp_q);

            let mut sum_over_kterm1 = 0.;
            let mut sum_over_kterm2 = 0.;
            let mut sum_over_kterm1_s = 0.;
            let mut sum_over_kterm2_s = 0.;

            for (k, (f_31k, f_33k)) in izip!(
                kaula.polynomials.inclination_function_f_3mp[1],
                kaula.polynomials.inclination_function_f_3mp[3]
            )
            .enumerate()
            {
                // Sum over p for each m
                let mut sum_over_jterm1 = 0.;
                let mut sum_over_jterm2 = 0.;
                let mut sum_over_jterm1_s = 0.;
                let mut sum_over_jterm2_s = 0.;
                let sum_g_3kj = kaula.polynomials.eccentricity_function_g_3pq[k];

                for (j, g_3kj) in sum_g_3kj.iter().take(q_max).skip(q_min).enumerate() {
                    let phase_alpha_3 = compute_phase_alpha_3(
                        p,
                        q_min + q,
                        k,
                        q_min + j,
                        mean_anomaly,
                        spin_angle,
                        argument_perihelion,
                        longitude_of_ascending_node,
                        heliocentric_varphi,
                    );
                    let phase_alpha_4 = compute_phase_alpha_4(
                        p,
                        q_min + q,
                        k,
                        q_min + j,
                        mean_anomaly,
                        spin_angle,
                        argument_perihelion,
                        longitude_of_ascending_node,
                        heliocentric_varphi,
                    );

                    let cos_alpha_1 = cos!(phase_alpha_3);
                    let sin_alpha_1 = sin!(phase_alpha_3);
                    let cos_alpha_2 = cos!(phase_alpha_4);
                    let sin_alpha_2 = sin!(phase_alpha_4);

                    sum_over_jterm1 += g_3kj * (sin_alpha_1 * rek2_22pq - cos_alpha_1 * imk2_22pq);
                    sum_over_jterm2 += g_3kj * (sin_alpha_2 * rek2_22pq - cos_alpha_2 * imk2_22pq);
                    sum_over_jterm1_s -= g_3kj * imk2_22pq;
                    sum_over_jterm2_s -= g_3kj * imk2_22pq;
                }
                sum_over_kterm1 += f_33k * sum_over_jterm1;
                sum_over_kterm2 += f_31k * sum_over_jterm2;
                sum_over_kterm1_s += f_33k * sum_over_jterm1_s;
                sum_over_kterm2_s += f_31k * sum_over_jterm2_s;
            }
            sum_over_q += g_2pq * (sum_over_kterm1 + 2. * sum_over_kterm2);
            sum_over_q_s += g_2pq * (sum_over_kterm1_s + 2. * sum_over_kterm2_s);
        }
        sum_over_p += f_22p * sum_over_q;
        sum_over_p_s += f_22p * sum_over_q_s;
    }
    let term_m2 = -5. / (48. * sqrt!(6_f64)) * sum_over_p;
    let term_m2_s = -5. / (48. * sqrt!(6_f64)) * sum_over_p_s;

    let cste_3d = if central_body {
        calculate_base_constant(tidal_host_particle, particle, 1.0, semi_major_axis)
    } else {
        calculate_base_constant(particle, tidal_host_particle, 1.0, semi_major_axis)
    };

    let orthogonal_force = cste_3d * (term_m1 + term_m2);
    let orthogonal_force_secular = cste_3d * (term_m1_s + term_m2_s);
    (orthogonal_force, orthogonal_force_secular)
}
