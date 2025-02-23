use super::super::super::super::Particle;
use super::{
    calculate_base_constant, compute_phase_alpha_1, compute_phase_alpha_2, compute_phase_beta,
    select_eccentricty_order_q,
};
use itertools::izip;

pub fn calculate_normal_component_of_the_tidal_force_3d(
    tidal_host_particle: &Particle,
    particle: &Particle,
    keplerian_elements: (f64, f64, f64, f64, f64, f64, f64, f64),
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
    ) = keplerian_elements;
    let argument_perihelion = longitude_perihelion - longitude_of_ascending_node;

    // 3D case
    let heliocentric_varphi = 0.;
    let spin_angle = 0.;

    let order = select_eccentricty_order_q(eccentricity);
    let q_min = 7 - order;
    let q_max = 2 * order + 1;

    let mut sum_over_p = 0.;
    let mut sum_over_p_s = 0.;
    for (p, f_20p) in kaula.polynomials.inclination_function_f_2mp[0]
        .iter()
        .enumerate()
    {
        let mut sum_over_q = 0.;
        let mut sum_over_q_s = 0.;
        let sum_g_2pq = kaula.polynomials.eccentricity_function_g_2pq[p];

        for (q, g_2pq) in sum_g_2pq.iter().skip(q_min).take(q_max).enumerate() {
            let tmp_q = q + 7 - order;
            let rek2_20pq = kaula.love_numbers.real(0, p, tmp_q);
            let imk2_20pq = kaula.love_numbers.imaginary(0, p, tmp_q);

            let mut sum_over_k = 0.;
            let mut sum_over_k_s = 0.;

            for (k, f_21k) in kaula.polynomials.inclination_function_f_2mp[1]
                .iter()
                .enumerate()
            {
                let mut sum_over_j = 0.;
                let mut sum_over_j_s = 0.;
                let sum_g_2kj = kaula.polynomials.eccentricity_function_g_2pq[k];

                for (j, g_2kj) in sum_g_2kj.iter().skip(q_min).take(q_max).enumerate() {
                    let phase_beta = compute_phase_beta(
                        p as i32,
                        q as i32 - order as i32,
                        k as i32,
                        j as i32 - order as i32,
                        mean_anomaly,
                        spin_angle, // 0.0
                        argument_perihelion,
                        longitude_of_ascending_node,
                        heliocentric_varphi, // 0.0
                    );
                    let phase_alpha_1 = compute_phase_alpha_1(
                        p as i32,
                        q as i32 - order as i32,
                        k as i32,
                        j as i32 - order as i32,
                        mean_anomaly,
                        spin_angle, // 0.0
                        argument_perihelion,
                        longitude_of_ascending_node,
                        heliocentric_varphi, // 0.0
                    );

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
    for (p, f_21p) in kaula.polynomials.inclination_function_f_2mp[1]
        .iter()
        .enumerate()
    {
        let mut sum_over_q = 0.;
        let mut sum_over_q_s = 0.;
        let sum_g_2pq = kaula.polynomials.eccentricity_function_g_2pq[p];

        for (q, g_2pq) in sum_g_2pq.iter().skip(q_min).take(q_max).enumerate() {
            let tmp_q = q + 7 - order;
            let rek2_21pq = kaula.love_numbers.real(1, p, tmp_q);
            let imk2_21pq = kaula.love_numbers.imaginary(1, p, tmp_q);

            let mut sum_over_k_m2 = 0.;
            let mut sum_over_k_m0 = 0.;
            let mut sum_over_k_m2_s = 0.;
            let mut sum_over_k_m0_s = 0.;

            for (k, (f_20k, f_22k)) in izip!(
                kaula.polynomials.inclination_function_f_2mp[0],
                kaula.polynomials.inclination_function_f_2mp[2],
            )
            .enumerate()
            {
                // sum over p for each m
                let mut sum_over_j_p2 = 0.;
                let mut sum_over_j_p0 = 0.;
                let mut sum_over_j_p2_s = 0.;
                let mut sum_over_j_p0_s = 0.;
                let sum_g_2kj = kaula.polynomials.eccentricity_function_g_2pq[k];

                for (j, g_2kj) in sum_g_2kj.iter().skip(q_min).take(q_max).enumerate() {
                    let phase_alpha_1 = compute_phase_alpha_1(
                        p as i32,
                        q as i32 - order as i32,
                        k as i32,
                        j as i32 - order as i32,
                        mean_anomaly,
                        spin_angle,
                        argument_perihelion,
                        longitude_of_ascending_node,
                        heliocentric_varphi,
                    );
                    let phase_alpha_2 = compute_phase_alpha_2(
                        p as i32,
                        q as i32 - order as i32,
                        k as i32,
                        j as i32 - order as i32,
                        mean_anomaly,
                        spin_angle,
                        argument_perihelion,
                        longitude_of_ascending_node,
                        heliocentric_varphi,
                    );

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
    for (p, f_22p) in kaula.polynomials.inclination_function_f_2mp[2]
        .iter()
        .enumerate()
    {
        let mut sum_over_q = 0.;
        let mut sum_over_q_s = 0.;
        let sum_g_2pq = kaula.polynomials.eccentricity_function_g_2pq[p];

        for (q, g_2pq) in sum_g_2pq.iter().skip(q_min).take(q_max).enumerate() {
            let tmp_q = q + 7 - order;
            let rek2_22pq = kaula.love_numbers.real(2, p, tmp_q);
            let imk2_22pq = kaula.love_numbers.imaginary(2, p, tmp_q);

            let mut sum_over_k = 0.;
            let mut sum_over_k_s = 0.;

            for (k, f_21k) in kaula.polynomials.inclination_function_f_2mp[1]
                .iter()
                .enumerate()
            {
                let mut sum_over_j = 0.;
                let mut sum_over_j_s = 0.;

                let sum_g_2kj = kaula.polynomials.eccentricity_function_g_2pq[k];

                for (j, g_2kj) in sum_g_2kj.iter().skip(q_min).take(q_max).enumerate() {
                    let phase_alpha_2 = compute_phase_alpha_2(
                        p as i32,
                        q as i32 - order as i32,
                        k as i32,
                        j as i32 - order as i32,
                        mean_anomaly,
                        spin_angle,
                        argument_perihelion,
                        longitude_of_ascending_node,
                        heliocentric_varphi,
                    );

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

    let cste = if central_body {
        calculate_base_constant(
            tidal_host_particle,
            particle,
            tidal_host_particle.heliocentric_distance,
            semi_major_axis,
        )
    } else {
        calculate_base_constant(
            particle,
            tidal_host_particle,
            particle.heliocentric_distance,
            semi_major_axis,
        )
    };

    let normal_force = cste * (term_m0 + term_m1 + term_m2);
    let normal_force_secular = cste * (term_m0_s + term_m1_s + term_m2_s);

    (normal_force, normal_force_secular)
}
