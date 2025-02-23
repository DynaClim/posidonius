use super::super::super::super::Particle;
use super::{alpha_pqkj, calculate_base_constant, select_eccentricty_order_q};
use crate::{TidalModel, TidesEffect};
use itertools::izip;

pub fn calculate_radial_component_of_the_tidal_force_2d(
    tidal_host_particle: &Particle,
    particle: &Particle,
    keplerian_elements: (f64, f64, f64, f64, f64, f64, f64, f64),
    central_body: bool,
) -> (f64, f64) {
    // TODO pass in kaula
    let &(TidesEffect::CentralBody(TidalModel::Kaula(ref kaula_param))
    | TidesEffect::OrbitingBody(TidalModel::Kaula(ref kaula_param))) = &particle.tides.effect
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
        _orbital_period,
    ) = keplerian_elements;

    let argument_perihelion = longitude_perihelion - longitude_of_ascending_node;

    let cste = if central_body {
        -calculate_base_constant(
            particle,
            tidal_host_particle,
            -tidal_host_particle.heliocentric_distance,
            semi_major_axis,
        )
    } else {
        -calculate_base_constant(
            particle,
            tidal_host_particle,
            particle.heliocentric_distance,
            semi_major_axis,
        )
    };

    let radial_force;
    let radial_force_secular;
    // if the inclination btw the equatorial plane and the orbital plane is negligible
    if eccentricity == 0.0 {
        // If circular coplanar orbit
        let rek2_2010 = kaula_param.love_numbers.real(0, 1, 0);
        let imk2_2200 = kaula_param.love_numbers.imaginary(2, 0, 0);

        radial_force = cste * ((3. / 4.) * rek2_2010 + (9. / 4.) * imk2_2200);
        radial_force_secular = radial_force;
    } else {
        // If eccentric orbit
        let mut sum_over_q = 0.;
        let mut sum_over_q_secular = 0.;

        let order = select_eccentricty_order_q(eccentricity);
        let q_min = 7 - order;
        let q_max = 2 * order + 1;

        for (q, (g_20q, g_21q)) in izip!(
            kaula_param.polynomials.eccentricity_function_g_2pq[0],
            kaula_param.polynomials.eccentricity_function_g_2pq[1],
        )
        .skip(q_min)
        .take(q_max)
        .enumerate()
        {
            let tmp_q = q + 7 - order;
            let rek2_201q = kaula_param.love_numbers.real(0, 1, tmp_q);
            let imk2_201q = kaula_param.love_numbers.imaginary(0, 1, tmp_q);
            let rek2_220q = kaula_param.love_numbers.real(2, 0, tmp_q);
            let imk2_220q = kaula_param.love_numbers.imaginary(2, 0, tmp_q);

            let mut sum_over_j_1 = 0.;
            let mut sum_over_j_3 = 0.;

            let mut sum_over_j_1_secular = 0.;
            let mut sum_over_j_3_secular = 0.;

            for (j, (g_20j, g_21j)) in izip!(
                kaula_param.polynomials.eccentricity_function_g_2pq[0],
                kaula_param.polynomials.eccentricity_function_g_2pq[1]
            )
            .skip(q_min)
            .take(q_max)
            .enumerate()
            {
                let alpha_qj = alpha_pqkj(
                    0,
                    q as i32 - order as i32,
                    0,
                    j as i32 - order as i32,
                    mean_anomaly,
                    argument_perihelion,
                );

                sum_over_j_1 += g_21j * (cos!(alpha_qj) * rek2_201q - sin!(alpha_qj) * imk2_201q);
                sum_over_j_3 += g_20j * (cos!(alpha_qj) * rek2_220q - sin!(alpha_qj) * imk2_220q);
                if q == j {
                    sum_over_j_1_secular += g_21j * rek2_201q;
                    sum_over_j_3_secular += g_20j * rek2_220q;
                }
            }

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
    particle: &Particle,
    keplerian_elements: (f64, f64, f64, f64, f64, f64, f64, f64),
    central_body: bool,
) -> (f64, f64) {
    // TODO pass in kaula
    let &(TidesEffect::CentralBody(TidalModel::Kaula(ref kaula_param))
    | TidesEffect::OrbitingBody(TidalModel::Kaula(ref kaula_param))) = &particle.tides.effect
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
        _orbital_period,
    ) = keplerian_elements;

    let argument_perihelion = longitude_perihelion - longitude_of_ascending_node;
    // if there is a non-negligible inclination (3D case)
    // Modification has been made but no testing has been done for 3D case (26/2)
    // let eccentricity_function_g_2pq = calculate_eccentricity_function_g_2pq(eccentricity);
    // let inclination_function_f_2mp = calculate_inclination_function_f_2mp(obliquity);
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

            let tmp_q = q + 7 - order;
            let rek2_20pq = kaula_param.love_numbers.real(0, p, tmp_q);
            let imk2_20pq = kaula_param.love_numbers.imaginary(0, p, tmp_q);
            let rek2_21pq = kaula_param.love_numbers.real(1, p, tmp_q);
            let imk2_21pq = kaula_param.love_numbers.imaginary(1, p, tmp_q);
            let rek2_22pq = kaula_param.love_numbers.real(2, p, tmp_q);
            let imk2_22pq = kaula_param.love_numbers.imaginary(2, p, tmp_q);

            let mut sum_over_k_m0 = 0.;
            let mut sum_over_k_m1 = 0.;
            let mut sum_over_k_m2 = 0.;

            for (k, (f_20k, f_21k, f_22k)) in izip!(
                kaula_param.polynomials.inclination_function_f_20p,
                kaula_param.polynomials.inclination_function_f_21p,
                kaula_param.polynomials.inclination_function_f_22p,
            )
            .enumerate()
            {
                let mut sum_over_j_m0 = 0.;
                let mut sum_over_j_m1 = 0.;
                let mut sum_over_j_m2 = 0.;
                let sum_g_2kj = kaula_param.polynomials.eccentricity_function_g_2pq[k];

                for (j, g_2kj) in sum_g_2kj.iter().skip(q_min).take(q_max).enumerate() {
                    let phase_alpha = alpha_pqkj(
                        p as i32,
                        q as i32 - order as i32,
                        k as i32,
                        j as i32 - order as i32,
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

    let radial_force =
        (sum_over_p_m0 + (1. / 3.) * sum_over_p_m1 + (1. / 12.) * sum_over_p_m2) * cste;
    let radial_force_secular =
        (sum_over_p_m0_s + (1. / 3.) * sum_over_p_m1_s + (1. / 12.) * sum_over_p_m2_s) * cste;
    (radial_force, radial_force_secular)
}
