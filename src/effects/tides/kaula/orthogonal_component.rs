use super::super::super::super::Particle;
use super::super::super::super::constants::{DAY, G, TWO_PI};
use super::{
    alpha_pqkj, calculate_kaula_numbers, calculate_tidal_excitation_frequency_mode_sigma_2mpq,
    compute_phase_alpha_3, compute_phase_alpha_4, select_eccentricty_order_q,
};
use crate::{TidalModel, TidesEffect};
use itertools::izip;

// Ortho-radial (the e_{\varphi}) component of tidal force
pub fn calculate_orthogonal_component_of_the_tidal_force_2d(
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

    let tem_radius;

    let distance;
    let coplanar_distance;
    let sin_theta;
    let cste_2d;
    if central_body {
        // inclination = tools::calculate_inclination_orbital_equatorial_plane(tidal_host_particle.heliocentric_position, tidal_host_particle.heliocentric_velocity, tidal_host_particle.spin);
        tem_radius = -tidal_host_particle.heliocentric_distance;
        distance = sqrt!(
            tidal_host_particle.tides.coordinates.position.x.powi(2)
                + tidal_host_particle.tides.coordinates.position.y.powi(2)
                + tidal_host_particle.tides.coordinates.position.z.powi(2)
        );
        coplanar_distance = sqrt!(
            tidal_host_particle.tides.coordinates.position.x.powi(2)
                + tidal_host_particle.tides.coordinates.position.y.powi(2)
        );
        sin_theta = coplanar_distance / distance;
        cste_2d = -(G * particle.mass.powi(2) * tidal_host_particle.radius.powi(5))
            / (semi_major_axis.powi(6) * tem_radius * sin_theta);
    } else {
        tem_radius = particle.heliocentric_distance;
        // inclination = tools::calculate_inclination_orbital_equatorial_plane(particle.heliocentric_position, particle.heliocentric_velocity, particle.spin);
        distance = sqrt!(
            particle.tides.coordinates.position.x.powi(2)
                + particle.tides.coordinates.position.y.powi(2)
                + particle.tides.coordinates.position.z.powi(2)
        );
        coplanar_distance = sqrt!(
            particle.tides.coordinates.position.x.powi(2)
                + particle.tides.coordinates.position.y.powi(2)
        );
        sin_theta = coplanar_distance / distance;
        cste_2d = -(G * tidal_host_particle.mass.powi(2) * particle.radius.powi(5))
            / (semi_major_axis.powi(6) * tem_radius * sin_theta);
    }

    let argument_perihelion = longitude_perihelion - longitude_of_ascending_node;
    // Orbital mean motion in [rad.s^-1]
    let orbital_frequency = TWO_PI / (orbital_period * DAY);

    // Stellar/planetary spin in [rad.s^-1]
    let spin = sqrt!(particle.norm_spin_vector_2) / DAY;

    // Cartesian angles
    // summation
    let orthogonal_force;
    let orthogonal_force_secular;

    if eccentricity == 0. {
        let freq_2200 = calculate_tidal_excitation_frequency_mode_sigma_2mpq(
            2.,
            0.,
            0.,
            spin,
            orbital_frequency,
        );
        let (_rek2_2200, imk2_2200) =
            calculate_kaula_numbers(freq_2200, kaula_param, central_body, spin);
        orthogonal_force = cste_2d * (3. / 2.) * imk2_2200;
        orthogonal_force_secular = orthogonal_force;
    } else {
        kaula_param
            .polynomials
            .calculate_eccentricity_function_g_20q(eccentricity);
        let mut sum_over_q = 0.;
        let mut sum_over_q_secular = 0.;
        let order = select_eccentricty_order_q(eccentricity);
        let q_min = 7 - order;
        let q_max = 2 * order + 1;

        // for q in -7..8{
        for (q, g_20q) in kaula_param
            .polynomials
            .eccentricity_function_g_20q
            .iter()
            .skip(q_min)
            .take(q_max)
            .enumerate()
        {
            let freq_220q = calculate_tidal_excitation_frequency_mode_sigma_2mpq(
                2.,
                0.,
                q as f64 - order as f64,
                spin,
                orbital_frequency,
            );
            let (rek2_220q, imk2_220q) =
                calculate_kaula_numbers(freq_220q, kaula_param, central_body, spin);

            let mut sum_over_j_2 = 0.;
            let mut sum_over_j_2_secular = 0.;
            for (j, g_20j) in kaula_param
                .polynomials
                .eccentricity_function_g_20q
                .iter()
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

    // Stellar/planetary spin in [rad.s^-1]
    let spin = sqrt!(particle.norm_spin_vector_2) / DAY;

    // Cartesian angles
    // summation
    let orthogonal_force;
    let orthogonal_force_secular;
    let heliocentric_varphi = 0.;
    let spin_angle = 0.;

    let order = select_eccentricty_order_q(eccentricity);
    let q_min = 7 - order;
    let q_max = 2 * order + 1;

    let mut sum_over_p;
    let mut sum_over_p_s;

    sum_over_p = 0.;
    sum_over_p_s = 0.;
    let c1 = 1. / 6.;
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

            let mut sum_over_kterm1 = 0.;
            let mut sum_over_kterm2 = 0.;
            let mut sum_over_kterm1_s = 0.;
            let mut sum_over_kterm2_s = 0.;

            for (k, (f_30k, f_32k)) in izip!(
                kaula_param.polynomials.inclination_function_f_30p,
                kaula_param.polynomials.inclination_function_f_32p,
            )
            .enumerate()
            {
                // Sum over p for each m
                let mut sum_over_jterm1 = 0.;
                let mut sum_over_jterm2 = 0.;
                let mut sum_over_jterm1_s = 0.;
                let mut sum_over_jterm2_s = 0.;
                let sum_g_3kj = kaula_param.polynomials.eccentricity_function_g_3pq[k];

                for (j, g_3kj) in sum_g_3kj.iter().skip(q_min).take(q_max).enumerate() {
                    let phase_alpha_3 = compute_phase_alpha_3(
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
                    let phase_alpha_4 = compute_phase_alpha_4(
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
    for (p, f_21p) in kaula_param
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

            let mut sum_over_kterm1 = 0.;
            let mut sum_over_kterm2 = 0.;
            let mut sum_over_kterm1_s = 0.;
            let mut sum_over_kterm2_s = 0.;

            for (k, (f_31k, f_33k)) in izip!(
                kaula_param.polynomials.inclination_function_f_31p,
                kaula_param.polynomials.inclination_function_f_33p
            )
            .enumerate()
            {
                // Sum over p for each m
                let mut sum_over_jterm1 = 0.;
                let mut sum_over_jterm2 = 0.;
                let mut sum_over_jterm1_s = 0.;
                let mut sum_over_jterm2_s = 0.;
                let sum_g_3kj = kaula_param.polynomials.eccentricity_function_g_3pq[k];

                for (j, g_3kj) in sum_g_3kj.iter().skip(q_min).take(q_max).enumerate() {
                    let phase_alpha_3 = compute_phase_alpha_3(
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
                    let phase_alpha_4 = compute_phase_alpha_4(
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
        sum_over_p += f_21p * sum_over_q;
        sum_over_p_s += f_21p * sum_over_q_s;
    }
    let term_m2 = -5. / (48. * sqrt!(6_f64)) * sum_over_p;
    let term_m2_s = -5. / (48. * sqrt!(6_f64)) * sum_over_p_s;

    let cste_3d = if central_body {
        // inclination = tools::calculate_inclination_orbital_equatorial_plane(tidal_host_particle.heliocentric_position, tidal_host_particle.heliocentric_velocity, tidal_host_particle.spin);
        -(G * particle.mass.powi(2) * tidal_host_particle.radius.powi(5))
            / (semi_major_axis.powi(7))
    } else {
        // inclination = tools::calculate_inclination_orbital_equatorial_plane(particle.heliocentric_position, particle.heliocentric_velocity, particle.spin);
        -(G * tidal_host_particle.mass.powi(2) * particle.radius.powi(5))
            / (semi_major_axis.powi(7))
    };

    orthogonal_force = cste_3d * (term_m1 + term_m2);
    orthogonal_force_secular = cste_3d * (term_m1_s + term_m2_s); // cste *sum_over_p_secular;
    (orthogonal_force, orthogonal_force_secular)
}
