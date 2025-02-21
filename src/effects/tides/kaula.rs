// Implemented by Alexandre Revol alexandre.revol@unige.ch
use super::super::super::Axes;
use super::super::super::Particle;
use super::super::super::constants::G;
use super::super::super::tools;
use super::common::TidalModel;
use super::common::TidesEffect;
use serde::{Deserialize, Serialize};

pub mod polynomials;
use polynomials::Polynomials;

mod radial_component;
use radial_component::{
    calculate_radial_component_of_the_tidal_force_2d,
    calculate_radial_component_of_the_tidal_force_3d,
};

mod normal_component;
use normal_component::calculate_normal_component_of_the_tidal_force_3d;

mod orthogonal_component;
use orthogonal_component::{
    calculate_orthogonal_component_of_the_tidal_force_2d,
    calculate_orthogonal_component_of_the_tidal_force_3d,
};

// Structure declare the intput parameters needed for the Kaula model calculation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct KaulaParameters {
    // The tidal Love numbers function of the excitation frequency
    pub love_number_excitation_frequency: Vec<f64>, // The excitation frequency
    pub real_part_love_number: Vec<f64>,            // The real part of the complex love number
    pub imaginary_part_love_number: Vec<f64>,       // The imaginary part of the complex love number
    pub num_datapoints: f64, // The number of point stored in the 32 by 32 array
    // If we are simulating stellar tide, then we need the value of the spectrum spin rate,
    // which was an input value of stellar spin rate from the computation of the spectrum.
    pub stellar_tide: bool,      // Specified the type of tides
    pub spectrum_spin_rate: f64, // Specified the initial spin rate of the star (stellar tide specific)
    #[serde(default)]
    pub polynomials: Polynomials,
    pub kaula_tidal_force: Axes,
}

pub fn calculate_tidal_force(
    tidal_host_particle: &mut Particle,
    particle: &mut Particle,
    central_body: bool,
) -> Axes {
    //// Planetary tide component
    //let central_body = false;

    //// TODO: Reconsider how to compute the stellar tide here instead of 'src/effects/tides/common.rs' while allowing a mixture of tidal models
    //let mut tidal_force = tidal_force_due_to_planetary_tide;
    //if matches!(tidal_host_particle.tides.effect, TidesEffect::CentralBody(TidalModel::Kaula(_))) {
    //// Stellar tide component
    //let central_body = true;
    //let tidal_force_due_to_stellar_tide = calculate_tidal_force_component(particle, tidal_host_particle, central_body);
    //tidal_force.x -= tidal_force_due_to_stellar_tide.x;
    //tidal_force.y -= tidal_force_due_to_stellar_tide.y;
    //tidal_force.z -= tidal_force_due_to_stellar_tide.z;
    //}
    //tidal_force
    calculate_tidal_force_component(tidal_host_particle, particle, central_body)
}

fn calculate_2d_or_3d_components(
    tidal_host_particle: &Particle,
    particle: &mut Particle,
    central_body: bool,
) -> (f64, f64, f64, f64, f64, f64) {
    // !central_body ==> planetary tide ==> particle is the planet
    // central_body ==> stellar tide ==> tidal_host_particle is the planet
    let obliquity;
    let keplerian_elements;
    let gm = G * (tidal_host_particle.mass + particle.mass);
    if central_body {
        obliquity = tools::calculate_inclination_orbital_equatorial_plane(
            tidal_host_particle.heliocentric_position,
            tidal_host_particle.heliocentric_velocity,
            particle.spin,
        );
        keplerian_elements = tools::calculate_keplerian_orbital_elements(
            gm,
            tidal_host_particle.heliocentric_position,
            tidal_host_particle.heliocentric_velocity,
        );
    } else {
        obliquity = tools::calculate_inclination_orbital_equatorial_plane(
            particle.heliocentric_position,
            particle.heliocentric_velocity,
            particle.spin,
        );
        keplerian_elements = tools::calculate_keplerian_orbital_elements(
            gm,
            particle.heliocentric_position,
            particle.heliocentric_velocity,
        );
    }
    // Keplerian elements
    let (
        _semi_major_axis,
        _perihelion_distance,
        eccentricity,
        _inclination,
        _longitude_perihelion,
        _longitude_of_ascending_node,
        _mean_anomaly,
        _orbital_period,
    ) = keplerian_elements;

    let radial_component_of_the_tidal_force;
    let radial_component_of_the_tidal_force_secular;
    let normal_component_of_the_tidal_force;
    let normal_component_of_the_tidal_force_secular;
    let orthogonal_component_of_the_tidal_force;
    let orthogonal_component_of_the_tidal_force_secular;


    let &mut (TidesEffect::CentralBody(TidalModel::Kaula(ref mut kaula_param))
    | TidesEffect::OrbitingBody(TidalModel::Kaula(ref mut kaula_param))) =
        &mut particle.tides.effect
    else {
        unreachable!()
    };

    // Update the kaula eccentricity, used by both 2D and 3D
    kaula_param.polynomials.update_eccentricity(eccentricity);

    // Decide if this is a 2d or 3d case
    if obliquity <= 1.0e-8 {
        // 2D case
        (
            orthogonal_component_of_the_tidal_force,
            orthogonal_component_of_the_tidal_force_secular,
        ) = calculate_orthogonal_component_of_the_tidal_force_2d(
            tidal_host_particle,
            particle,
            keplerian_elements,
            central_body,
        );
        (
            radial_component_of_the_tidal_force,
            radial_component_of_the_tidal_force_secular,
        ) = calculate_radial_component_of_the_tidal_force_2d(
            tidal_host_particle,
            particle,
            keplerian_elements,
            central_body,
        );
        (
            normal_component_of_the_tidal_force,
            normal_component_of_the_tidal_force_secular,
        ) = (0., 0.);
    } else {
        // 3D case

        // Update the kaula inclination, used only by 3D
        kaula_param.polynomials.update_inclination(obliquity);

        (
            orthogonal_component_of_the_tidal_force,
            orthogonal_component_of_the_tidal_force_secular,
        ) = calculate_orthogonal_component_of_the_tidal_force_3d(
            tidal_host_particle,
            particle,
            keplerian_elements,
            central_body,
        );
        (
            radial_component_of_the_tidal_force,
            radial_component_of_the_tidal_force_secular,
        ) = calculate_radial_component_of_the_tidal_force_3d(
            tidal_host_particle,
            particle,
            keplerian_elements,
            central_body,
        );
        (
            normal_component_of_the_tidal_force,
            normal_component_of_the_tidal_force_secular,
        ) = calculate_normal_component_of_the_tidal_force_3d(
            tidal_host_particle,
            particle,
            keplerian_elements,
            central_body,
        );
    }

    (
        radial_component_of_the_tidal_force,
        radial_component_of_the_tidal_force_secular,
        normal_component_of_the_tidal_force,
        normal_component_of_the_tidal_force_secular,
        orthogonal_component_of_the_tidal_force,
        orthogonal_component_of_the_tidal_force_secular,
    )
}

fn cartesian_projection_of_spherical_coordinates(
    tidal_host_particle: &Particle,
    particle: &mut Particle,
    central_body: bool,
    normal_component: f64,
    orthogonal_component: f64,
    radial_component: f64) -> Axes
{
    // Spherical coordinate
    // The following elements correspond to the coordinate in the spherical coordinate
    // The coplanar distance is the radial distance projected in the x-y plane
    // The theta angle is the angle between the radial distance vector with respect to the z axis
    // The phi angle is the angle between the coplanar distance with respect to the x axis
    let cos_phi;
    let cos_theta;
    let sin_phi;
    let coplanar_distance;
    let radial_distance;

    if central_body {
        radial_distance = tidal_host_particle.tides.parameters.internal.distance;
        coplanar_distance = sqrt!(
            tidal_host_particle.tides.coordinates.position.x.powi(2)
                + tidal_host_particle.tides.coordinates.position.y.powi(2)
        );
        cos_theta = tidal_host_particle.tides.coordinates.position.z / radial_distance;
        cos_phi = tidal_host_particle.tides.coordinates.position.x / coplanar_distance;
        sin_phi = tidal_host_particle.tides.coordinates.position.y / coplanar_distance;
    } else {
        radial_distance = particle.tides.parameters.internal.distance;
        coplanar_distance = sqrt!(
            particle.tides.coordinates.position.x.powi(2)
                + particle.tides.coordinates.position.y.powi(2)
        );
        cos_theta = particle.tides.coordinates.position.z / radial_distance;
        cos_phi = particle.tides.coordinates.position.x / coplanar_distance;
        sin_phi = particle.tides.coordinates.position.y / coplanar_distance;
    }
    let sin_theta = coplanar_distance / radial_distance;

    let tidal_force_x = radial_component * sin_theta * cos_phi
        + normal_component * cos_theta * cos_phi
        - orthogonal_component * sin_phi;
    let tidal_force_y = radial_component * sin_theta * sin_phi
        + normal_component * cos_theta * sin_phi
        + orthogonal_component * cos_phi;
    let tidal_force_z = radial_component * cos_theta
        - normal_component * sin_theta;

    Axes::from(tidal_force_x, tidal_force_y, tidal_force_z)

}

fn calculate_tidal_force_component(
    tidal_host_particle: &mut Particle,
    particle: &mut Particle,
    central_body: bool,
) -> Axes {

    // TODO precompute polynomials/love numbers here
    // (used by each of the three component calculations below)

    // Spherical component of the tidal force
    // The radial component is the force applicated through the radial axis
    // The normal component act on the co longitude axis
    // The orthogonal component act on the co latitude axis
    let (
        radial_component_of_the_tidal_force,
        radial_component_of_the_tidal_force_secular,
        normal_component_of_the_tidal_force,
        normal_component_of_the_tidal_force_secular,
        orthogonal_component_of_the_tidal_force,
        orthogonal_component_of_the_tidal_force_secular,
    ) = calculate_2d_or_3d_components(
        tidal_host_particle,
        particle,
        central_body,
    );
    // Required for denergy_dt calculation
    if central_body {
        tidal_host_particle
            .tides
            .parameters
            .internal
            .orthogonal_component_of_the_tidal_force_due_to_stellar_tide =
            orthogonal_component_of_the_tidal_force;
    } else {
        particle
            .tides
            .parameters
            .internal
            .orthogonal_component_of_the_tidal_force_due_to_planetary_tide =
            orthogonal_component_of_the_tidal_force;
    }

    // Cartesian tidal force computed by projection of the spherical coordinates
    let components = cartesian_projection_of_spherical_coordinates(
        tidal_host_particle,
        particle,
        central_body,
        normal_component_of_the_tidal_force,
        orthogonal_component_of_the_tidal_force,
        radial_component_of_the_tidal_force,
    );


    // Secular part of the tidal torque (simplified from the rapid varying phases)
    let secular_projection = cartesian_projection_of_spherical_coordinates(
        tidal_host_particle,
        particle,
        central_body,
        normal_component_of_the_tidal_force_secular,
        orthogonal_component_of_the_tidal_force_secular,
        radial_component_of_the_tidal_force_secular,
    );

    let &mut (TidesEffect::CentralBody(TidalModel::Kaula(ref mut kaula_param))
    | TidesEffect::OrbitingBody(TidalModel::Kaula(ref mut kaula_param))) =
        &mut particle.tides.effect
    else {
        unreachable!()
    };

    kaula_param.kaula_tidal_force.copy_from(&secular_projection);

    components

}

pub fn select_eccentricty_order_q(eccentricity: f64) -> usize {
    match () {
        // Select the order of the summation q over the eccentricity function G_lpq
        () if (eccentricity > 0.3) => 7,  // -7 < q < 7
        () if (eccentricity > 0.25) => 6, // -6 < q < 6
        () if (eccentricity > 0.2) => 5,  // -5 < q < 5
        () if (eccentricity > 0.15) => 4, // -4 < q < 4
        () if (eccentricity > 0.1) => 3,  // -3 < q < 3
        () if (eccentricity <= 0.1) => 2, // -2 < q < 2
        () => 0,
    }
}

// Calculate tidal torque due to tidal forces
pub fn calculate_torque_due_to_tides(
    tidal_host_particle: &Particle,
    particle: &Particle,
    central_body: bool,
) -> Axes {
    let position_x;
    let position_y;
    let position_z;
    let tidal_force_x;
    let tidal_force_y;
    let tidal_force_z;

    if central_body {
        let kaula_tidal_force = match &tidal_host_particle.tides.effect {
            TidesEffect::CentralBody(TidalModel::Kaula(params)) => params.kaula_tidal_force,
            _ => unreachable!(),
        };
        tidal_force_x = kaula_tidal_force.x;
        tidal_force_y = kaula_tidal_force.y;
        tidal_force_z = kaula_tidal_force.z;
        position_x = -particle.tides.coordinates.position.x;
        position_y = -particle.tides.coordinates.position.y;
        position_z = -particle.tides.coordinates.position.z;
    } else {
        let kaula_tidal_force = match &particle.tides.effect {
            TidesEffect::OrbitingBody(TidalModel::Kaula(params)) => params.kaula_tidal_force,
            _ => unreachable!(),
        };
        tidal_force_x = kaula_tidal_force.x;
        tidal_force_y = kaula_tidal_force.y;
        tidal_force_z = kaula_tidal_force.z;
        position_x = particle.tides.coordinates.position.x;
        position_y = particle.tides.coordinates.position.y;
        position_z = particle.tides.coordinates.position.z;
    }
    // The negative sign of the position vector for stellar tide: Torque = r cross F
    // The r here should be the vector point from thr primary (perturber) to the secondary (perturbed)
    // Thus we added a minus sign here as particle.tides.coordinates.position.xyx is always heliocentric.

    // Let the torque be the cross product of the radial distance vector and the tidal force vector
    let torque_due_to_tides_x = position_y * tidal_force_z - position_z * tidal_force_y;
    let torque_due_to_tides_y = position_z * tidal_force_x - position_x * tidal_force_z;
    let torque_due_to_tides_z = position_x * tidal_force_y - position_y * tidal_force_x;

    Axes::from(
        torque_due_to_tides_x,
        torque_due_to_tides_y,
        torque_due_to_tides_z,
    )
}

#[allow(dead_code)]
// Calculate the associated Legendre polynomials P_{l=2}^{m}
pub fn calculate_associated_legendre_polynomials_p_2m(x: f64) -> [f64; 3] {
    let mut p_2m = [0.0; 3];
    let x_2 = x.powi(2);

    p_2m[0] = 0.5 * (3. * x_2 - 1.);
    p_2m[1] = -3. * x * (sqrt!(1. - x_2));
    p_2m[2] = 3. * (1. - x_2);

    p_2m
}

// Calculate the tidal excitation frequency mode \sigma_{2mpq}
pub fn calculate_tidal_excitation_frequency_mode_sigma_2mpq(
    m: f64,
    p: f64,
    q: f64,
    spin: f64,
    orbital_frequency: f64,
) -> f64 {
    (2.0 - 2. * p + q) * orbital_frequency - m * spin
}

// Calculate the phases of the 2-Kaula transformed tidal forces
pub fn alpha_pqkj(
    p: f64,
    q: f64,
    k: f64,
    j: f64,
    mean_anomaly: f64,
    argument_perihelion: f64,
) -> f64 {
    (2. * p - 2. * k + j - q) * mean_anomaly + 2. * (p - k) * argument_perihelion
}

// Calculate the phases of the 2-Kaula transformed tidal forces
pub fn compute_phase_beta(
    p: f64,
    q: f64,
    k: f64,
    j: f64,
    mean_anomaly: f64,
    spin_angle: f64,
    argument_perihelion: f64,
    longitude_ascending_node: f64,
    heliocentric_varphi: f64,
) -> f64 {
    (4. - 2. * p - 2. * k + q - j) * mean_anomaly
        + (4. - 2. * (k + p)) * argument_perihelion
        + spin_angle
        - longitude_ascending_node
        + heliocentric_varphi
}

pub fn compute_phase_alpha_1(
    p: f64,
    q: f64,
    k: f64,
    j: f64,
    mean_anomaly: f64,
    spin_angle: f64,
    argument_perihelion: f64,
    longitude_ascending_node: f64,
    heliocentric_varphi: f64,
) -> f64 {
    (2. * k - 2. * p + q - j) * mean_anomaly + 2. * (k - p) * argument_perihelion + spin_angle
        - longitude_ascending_node
        - heliocentric_varphi
}

pub fn compute_phase_alpha_2(
    p: f64,
    q: f64,
    k: f64,
    j: f64,
    mean_anomaly: f64,
    spin_angle: f64,
    argument_perihelion: f64,
    longitude_ascending_node: f64,
    heliocentric_varphi: f64,
) -> f64 {
    (2. * k - 2. * p + q - j) * mean_anomaly + 2. * (k - p) * argument_perihelion - spin_angle
        + longitude_ascending_node
        + heliocentric_varphi
}

pub fn compute_phase_alpha_3(
    p: f64,
    q: f64,
    k: f64,
    j: f64,
    mean_anomaly: f64,
    spin_angle: f64,
    argument_perihelion: f64,
    longitude_ascending_node: f64,
    heliocentric_varphi: f64,
) -> f64 {
    (2. * k - 2. * p + q - j - 1.) * mean_anomaly
        + 2. * (k - p - 1.) * argument_perihelion
        + spin_angle
        - longitude_ascending_node
        - heliocentric_varphi
}

pub fn compute_phase_alpha_4(
    p: f64,
    q: f64,
    k: f64,
    j: f64,
    mean_anomaly: f64,
    spin_angle: f64,
    argument_perihelion: f64,
    longitude_ascending_node: f64,
    heliocentric_varphi: f64,
) -> f64 {
    (2. * k - 2. * p + q - j - 1.) * mean_anomaly + 2. * (k - p - 1.) * argument_perihelion
        - spin_angle
        + longitude_ascending_node
        - heliocentric_varphi
}

// Find the real part and the imaginary part of the Love number associated with
// the excitation frequenccy wk2
pub fn calculate_kaula_numbers(
    mut wk2: f64,
    kaula_param: &KaulaParameters,
    central_body: bool,
    stellar_spin_rate: f64,
) -> (f64, f64) {
    // Planets have symmetric tidal response, stars do not.
    let parity = !central_body & (wk2 < 0.0);
    if parity {
        wk2 = abs!(wk2);
    }

    // These small part of modification is only for stellar tide, kaula with a fixed spectrum
    // , also why I declare the new input as "stellar_spin_rate".
    if kaula_param.stellar_tide {
        // Days
        wk2 *= kaula_param.spectrum_spin_rate / stellar_spin_rate;
    }

    let mut im_k2;
    let re_k2;
    // Find the index of the closest match to use for the interpolation
    if wk2 <= kaula_param.love_number_excitation_frequency[0] {
        // If wk2 is less than or equal to the first element, take the first value
        im_k2 = kaula_param.imaginary_part_love_number[0];
        re_k2 = kaula_param.real_part_love_number[0];
    } else if wk2
        >= kaula_param.love_number_excitation_frequency
            [kaula_param.love_number_excitation_frequency.len() - 1]
    {
        // If wk2 is greater than or equal to the last element, take the last value
        im_k2 = kaula_param.imaginary_part_love_number
            [kaula_param.love_number_excitation_frequency.len() - 1];
        re_k2 = kaula_param.real_part_love_number
            [kaula_param.love_number_excitation_frequency.len() - 1];
    } else {
        // Find the index of the closest match to use for the interpolation
        match kaula_param
            .love_number_excitation_frequency
            .binary_search_by(|val| val.total_cmp(&wk2))
        {
            Ok(i) => {
                // Exact match found: love_number[i] == wk2
                im_k2 = kaula_param.imaginary_part_love_number[i];
                re_k2 = kaula_param.real_part_love_number[i];
            }
            Err(i) => {
                // wk2 is between love_number[i - 1] and love_number[i]
                let prev_freq = kaula_param.love_number_excitation_frequency[i - 1];
                let next_freq = kaula_param.love_number_excitation_frequency[i];
                let delta = (wk2 - prev_freq) / (next_freq - prev_freq);
                im_k2 = (1.0 - delta) * kaula_param.imaginary_part_love_number[i - 1]
                    + delta * kaula_param.imaginary_part_love_number[i];
                re_k2 = (1.0 - delta) * kaula_param.real_part_love_number[i - 1]
                    + delta * kaula_param.real_part_love_number[i];
            }
        }
    }

    if kaula_param.stellar_tide {
        let spectrum_spin_rate = kaula_param.spectrum_spin_rate;
        im_k2 *= (stellar_spin_rate / spectrum_spin_rate).powi(2);
    }

    if parity {
        (-re_k2, -im_k2)
    } else {
        (-re_k2, im_k2)
    }
}
