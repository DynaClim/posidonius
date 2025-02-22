use serde::{Deserialize, Serialize};
use serde_big_array::BigArray;

use super::select_eccentricty_order_q;

// Real and imaginary love numbers calculated each timestep.
#[derive(Serialize, Copy, Deserialize, PartialEq, Debug, Clone)]
#[serde(default)]
pub struct LoveNumber {
    #[serde(with = "BigArray")]
    spectrum_excitation_frequency: [f64; 1024],
    #[serde(with = "BigArray")]
    spectrum_real_part: [f64; 1024],
    #[serde(with = "BigArray")]
    spectrum_imaginary_part: [f64; 1024],

    // Cache of the tidal frequency
    #[serde(with = "BigArray")]
    freq: [f64; 145],
    #[serde(with = "BigArray")]
    real: [f64; 145],
    #[serde(with = "BigArray")]
    imaginary: [f64; 145],
}

impl Default for LoveNumber {
    fn default() -> Self {
        Self {
            spectrum_excitation_frequency: [0.; 1024],
            spectrum_real_part: [0.; 1024],
            spectrum_imaginary_part: [0.; 1024],
            freq: [0.; 145],
            real: [0.; 145],
            imaginary: [0.; 145],
        }
    }
}

impl LoveNumber {
    pub fn new_from(
        spectrum_excitation_frequency: [f64; 1024],
        spectrum_real_part: [f64; 1024],
        spectrum_imaginary_part: [f64; 1024],
    ) -> Self {
        Self {
            spectrum_excitation_frequency,
            spectrum_real_part,
            spectrum_imaginary_part,
            ..Default::default()
        }
    }
    // Maps a set of 3D coordinates into a 1D index.
    fn map_3d_to_1d(m: usize, m_max: usize, p: usize, p_max: usize, q: usize) -> usize {
        (q * m_max * p_max) + (p * m_max) + m
    }

    /// Fetches the real love number from the cache for index of tuple (m, p, q).
    pub(crate) fn real(&self, m: usize, p: usize, q: usize) -> f64 {
        // The cached data is stored in a 1D array, so the 3D coordinates are mapped to the 1D index.
        // m_max and p_max are both 3
        self.real[Self::map_3d_to_1d(m, 3, p, 3, q)]
    }

    /// Fetches the imaginary love number from the cache for index of tuple (m, p, q).
    pub(crate) fn imaginary(&self, m: usize, p: usize, q: usize) -> f64 {
        // The cached data is stored in a 1D array, so the 3D coordinates are mapped to the 1D index.
        // m_max and p_max are both 3
        self.imaginary[Self::map_3d_to_1d(m, 3, p, 3, q)]
    }

    pub fn refresh_cache_full(
        &mut self,
        central_body: bool,
        is_stellar_tide: bool,
        spectrum_spin_rate: f64,
        spin_rate: f64,
        orbital_frequency: f64,
        eccentricity: f64,
    ) {
        // refresh m = 0..3, p = 0..3, q bounded by eccentricty order
        let (m_min, m_max) = (0, 3);
        let (p_min, p_max) = (0, 3);
        self.refresh_cache(
            central_body,
            is_stellar_tide,
            spectrum_spin_rate,
            spin_rate,
            orbital_frequency,
            eccentricity,
            m_min,
            m_max,
            p_min,
            p_max,
        );
    }
    pub fn refresh_cache_partial(
        &mut self,
        central_body: bool,
        is_stellar_tide: bool,
        spectrum_spin_rate: f64,
        spin_rate: f64,
        orbital_frequency: f64,
        eccentricity: f64,
    ) {
        // refresh m = 0, p = 1, q = bounded by eccentricty order
        let (m_min, m_max) = (0, 1);
        let (p_min, p_max) = (1, 2);
        self.refresh_cache(
            central_body,
            is_stellar_tide,
            spectrum_spin_rate,
            spin_rate,
            orbital_frequency,
            eccentricity,
            m_min,
            m_max,
            p_min,
            p_max,
        );

        // refresh m = 2, p = 0, q = bounded by eccentricty order
        let (m_min, m_max) = (2, 3);
        let (p_min, p_max) = (0, 1);
        self.refresh_cache(
            central_body,
            is_stellar_tide,
            spectrum_spin_rate,
            spin_rate,
            orbital_frequency,
            eccentricity,
            m_min,
            m_max,
            p_min,
            p_max,
        );
    }
    /// Recomputes all the love number values.
    // Called at each time step to cache love numbers for that iteration, to prevent duplicate calculations.
    fn refresh_cache(
        &mut self,
        central_body: bool,
        is_stellar_tide: bool,
        spectrum_spin_rate: f64,
        spin_rate: f64,
        orbital_frequency: f64,
        eccentricity: f64,
        m_min: i32,
        m_max: i32,
        p_min: i32,
        p_max: i32,
    ) {
        let order = select_eccentricty_order_q(eccentricity);
        let q_min = 7 - order;
        let q_max = 2 * order + 1;

        for (q, q_idx) in (0..14).skip(q_min).take(q_max).enumerate() {
            for p in p_min..p_max {
                for m in m_min..m_max {
                    let wk2 = calculate_tidal_excitation_frequency_mode_sigma_2mpq(
                        m,
                        p,
                        q as i32 - (order as i32),
                        spin_rate,
                        orbital_frequency,
                    );

                    let prev_wk2 =
                        self.freq[Self::map_3d_to_1d(m as usize, 3, p as usize, 3, q_idx)];
                    // TODO Don't update the love numbers cache if the change in wk2 is minimal
                    //                    if prev_wk2 != 0. && abs!(abs!(prev_wk2) - abs!(wk2)) < 1e-14 {
                    //                        todo!();
                    //                        continue;
                    //                    }
                    let (real_k2, imaginary_k2) = self.get_love_numbers(
                        central_body,
                        is_stellar_tide,
                        spectrum_spin_rate,
                        spin_rate,
                        wk2,
                    );

                    self.freq[Self::map_3d_to_1d(m as usize, 3, p as usize, 3, q_idx)] = wk2;
                    self.real[Self::map_3d_to_1d(m as usize, 3, p as usize, 3, q_idx)] = real_k2;
                    self.imaginary[Self::map_3d_to_1d(m as usize, 3, p as usize, 3, q_idx)] =
                        imaginary_k2;
                }
            }
        }
    }

    fn get_love_numbers(
        &self,
        central_body: bool,
        is_stellar_tide: bool,
        spectrum_spin_rate: f64,
        spin_rate: f64,
        mut wk2: f64,
    ) -> (f64, f64) {
        // Only for kaula stellar tide with a fixed spectrum
        // Taking into account of the real time evolution of the stellar spin rate
        // by modifying the tidal frequency, and rescale the love number (below)
        // instead of modifying the spectrum every time step.
        if is_stellar_tide {
            // Days
            wk2 *= spectrum_spin_rate / spin_rate;
        }

        // Planets have symmetric tidal response, stars do not.
        let parity = !central_body & (wk2 < 0.0);
        if parity {
            wk2 = abs!(wk2);
        }

        let (mut real_k2, mut imaginary_k2) = self.interpolate_love_numbers(wk2);

        real_k2 *= -1.;

        if parity {
            imaginary_k2 *= -1.;
        }

        // Rescale the love number
        if is_stellar_tide {
            imaginary_k2 *= (spin_rate / spectrum_spin_rate).powi(2);
        }

        (real_k2, imaginary_k2)
    }

    // Find the real part and the imaginary part of the Love number associated with
    // the excitation frequenccy wk2
    fn interpolate_love_numbers(&self, wk2: f64) -> (f64, f64) {
        let im_k2;
        let re_k2;
        // Find the index of the closest match to use for the interpolation
        if wk2 <= self.spectrum_excitation_frequency[0] {
            // If wk2 is less than or equal to the first element, take the first value
            im_k2 = self.spectrum_imaginary_part[0];
            re_k2 = self.spectrum_real_part[0];
        } else if wk2
            >= self.spectrum_excitation_frequency[self.spectrum_excitation_frequency.len() - 1]
        {
            // If wk2 is greater than or equal to the last element, take the last value
            im_k2 = self.spectrum_imaginary_part[self.spectrum_excitation_frequency.len() - 1];
            re_k2 = self.spectrum_real_part[self.spectrum_excitation_frequency.len() - 1];
        } else {
            // Find the index of the closest match to use for the interpolation
            match self
                .spectrum_excitation_frequency
                .binary_search_by(|val| val.total_cmp(&wk2))
            {
                Ok(i) => {
                    // Exact match found: love_number[i] == wk2
                    im_k2 = self.spectrum_imaginary_part[i];
                    re_k2 = self.spectrum_real_part[i];
                }
                Err(i) => {
                    // wk2 is between love_number[i - 1] and love_number[i]
                    let prev_freq = self.spectrum_excitation_frequency[i - 1];
                    let next_freq = self.spectrum_excitation_frequency[i];
                    let delta = (wk2 - prev_freq) / (next_freq - prev_freq);
                    im_k2 = (1.0 - delta) * self.spectrum_imaginary_part[i - 1]
                        + delta * self.spectrum_imaginary_part[i];
                    re_k2 = (1.0 - delta) * self.spectrum_real_part[i - 1]
                        + delta * self.spectrum_real_part[i];
                }
            }
        }

        (re_k2, im_k2)
    }
}

// Calculate the tidal excitation frequency mode \sigma_{2mpq}
pub fn calculate_tidal_excitation_frequency_mode_sigma_2mpq(
    m: i32,
    p: i32,
    q: i32,
    spin: f64,
    orbital_frequency: f64,
) -> f64 {
    let x = f64!(2 - 2 * p + q) * orbital_frequency - f64!(m) * spin;
    x
}
