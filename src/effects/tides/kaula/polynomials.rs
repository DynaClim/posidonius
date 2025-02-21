use serde::{Deserialize, Serialize};

#[derive(Debug, Copy, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct Polynomials {
    pub eccentricity_function_g_2pq: [[f64; 15]; 3],
    pub eccentricity_function_g_20q: [f64; 15],
    pub eccentricity_function_g_21q: [f64; 15],
    pub eccentricity_function_g_3pq: [[f64; 15]; 4],

    pub inclination_function_f_20p: [f64; 3],
    pub inclination_function_f_21p: [f64; 3],
    pub inclination_function_f_22p: [f64; 3],
    pub inclination_function_f_30p: [f64; 4],
    pub inclination_function_f_31p: [f64; 4],
    pub inclination_function_f_32p: [f64; 4],
    pub inclination_function_f_33p: [f64; 4],
}

impl Polynomials {
    // Constructor function that creates a Polynomials instance with all values set to 0.0
    pub fn new() -> Self {
        Polynomials {
            eccentricity_function_g_2pq: [[0.0; 15]; 3],
            eccentricity_function_g_20q: [0.0; 15],
            eccentricity_function_g_21q: [0.0; 15],
            eccentricity_function_g_3pq: [[0.0; 15]; 4],

            inclination_function_f_20p: [0.0; 3],
            inclination_function_f_21p: [0.0; 3],
            inclination_function_f_22p: [0.0; 3],
            inclination_function_f_30p: [0.0; 4],
            inclination_function_f_31p: [0.0; 4],
            inclination_function_f_32p: [0.0; 4],
            inclination_function_f_33p: [0.0; 4],
        }
    }
    pub fn update_inclination(&mut self, obliquity: f64) {
        self
        .calculate_inclination_function_f_21p(obliquity);
        self
        .calculate_inclination_function_f_22p(obliquity);
        self
        .calculate_inclination_function_f_30p(obliquity);
        self
        .calculate_inclination_function_f_31p(obliquity);
        self
        .calculate_inclination_function_f_32p(obliquity);
        self
        .calculate_inclination_function_f_33p(obliquity);
    }

    pub fn update_eccentricity(&mut self, eccentricity: f64) {
        self
        .calculate_eccentricity_function_g_2pq(eccentricity);
        self
        .calculate_eccentricity_function_g_20q(eccentricity);
        self
        .calculate_eccentricity_function_g_21q(eccentricity);
        self
        .calculate_eccentricity_function_g_3pq(eccentricity);
    
    }

    #[allow(dead_code)]
    // The inclination function of the Kaula 1961 developpement Flmp(i)
    pub fn calculate_inclination_function_f_2mp(inclination: f64) -> [[f64; 3]; 3] {
        let mut inclination_function = [[0.; 3]; 3];
        // in this notation:
        // 0 is q=-2
        // 1 is q=-1
        // 2 is q=0
        // 3 is q=1
        //4 is q=2 ... cf Kaula 64 Table 3
        let sin_incl = sin!(inclination);
        let sin_incl_2 = sin_incl * sin_incl;
        let cos_incl = cos!(inclination);
        let borrowed_slice = &mut inclination_function[..];
        let [first, second, third] = borrowed_slice else {
            unreachable!()
        };

        first[0] = -(3. / 8.) * sin_incl_2;
        first[1] = (3. / 4.) * sin_incl_2 - (1. / 2.);
        first[2] = -(3. / 8.) * sin_incl_2;

        second[0] = (3. / 4.) * sin_incl * (1. + cos_incl);
        second[1] = -(3. / 2.) * sin_incl * cos_incl;
        second[2] = (3. / 4.) * sin_incl * (cos_incl - 1.);

        third[0] = (3. / 4.) * (1. + cos_incl).powi(2);
        third[1] = (3. / 2.) * sin_incl.powi(2);
        third[2] = (3. / 4.) * (1. - cos_incl).powi(2);

        inclination_function
    }
    // F2mp
    // The inclination function of the Kaula 1961 developpement Flmp(i)
    pub fn calculate_inclination_function_f_20p(&mut self, inclination: f64) {
        // in this notation:
        // 0 is q=-2
        // 1 is q=-1
        // 2 is q=0
        // 3 is q=1
        //4 is q=2 ... cf Kaula 64 Table 3
        let sin_incl = sin!(inclination);
        let sin_incl_2 = sin_incl * sin_incl;

        self.inclination_function_f_20p[0] = -(3. / 8.) * sin_incl_2;
        self.inclination_function_f_20p[1] = (3. / 4.) * sin_incl_2 - (1. / 2.);
        self.inclination_function_f_20p[2] = -(3. / 8.) * sin_incl_2;
    }
    // The inclination function of the Kaula 1961 developpement Flmp(i)
    pub fn calculate_inclination_function_f_21p(&mut self, inclination: f64) {
        let sin_incl = sin!(inclination);
        let cos_incl = cos!(inclination);

        self.inclination_function_f_21p[0] = (3. / 4.) * sin_incl * (1. + cos_incl);
        self.inclination_function_f_21p[1] = -(3. / 2.) * sin_incl * cos_incl;
        self.inclination_function_f_21p[2] = (3. / 4.) * sin_incl * (cos_incl - 1.);
    }
    // The inclination function of the Kaula 1961 developpement Flmp(i)
    pub fn calculate_inclination_function_f_22p(&mut self, inclination: f64) {
        let sin_incl = sin!(inclination);
        let cos_incl = cos!(inclination);

        self.inclination_function_f_22p[0] = (3. / 4.) * (1. + cos_incl).powi(2);
        self.inclination_function_f_22p[1] = (3. / 2.) * sin_incl.powi(2);
        self.inclination_function_f_22p[2] = (3. / 4.) * (1. - cos_incl).powi(2);
    }
    // F3pmp
    // The inclination function of the Kaula 1961 developpement Flmp(i)
    pub fn calculate_inclination_function_f_30p(&mut self, inclination: f64) {
        let sin_incl = sin!(inclination);
        let sin_3_factor_1 = (5. / 16.) * sin_incl.powi(3);
        let sin_factor_2 = (3. / 4.) * sin_incl;

        self.inclination_function_f_30p[0] = -sin_3_factor_1;
        self.inclination_function_f_30p[1] = sin_3_factor_1 - sin_factor_2;
        self.inclination_function_f_30p[2] = -sin_3_factor_1 + sin_factor_2;
        self.inclination_function_f_30p[3] = -self.inclination_function_f_30p[0];
    }
    // The inclination function of the Kaula 1961 developpement Flmp(i)
    pub fn calculate_inclination_function_f_31p(&mut self, inclination: f64) {
        let sin_incl = sin!(inclination);
        let sin_2_factor_1 = (15. / 16.) * sin_incl.powi(2);
        let sin_2_factor_43 = sin_2_factor_1 - (3. / 4.);
        let cos_incl = cos!(inclination);
        let cos_incl_factor_1 = 1.0 + 3.0 * cos_incl;
        let cos_incl_factor_2 = 1.0 - 3.0 * cos_incl;

        self.inclination_function_f_31p[0] = -sin_2_factor_1 * cos_incl_factor_1;
        self.inclination_function_f_31p[1] = sin_2_factor_43 * cos_incl_factor_1;
        self.inclination_function_f_31p[2] = sin_2_factor_43 * cos_incl_factor_2;
        self.inclination_function_f_31p[3] = -sin_2_factor_1 * cos_incl_factor_2;
    }
    // The inclination function of the Kaula 1961 developpement Flmp(i)
    pub fn calculate_inclination_function_f_32p(&mut self, inclination: f64) {
        let sin_incl = sin!(inclination);
        let sin_2_factor_1 = (15. / 8.) * sin_incl.powi(2);
        let cos_incl = cos!(inclination);
        let cos_incl_2 = cos_incl.powi(2);

        self.inclination_function_f_32p[0] = sin_2_factor_1 * (1.0 + cos_incl).powi(2);
        self.inclination_function_f_32p[1] =
            sin_2_factor_1 * (1.0 - 2.0 * cos_incl - 3.0 * cos_incl_2);
        self.inclination_function_f_32p[2] =
            -sin_2_factor_1 * (1.0 + 2.0 * cos_incl - 3.0 * cos_incl_2);
        self.inclination_function_f_32p[3] = -sin_2_factor_1 * (1.0 - cos_incl).powi(2);
    }
    // The inclination function of the Kaula 1961 developpement Flmp(i)
    pub fn calculate_inclination_function_f_33p(&mut self, inclination: f64) {
        let sin_incl = sin!(inclination);
        let sin_2_factor_1 = (48. / 8.) * sin_incl.powi(2);
        let cos_incl = cos!(inclination);
        let cos_incl_factor_1 = 1.0 + cos_incl;
        let cos_incl_factor_2 = 1.0 - cos_incl;

        self.inclination_function_f_33p[0] = (15. / 8.) * cos_incl_factor_1.powi(3);
        self.inclination_function_f_33p[1] = sin_2_factor_1 * cos_incl_factor_1;
        self.inclination_function_f_33p[2] = sin_2_factor_1 * cos_incl_factor_2;
        self.inclination_function_f_33p[3] = (15. / 8.) * cos_incl_factor_2.powi(3);
    }
    // The eccentricity function of the Kaula 1961 developpement Glpq(e) from the Hensen coeff [?] (see Cayley table [?])
    pub fn calculate_eccentricity_function_g_2pq(&mut self, eccentricity: f64) {
        // in this notation:
        // [0] is -7
        // [1] is -6
        // [2] is -5
        // [3] is -4
        // [4] is -3
        // [5] is -2
        // [6] is -1
        // [7] is 0
        // [8] is 1
        // [9] is 2
        // [10] is 3
        // [11] is 4
        // [12] is 5
        // [13] is 6
        // [14] is 7

        let ecc = eccentricity;
        let ecc_2 = ecc * ecc;
        let ecc_3 = ecc_2 * ecc;
        let ecc_4 = ecc_3 * ecc;
        let ecc_5 = ecc_4 * ecc;
        let ecc_6 = ecc_5 * ecc;
        let ecc_7 = ecc_6 * ecc;

        let borrowed_slice = &mut self.eccentricity_function_g_2pq[..];
        let [first, second, third] = borrowed_slice else {
            unreachable!()
        };

        first[0] = (15_625. / 129_024.) * ecc_7;
        first[1] = (4. / 45.) * ecc_6;
        first[2] = (81. / 1280.) * ecc_5 + (81. / 2048.) * ecc_7;
        first[3] = (1. / 24.) * ecc_4 + (7. / 240.) * ecc_6;
        first[4] = (1. / 48.) * ecc_3 + (11. / 768.) * ecc_5 + (313. / 30_720.) * ecc_7;
        first[5] = 0.;
        first[6] = -0.5 * ecc + (1. / 16.) * ecc_3 - (5. / 384.) * ecc_5 - (143. / 18_432.) * ecc_7;
        first[7] = 1. - (5. / 2.) * ecc_2 + (13. / 16.) * ecc_4 - (35. / 288.) * ecc_6;
        first[8] = (7. / 2.) * ecc - (123. / 16.) * ecc_3 + (489. / 128.) * ecc_5
            - (1763. / 2048.) * ecc_7;
        first[9] = (17. / 2.) * ecc_2 - (115. / 16.) * ecc_4 + (601. / 48.) * ecc_6;
        first[10] = (845. / 48.) * ecc_3 - (32_525. / 768.) * ecc_5 + (208_225. / 6144.) * ecc_7;
        first[11] = (533. / 16.) * ecc_4 - (13_827. / 160.) * ecc_6;
        first[12] = (228_347. / 3840.) * ecc_5 - (3_071_075. / 18_432.) * ecc_7;
        first[13] = (73_369. / 720.) * ecc_6;
        first[14] = (12_144_273. / 71_680.) * ecc_7;

        second[0] = (432_091. / 30_720.) * ecc_7;
        second[1] = (3167. / 320.) * ecc_6;
        second[2] = (1773. / 256.) * ecc_5 - (4987. / 6144.) * ecc_7;
        second[3] = (77. / 16.) * ecc_4 + (129. / 160.) * ecc_6;
        second[4] = (53. / 16.) * ecc_3 + (393. / 256.) * ecc_5 + (24_753. / 10_240.) * ecc_7;
        second[5] = (9. / 4.) * ecc_2 + (7. / 4.) * ecc_4 + (141. / 64.) * ecc_6;
        second[6] = (3. / 2.) * ecc
            + (27. / 16.) * ecc_3
            + (261. / 128.) * ecc_5
            + (14_309. / 6144.) * ecc_7;
        second[7] = (1. - ecc_2).powf(-3. / 2.);
        // array reflected here
        let (left, right) = second.split_at_mut(8);
        right.copy_from_slice(&left[0..=6]);
        right.reverse();

        // Copy in reverse order
        third.copy_from_slice(&first[..]);
        third.reverse();
    }
    // The eccentricity function of the Kaula 1961 developpement G20q(e) from the Hensen coeff [?] (see Cayley table [?])
    pub fn calculate_eccentricity_function_g_20q(&mut self, eccentricity: f64) {
        // in this notation:
        // [0] is -7
        // [1] is -6
        // [2] is -5
        // [3] is -4
        // [4] is -3
        // [5] is -2
        // [6] is -1
        // [7] is 0
        // [8] is 1
        // [9] is 2
        // [10] is 3
        // [11] is 4
        // [12] is 5
        // [13] is 6
        // [14] is 7

        let ecc = eccentricity;
        let ecc_2 = ecc * ecc;
        let ecc_3 = ecc_2 * ecc;
        let ecc_4 = ecc_3 * ecc;
        let ecc_5 = ecc_4 * ecc;
        let ecc_6 = ecc_5 * ecc;
        let ecc_7 = ecc_6 * ecc;

        self.eccentricity_function_g_20q[0] = (15_625. / 129_024.) * ecc_7;
        self.eccentricity_function_g_20q[1] = (4. / 45.) * ecc_6;
        self.eccentricity_function_g_20q[2] = (81. / 1280.) * ecc_5 + (81. / 2048.) * ecc_7;
        self.eccentricity_function_g_20q[3] = (1. / 24.) * ecc_4 + (7. / 240.) * ecc_6;
        self.eccentricity_function_g_20q[4] =
            (1. / 48.) * ecc_3 + (11. / 768.) * ecc_5 + (313. / 30_720.) * ecc_7;
        self.eccentricity_function_g_20q[5] = 0.;
        self.eccentricity_function_g_20q[6] =
            -0.5 * ecc + (1. / 16.) * ecc_3 - (5. / 384.) * ecc_5 - (143. / 18_432.) * ecc_7;
        self.eccentricity_function_g_20q[7] =
            1. - (5. / 2.) * ecc_2 + (13. / 16.) * ecc_4 - (35. / 288.) * ecc_6;
        self.eccentricity_function_g_20q[8] = (7. / 2.) * ecc - (123. / 16.) * ecc_3
            + (489. / 128.) * ecc_5
            - (1763. / 2048.) * ecc_7;
        self.eccentricity_function_g_20q[9] =
            (17. / 2.) * ecc_2 - (115. / 16.) * ecc_4 + (601. / 48.) * ecc_6;
        self.eccentricity_function_g_20q[10] =
            (845. / 48.) * ecc_3 - (32_525. / 768.) * ecc_5 + (208_225. / 6144.) * ecc_7;
        self.eccentricity_function_g_20q[11] = (533. / 16.) * ecc_4 - (13_827. / 160.) * ecc_6;
        self.eccentricity_function_g_20q[12] =
            (228_347. / 3840.) * ecc_5 - (3_071_075. / 18_432.) * ecc_7;
        self.eccentricity_function_g_20q[13] = (73_369. / 720.) * ecc_6;
        self.eccentricity_function_g_20q[14] = (12_144_273. / 71_680.) * ecc_7;
    }
    // The eccentricity function of the Kaula 1961 developpement G21q(e) from the Hensen coeff [?] (see Cayley table [?])
    pub fn calculate_eccentricity_function_g_21q(&mut self, eccentricity: f64) {
        // in this notation:
        // [0] is -7
        // [1] is -6
        // [2] is -5
        // [3] is -4
        // [4] is -3
        // [5] is -2
        // [6] is -1
        // [7] is 0
        // [8] is 1
        // [9] is 2
        // [10] is 3
        // [11] is 4
        // [12] is 5
        // [13] is 6
        // [14] is 7

        let ecc = eccentricity;
        let ecc_2 = ecc * ecc;
        let ecc_3 = ecc_2 * ecc;
        let ecc_4 = ecc_3 * ecc;
        let ecc_5 = ecc_4 * ecc;
        let ecc_6 = ecc_5 * ecc;
        let ecc_7 = ecc_6 * ecc;

        self.eccentricity_function_g_21q[0] = (432_091. / 30_720.) * ecc_7;
        self.eccentricity_function_g_21q[1] = (3167. / 320.) * ecc_6;
        self.eccentricity_function_g_21q[2] = (1773. / 256.) * ecc_5 - (4987. / 6144.) * ecc_7;
        self.eccentricity_function_g_21q[3] = (77. / 16.) * ecc_4 + (129. / 160.) * ecc_6;
        self.eccentricity_function_g_21q[4] =
            (53. / 16.) * ecc_3 + (393. / 256.) * ecc_5 + (24_753. / 10_240.) * ecc_7;
        self.eccentricity_function_g_21q[5] =
            (9. / 4.) * ecc_2 + (7. / 4.) * ecc_4 + (141. / 64.) * ecc_6;
        self.eccentricity_function_g_21q[6] = (3. / 2.) * ecc
            + (27. / 16.) * ecc_3
            + (261. / 128.) * ecc_5
            + (14_309. / 6144.) * ecc_7;
        self.eccentricity_function_g_21q[7] = (1. - ecc_2).powf(-3. / 2.);
        self.eccentricity_function_g_21q[8] = self.eccentricity_function_g_21q[6];
        self.eccentricity_function_g_21q[9] = self.eccentricity_function_g_21q[5];
        self.eccentricity_function_g_21q[10] = self.eccentricity_function_g_21q[4];
        self.eccentricity_function_g_21q[11] = self.eccentricity_function_g_21q[3];
        self.eccentricity_function_g_21q[12] = self.eccentricity_function_g_21q[2];
        self.eccentricity_function_g_21q[13] = self.eccentricity_function_g_21q[1];
        self.eccentricity_function_g_21q[14] = self.eccentricity_function_g_21q[0];
    }
    // The eccentricity function of the Kaula 1961 developpement Glpq(e) from the Hensen coeff [?] (see Cayley table [?])
    pub fn calculate_eccentricity_function_g_3pq(&mut self, eccentricity: f64) {
        // in this notation:
        // [0] is -7
        // [1] is -6
        // [2] is -5
        // [3] is -4
        // [4] is -3
        // [5] is -2
        // [6] is -1
        // [7] is 0
        // [8] is 1
        // [9] is 2
        // [10] is 3
        // [11] is 4
        // [12] is 5
        // [13] is 6
        // [14] is 7

        let ecc = eccentricity;
        let ecc_2 = ecc.powi(2);
        let ecc_3 = ecc.powi(3);
        let ecc_4 = ecc.powi(4);
        let ecc_5 = ecc.powi(5);
        let ecc_6 = ecc.powi(6);
        let ecc_7 = ecc.powi(7);

        let borrowed_slice = &mut self.eccentricity_function_g_3pq[..];
        let [first, second, third, fourth] = borrowed_slice else {
            unreachable!()
        };

        first[0] = (8. / 315.) * ecc_7;
        first[1] = (81. / 5120.) * ecc_6;
        first[2] = (1. / 120.) * ecc_5 + (13. / 1440.) * ecc_7;
        first[3] = (1. / 384.) * ecc_4 + (1. / 384.) * ecc_6;
        first[4] = 0.;
        first[5] = (1. / 8.) * ecc_2 + (1. / 48.) * ecc_4 + (55. / 3072.) * ecc_6;
        first[6] = -ecc + (5. / 4.) * ecc_3 - (7. / 48.) * ecc_5 + (23. / 288.) * ecc_7;
        first[7] = 1. - 6. * ecc_2 + (423. / 64.) * ecc_4 - (125. / 64.) * ecc_6;
        first[8] = 5. * ecc - 22. * ecc_3 + (607. / 24.) * ecc_5 - (98. / 9.) * ecc_7;
        first[9] = (127. / 8.) * ecc_2 - (3065. / 48.) * ecc_4 + (243_805. / 3072.) * ecc_6;
        first[10] = (163. / 4.) * ecc_3 - (2577. / 16.) * ecc_5 + (1089. / 5.) * ecc_7;
        first[11] = (35_413. / 384.) * ecc_4 - (709_471. / 1920.) * ecc_6;
        first[12] = (23_029. / 120.) * ecc_5 - (35_614. / 45.) * ecc_7;
        first[13] = (385_095. / 1024.) * ecc_6;
        first[14] = (44_377. / 63.) * ecc_7;

        second[0] = (16_337. / 2240.) * ecc_7;
        second[1] = (48_203. / 9240.) * ecc_6;
        second[2] = (899. / 240.) * ecc_5 + (2441. / 480.) * ecc_7;
        second[3] = (343. / 128.) * ecc_4 + (2819. / 640.) * ecc_6;
        second[4] = (23. / 12.) * ecc_3 + (89. / 24.) * ecc_5 + (5663. / 960.) * ecc_7;
        second[5] = (11. / 8.) * ecc_2 + (49. / 16.) * ecc_4 + (15_665. / 3072.) * ecc_6;
        second[6] = ecc * (1. - ecc_2).powf(-5. / 2.);
        second[7] = 1. + (1. / 2.) * ecc_2 + (239. / 64.) * ecc_4 - (3323. / 576.) * ecc_6;
        second[8] = 3. * ecc + (11. / 4.) * ecc_3 + (245. / 48.) * ecc_5 + (463. / 64.) * ecc_7;
        second[9] = (53. / 8.) * ecc_2 + (39. / 16.) * ecc_4 + (7041. / 1024.) * ecc_6;
        second[10] = (163. / 4.) * ecc_3 - (2577. / 16.) * ecc_5 + (1089. / 5.) * ecc_7;
        second[11] = (35_413. / 384.) * ecc_4 - (709_471. / 1920.) * ecc_6;
        second[12] = (23_029. / 120.) * ecc_5 - (35_614. / 45.) * ecc_7;
        second[13] = (385_095. / 1024.) * ecc_6;
        second[14] = (44_377. / 63.) * ecc_7;

        // Copy in reverse order
        third.copy_from_slice(&second[..]);
        third.reverse();

        // Copy in reverse order
        fourth.copy_from_slice(&first[..]);
        fourth.reverse();
    }
}
