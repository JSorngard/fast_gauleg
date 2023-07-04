//! *******************************************
//!    Copyright (C) 2014 by Ignace Bogaert   *
//! *******************************************
//! This software package is based on the paper
//!    I. Bogaert, "Iteration-Free Computation of Gauss-Legendre Quadrature Nodes and Weights",
//!    to be published in the SIAM Journal of Scientific Computing.
//!
//! The main features of this software are:
//! - Speed: due to the simple formulas and the O(1) complexity computation of individual Gauss-Legendre
//!   quadrature nodes and weights. This makes it compatible with parallel computing paradigms.
//! - Accuracy: the error on the nodes and weights is within a few ulps (see the paper for details).
//!
//! Disclaimer:
//! THIS SOFTWARE IS PROVIDED "AS IS" AND ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
//! WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR
//! CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
//! BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
//! HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
//! OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//!
//! It was ported to Rust and extended by Johanna Sörngård in 2023.

use crate::data::{CL, EVEN_THETA_ZEROS, EVEN_WEIGHTS, J1, JZ, ODD_THETA_ZEROS, ODD_WEIGHTS};
use core::cmp::Ordering;
use std::f64::consts::PI;

pub fn gauleg(points: &mut [QuadPair]) {
    assert!(!points.is_empty());

    let l = points.len();
    for (i, point) in points.iter_mut().enumerate() {
        *point = QuadPair::new(l, i + 1);
    }
}

/// This function computes the kth zero of the BesselJ(0,x)
fn besselj0_zero(k: usize) -> f64 {
    if k > 20 {
        let z: f64 = PI * (k as f64 - 0.25);
        let r = 1.0 / z;
        let r2 = r * r;
        z + r
            * (0.125
                + r2 * (-8.072_916_666_666_667e-2
                    + r2 * (0.246_028_645_833_333_34
                        + r2 * (-1.824_438_767_206_101
                            + r2 * (25.336_414_797_343_906
                                + r2 * (-567.644_412_135_183_4
                                    + r2 * (18_690.476_528_232_066
                                        + r2 * (-8.493_535_802_991_488e5
                                            + 5.092_254_624_022_268e7 * r2))))))))
    } else {
        JZ[k - 1]
    }
}

/// This function computes the square of BesselJ(1, BesselZero(0,k))
fn besselj1_squared(k: usize) -> f64 {
    if k > 21 {
        let x: f64 = 1.0 / (k as f64 - 0.25);
        let x2 = x * x;
        x * (0.202_642_367_284_675_55
            + x2 * x2
                * (-3.033_804_297_112_902_7e-4
                    + x2 * (1.989_243_642_459_693e-4
                        + x2 * (-2.289_699_027_721_116_6e-4
                            + x2 * (4.337_107_191_307_463e-4
                                + x2 * (-1.236_323_497_271_754e-3
                                    + x2 * (4.961_014_232_688_831_4e-3
                                        + x2 * (-2.668_373_937_023_237_7e-2
                                            + 0.185_395_398_206_345_62 * x2))))))))
    } else {
        J1[k - 1]
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct QuadPair {
    theta: f64,
    weight: f64,
}

impl QuadPair {
    fn new(n: usize, k: usize) -> Self {
        assert!(k > 0);
        assert!(n >= k);
        if n < 101 {
            Self::gl_pair_tabulated(n, k - 1)
        } else if 2 * k - 1 > n {
            let mut p = Self::gl_pair_s(n, n - k + 1);
            p.theta = PI - p.theta;
            p
        } else {
            Self::gl_pair_s(n, k)
        }
    }

    pub fn x(&self) -> f64 {
        self.theta.cos()
    }

    pub const fn w(&self) -> f64 {
        self.weight
    }

    /// Compute a node-weight pair, with k limited to half the range
    fn gl_pair_s(n: usize, k: usize) -> Self {
        // First get the Bessel zero
        let w: f64 = 1.0 / (n as f64 + 0.5);
        let nu = besselj0_zero(k);
        let theta = w * nu;
        let x = theta * theta;

        // Get the asymptotic BesselJ(1, nu) squared
        let b = besselj1_squared(k);

        // Get the Chebyshev interpolants for the nodes...
        let sf1t = (((((-1.290_529_962_742_805_1e-12 * x + 2.407_246_858_643_301_3e-10) * x
            - 3.131_486_546_359_920_4e-8)
            * x
            + 2.755_731_689_620_612_4e-6)
            * x
            - 1.488_095_237_139_091_4e-4)
            * x
            + 4.166_666_666_651_934e-3)
            * x
            - 4.166_666_666_666_63e-2;
        let sf2t = (((((2.206_394_217_818_71e-9 * x - 7.530_367_713_737_693e-8) * x
            + 1.619_692_594_538_362_7e-6)
            * x
            - 2.533_003_260_082_32e-5)
            * x
            + 2.821_168_860_575_604_5e-4)
            * x
            - 2.090_222_483_878_529e-3)
            * x
            + 8.159_722_217_729_322e-3;
        let sf3t = (((((-2.970_582_253_755_262_3e-8 * x + 5.558_453_302_237_962e-7) * x
            - 5.677_978_413_568_331e-6)
            * x
            + 4.184_981_003_295_046e-5)
            * x
            - 2.513_952_932_839_659e-4)
            * x
            + 1.286_541_985_428_451_3e-3)
            * x
            - 4.160_121_656_202_043e-3;

        // ...and for the weights
        let wsf1t = ((((((((-2.209_028_610_446_166_4e-14 * x + 2.303_657_268_603_773_8e-12)
            * x
            - 1.752_577_007_354_238e-10)
            * x
            + 1.037_560_669_279_168e-8)
            * x
            - 4.639_686_475_532_213e-7)
            * x
            + 1.496_445_936_250_286_4e-5)
            * x
            - 3.262_786_595_944_122e-4)
            * x
            + 4.365_079_365_075_981e-3)
            * x
            - 3.055_555_555_555_53e-2)
            * x
            + 8.333_333_333_333_333e-2;
        let wsf2t = (((((((3.631_174_121_526_548e-12 * x + 7.676_435_450_698_932e-11) * x
            - 7.129_128_572_336_422e-9)
            * x
            + 2.114_838_806_859_471_6e-7)
            * x
            - 3.818_179_186_800_454e-6)
            * x
            + 4.659_695_306_949_684e-5)
            * x
            - 4.072_971_856_113_357_5e-4)
            * x
            + 2.689_594_356_947_297e-3)
            * x
            - 1.111_111_111_112_149_2e-2;
        let wsf3t = (((((((2.018_267_912_567_033e-9 * x - 4.386_471_225_202_067e-8) * x
            + 5.088_983_472_886_716e-7)
            * x
            - 3.979_333_165_191_352_5e-6)
            * x
            + 2.005_593_263_964_583_4e-5)
            * x
            - 4.228_880_592_829_212e-5)
            * x
            - 1.056_460_502_540_761_4e-4)
            * x
            - 9.479_693_089_585_773e-5)
            * x
            + 6.569_664_899_264_848e-3;

        // Then refine with the paper expansions
        let nu_o_sin = nu / theta.sin();
        let b_nu_o_sin = b * nu_o_sin;
        let w_inv_sinc = w * w * nu_o_sin;
        let wis2 = w_inv_sinc * w_inv_sinc;

        // Finally compute the node and the weight
        let theta = w * (nu + theta + w_inv_sinc * (sf1t + wis2 * (sf2t + wis2 * sf3t)));
        let deno = b_nu_o_sin + b_nu_o_sin * wis2 * (wsf1t + wis2 * (wsf2t + wis2 * wsf3t));
        let weight = 2.0 * w / deno;

        Self { theta, weight }
    }

    /// Returns tabulated theta and weight values: valid for l <= 100
    fn gl_pair_tabulated(l: usize, k: usize) -> Self {
        // Odd Legendre degree
        let (theta, weight) = if l % 2 == 1 {
            // originally l & 1
            let l2 = (l - 1) / 2;
            match k.cmp(&l2) {
                Ordering::Equal => (PI / 2.0, 2.0 / (CL[l] * CL[l])),
                Ordering::Less => (
                    ODD_THETA_ZEROS[l2 - 1][l2 - k - 1],
                    ODD_WEIGHTS[l2 - 1][l2 - k - 1],
                ),
                Ordering::Greater => (
                    PI - ODD_THETA_ZEROS[l2 - 1][k - l2 - 1],
                    ODD_WEIGHTS[l2 - 1][k - l2 - 1],
                ),
            }
        // Even Legendre degree
        } else {
            let l2 = l / 2;
            match k.cmp(&l2) {
                Ordering::Less => (
                    EVEN_THETA_ZEROS[l2 - 1][l2 - k - 1],
                    EVEN_WEIGHTS[l2 - 1][l2 - k - 1],
                ),
                Ordering::Equal | Ordering::Greater => (
                    PI - EVEN_THETA_ZEROS[l2 - 1][k - l2],
                    EVEN_WEIGHTS[l2 - 1][k - l2],
                ),
            }
        };
        Self { theta, weight }
    }
}
