use crate::data::{CL, EVEN_THETA_ZEROS, EVEN_WEIGHTS, J1, JZ, ODD_THETA_ZEROS, ODD_WEIGHTS};

use core::{cmp::Ordering, num::NonZeroUsize};
use std::f64::consts::PI;

#[cfg(feature = "parallel")]
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator,
};
#[cfg(feature = "serde_support")]
use serde::{Deserialize, Serialize};

/// Generate a [`Vec`] of [`GlqPair`]s for manual integration. The pairs are ordered
/// by their x-position.
/// # Example
/// ```
/// # use fast_gauleg::glq_pairs;
/// # use approx::assert_relative_eq;
/// let f: fn(f64) -> f64 = |x| 2.0 * x * x + 1.0;
/// let res: f64 = glq_pairs(3.try_into().unwrap())
///     .into_iter()
///     .map(|pair| 0.5 * pair.weight() * f(0.5 * pair.position() + 0.5))
///     .sum();
/// assert_relative_eq!(res, 2.0 / 3.0 + 1.0);
/// ```
#[must_use = "the function returns a new value and does not modify the input"]
pub fn glq_pairs(nodes: NonZeroUsize) -> Vec<GlqPair> {
    (1..=nodes.get())
        .map(|k| GlqPair::new(nodes.into(), k))
        .collect()
}

/// Writes [`GlqPair`]s to an already allocated slice for manual integration.
/// Does nothing if the slice is empty.
pub fn write_glq_pairs(points: &mut [GlqPair]) {
    let l = points.len();
    for (i, point) in points.iter_mut().enumerate() {
        *point = GlqPair::new(l, i + 1);
    }
}

#[cfg(feature = "parallel")]
/// Same as [`glq_pairs`] but parallel.
#[must_use = "the function returns a new value and does not modify the input"]
pub fn par_glq_pairs(points: NonZeroUsize) -> Vec<GlqPair> {
    (1..=points.get())
        .into_par_iter()
        .map(|k| GlqPair::new(points.into(), k))
        .collect()
}

#[cfg(feature = "parallel")]
/// Same as [`write_glq_pairs`] but parallel.
pub fn par_write_glq_pairs(points: &mut [GlqPair]) {
    let l = points.len();
    points
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, point)| *point = GlqPair::new(l, i + 1));
}

/// This function computes the `k`th zero of j_0
#[rustfmt::skip]
#[inline]
#[must_use]
fn bessel_j0_zero(k: usize) -> f64 {
    if k > 20 {
        let z: f64 = PI * (k as f64 - 0.25);
        let r = 1.0 / z;
        let r2 = r * r;
        z + r * (0.125 + r2 * (-8.072_916_666_666_667e-2 + r2 * (0.246_028_645_833_333_34 + r2 * (-1.824_438_767_206_101 + r2 * (25.336_414_797_343_906 + r2 * (-567.644_412_135_183_4 + r2 * (18_690.476_528_232_066  + r2 * (-8.493_535_802_991_488e5 + 5.092_254_624_022_268e7 * r2))))))))
    } else {
        JZ[k - 1]
    }
}

/// This function computes j_1(`k`th zero of j_0)^2
#[rustfmt::skip]
#[inline]
#[must_use]
fn bessel_j1_squared(k: usize) -> f64 {
    if k > 21 {
        let x: f64 = 1.0 / (k as f64 - 0.25);
        let x2 = x * x;
        x * (0.202_642_367_284_675_55 + x2 * x2 * (-3.033_804_297_112_902_7e-4 + x2 * (1.989_243_642_459_693e-4 + x2 * (-2.289_699_027_721_116_6e-4 + x2 * (4.337_107_191_307_463e-4 + x2 * (-1.236_323_497_271_754e-3 + x2 * (4.961_014_232_688_831_4e-3 + x2 * (-2.668_373_937_023_237_7e-2 + 0.185_395_398_206_345_62 * x2))))))))
    } else {
        J1[k - 1]
    }
}

/// A Gauss-Legendre node-weight pair used for manual integration.
/// # Example
/// Integrate `f(x) = e^x` in the interval `[-1, 1]`.
/// ```
/// # use fast_gauleg::GlqPair;
/// let n = 9;
/// assert_eq!(
///     (1..=n).map(|k| {
///         let p = GlqPair::new(n, k);
///         p.weight() * p.position().exp()
///     }).sum::<f64>(),
///     1.0_f64.exp() - (-1.0_f64).exp(),
/// )
/// ```
/// Integrate `f(x) = x^2 - x - 1` in the interval `[0, 1]`.
/// ```
/// # use fast_gauleg::GlqPair;
/// let n = 3;
/// let f = |x| x * x - x - 1.0;
/// assert_eq!(
///     (1..=n).map(|k| {
///         let p = GlqPair::new(n, k);
///         0.5 * p.weight() * f(0.5 * p.position() + 0.5)
///     }).sum::<f64>(),
///     -7.0 / 6.0,
/// );
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct GlqPair {
    position: f64,
    weight: f64,
}

impl GlqPair {
    /// Returns the `k`th node-weight pair associated with the `n`-node quadrature rule.
    /// # Panic
    /// Panics if `k = 0` or `n < k`.
    #[must_use = "the associated method returns a new GlqPair and does not modify the inputs"]
    pub fn new(n: usize, k: usize) -> Self {
        GlqThetaWeightPair::new(n, k).into()
    }

    /// Returns the x-position.
    #[inline]
    #[must_use = "the method returns a value and does not modify `self`"]
    pub const fn position(&self) -> f64 {
        self.position
    }

    /// Returns the weight.
    #[inline]
    #[must_use = "the method returns a value and does not modify `self`"]
    pub const fn weight(&self) -> f64 {
        self.weight
    }
}

impl core::convert::From<GlqThetaWeightPair> for GlqPair {
    #[inline]
    #[must_use = "`value` will be dropped if the result is not used"]
    fn from(value: GlqThetaWeightPair) -> Self {
        Self {
            position: value.theta.cos(),
            weight: value.weight,
        }
    }
}

/// A Gauss-Legendre node-weight pair in theta-space.
struct GlqThetaWeightPair {
    theta: f64,
    weight: f64,
}

impl GlqThetaWeightPair {
    /// Compute a new GlqPair in theta-space
    /// # Panic
    /// Panics if `k = 0` or `n < k`.
    #[must_use]
    fn new(n: usize, k: usize) -> Self {
        assert_ne!(k, 0);
        assert!(n >= k);
        if n <= 100 {
            Self::gl_pair_tabulated(n, k - 1)
        } else if 2 * k - 1 > n {
            let mut p = Self::gl_pair_computed(n, n - k + 1);
            p.theta = PI - p.theta;
            p
        } else {
            Self::gl_pair_computed(n, k)
        }
    }

    /// Compute a node-weight pair, with k limited to half the range
    #[rustfmt::skip]
    #[must_use]
    fn gl_pair_computed(n: usize, k: usize) -> Self {
        // First get the j_0 zero
        let w: f64 = 1.0 / (n as f64 + 0.5);
        let nu = bessel_j0_zero(k);
        let theta = w * nu;
        let x = theta * theta;

        // Get the asymptotic j_1(nu) squared
        let b = bessel_j1_squared(k);

        // Get the Chebyshev interpolants for the nodes...
        let sf1t = (((((-1.290_529_962_742_805_1e-12 * x + 2.407_246_858_643_301_3e-10) * x - 3.131_486_546_359_920_4e-8) * x + 2.755_731_689_620_612_4e-6) * x - 1.488_095_237_139_091_4e-4) * x + 4.166_666_666_651_934e-3) * x - 4.166_666_666_666_63e-2;
        let sf2t = (((((2.206_394_217_818_71e-9 * x - 7.530_367_713_737_693e-8) * x + 1.619_692_594_538_362_7e-6) * x - 2.533_003_260_082_32e-5) * x + 2.821_168_860_575_604_5e-4) * x - 2.090_222_483_878_529e-3) * x + 8.159_722_217_729_322e-3;
        let sf3t = (((((-2.970_582_253_755_262_3e-8 * x + 5.558_453_302_237_962e-7) * x - 5.677_978_413_568_331e-6) * x + 4.184_981_003_295_046e-5) * x - 2.513_952_932_839_659e-4) * x + 1.286_541_985_428_451_3e-3) * x - 4.160_121_656_202_043e-3;

        // ...and weights
        let wsf1t = ((((((((-2.209_028_610_446_166_4e-14 * x + 2.303_657_268_603_773_8e-12) * x - 1.752_577_007_354_238e-10) * x + 1.037_560_669_279_168e-8) * x - 4.639_686_475_532_213e-7) * x + 1.496_445_936_250_286_4e-5) * x - 3.262_786_595_944_122e-4) * x + 4.365_079_365_075_981e-3) * x - 3.055_555_555_555_53e-2) * x + 8.333_333_333_333_333e-2;
        let wsf2t = (((((((3.631_174_121_526_548e-12 * x + 7.676_435_450_698_932e-11) * x - 7.129_128_572_336_422e-9) * x + 2.114_838_806_859_471_6e-7) * x - 3.818_179_186_800_454e-6) * x + 4.659_695_306_949_684e-5) * x - 4.072_971_856_113_357_5e-4) * x + 2.689_594_356_947_297e-3) * x - 1.111_111_111_112_149_2e-2;
        let wsf3t = (((((((2.018_267_912_567_033e-9 * x - 4.386_471_225_202_067e-8) * x + 5.088_983_472_886_716e-7) * x - 3.979_333_165_191_352_5e-6) * x + 2.005_593_263_964_583_4e-5) * x - 4.228_880_592_829_212e-5) * x - 1.056_460_502_540_761_4e-4) * x - 9.479_693_089_585_773e-5) * x + 6.569_664_899_264_848e-3;

        // Then refine with the expansions from the paper
        let nu_o_sin = nu / theta.sin();
        let b_nu_o_sin = b * nu_o_sin;
        let w_inv_sinc = w * w * nu_o_sin;
        let wis2 = w_inv_sinc * w_inv_sinc;

        // Compute the theta-node and the weight
        let theta = w * (nu + theta * w_inv_sinc * (sf1t + wis2 * (sf2t + wis2 * sf3t)));
        let weight = 2.0 * w / (b_nu_o_sin + b_nu_o_sin * wis2 * (wsf1t + wis2 * (wsf2t + wis2 * wsf3t)));

        Self { theta, weight }
    }

    /// Returns tabulated theta and weight values, valid for l <= 100
    #[must_use]
    fn gl_pair_tabulated(l: usize, k: usize) -> Self {
        // Odd Legendre degree
        let (theta, weight) = if l % 2 == 1 {
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

#[cfg(test)]
mod test {
    use super::GlqPair;
    use approx::assert_relative_eq;

    #[test]
    fn check_manual_integrations() {
        let l = 9;
        assert_eq!(
            (1..=l)
                .map(|k| {
                    let temp = GlqPair::new(l, k);
                    temp.weight() * temp.position().exp()
                })
                .sum::<f64>(),
            (1.0_f64).exp() - (-1.0_f64).exp(),
        );

        let l = 600;
        assert_relative_eq!(
            (1..=l)
                .map(|k| {
                    let temp = GlqPair::new(l, k);
                    temp.weight() * (1000.0 * temp.position()).cos()
                })
                .sum::<f64>(),
            (1000.0_f64).sin() / 500.0,
            epsilon = 1e-14,
        );

        let l = 1_000_000;
        assert_relative_eq!(
            (1..=l)
                .map(|k| {
                    let temp = GlqPair::new(l, k);
                    0.5 * temp.weight() * (0.5 * (temp.position() + 1.0)).ln()
                })
                .sum::<f64>(),
            -1.0,
            epsilon = 1e-12,
        );
    }
}
