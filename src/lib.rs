//! This crate contains tools for numerical integration with [Gauss-Legendre quadrature](https://en.wikipedia.org/wiki/Gauss%E2%80%93Legendre_quadrature).
//! This is a method that allows integration of polynomial functions with very few evaluation points.
//!
//! The nodes and weights used during integration are computed with the fast and parallelizable method developed by
//! [Ignace Bogaert](https://doi.org/10.1137/140954969).
//! # Examples
//! Integrate degree five polynomials while only evaluating them at three points:
//! ```
//! use gl_quadrature::glq_integrate;
//! // This macro is used in the docs of this crate to compare floating point values.
//! // The assertion succeeds if the two values are within floating point error of each other,
//! // or their relative difference is within an optional epsilon.
//! use approx::assert_relative_eq;
//! use core::num::NonZeroUsize;
//!
//! let pts = NonZeroUsize::new(3).unwrap();
//!
//! assert_relative_eq!(
//!     glq_integrate(-1.0, 1.0, |x| 0.25 * (3.0 * x.powf(2.0) - 1.0) * (5.0 * x.powf(3.0) - 3.0 * x), pts),
//!     0.0,
//! );
//! assert_relative_eq!(
//!     glq_integrate(-5.0, 2.0, |x| 0.125 * (63.0 * x.powf(5.0) - 70.0 * x.powf(3.0) + 15.0 * x), pts),
//!     -305781.0 / 16.0,
//!     // The exponent in the epsilon is always chosen
//!     // to be as small as possible
//!     // while still passing the assertion.
//!     epsilon = 1e-10,
//! );
//! ```
//! Integrate a trancendental function:
//! ```
//! # use approx::assert_relative_eq;
//! # use gl_quadrature::glq_integrate;
//! assert_relative_eq!(
//!     glq_integrate(0.0, 1.0, |x| (x + 1.0).ln().sin(), 10.try_into().unwrap()),
//!     0.5 - f64::ln(2.0).cos() + f64::ln(2.0).sin(),
//! );   
//! ```
//! Divergences can be hard to integrate with this method.
//! Integration with many points can compensate for this, and is still fast
//! ```
//! # use approx::assert_relative_eq;
//! # use gl_quadrature::glq_integrate;
//! # #[cfg(feature = "parallel")]
//! # use gl_quadrature::par_glq_integrate;
//! assert_relative_eq!(
//!     glq_integrate(0.0, 1.0, |x| x.ln(), 10.try_into().unwrap()),
//!     -1.0,
//!     epsilon = 1e-2,
//! );
//! assert_relative_eq!(
//!     glq_integrate(0.0, 1.0, |x| x.ln(), 1_000_000.try_into().unwrap()),
//!     -1.0,
//!     epsilon = 1e-12,
//! );
//! // Very large calculations can be done in parallel (needs the `parallel` feature)
//! # #[cfg(feature = "parallel")]
//! assert_relative_eq!(
//!     par_glq_integrate(0.0, 1.0, |x| x.ln(), 100_000_000.try_into().unwrap()),
//!     -1.0,
//!     epsilon = 1e-15,
//! );
//! ```
//! If many integrals need to be computed the crate provides [`GlqIntegrator`], which reuses
//! the calculation of the quadrature nodes and weights:
//! ```
//! # use approx::assert_relative_eq;
//! use gl_quadrature::GlqIntegrator;
//! let integrator = GlqIntegrator::new(10.try_into().unwrap());
//! assert_relative_eq!(
//!     integrator.integrate(0.0, 2.0 * std::f64::consts::PI, |theta| theta.sin() * theta.cos()),
//!     0.0,
//!     epsilon = 1e-15,
//! );
//! for n in 0..=19 {
//!     assert_relative_eq!(
//!         integrator.integrate(-1.0, 1.0, |x| x.powf(n.into())),
//!         2.0 * f64::from(n % 2 == 0) / f64::from(n + 1),
//!     );
//! }
//! ```

#![cfg_attr(docsrs, feature(doc_cfg))]

#[rustfmt::skip]
mod data;
mod glq_nodes_and_weights;
pub use glq_nodes_and_weights::GlqPair;
use glq_nodes_and_weights::{new_gauleg, write_gauleg};
#[cfg(feature = "parallel")]
use glq_nodes_and_weights::{par_new_gauleg, par_write_gauleg};

use core::num::NonZeroUsize;

#[cfg(feature = "parallel")]
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
#[cfg(feature = "serde_support")]
use serde::{Deserialize, Serialize};

/// An object that can integrate `Fn(f64) -> f64` functions and closures.
/// # Examples
/// Integrate degree 5 polynomials with only 3 evaluation points:
/// ```
/// # use gl_quadrature::GlqIntegrator;
/// use approx::assert_relative_eq;
/// let integrator = GlqIntegrator::new(3.try_into().unwrap());
/// assert_relative_eq!(
///     integrator.integrate(0.0, 1.0, |x| x.powf(5.0)),
///     1.0 / 6.0,
/// );
/// assert_relative_eq!(
///     integrator.integrate(-1.0, 1.0, |x| x.powf(5.0) - 2.0 * x.powf(4.0) + 1.0),
///     6.0 / 5.0,
/// );
/// ```
/// Non-polynomial functions need more points to evaluate correctly
/// ```
/// # use gl_quadrature::GlqIntegrator;
/// # use approx::assert_relative_eq;
/// let mut integrator = GlqIntegrator::new(3.try_into().unwrap());
/// assert_relative_eq!(
///     integrator.integrate(0.0, std::f64::consts::PI, |x| x.sin()),
///     2.0,
///     epsilon = 1e-2 // Very bad accuracy
/// );
/// integrator.change_number_of_points(58.try_into().unwrap());
/// assert_relative_eq!(
///     integrator.integrate(0.0, std::f64::consts::PI, |x| x.sin()),
///     2.0,
///     epsilon = 1e-15 // Much better!
/// );
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct GlqIntegrator {
    xs_and_ws: Vec<GlqPair>,
    points: NonZeroUsize,
}

impl GlqIntegrator {
    /// Creates a new integrator that integrates functions using the given number
    /// of evaluation points.
    #[must_use = "associated method returns a new instance and does not modify the input value"]
    pub fn new(points: NonZeroUsize) -> Self {
        let xs_and_ws = new_gauleg(points);
        Self { xs_and_ws, points }
    }

    #[cfg(feature = "parallel")]
    #[cfg_attr(docsrs, doc(cfg(feature = "parallel")))]
    /// Same as [`new`](GlqIntegrator::new) but parallel.
    #[must_use = "associated method returns a new instance and does not modify the input value"]
    pub fn par_new(points: NonZeroUsize) -> Self {
        let xs_and_ws = par_new_gauleg(points);
        Self { xs_and_ws, points }
    }

    /// Integrates the given function over the given domain.
    #[must_use = "the method returns a new value and does not modify `self` or the inputs"]
    pub fn integrate<F>(&self, start: f64, end: f64, f: F) -> f64
    where
        F: Fn(f64) -> f64,
    {
        let width = (end - start) * 0.5;
        let offset = (start + end) * 0.5;
        self.xs_and_ws
            .iter()
            .map(|p| p.weight() * f(width * p.position() + offset))
            .sum::<f64>()
            * width
    }

    #[cfg(feature = "parallel")]
    #[cfg_attr(docsrs, doc(cfg(feature = "parallel")))]
    /// Same as [`integrate`](GlqIntegrator::integrate) but parallel.
    #[must_use = "the method returns a new value and does not modify `self` or the inputs"]
    pub fn par_integrate<F>(&self, start: f64, end: f64, f: F) -> f64
    where
        F: Fn(f64) -> f64 + Send + Sync,
    {
        let width = (end - start) * 0.5;
        let offset = (start + end) * 0.5;
        self.xs_and_ws
            .par_iter()
            .map(|p| p.weight() * f(width * p.position() + offset))
            .sum::<f64>()
            * width
    }

    /// Returns the number of points in the integration domain
    #[must_use = "the method returns a value and does not modify `self`"]
    #[inline(always)]
    pub const fn points(&self) -> NonZeroUsize {
        self.points
    }

    /// Changes the number of points used during integration.
    /// If the number is not increased the old allocation is reused.
    pub fn change_number_of_points(&mut self, new_points: NonZeroUsize) {
        self.xs_and_ws.resize(new_points.into(), GlqPair::default());
        write_gauleg(&mut self.xs_and_ws);
        self.points = new_points;
    }

    #[cfg(feature = "parallel")]
    #[cfg_attr(docsrs, doc(cfg(feature = "parallel")))]
    /// Same as [`change_number_of_points`](GlqIntegrator::change_number_of_points) but parallel.
    pub fn par_change_number_of_points(&mut self, new_points: NonZeroUsize) {
        self.xs_and_ws.resize(new_points.into(), GlqPair::default());
        par_write_gauleg(&mut self.xs_and_ws);
        self.points = new_points;
    }

    /// Converts self into a `Vec` of [`GlqPair`]s.
    #[inline]
    #[must_use = "`self` will be dropped if the result is not used"]
    pub fn into_glq_pairs(self) -> Vec<GlqPair> {
        self.xs_and_ws
    }

    /// Return a slice of the underlying [`GlqPair`]s.
    #[inline]
    #[must_use = "this method returns a new value and does not modify the original"]
    pub fn as_glq_pairs(&self) -> &[GlqPair] {
        &self.xs_and_ws
    }
}

/// Integrates the given function over the interval `[start, end]`
/// using [Gauss-Legendre quadrature](https://en.wikipedia.org/wiki/Gauss%E2%80%93Legendre_quadrature)
/// with `points` evaluation points.
/// # Example
/// ```
/// # use gl_quadrature::glq_integrate;
/// use approx::assert_relative_eq;
/// use core::num::NonZeroUsize;
/// // Integrate degree 2 and 3 polynomials with only 2 points:
/// let pts = NonZeroUsize::new(2).unwrap();
/// assert_relative_eq!(
///     glq_integrate(0.0, 1.0, |x| x * x, pts),
///     1.0 / 3.0,
/// );
/// assert_eq!(
///     glq_integrate(-1.0, 1.0, |x| 0.5 * (3.0 * x * x - 1.0) * x, pts),
///     0.0
/// );
/// // Non-polynomials need more points to evaluate correctly:
/// const END: f64 = 10.0;
/// assert_relative_eq!(
///     glq_integrate(0.0, END, |x| x * (-x).exp(), 13.try_into().unwrap()),
///     (1.0 - (1.0 + END) * (-END).exp()),
/// );
/// ```
#[must_use = "the function returns a value and does not modify its inputs"]
pub fn glq_integrate<F>(start: f64, end: f64, f: F, points: NonZeroUsize) -> f64
where
    F: Fn(f64) -> f64,
{
    let width = (end - start) * 0.5;
    let offset = (start + end) * 0.5;
    new_gauleg(points)
        .into_iter()
        .map(|p| p.weight() * f(width * p.position() + offset))
        .sum::<f64>()
        * width
}

#[cfg(feature = "parallel")]
#[cfg_attr(docsrs, doc(cfg(feature = "parallel")))]
/// Same as [`glq_integrate`] but parallel.
#[must_use = "the function returns a value and does not modify its inputs"]
pub fn par_glq_integrate<F>(start: f64, end: f64, f: F, points: NonZeroUsize) -> f64
where
    F: Fn(f64) -> f64 + Send + Sync,
{
    let width = (end - start) * 0.5;
    let offset = (start + end) * 0.5;
    par_new_gauleg(points)
        .into_par_iter()
        .map(|p| p.weight() * f(width * p.position() + offset))
        .sum::<f64>()
        * width
}

#[cfg(test)]
mod test {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn check_integrator() {
        const NUMBER_OF_POINTS: usize = 100;
        const X1: f64 = 0.0;
        const X2: f64 = 10.0;

        let integrator = GlqIntegrator::new(NUMBER_OF_POINTS.try_into().unwrap());
        assert_relative_eq!(
            integrator.integrate(X1, X2, |x| x * (-x).exp()),
            1.0 - (1.0 + X2) * (-X2).exp(),
            epsilon = 1e-14
        );
        assert_relative_eq!(
            integrator.integrate(X1, X2, |x| x),
            X2 * X2 / 2.0,
            epsilon = 1e-12
        );

        const X3: f64 = 100.0;
        assert_relative_eq!(
            integrator.integrate(X1, X3, |x| x.cos()),
            X3.sin(),
            epsilon = 1e-12
        );
    }

    #[test]
    fn check_large_integral() {
        assert_relative_eq!(
            glq_integrate(0.0, 1.0, |x| x.ln(), 1_000_000.try_into().unwrap()),
            -1.0,
            epsilon = 1e-6
        );
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_parallel_glq_integrate() {
        use super::par_glq_integrate;
        assert_relative_eq!(
            par_glq_integrate(0.0, 1.0, |x| x * x, 3.try_into().unwrap()),
            1.0 / 3.0,
        );
        assert_relative_eq!(
            par_glq_integrate(
                -1.0,
                1.0,
                |x| 0.5 * (3.0 * x * x - 1.0) * x,
                3.try_into().unwrap()
            ),
            0.0
        );
        const END: f64 = 10.0;
        assert_relative_eq!(
            par_glq_integrate(0.0, END, |x| x * (-x).exp(), 13.try_into().unwrap()),
            (1.0 - (1.0 + END) * (-END).exp()),
        );
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_parallel_glqintegrator() {
        let mut integrator = GlqIntegrator::par_new(3.try_into().unwrap());
        assert_relative_eq!(
            integrator.par_integrate(0.0, 1.0, |x| x.powf(5.0)),
            1.0 / 6.0,
        );
        assert_relative_eq!(
            integrator.par_integrate(-1.0, 1.0, |x| x.powf(5.0) - 2.0 * x.powf(4.0) + 1.0),
            6.0 / 5.0,
        );
        integrator.par_change_number_of_points(58.try_into().unwrap());
        assert_relative_eq!(
            integrator.par_integrate(0.0, std::f64::consts::PI, |x| x.sin()),
            2.0,
            epsilon = 1e-15
        );
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_large_parallel_integration() {
        let integrator = GlqIntegrator::par_new(100_000_000.try_into().unwrap());
        assert_relative_eq!(
            integrator.par_integrate(0.0, 1.0, |x| x.ln()),
            -1.0,
            epsilon = 1e-14,
        );
    }
}
