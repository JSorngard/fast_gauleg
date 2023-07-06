//! This crate contains tools for numerical integration using [Gauss-Legendre quadrature](https://en.wikipedia.org/wiki/Gauss%E2%80%93Legendre_quadrature).
//! This is a method that allows integration of polynomial functions using very few evaluation points.
//! A quadrature using `n` points can exactly integrate a polynomial of degree `2n - 1` in the interval `[-1, 1]`.
//! Non-polynomials will need more evaluation points, and the answer will be less accurate
//! the less polynomial-like the given function is and the more it violates the degree bound.
//!
//! The weights and abscissas used during integration are computed using the method developed by
//! [Ignace Bogaert](https://www.researchgate.net/publication/262672564_Iteration-Free_Computation_of_Gauss-Legendre_Quadrature_Nodes_and_Weights).
//! # Examples
//! Integrate a degree five polynomial while only evaluating it at three points:
//! ```
//! use gauss_legendre_quadrature::quad;
//! // This macro is used in the docs of this crate to compare floating point values.
//! // The assertion succeeds if the two values are within floating point error of each other,
//! // or within an optional epsilon.
//! use approx::assert_relative_eq;
//! use core::num::NonZeroUsize;
//!
//! let pts = NonZeroUsize::new(3).unwrap();
//!
//! // Check the orthogonality of Legendre polynomials of degree 2 and 3:
//! assert_relative_eq!(
//!     quad(-1.0, 1.0, |x| 0.25 * (3.0 * x.powf(2.0) - 1.0) * (5.0 * x.powf(3.0) - 3.0 * x), pts),
//!     0.0,
//! );
//! // Integrating more complicated polynomials or integrating outside [-1, 1] can reduce accuracy
//! assert_relative_eq!(
//!     quad(-5.0, 2.0, |x| 0.125 * (63.0 * x.powf(5.0) - 70.0 * x.powf(3.0) + 15.0 * x), pts),
//!     -305781.0 / 16.0,
//!     // The exponent in the epsilon is always chosen
//!     // to be as small as possible
//!     // while still passing the assertion.
//!     epsilon = 1e-11,
//! );
//!```
//! Integrate a trancendental function:
//! ```
//! # use approx::assert_relative_eq;
//! # use gauss_legendre_quadrature::quad;
//! assert_relative_eq!(
//!     quad(0.0, 1.0, |x| (x + 1.0).ln().sin(), 10.try_into().unwrap()),
//!     0.5 - f64::ln(2.0).cos() + f64::ln(2.0).sin(),
//! );   
//! ```
//! Divergences can be hard to integrate. Integration with many points can compensate for this, and is still fast
//! ```
//! # use approx::assert_relative_eq;
//! # use gauss_legendre_quadrature::quad;
//! assert_relative_eq!(
//!     quad(0.0, 1.0, |x| x.ln(), 10.try_into().unwrap()),
//!     -1.0,
//!     epsilon = 1e-2,
//! );
//! assert_relative_eq!(
//!     quad(0.0, 1.0, |x| x.ln(), 1_000_000.try_into().unwrap()),
//!     -1.0,
//!     epsilon = 1e-12,
//! );
//! ```
//! If many integrations need to be done the crate provides [`GLQIntegrator`], which allows
//! the calculation of the quadrature weights and abscissas to be reused:
//! ```
//! # use approx::assert_relative_eq;
//! use gauss_legendre_quadrature::GLQIntegrator;
//! let integrator = GLQIntegrator::new(10.try_into().unwrap());
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

use core::num::NonZeroUsize;
mod data;
mod fastgl;
use fastgl::{modify_gauleg, new_gauleg, QuadPair};

/// An object that can integrate `Fn(f64) -> f64` functions and closures.
/// If instantiated with `n` points it can integrate polynomials of degree `2n - 1` exactly.
/// It is less accurate the less polynomial-like the given function is, and the less it conforms to the degree-bound.
/// # Examples
/// Integrate degree 5 polynomials with only 3 evaluation points:
/// ```
/// # use gauss_legendre_quadrature::GLQIntegrator;
/// use approx::assert_relative_eq;
/// let integrator = GLQIntegrator::new(3.try_into().unwrap());
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
/// # use gauss_legendre_quadrature::GLQIntegrator;
/// # use approx::assert_relative_eq;
/// let mut integrator = GLQIntegrator::new(3.try_into().unwrap());
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
pub struct GLQIntegrator {
    xs_and_ws: Vec<QuadPair>,
    points: NonZeroUsize,
}

impl GLQIntegrator {
    #[must_use = "function returns a new instance and does not modify the input values"]
    /// Creates a new integrator that integrates functions over the given domain.
    pub fn new(points: NonZeroUsize) -> Self {
        let xs_and_ws = new_gauleg(points);
        Self { xs_and_ws, points }
    }

    /// Integrates the given function over the given domain.
    pub fn integrate<F: Fn(f64) -> f64>(&self, start: f64, end: f64, f: F) -> f64 {
        self.xs_and_ws
            .iter()
            .map(|p| p.w() * f((end - start) * 0.5 * p.x() + (start + end) * 0.5))
            .sum::<f64>()
            * (end - start)
            * 0.5
    }

    /// Returns the number of points in the integration domain
    #[must_use = "the method returns a value and does not modify `self` or its inputs"]
    #[inline(always)]
    pub const fn points(&self) -> NonZeroUsize {
        self.points
    }

    /// Changes the number of points used during integration.
    /// If the number is not increased the old allocation is reused.
    pub fn change_number_of_points(&mut self, new_points: NonZeroUsize) {
        self.xs_and_ws.resize(new_points.get(), QuadPair::default());
        modify_gauleg(&mut self.xs_and_ws);
        self.points = new_points;
    }
}

/// Integrates the given function over the interval `[start, end]`
/// using [Gauss-Legendre quadrature](https://en.wikipedia.org/wiki/Gauss%E2%80%93Legendre_quadrature)
/// with `points` points. With `n` points it can integrate polynomials of degree `2n - 1` exactly.
/// The result will be less accurate the less polynomial-like the function is,
/// and the less it adheres to the degree bound.
/// # Example
/// ```
/// # use gauss_legendre_quadrature::quad;
/// use approx::assert_relative_eq;
/// use core::num::NonZeroUsize;
/// // Integrate degree 2 and 3 polynomials with only 2 points:
/// let pts = NonZeroUsize::new(2).unwrap();
/// assert_relative_eq!(
///     quad(0.0, 1.0, |x| x * x, pts),
///     1.0 / 3.0,
/// );
/// assert_relative_eq!(
///     quad(-1.0, 1.0, |x| 0.5 * (3.0 * x * x - 1.0) * x, pts),
///     0.0
/// );
/// // Non-polynomials need more points to evaluate correctly:
/// const END: f64 = 10.0;
/// assert_relative_eq!(
///     quad(0.0, END, |x| x * (-x).exp(), 13.try_into().unwrap()),
///     (1.0 - (1.0 + END) * (-END).exp()),
/// );
/// ```
pub fn quad<F: Fn(f64) -> f64>(start: f64, end: f64, f: F, points: NonZeroUsize) -> f64 {
    let xs_and_ws = new_gauleg(points);
    xs_and_ws
        .into_iter()
        .map(|p| 0.5 * (end - start) * p.w() * f(0.5 * (end - start) * p.x() + 0.5 * (start + end)))
        .sum()
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

        let integrator = GLQIntegrator::new(NUMBER_OF_POINTS.try_into().unwrap());
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
            quad(0.0, 1.0, |x| x.ln(), 1_000_000.try_into().unwrap()),
            -1.0,
            epsilon = 1e-6
        );
    }
}
