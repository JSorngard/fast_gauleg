use core::num::NonZeroUsize;

/// An object that can integrate `fn(f64) -> f64` functions.
/// If instantiated with `n` points it can integrate polynomials of degree `2n - 1` exactly.
/// It is less accurate the less polynomial-like the given function is, and the less it conforms to the degree-bound.
/// # Examples
/// Integrate degree 5 polynomials with only 3 evaluation points:
/// ```
/// # use gauss_legendre_quadrature::GLQIntegrator;
/// # use approx::assert_relative_eq;
/// let integrator = GLQIntegrator::new(3.try_into().unwrap());
/// assert_relative_eq!(
///     integrator.integrate(0.0, 1.0, |x| x.powf(5.0)),
///     1.0 / 6.0,
///     epsilon = 1e-15,
/// );
/// assert_relative_eq!(
///     integrator.integrate(-1.0, 1.0, |x| x.powf(5.0) - 2.0 * x.powf(4.0) + 1.0),
///     6.0 / 5.0,
///     epsilon = 1e-14, // Slightly less accurate
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
///     epsilon = 0.01 // Very bad accuracy
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
    // We use only one allocation for the abscissas and weights
    // to increase the chances that both end up in the cache during
    // the same fetch from memory when the number of points is small.
    xs_and_ws: Vec<f64>,
    points: NonZeroUsize,
}

impl GLQIntegrator {
    #[must_use = "function returns a new instance and does not modify the input values"]
    /// Creates a new integrator that integrates functions over the given domain.
    pub fn new(points: NonZeroUsize) -> Self {
        let mut xs_and_ws = vec![0.0; 2 * points.get()];
        let (xs, ws) = xs_and_ws.split_at_mut(points.into());
        gauleg(-1.0, 1.0, xs, ws);
        Self { xs_and_ws, points }
    }

    /// Integrates the given function over the given domain.
    pub fn integrate(&self, start: f64, end: f64, f: fn(f64) -> f64) -> f64 {
        let (xs, ws) = self.xs_and_ws.split_at(self.points.into());
        xs.iter()
            .zip(ws.iter())
            .map(|(x, w)| w * f((end - start) * 0.5 * x + (start + end) * 0.5))
            .sum::<f64>()
            * (end - start)
            * 0.5
    }

    /// Returns a slice of the integrators abscissas.
    pub fn abscissas(&self) -> &[f64] {
        &self.xs_and_ws[..self.points.get()]
    }

    /// Returns a slice of the integrators weights. The function values are multiplied by these
    /// numbers before they are summed.
    pub fn weights(&self) -> &[f64] {
        &self.xs_and_ws[self.points.get()..]
    }

    /// Returns the number of points in the integration domain
    #[must_use = "the method returns a value and does not modify `self` or its inputs"]
    #[inline(always)]
    pub const fn points(&self) -> NonZeroUsize {
        self.points
    }

    /// Changes the number of points used during integration.
    /// If the number is not increased the old allocation is reused.
    pub fn change_number_of_points(&mut self, points: NonZeroUsize) {
        self.xs_and_ws.resize(2 * points.get(), 0.0);
        let (xs, ws) = self.xs_and_ws.split_at_mut(points.into());
        gauleg(-1.0, 1.0, xs, ws);
        self.points = points;
    }
}

/// Computes the weights and abscissas used when integrating a function
/// in the domain [x1, x2] using Gauss-Legendre quadrature.
/// # Panic
/// Panics if the lengths of the slices are different
pub fn gauleg(x1: f64, x2: f64, x: &mut [f64], w: &mut [f64]) {
    // This function is ported Fortran code, and is not very ideomatic.
    // The original code can be found in the book Numerical Recipes: http://numerical.recipes/
    assert_eq!(x.len(), w.len());

    const EPS: f64 = 1e-14;

    let n = x.len();
    let nf = n as f64;
    let xm = 0.5 * (x2 + x1);
    let xl = 0.5 * (x2 - x1);

    for i in 0..(n + 1) / 2 {
        let mut z = (std::f64::consts::PI * (i as f64 + 0.75) / (nf + 0.5)).cos();
        let mut pp: f64;
        loop {
            let mut p1 = 1.0;
            let mut p2 = 0.0;
            for j in 0..n {
                let p3 = p2;
                p2 = p1;
                p1 = (((2 * j + 1) as f64) * z * p2 - (j as f64) * p3) / (j as f64 + 1.0);
            }
            pp = nf * (z * p1 - p2) / (z * z - 1.0);
            let z1 = z;
            z = z1 - p1 / pp;
            if (z - z1).abs() <= EPS {
                break;
            }
        }
        x[i] = xm - xl * z;
        x[n - 1 - i] = xm + xl * z;
        w[i] = 2.0 * xl / ((1.0 - z * z) * pp * pp);
        w[n - 1 - i] = w[i]
    }
}

/// Integrates the given function from `start` to `end`
/// using Gauss-Legendre quadrature with `points` points.
/// With `n` points it can integrate polynomials of degree `2n - 1` exactly.
/// The result will be less accurate the less polynomial-like the function is,
/// and the less it adheres to the degree bound.
/// # Example
/// ```
/// # use gauss_legendre_quadrature::quad;
/// # use approx::assert_relative_eq;
/// // Integrate degree 2 and 3 polynomials with only 2 points:
/// assert_relative_eq!(
///     quad(0.0, 1.0, 2.try_into().unwrap(), |x| x * x),
///     1.0 / 3.0,
/// );
/// assert_relative_eq!(
///     quad(-1.0, 1.0, 2.try_into().unwrap(), |x| 0.5 * (3.0 * x * x - 1.0) * x),
///     0.0
/// );
/// // Non-polynomials need more points to evaluate correctly:
/// const END: f64 = 10.0;
/// assert_relative_eq!(
///     quad(0.0, END, 13.try_into().unwrap(), |x| x * (-x).exp()),
///     (1.0 - (1.0 + END) * (-END).exp()),
///     epsilon = 1e-14,
/// );
/// ```
pub fn quad(start: f64, end: f64, points: NonZeroUsize, f: fn(f64) -> f64) -> f64 {
    let mut xs = vec![0.0; points.into()];
    let mut ws = vec![0.0; points.into()];
    gauleg(start, end, &mut xs, &mut ws);
    xs.into_iter()
        .zip(ws.into_iter())
        .map(|(x, w)| w * f(x))
        .sum()
}

#[cfg(test)]
mod test {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn check_gauss_legendre_quadrature() {
        const NUMBER_OF_POINTS: usize = 100;
        const X1: f64 = 0.0;
        const X2: f64 = 10.0;

        let mut xs = [0.0; NUMBER_OF_POINTS];
        let mut ws = [0.0; NUMBER_OF_POINTS];

        gauleg(X1, X2, &mut xs, &mut ws);

        fn func(x: f64) -> f64 {
            x * (-x).exp()
        }

        // integrate func from X1 to X2.
        assert_relative_eq!(
            1.0 - (1.0 + X2) * (-X2).exp(),
            quad(X1, X2, NUMBER_OF_POINTS.try_into().unwrap(), func),
            epsilon = 1e-14,
        );
    }

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
}
