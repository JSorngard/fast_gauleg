#[cfg(feature = "rayon")]
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

/// An object that can integrate `FnMut(f64) -> f64` functions over its domain.
/// If the `rayon` feature is enabled it can also integrate `Fn(f64) -> f64` functions over its domain in parallel.
/// Useful if you need to integrate many functions over the same domain.
/// If instantiated with `n` points it can integrate polynomials of degree `2n - 1` exactly.
/// It is less accurate the less polynomial-like the given funciton is, and the less it conforms to the degree-bound.
#[derive(Debug, Clone, PartialEq)]
pub struct GLQIntegrator {
    start: f64,
    end: f64,
    // We use only one allocation for the abscissas and weights
    // to increase that chances that both end up in the cache during
    // the same fetch from memory when the number of points is small.
    xs_and_ws: Vec<f64>,
    points: usize,
}

impl GLQIntegrator {
    #[must_use = "function returns a new instance and does not modify the input values"]
    /// Creates a new integrator that integrates functions over the given domain.
    pub fn new(start: f64, end: f64, points: usize) -> Self {
        let mut xs_and_ws = vec![0.0; 2 * points];
        let (xs, ws) = xs_and_ws.split_at_mut(points);
        gauleg(start, end, xs, ws);
        Self {
            start,
            end,
            xs_and_ws,
            points,
        }
    }

    /// Integrates the given function over `self`'s domain. The given closure will be called
    /// once for each point in the domain.
    /// # Example
    /// Integrate degree 5 polynomials with only 3 evaluation points:
    /// ```
    /// # use gauss_legendre_quadrature::GLQIntegrator;
    /// # use approx::assert_relative_eq;
    /// let integrator = GLQIntegrator::new(0.0, 1.0, 3);
    /// assert_relative_eq!(
    ///     integrator.integrate(|x| x.powf(5.0)),
    ///     1.0 / 6.0,
    ///     epsilon = 1e-15,
    /// );
    /// assert_relative_eq!(
    ///     integrator.integrate(|x| x.powf(5.0) - 2.0 * x.powf(4.0) + 1.0),
    ///     1.0 / 6.0 - 2.0 / 5.0 + 1.0,
    ///     epsilon = 1e-14, // Slightly less accurate
    /// );
    /// ```
    pub fn integrate<F>(&self, mut f: F) -> f64
    where
        F: FnMut(f64) -> f64,
    {
        let (xs, ws) = self.xs_and_ws.split_at(self.points);
        xs.iter().zip(ws.iter()).map(|(x, w)| w * f(*x)).sum()
    }

    /// Returns a slice of the integrators abscissas, the points at which the integrator
    /// evaluates the functions it integrates.
    pub fn abscissas(&self) -> &[f64] {
        &self.xs_and_ws[..self.points]
    }

    /// Returns a slice of the integrators weights. The function values are multiplied by these
    /// numbers before they are summed.
    pub fn weights(&self) -> &[f64] {
        &self.xs_and_ws[self.points..]
    }

    /// Integrates a function that returns the given values at the integrator's abscissas.
    /// This allows pre-computing function values, and then integrating.
    /// # Example
    /// ```
    /// # use gauss_legendre_quadrature::GLQIntegrator;
    /// # use approx::assert_relative_eq;
    /// let integrator = GLQIntegrator::new(0.0, std::f64::consts::PI, 65);
    /// let f_vals: Vec<f64> = integrator.abscissas().iter().map(|x| x.sin()).collect();
    /// assert_relative_eq!(integrator.integrate_slice(&f_vals), 2.0);
    /// ```
    /// # Panic
    /// Panics if the length of the given slice is not the same as the number of points in the integrator.
    #[must_use = "the method returns a value and does not modify `self` or its inputs"]
    pub fn integrate_slice(&self, f_vals: &[f64]) -> f64 {
        assert_eq!(f_vals.len(), self.points);
        self.xs_and_ws
            .iter()
            .skip(self.points)
            .enumerate()
            .map(|(i, w)| w * f_vals[i])
            .sum()
    }

    #[cfg(feature = "rayon")]
    /// Integrates a function that returns the given values at the integrator's abscissas  (in parallel).
    /// This allows pre-computing function values, and then integrating.
    /// # Panic
    /// Panics if the length of the given slice is not the same as the number of points in the integrator.
    #[must_use = "the method returns a value and does not modify `self` or its inputs"]
    pub fn par_integrate_slice(&self, f_vals: &[f64]) -> f64 {
        assert_eq!(f_vals.len(), self.points);
        self.xs_and_ws
            .par_iter()
            .skip(self.points)
            .enumerate()
            .map(|(i, w)| w * f_vals[i])
            .sum()
    }

    #[cfg(feature = "rayon")]
    /// Integrates the given function over `self`'s domain in parallel.
    #[must_use = "the method returns a value and does not modify `self` or its inputs"]
    pub fn par_integrate<F>(&self, f: F) -> f64
    where
        F: Fn(f64) -> f64 + Sync,
    {
        let (xs, ws) = self.xs_and_ws.split_at(self.points);
        xs.par_iter()
            .zip(ws.par_iter())
            .map(|(x, w)| w * f(*x))
            .sum()
    }

    /// Returns the first point of the integration domain
    #[must_use = "the method returns a value and does not modify `self` or its inputs"]
    #[inline(always)]
    pub const fn start(&self) -> f64 {
        self.start
    }

    /// Returns the last point of the integration domain
    #[must_use = "the method returns a value and does not modify `self` or its inputs"]
    #[inline(always)]
    pub const fn end(&self) -> f64 {
        self.end
    }

    /// Returns the number of points in the integration domain
    #[must_use = "the method returns a value and does not modify `self` or its inputs"]
    #[inline(always)]
    pub const fn points(&self) -> usize {
        self.points
    }

    /// Change the domain of the integrator
    pub fn change_domain(&mut self, start: f64, end: f64) {
        let (xs, ws) = self.xs_and_ws.split_at_mut(self.points);
        gauleg(start, end, xs, ws);
        self.start = start;
        self.end = end;
    }

    /// Changes the number of points used during integration.
    /// If the number is not increased the old allocation is reused.
    pub fn change_number_of_points(&mut self, points: usize) {
        self.xs_and_ws.resize(2 * points, 0.0);
        let (xs, ws) = self.xs_and_ws.split_at_mut(points);
        gauleg(self.start, self.end, xs, ws);
        self.points = points;
    }

    /// Change the number of integration points and the integration domain. If the number of integration points
    /// is not increased the old allocation is reused.
    pub fn change_number_of_points_and_domain(&mut self, start: f64, end: f64, points: usize) {
        self.xs_and_ws.resize(2 * points, 0.0);
        let (xs, ws) = self.xs_and_ws.split_at_mut(points);
        gauleg(start, end, xs, ws);
        self.start = start;
        self.end = end;
        self.points = points;
    }
}

/// Computes the weights and abscissas used when integrating a function
/// in the domain [x1, x2] using Gauss-Legendre quadrature.
/// # Panic
/// Panics if the lengths of the slices are different
pub fn gauleg(x1: f64, x2: f64, x: &mut [f64], w: &mut [f64]) {
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
/// # Example
/// ```
/// # use gauss_legendre_quadrature::quad;
/// fn f(x: f64) -> f64 {
///     x * (-x).exp()
/// }
/// let end = 10.0;
/// assert!(
///     (quad(0.0, end, 100, f) - (1.0 - (1.0 + end) * (-end).exp())).abs() < 1e-14);
/// ```
pub fn quad<F>(start: f64, end: f64, points: usize, mut f: F) -> f64
where
    F: FnMut(f64) -> f64,
{
    let mut xs = vec![0.0; points];
    let mut ws = vec![0.0; points];
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
            quad(X1, X2, NUMBER_OF_POINTS, func),
            epsilon = 1e-14,
        );
    }

    #[test]
    fn check_integrator() {
        const NUMBER_OF_POINTS: usize = 100;
        const X1: f64 = 0.0;
        const X2: f64 = 10.0;

        let mut integrator = GLQIntegrator::new(X1, X2, NUMBER_OF_POINTS);
        assert_relative_eq!(
            integrator.integrate(|x| x * (-x).exp()),
            1.0 - (1.0 + X2) * (-X2).exp(),
            epsilon = 1e-14
        );
        assert_relative_eq!(integrator.integrate(|x| x), X2 * X2 / 2.0, epsilon = 1e-12);

        const X3: f64 = 100.0;
        integrator.change_domain(X1, X3);
        assert_relative_eq!(integrator.integrate(|x| x.cos()), X3.sin(), epsilon = 1e-12);
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn check_parallel_integrator() {
        const NUMBER_OF_POINTS: usize = 100;
        const X1: f64 = 0.0;
        const X2: f64 = 10.0;
        let integrator = GLQIntegrator::new(X1, X2, NUMBER_OF_POINTS);
        assert_relative_eq!(
            integrator.par_integrate(|x| x.cos()),
            X2.sin(),
            epsilon = 1e-14
        );
    }
}
