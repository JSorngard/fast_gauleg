#[cfg(feature = "rayon")]
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

/// An object that can integrate any `FnMut(f64) -> f64` function over its domain.
/// If the `rayon` feature is enabled it can also integrate any `Fn(f64) -> f64` function over its domain in parallel.
/// Useful if you need to integrate many functions over the same domain.
#[derive(Debug, Clone, PartialEq)]
pub struct QuadratureIntegrator {
    start: f64,
    end: f64,
    xs_and_ws: Vec<f64>,
    points: usize,
}

impl QuadratureIntegrator {
    #[must_use = "function returns a new instance and does not modify the input values"]
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
    pub fn integrate<F>(&self, mut f: F) -> f64
    where
        F: FnMut(f64) -> f64,
    {
        let (xs, ws) = self.xs_and_ws.split_at(self.points);
        xs.iter().zip(ws.iter()).map(|(x, w)| w * f(*x)).sum()
    }

    #[cfg(feature = "rayon")]
    /// Integrates the given function over `self`'s domain in parallel.
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
/// # use numerical_recipes::gauss_legendre_quadrature;
/// fn f(x: f64) -> f64 {
///     x * (-x).exp()
/// }
/// let end = 10.0;
/// assert!(
///     (gauss_legendre_quadrature(0.0, end, 100, f) - (1.0 - (1.0 + end) * (-end).exp())).abs() < 1e-14);
/// ```
pub fn gauss_legendre_quadrature<F>(start: f64, end: f64, points: usize, mut f: F) -> f64
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
            gauss_legendre_quadrature(X1, X2, NUMBER_OF_POINTS, func),
            epsilon = 1e-14,
        );
    }

    #[test]
    fn check_integrator() {
        const NUMBER_OF_POINTS: usize = 100;
        const X1: f64 = 0.0;
        const X2: f64 = 10.0;

        let integrator = QuadratureIntegrator::new(X1, X2, NUMBER_OF_POINTS);
        assert_relative_eq!(
            integrator.integrate(|x| x * (-x).exp()),
            1.0 - (1.0 + X2) * (-X2).exp(),
            epsilon = 1e-14
        );
        assert_relative_eq!(integrator.integrate(|x| x), X2 * X2 / 2.0, epsilon = 1e-12);
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn check_parallel_integrator() {
        const NUMBER_OF_POINTS: usize = 100;
        const X1: f64 = 0.0;
        const X2: f64 = 10.0;
        let integrator = QuadratureIntegrator::new(X1, X2, NUMBER_OF_POINTS);
        assert_relative_eq!(
            integrator.par_integrate(|x| x.cos()),
            X2.sin(),
            epsilon = 1e-14
        );
    }
}
