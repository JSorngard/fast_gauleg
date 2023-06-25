use std::f64::consts::PI;

type Float = f64;

#[derive(Debug, Clone, PartialEq)]
pub struct QuadratureIntegrator {
    start: Float,
    end: Float,
    xs_and_ws: Vec<Float>,
    points: usize,
}

impl QuadratureIntegrator {
    pub fn new(start: Float, end: Float, points: usize) -> Self {
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

    pub fn integrate<F>(&self, mut f: F) -> Float
    where
        F: FnMut(Float) -> Float,
    {
        let (xs, ws) = self.xs_and_ws.split_at(self.points);
        xs.iter().zip(ws.iter()).map(|(x, w)| w * f(*x)).sum()
    }
}

/// Computes the weights and abscissas used when integrating a function
/// in the domain [x1, x2] using Gauss-Legendre quadrature.
/// # Panic
/// Panics if the lengths of the slices are different
pub fn gauleg(x1: Float, x2: Float, x: &mut [Float], w: &mut [Float]) {
    assert_eq!(x.len(), w.len());

    const EPS: Float = 1e-14;

    let n = x.len();
    let nf = n as Float;
    let xm = 0.5 * (x2 + x1);
    let xl = 0.5 * (x2 - x1);

    let mut z: Float;
    let mut p1: Float;
    let mut p2: Float;
    let mut p3: Float;
    let mut pp: Float;
    for i in 0..(n + 1) / 2 {
        z = (PI * (i as Float + 0.75) / (nf + 0.5)).cos();
        loop {
            p1 = 1.0;
            p2 = 0.0;
            for j in 0..n {
                p3 = p2;
                p2 = p1;
                p1 = (((2 * j + 1) as Float) * z * p2 - (j as Float) * p3) / (j as Float + 1.0);
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
/// using Gauss-Legendre quadrature with `number_of_points` points.
/// # Example
/// ```
/// # use numerical_recipes::gauss_legendre_quadrature;
/// fn f(x: f64) -> f64 {
///     x * (-x).exp()
/// }
/// let end = 10.0;
/// assert!(
///     (gauss_legendre_quadrature(0.0, end, f, 100) - (1.0 - (1.0 + end) * (-end).exp())).abs() < 1e-14);
/// ```
pub fn gauss_legendre_quadrature(
    start: Float,
    end: Float,
    function_to_integrate: fn(Float) -> Float,
    number_of_points: usize,
) -> Float {
    let mut xs = vec![0.0; number_of_points];
    let mut ws = vec![0.0; number_of_points];
    gauleg(start, end, &mut xs, &mut ws);
    xs.into_iter()
        .zip(ws.into_iter())
        .map(|(x, w)| w * function_to_integrate(x))
        .sum()
}

#[cfg(test)]
mod test {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn check_gauss_legendre_quadrature() {
        const NUMBER_OF_POINTS: usize = 100;
        const X1: Float = 0.0;
        const X2: Float = 10.0;

        let mut xs = [0.0; NUMBER_OF_POINTS];
        let mut ws = [0.0; NUMBER_OF_POINTS];

        gauleg(X1, X2, &mut xs, &mut ws);

        fn func(x: Float) -> Float {
            x * (-x).exp()
        }

        // integrate func from X1 to X2.
        assert_relative_eq!(
            1.0 - (1.0 + X2) * Float::exp(-X2),
            gauss_legendre_quadrature(X1, X2, func, NUMBER_OF_POINTS),
            epsilon = 1e-14,
        );
    }
}
