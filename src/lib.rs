use std::f64::consts::PI;

pub fn gauleg<const N: usize>(x1: f64, x2: f64, x: &mut [f64; N], w: &mut [f64; N]) {
    const EPS: f64 = 1e-14;
    let nf = N as f64;
    let xm = 0.5 * (x2 + x1);
    let xl = 0.5 * (x2 - x1);

    let mut z: f64;
    let mut p1: f64;
    let mut p2: f64;
    let mut p3: f64;
    let mut pp: f64;
    for i in 0..(N + 1) / 2 {
        z = (PI * (i as f64 + 0.75) / (nf + 0.5)).cos();
        loop {
            p1 = 1.0;
            p2 = 0.0;
            for j in 0..N {
                p3 = p2;
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
        x[N - 1 - i] = xm + xl * z;
        w[i] = 2.0 * xl / ((1.0 - z * z) * pp * pp);
        w[N - 1 - i] = w[i]
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn check_gauleg() {
        const NUMBER_OF_POINTS: usize = 100;
        const X1: f64 = 0.0;
        const X2: f64 = 10.0;

        let mut xs = [0.0; NUMBER_OF_POINTS];
        let mut ws = [0.0; NUMBER_OF_POINTS];

        gauleg(X1, X2, &mut xs, &mut ws);

        fn func(x: f64) -> f64 {
            x * f64::exp(-x)
        }

        // integrate func from 0 to 10.
        assert_relative_eq!(
            xs.map(func)
                .into_iter()
                .zip(ws.into_iter())
                .map(|(x, w)| w * x)
                .sum::<f64>(),
            1.0 - (1.0 + X2) * f64::exp(-X2),
            epsilon = 1e-14,
        );
    }
}
