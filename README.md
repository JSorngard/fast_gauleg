# gl_quadrature

A crate for numerical integration with [Gauss-Legendre quadrature](https://en.wikipedia.org/wiki/Gauss%E2%80%93Legendre_quadrature).
This technique allows for the computation of integrals of polynomials with very few evaluation points, and can be used for non-polynomials as well at possibly reduced accuracy.
A quadrature rule with `n` points can integrate polynomials of degree `2n - 1` exactly.

This crate is based on the paper ["Iteration-Free Computation of Gauss-Legendre Quadrature Nodes and Weights"](https://doi.org/10.1137/140954969) by I. Bogaert. This computation method allows for the computation of node-weight pairs in O(1) time complexity, and optionally in parallel.

# Examples

Integrate a degree 5 polynomial while only evaluating it at three points:
```rust
use fast_gauleg::glq_integrate;
use approx::assert_relative_eq;

assert_relative_eq!(
    glq_integrate(-1.0, 1.0, |x| 0.125 * (63.0 * x.powf(5.0) - 73.0 * x.powf(3.0) + 15.0 * x), 3.try_into().unwrap()),
    0.0,
);
```
Very large quadrature rules are feasible, and can be evaluated in parallel:
```rust
assert_relative_eq!(
    par_glq_integrate(0.0, 1.0, |x| x.ln(), 100_000_000.try_into().unwrap()),
    -1.0
);
```