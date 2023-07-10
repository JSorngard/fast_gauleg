# fast_gauleg

A crate for numerical integration with [Gauss-Legendre quadrature](https://en.wikipedia.org/wiki/Gauss%E2%80%93Legendre_quadrature). This crate uses the method described in the paper ["Iteration-Free Computation of Gauss-Legendre Quadrature Nodes and Weights"](https://doi.org/10.1137/140954969) by I. Bogaert which enables computation of node-weight pairs in O(1) time complexity (and optionally in parallel) while maintaining an accuracy of a few ulps in the nodes and weights (see paper for details).

The integrals are computed with the formula  
![integral_approximation](readme_images/integral_approx.svg)  
where `n` is a chosen integer, `a` and `b` are finite numbers, and `w_n` and `x_n` are specific numbers derived from Legendre polynomials.

# Examples

Integrate a degree 5 polynomial while only evaluating it at three points:
```rust
use fast_gauleg::glq_integrate;
use approx::assert_relative_eq;

assert_relative_eq!(
    glq_integrate(-1.0, 1.0, |x| x.powf(5.0), 3.try_into().unwrap()),
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
This is because the computation of individual node-weight pairs is O(1)
```rust
use fast_gauleg::GlqPair;
// This:
let a = GlqPair::new(1_000_000_000, 500_000_000);
// Is as fast as this:
let b = GlqPair::new(24, 12);
```
