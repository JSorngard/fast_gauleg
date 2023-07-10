# gl_quadrature

A crate for numerical integration with [Gauss-Legendre quadrature](https://en.wikipedia.org/wiki/Gauss%E2%80%93Legendre_quadrature).
This technique allows for the computation of integrals of polynomials with very few evaluation points, and can be used for non-polynomials as well at possibly reduced accuracy.
A quadrature rule with `n` points can integrate polynomials of degree `2n - 1` exactly.

This crate is based on the paper ["Iteration-Free Computation of Gauss-Legendre Quadrature Nodes and Weights"](https://doi.org/10.1137/140954969) by I. Bogaert. This computation method allows for the computation of node-weight pairs in O(1) time complexity, and optionally in parallel.
