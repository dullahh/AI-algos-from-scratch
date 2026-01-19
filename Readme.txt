Machine Learning Algorithms From Scratch (NumPy & JAX)

Hi!! This repository documents my process of learning and implementing core ML optimisation algorithms from scratch.

Rather than relying on high-level ML libraries, I focus on building each algorithm manually in order to develop a strong understanding of the underlying mathematics, gradient mechanics, and computational trade-offs.


NumPy Implementations:

Each algorithm is first implemented using NumPy.
This provides a clear, imperative baseline that closely mirrors the mathematical derivations typically taught in theory.

The NumPy versions prioritise:

explicit matrix and vector operations

transparent gradient calculations

readable step-by-step training loops


JAX Implementations:

I then re-implement selected algorithms in JAX to explore a more functional and performance-oriented programming model.

The JAX implementations focus on:

- explicit pseudo-random number generation (PRNG keys)

- immutability and pure functions(for determinism)

- JIT compilation using jax.jit

- loop compilation using lax.scan where appropriate

This highlights how the same optimisation algorithms must be expressed differently to enable compilation, vectorisation, and accelerator support.


Algorithms Implemented (So Far)

- Optimisation Algorithms
  - Stochastic Gradient Descent (SGD)
    - NumPy implementation
    - JAX implementation (including a fully JIT-compiled training loop)
  - Mini-Batch Gradient Descent (MBGD)
    - NumPy implementation
    - JAX implementation with explicit batching and functional randomness(without jax.jit usage)


Planned extensions:

- Momentum-based Gd
- RMSProp


