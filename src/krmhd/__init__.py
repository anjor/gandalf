"""
KRMHD: Kinetic Reduced Magnetohydrodynamics Spectral Solver

A spectral code for simulating turbulence in weakly collisional magnetized plasmas
with kinetic effects such as Landau damping and finite Larmor radius corrections.

The KRMHD model describes the evolution of:
- Active (Alfvénic) fields: stream function φ and parallel vector potential A∥
- Passive (slow mode) fields: parallel magnetic field δB∥ and electron density/pressure

Key features:
- Fourier spectral methods with 2/3 dealiasing
- JAX-based implementation with Metal GPU acceleration
- Functional programming style for clarity and composability
- Full type annotations for correctness

Physical processes:
- Poisson bracket nonlinearities: {f,g} = ẑ·(∇f × ∇g)
- Parallel electron kinetic effects (Landau damping)
- Spectral energy cascade from injection to dissipation scales

This is a modern Python rewrite of the legacy GANDALF Fortran+CUDA code.
"""

__version__ = "0.1.0"
__author__ = "anjor"
__email__ = "anjor@umd.edu"

__all__ = ["__version__"]
