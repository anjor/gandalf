"""
KRMHD: Kinetic Reduced Magnetohydrodynamics Spectral Solver

A spectral code for simulating turbulence in weakly collisional magnetized plasmas
with kinetic effects such as Landau damping and finite Larmor radius corrections.

The KRMHD model describes the evolution of:
- Active (Alfvénic) fields: stream function φ and parallel vector potential A∥
- Passive (slow mode) fields: parallel magnetic field δB∥ and electron density/pressure

Key features:
- Fourier spectral methods (2D and 3D) with 2/3 dealiasing
- JAX-based implementation with Metal GPU acceleration
- Functional programming style for clarity and composability
- Full type annotations for correctness

Physical processes:
- Poisson bracket nonlinearities: {f,g} = ẑ·(∇⊥f × ∇⊥g)
- Parallel electron kinetic effects (Landau damping)
- Spectral energy cascade from injection to dissipation scales
- Field line following for curved magnetic field geometry

This is a modern Python rewrite of the legacy GANDALF Fortran+CUDA code.
"""

__version__ = "0.1.0"
__author__ = "anjor"
__email__ = "anjor@umd.edu"

from krmhd.spectral import (
    SpectralGrid2D,
    SpectralGrid3D,
    SpectralField2D,
    SpectralField3D,
    derivative_x,
    derivative_y,
    derivative_z,
    laplacian,
    dealias,
)

from krmhd.hermite import (
    hermite_polynomial,
    hermite_polynomials_all,
    hermite_normalization,
    hermite_basis_function,
    distribution_to_moments,
    moments_to_distribution,
    check_orthogonality,
)

from krmhd.physics import (
    KRMHDState,
    poisson_bracket_2d,
    poisson_bracket_3d,
    physical_to_elsasser,
    elsasser_to_physical,
    z_plus_rhs,
    z_minus_rhs,
    initialize_hermite_moments,
    initialize_alfven_wave,
    initialize_kinetic_alfven_wave,
    initialize_random_spectrum,
    energy,
)

from krmhd.timestepping import (
    krmhd_rhs,
    rk4_step,
    compute_cfl_timestep,
)

__all__ = [
    "__version__",
    # Spectral infrastructure
    "SpectralGrid2D",
    "SpectralGrid3D",
    "SpectralField2D",
    "SpectralField3D",
    "derivative_x",
    "derivative_y",
    "derivative_z",
    "laplacian",
    "dealias",
    # Hermite basis for kinetic physics
    "hermite_polynomial",
    "hermite_polynomials_all",
    "hermite_normalization",
    "hermite_basis_function",
    "distribution_to_moments",
    "moments_to_distribution",
    "check_orthogonality",
    # Physics and state
    "KRMHDState",
    "poisson_bracket_2d",
    "poisson_bracket_3d",
    "physical_to_elsasser",
    "elsasser_to_physical",
    "z_plus_rhs",
    "z_minus_rhs",
    "initialize_hermite_moments",
    "initialize_alfven_wave",
    "initialize_kinetic_alfven_wave",
    "initialize_random_spectrum",
    "energy",
    # Time integration
    "krmhd_rhs",
    "rk4_step",
    "compute_cfl_timestep",
]
