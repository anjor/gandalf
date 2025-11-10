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
    initialize_orszag_tang,
    energy,
)

from krmhd.timestepping import (
    krmhd_rhs,
    gandalf_step,
    compute_cfl_timestep,
)

from krmhd.diagnostics import (
    energy_spectrum_1d,
    energy_spectrum_perpendicular,
    energy_spectrum_parallel,
    EnergyHistory,
    plot_state,
    plot_energy_history,
    plot_energy_spectrum,
)

from krmhd.forcing import (
    gaussian_white_noise_fourier,
    force_alfven_modes,
    force_slow_modes,
    compute_energy_injection_rate,
)

from krmhd.config import (
    GridConfig,
    PhysicsConfig,
    InitialConditionConfig,
    ForcingConfig,
    TimeIntegrationConfig,
    IOConfig,
    SimulationConfig,
    decaying_turbulence_config,
    driven_turbulence_config,
    orszag_tang_config,
)

from krmhd.io import (
    save_checkpoint,
    load_checkpoint,
    save_timeseries,
    load_timeseries,
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
    "initialize_orszag_tang",
    "energy",
    # Time integration
    "krmhd_rhs",
    "gandalf_step",
    "compute_cfl_timestep",
    # Diagnostics
    "energy_spectrum_1d",
    "energy_spectrum_perpendicular",
    "energy_spectrum_parallel",
    "EnergyHistory",
    "plot_state",
    "plot_energy_history",
    "plot_energy_spectrum",
    # Forcing
    "gaussian_white_noise_fourier",
    "force_alfven_modes",
    "force_slow_modes",
    "compute_energy_injection_rate",
    # Configuration
    "GridConfig",
    "PhysicsConfig",
    "InitialConditionConfig",
    "ForcingConfig",
    "TimeIntegrationConfig",
    "IOConfig",
    "SimulationConfig",
    "decaying_turbulence_config",
    "driven_turbulence_config",
    "orszag_tang_config",
    # I/O
    "save_checkpoint",
    "load_checkpoint",
    "save_timeseries",
    "load_timeseries",
]
