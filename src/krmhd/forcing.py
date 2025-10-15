"""
Forcing mechanisms for driven turbulence simulations (Thesis §2.5.1).

This module implements Gaussian white noise forcing for sustained turbulent cascades.
The forcing is applied in Fourier space at specified wavenumber bands to inject
energy at large scales while allowing inertial-range cascade to develop.

Key features:
- Band-limited forcing: inject energy only at k ∈ [k_min, k_max]
- Alfvén sector forcing: drives perpendicular velocity u⊥ only, NOT δB⊥
- Slow mode forcing: optional independent forcing for δB∥ and kinetic moments
- Energy injection rate diagnostics for monitoring energy balance

Physics context:
    Driven turbulence reaches steady state when energy injection balances dissipation:
    ε_inj = ∫ F · u dk = ε_diss = η ∫ k² |u|² dk

    The forcing is white noise in time (uncorrelated between timesteps) and
    band-limited in k-space to concentrate energy injection at large scales.

Example usage:
    >>> key = jax.random.PRNGKey(42)
    >>> state_forced, key = force_alfven_modes(
    ...     state, amplitude=0.1, k_min=2.0, k_max=5.0, dt=0.01, key=key
    ... )
    >>> energy_rate = compute_energy_injection_rate(state, state_forced, dt=0.01)

Reference:
    - Thesis §2.5.1 - Forcing mechanisms for driven turbulence
    - Elenbaas, Schukraft & Schubert (2008) MNRAS - Gaussian random forcing
"""

from functools import partial
from typing import Tuple
import jax
import jax.numpy as jnp
from jax import Array

from krmhd.physics import KRMHDState, energy
from krmhd.spectral import SpectralGrid3D


@jax.jit
def _gaussian_white_noise_fourier_jit(
    kx: Array,
    ky: Array,
    kz: Array,
    amplitude: float,
    k_min: float,
    k_max: float,
    dt: float,
    real_part: Array,
    imag_part: Array,
) -> Array:
    """
    JIT-compiled core function for generating Gaussian white noise in Fourier space.

    Args:
        kx, ky, kz: Wavenumber arrays (broadcast-compatible shapes)
        amplitude: Forcing amplitude (energy injection rate ~ amplitude²)
        k_min: Minimum wavenumber for forcing band
        k_max: Maximum wavenumber for forcing band
        dt: Timestep (for proper dimensional scaling)
        real_part: Random normal samples for real part [Nz, Ny, Nx//2+1]
        imag_part: Random normal samples for imaginary part [Nz, Ny, Nx//2+1]

    Returns:
        Complex Fourier field with forcing at k ∈ [k_min, k_max]
    """
    # Broadcast wavenumbers to 3D grids for element-wise operations
    # Shape transformations: kx [Nx//2+1] → [1, 1, Nx//2+1]
    #                       ky [Ny] → [1, Ny, 1]
    #                       kz [Nz] → [Nz, 1, 1]
    kx_3d = kx[jnp.newaxis, jnp.newaxis, :]
    ky_3d = ky[jnp.newaxis, :, jnp.newaxis]
    kz_3d = kz[:, jnp.newaxis, jnp.newaxis]

    # Compute |k| for each mode
    k_mag = jnp.sqrt(kx_3d**2 + ky_3d**2 + kz_3d**2)

    # Create spectral mask: force only modes with k ∈ [k_min, k_max]
    forcing_mask = (k_mag >= k_min) & (k_mag <= k_max)

    # Scale amplitude by √dt for proper white noise in time
    # Energy injection: dE/dt ~ amplitude² (independent of dt)
    scale = amplitude / jnp.sqrt(dt)

    # Generate complex noise field
    # Note: JAX's irfftn automatically enforces Hermitian symmetry during inverse
    # transform, so the output will be exactly real-valued even if we don't
    # explicitly enforce f(-k) = f*(k) here. Special modes (kx=0, Nyquist) will
    # be handled correctly by the forcing mask (which zeros k=0) and JAX's FFT.
    noise = (real_part + 1j * imag_part) * scale

    # Apply spectral mask to localize forcing
    forced_field = noise * forcing_mask.astype(noise.dtype)

    # Zero out k=0 mode (no forcing of mean field)
    # k=0 is at [0, 0, 0] in rfft format
    forced_field = forced_field.at[0, 0, 0].set(0.0 + 0.0j)

    return forced_field


def gaussian_white_noise_fourier(
    grid: SpectralGrid3D,
    amplitude: float,
    k_min: float,
    k_max: float,
    dt: float,
    key: Array,
) -> Tuple[Array, Array]:
    """
    Generate band-limited Gaussian white noise in Fourier space.

    This function creates a random forcing field with Gaussian statistics,
    localized to wavenumbers k ∈ [k_min, k_max]. The forcing is white noise
    in time (uncorrelated between calls) and satisfies the reality condition
    for inverse FFT.

    The noise amplitude is scaled by 1/√dt to ensure that the energy injection
    rate ε_inj = ⟨F·u⟩ is independent of timestep size.

    Args:
        grid: SpectralGrid3D defining wavenumber arrays and grid dimensions
        amplitude: Forcing amplitude (sets energy injection rate ~ amplitude²)
        k_min: Minimum wavenumber for forcing band (typically ~ 2)
        k_max: Maximum wavenumber for forcing band (typically ~ 5-10)
        dt: Timestep size (for proper white noise scaling)
        key: JAX random key for reproducible random number generation

    Returns:
        noise_field: Complex forcing field in Fourier space [Nz, Ny, Nx//2+1]
        new_key: Updated JAX random key for next call

    Example:
        >>> grid = SpectralGrid3D.create(Nx=64, Ny=64, Nz=32)
        >>> key = jax.random.PRNGKey(42)
        >>> noise, key = gaussian_white_noise_fourier(
        ...     grid, amplitude=0.1, k_min=2.0, k_max=5.0, dt=0.01, key=key
        ... )
        >>> # Apply to different timestep
        >>> noise2, key = gaussian_white_noise_fourier(
        ...     grid, amplitude=0.1, k_min=2.0, k_max=5.0, dt=0.01, key=key
        ... )

    Physics:
        The forcing field F(k,t) has statistics:
        - ⟨F(k,t)⟩ = 0 (zero mean)
        - ⟨F(k,t)F*(k',t')⟩ = amplitude² δ(t-t') δ(k-k') for k ∈ [k_min, k_max]

        Energy injection rate: ε_inj = amplitude² × N_modes (independent of dt)
    """
    # Split key for real and imaginary parts
    key, subkey1, subkey2 = jax.random.split(key, 3)

    # Generate random Gaussian samples
    shape = (grid.Nz, grid.Ny, grid.Nx // 2 + 1)
    real_part = jax.random.normal(subkey1, shape, dtype=jnp.float32)
    imag_part = jax.random.normal(subkey2, shape, dtype=jnp.float32)

    # Call JIT-compiled kernel
    noise_field = _gaussian_white_noise_fourier_jit(
        grid.kx,
        grid.ky,
        grid.kz,
        amplitude,
        k_min,
        k_max,
        dt,
        real_part,
        imag_part,
    )

    return noise_field.astype(jnp.complex64), key


def force_alfven_modes(
    state: KRMHDState,
    amplitude: float,
    k_min: float,
    k_max: float,
    dt: float,
    key: Array,
) -> Tuple[KRMHDState, Array]:
    """
    Apply Gaussian white noise forcing to Alfvén modes (Elsasser variables).

    **CRITICAL PHYSICS**: This function forces z⁺ and z⁻ IDENTICALLY, which means:
    - φ = (z⁺ + z⁻)/2 is forced → drives perpendicular velocity u⊥ = ẑ × ∇φ
    - A∥ = (z⁺ - z⁻)/2 is NOT forced → avoids artificial perpendicular magnetic field

    This ensures that forcing drives velocity fluctuations only, preventing spurious
    large-scale magnetic reconnection that would occur if A∥ were forced.

    The forcing is additive: z⁺ → z⁺ + F, z⁻ → z⁻ + F (same F for both).

    Args:
        state: Current KRMHD state with Elsasser variables
        amplitude: Forcing amplitude (energy injection ~ amplitude²)
        k_min: Minimum forcing wavenumber (typically ~ 2)
        k_max: Maximum forcing wavenumber (typically ~ 5-10)
        dt: Timestep size (for white noise scaling)
        key: JAX random key

    Returns:
        new_state: State with forcing applied to z⁺ and z⁻
        new_key: Updated random key

    Example:
        >>> key = jax.random.PRNGKey(42)
        >>> # Force at large scales (k ~ 2-5)
        >>> state_forced, key = force_alfven_modes(
        ...     state, amplitude=0.1, k_min=2.0, k_max=5.0, dt=0.01, key=key
        ... )
        >>> # Verify φ changed but A∥ unchanged:
        >>> phi_old = (state.z_plus + state.z_minus) / 2
        >>> phi_new = (state_forced.z_plus + state_forced.z_minus) / 2
        >>> A_old = (state.z_plus - state.z_minus) / 2
        >>> A_new = (state_forced.z_plus - state_forced.z_minus) / 2
        >>> assert jnp.allclose(A_old, A_new)  # A∥ unchanged ✓

    Physics:
        By forcing z⁺ = z⁻ identically:
        - Δφ = (Δz⁺ + Δz⁻)/2 = F  (non-zero, drives u⊥)
        - ΔA∥ = (Δz⁺ - Δz⁻)/2 = 0  (zero, B⊥ unforced)

        This is the standard method for driving MHD turbulence without
        artificial reconnection (Elenbaas+ 2008, Schekochihin+ 2009).
    """
    # Generate single forcing field
    forcing, key = gaussian_white_noise_fourier(
        state.grid, amplitude, k_min, k_max, dt, key
    )

    # Apply IDENTICAL forcing to both Elsasser variables
    # This forces φ = (z⁺+z⁻)/2 only, leaving A∥ = (z⁺-z⁻)/2 unforced
    z_plus_new = state.z_plus + forcing
    z_minus_new = state.z_minus + forcing

    # Create new state with forced Elsasser variables
    new_state = KRMHDState(
        z_plus=z_plus_new,
        z_minus=z_minus_new,
        B_parallel=state.B_parallel,  # Passive field, not forced here
        g=state.g,  # Kinetic moments, not forced here
        M=state.M,
        beta_i=state.beta_i,
        v_th=state.v_th,
        nu=state.nu,
        Lambda=state.Lambda,
        time=state.time,  # Time unchanged (forcing is instantaneous)
        grid=state.grid,
    )

    return new_state, key


def force_slow_modes(
    state: KRMHDState,
    amplitude: float,
    k_min: float,
    k_max: float,
    dt: float,
    key: Array,
) -> Tuple[KRMHDState, Array]:
    """
    Apply Gaussian white noise forcing to slow modes (parallel magnetic field δB∥).

    This forcing is INDEPENDENT from Alfvén forcing and drives compressive/slow
    magnetosonic fluctuations. In KRMHD, slow modes are passively advected by
    the Alfvén flow, but can be independently forced for studying their dynamics.

    Args:
        state: Current KRMHD state
        amplitude: Forcing amplitude for slow modes
        k_min: Minimum forcing wavenumber
        k_max: Maximum forcing wavenumber
        dt: Timestep size
        key: JAX random key

    Returns:
        new_state: State with forcing applied to B∥
        new_key: Updated random key

    Example:
        >>> key = jax.random.PRNGKey(42)
        >>> # Force slow modes independently
        >>> state_forced, key = force_slow_modes(
        ...     state, amplitude=0.05, k_min=2.0, k_max=5.0, dt=0.01, key=key
        ... )

    Physics:
        Slow modes in KRMHD satisfy:
        ∂δB∥/∂t + {φ, δB∥} = -∇∥u∥ + D∇²δB∥ + F_slow

        The forcing F_slow is uncorrelated with Alfvén forcing and allows
        studying passive scalar advection, intermittency, and compressive effects.
    """
    # Generate forcing field for B∥
    forcing, key = gaussian_white_noise_fourier(
        state.grid, amplitude, k_min, k_max, dt, key
    )

    # Apply forcing to parallel magnetic field
    B_parallel_new = state.B_parallel + forcing

    # Create new state
    new_state = KRMHDState(
        z_plus=state.z_plus,  # Alfvén modes unchanged
        z_minus=state.z_minus,
        B_parallel=B_parallel_new,  # Forced slow mode
        g=state.g,  # Kinetic moments unchanged
        M=state.M,
        beta_i=state.beta_i,
        v_th=state.v_th,
        nu=state.nu,
        Lambda=state.Lambda,
        time=state.time,
        grid=state.grid,
    )

    return new_state, key


def compute_energy_injection_rate(
    state_before: KRMHDState,
    state_after: KRMHDState,
    dt: float,
) -> float:
    """
    Compute energy injection rate from forcing.

    This measures the rate at which forcing adds energy to the system:
    ε_inj = (E_after - E_before) / dt

    In steady-state driven turbulence, ε_inj ≈ ε_diss (energy balance).

    Args:
        state_before: State before forcing was applied
        state_after: State after forcing was applied
        dt: Timestep (forcing is applied instantaneously, but rate uses dt)

    Returns:
        Energy injection rate ε_inj (positive if forcing adds energy)

    Example:
        >>> key = jax.random.PRNGKey(42)
        >>> state_old = state
        >>> state_new, key = force_alfven_modes(state, 0.1, 2.0, 5.0, dt, key)
        >>> eps_inj = compute_energy_injection_rate(state_old, state_new, dt)
        >>> print(f"Energy injection: {eps_inj:.3e}")

    Physics:
        For white noise forcing with amplitude A at N_modes:
        ⟨ε_inj⟩ ≈ A² × N_modes (time-averaged)

        Individual realizations fluctuate around this mean value.
    """
    # Compute total energies before and after forcing
    energy_before = energy(state_before)["total"]
    energy_after = energy(state_after)["total"]

    # Energy injection rate
    energy_injection_rate = (energy_after - energy_before) / dt

    return float(energy_injection_rate)
