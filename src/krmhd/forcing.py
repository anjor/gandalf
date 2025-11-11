"""
Forcing mechanisms for driven turbulence simulations (Thesis §2.5.1).

This module implements Gaussian white noise forcing for sustained turbulent cascades.
The forcing is applied in Fourier space at specified wavenumber bands to inject
energy at large scales while allowing inertial-range cascade to develop.

Key features:
- Band-limited forcing: inject energy only at modes n ∈ [n_min, n_max]
- Alfvén sector forcing: drives perpendicular velocity u⊥ only, NOT δB⊥
- Slow mode forcing: optional independent forcing for δB∥ and kinetic moments
- Energy injection rate diagnostics for monitoring energy balance

Physics context:
    Driven turbulence reaches steady state when energy injection balances dissipation:
    ε_inj = ∫ F · u dk = ε_diss = η ∫ k² |u|² dk

    The forcing is white noise in time (uncorrelated between timesteps) and
    band-limited in k-space to concentrate energy injection at large scales.

    Mode numbers n are integers (n=1,2,3,...) that correspond to physical wavenumbers
    k = 2πn/L, where L is the domain size. This eliminates confusion about 2π factors.

Example usage:
    >>> key = jax.random.PRNGKey(42)
    >>> # Force modes n=1,2,3 (largest scales)
    >>> state_forced, key = force_alfven_modes(
    ...     state, amplitude=0.1, n_min=1, n_max=3, dt=0.01, key=key
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


def _mode_to_wavenumber(n: int, L: float) -> float:
    """
    Convert mode number to physical wavenumber.

    For a periodic domain of size L, the allowed Fourier modes are:
    k = 2πn / L, where n = 1, 2, 3, ... are integer mode numbers.

    Args:
        n: Mode number (integer, must be >= 1)
        L: Domain size in physical units

    Returns:
        Physical wavenumber k = 2πn/L

    Example:
        >>> # For unit domain (L=1.0)
        >>> k1 = _mode_to_wavenumber(1, 1.0)  # k₁ = 2π ≈ 6.28
        >>> k2 = _mode_to_wavenumber(2, 1.0)  # k₂ = 4π ≈ 12.57
    """
    if n <= 0:
        raise ValueError(f"Mode number must be positive, got n={n}")
    return 2.0 * jnp.pi * n / L


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

    # CRITICAL: Enforce Hermitian symmetry for rfft format
    # For real-valued fields in physical space, Fourier coefficients must satisfy:
    # f(kx, ky, kz) = f*(-kx, -ky, -kz)
    #
    # In rfft format (only kx >= 0 stored), special modes must be real:
    # 1. kx=0 plane: These modes are their own conjugate partners
    # 2. kx=Nyquist plane (if Nx even): Nyquist frequency must be real
    #
    # Without this enforcement, direct Fourier-space operations (like forcing)
    # can create non-Hermitian fields that violate reality condition.

    # Enforce reality on kx=0 plane (all ky, kz)
    forced_field = forced_field.at[:, :, 0].set(forced_field[:, :, 0].real.astype(forced_field.dtype))

    # Enforce reality on kx=Nyquist plane (if Nx is even)
    # For rfft: shape is [Nz, Ny, Nx//2+1]
    # Nyquist is at index Nx//2 if Nx is even
    Nx_rfft = forced_field.shape[2]  # This is Nx//2+1
    if Nx_rfft > 1:  # Have more than just kx=0 mode
        nyquist_idx = Nx_rfft - 1
        forced_field = forced_field.at[:, :, nyquist_idx].set(
            forced_field[:, :, nyquist_idx].real.astype(forced_field.dtype)
        )

    return forced_field


def gaussian_white_noise_fourier(
    grid: SpectralGrid3D,
    amplitude: float,
    n_min: int,
    n_max: int,
    dt: float,
    key: Array,
) -> Tuple[Array, Array]:
    """
    Generate band-limited Gaussian white noise in Fourier space.

    This function creates a random forcing field with Gaussian statistics,
    localized to mode numbers n ∈ [n_min, n_max]. The forcing is white noise
    in time (uncorrelated between calls) and satisfies the reality condition
    for inverse FFT.

    Mode numbers are converted to physical wavenumbers via k = 2πn/L_min,
    where L_min = min(Lx, Ly, Lz) ensures consistency in anisotropic domains.

    The noise amplitude is scaled by 1/√dt to ensure that the energy injection
    rate ε_inj = ⟨F·u⟩ is independent of timestep size.

    Args:
        grid: SpectralGrid3D defining wavenumber arrays and grid dimensions
        amplitude: Forcing amplitude (sets energy injection rate ~ amplitude²)
        n_min: Minimum mode number for forcing band (typically n=1 or n=2)
        n_max: Maximum mode number for forcing band (typically n=2 to n=5)
        dt: Timestep size (for proper white noise scaling)
        key: JAX random key for reproducible random number generation

    Returns:
        noise_field: Complex forcing field in Fourier space [Nz, Ny, Nx//2+1]
        new_key: Updated JAX random key for next call

    Example:
        >>> grid = SpectralGrid3D.create(Nx=64, Ny=64, Nz=32)
        >>> key = jax.random.PRNGKey(42)
        >>> # Force modes n=1,2,3 (largest scales)
        >>> noise, key = gaussian_white_noise_fourier(
        ...     grid, amplitude=0.1, n_min=1, n_max=3, dt=0.01, key=key
        ... )
        >>> # Apply to different timestep with same forcing band
        >>> noise2, key = gaussian_white_noise_fourier(
        ...     grid, amplitude=0.1, n_min=1, n_max=3, dt=0.01, key=key
        ... )

    Physics:
        The forcing field F(k,t) has statistics:
        - ⟨F(k,t)⟩ = 0 (zero mean)
        - ⟨F(k,t)F*(k',t')⟩ = amplitude² δ(t-t') δ(k-k') for modes n ∈ [n_min, n_max]

        Energy injection rate: ε_inj = amplitude² × N_modes (independent of dt)

        Mode number convention:
        - n=1: Fundamental mode (largest wavelength λ = L)
        - n=2: Second harmonic (λ = L/2)
        - n=3: Third harmonic (λ = L/3), etc.
    """
    # Input validation
    if not isinstance(n_min, int) or not isinstance(n_max, int):
        raise TypeError(f"Mode numbers must be integers, got n_min={type(n_min).__name__}, n_max={type(n_max).__name__}")
    if n_min <= 0:
        raise ValueError(f"n_min must be positive, got n_min={n_min}")
    if n_min >= n_max:
        raise ValueError(f"n_min must be < n_max, got n_min={n_min}, n_max={n_max}")
    if amplitude <= 0:
        raise ValueError(f"amplitude must be positive, got {amplitude}")
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")

    # Convert mode numbers to physical wavenumbers
    # Use minimum domain size for consistency in anisotropic domains
    L_min = min(grid.Lx, grid.Ly, grid.Lz)
    k_min = _mode_to_wavenumber(n_min, L_min)
    k_max = _mode_to_wavenumber(n_max, L_min)

    # Split key for real and imaginary parts
    key, subkey1, subkey2 = jax.random.split(key, 3)

    # Generate random Gaussian samples
    # Note: Using float32 for stochastic forcing is sufficient because:
    # 1. Random noise has intrinsic statistical uncertainty >> float32 precision (~10⁻⁷)
    # 2. Energy injection rate is time-averaged (individual realizations fluctuate)
    # 3. Matches grid.kx/ky/kz dtype (float32) AND state.z_plus dtype (complex64)
    #    for consistent precision throughout the codebase
    # 4. Reduces memory/compute cost with negligible impact on physics
    # 5. GANDALF energy conservation (0.0086% error) is not limited by float32
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
    n_min: int,
    n_max: int,
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
        n_min: Minimum mode number for forcing band (typically n=1 or n=2)
        n_max: Maximum mode number for forcing band (typically n=2 to n=5)
        dt: Timestep size (for white noise scaling)
        key: JAX random key

    Returns:
        new_state: State with forcing applied to z⁺ and z⁻
        new_key: Updated random key

    Example:
        >>> key = jax.random.PRNGKey(42)
        >>> # Force modes n=1,2,3 (largest scales)
        >>> state_forced, key = force_alfven_modes(
        ...     state, amplitude=0.1, n_min=1, n_max=3, dt=0.01, key=key
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

        Mode number convention: n=1 is the fundamental mode (k=2π/L),
        n=2 is the second harmonic (k=4π/L), etc.
    """
    # Input validation (gaussian_white_noise_fourier validates n_min, n_max, dt)
    if amplitude <= 0:
        raise ValueError(f"amplitude must be positive, got {amplitude}")

    # Generate single forcing field
    forcing, key = gaussian_white_noise_fourier(
        state.grid, amplitude, n_min, n_max, dt, key
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
    n_min: int,
    n_max: int,
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
        n_min: Minimum mode number for forcing band (typically n=1 or n=2)
        n_max: Maximum mode number for forcing band (typically n=2 to n=5)
        dt: Timestep size
        key: JAX random key

    Returns:
        new_state: State with forcing applied to B∥
        new_key: Updated random key

    Example:
        >>> key = jax.random.PRNGKey(42)
        >>> # Force slow modes at large scales
        >>> state_forced, key = force_slow_modes(
        ...     state, amplitude=0.05, n_min=1, n_max=3, dt=0.01, key=key
        ... )

    Physics:
        Slow modes in KRMHD satisfy:
        ∂δB∥/∂t + {φ, δB∥} = -∇∥u∥ + D∇²δB∥ + F_slow

        The forcing F_slow is uncorrelated with Alfvén forcing and allows
        studying passive scalar advection, intermittency, and compressive effects.

        Mode number convention: n=1 is the fundamental mode (k=2π/L),
        n=2 is the second harmonic (k=4π/L), etc.
    """
    # Input validation (gaussian_white_noise_fourier validates n_min, n_max, dt)
    if amplitude <= 0:
        raise ValueError(f"amplitude must be positive, got {amplitude}")

    # Generate forcing field for B∥
    forcing, key = gaussian_white_noise_fourier(
        state.grid, amplitude, n_min, n_max, dt, key
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
    Compute instantaneous energy injection rate from forcing.

    This measures the rate at which forcing adds energy to the system:
    ε_inj = (E_after - E_before) / dt

    **Important:** This is an INSTANTANEOUS measurement for a single forcing
    realization. For steady-state energy balance validation (ε_inj ≈ ε_diss),
    TIME-AVERAGE over many forcing realizations:

        ⟨ε_inj⟩ = (1/T) ∫₀ᵀ ε_inj(t) dt

    Individual realizations have O(1) variance due to stochastic forcing.

    Args:
        state_before: State before forcing was applied
        state_after: State after forcing was applied
        dt: Timestep (used ONLY for dimensional conversion: ΔE → ΔE/dt)
            Note: The physics of forcing is already complete in state_after.
            This parameter converts energy change to energy *rate* for convenience.

    Returns:
        Energy injection rate ε_inj for this realization (can be + or -)

    Example:
        >>> key = jax.random.PRNGKey(42)
        >>> # Single realization (fluctuates)
        >>> state_old = state
        >>> state_new, key = force_alfven_modes(state, 0.1, 2.0, 5.0, dt, key)
        >>> eps_inj = compute_energy_injection_rate(state_old, state_new, dt)
        >>> print(f"Instantaneous: {eps_inj:.3e}")
        >>>
        >>> # Time-averaged (steady-state validation)
        >>> eps_inj_list = []
        >>> for i in range(100):
        ...     state_old = state
        ...     state, key = force_alfven_modes(state, 0.1, 2.0, 5.0, dt, key)
        ...     eps_inj_list.append(compute_energy_injection_rate(state_old, state, dt))
        ...     state = gandalf_step(state, dt, eta=0.01, v_A=1.0)
        >>> eps_inj_avg = np.mean(eps_inj_list)
        >>> print(f"Time-averaged: {eps_inj_avg:.3e}")

    Physics:
        For white noise forcing with amplitude A at N_modes:
        ⟨ε_inj⟩ ≈ A² × N_modes (time-averaged expectation value)

        Individual realizations fluctuate: ε_inj(t) = ⟨ε_inj⟩ ± O(⟨ε_inj⟩)

        In steady state: ⟨ε_inj⟩ = ⟨ε_diss⟩ (energy balance)
    """
    # Input validation
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")

    # Compute total energies before and after forcing
    energy_before = energy(state_before)["total"]
    energy_after = energy(state_after)["total"]

    # Energy injection rate
    energy_injection_rate = (energy_after - energy_before) / dt

    return float(energy_injection_rate)
