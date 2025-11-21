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
def _gaussian_white_noise_fourier_perp_lowkz_jit(
    kx: Array,
    ky: Array,
    kz: Array,
    amplitude: float,
    kperp_min: float,
    kperp_max: float,
    kz_allowed: Array,
    dt: float,
    real_part: Array,
    imag_part: Array,
) -> Array:
    """
    JIT core for Gaussian white noise forcing restricted to a perpendicular band
    and a low-|kz| set, to respect RMHD ordering (k_perp >> k_parallel).

    Args:
        kx, ky, kz: 1D wavenumber arrays
        amplitude: Forcing amplitude (energy injection ~ amplitude²)
        kperp_min, kperp_max: Perpendicular band limits
        kz_allowed: Boolean mask over kz (shape [Nz]) selecting allowed kz planes
        dt: Timestep
        real_part, imag_part: Random normal arrays [Nz, Ny, Nx//2+1]

    Returns:
        Complex forcing field [Nz, Ny, Nx//2+1]
    """
    # Broadcast to grids
    kx_3d = kx[jnp.newaxis, jnp.newaxis, :]
    ky_3d = ky[jnp.newaxis, :, jnp.newaxis]
    kz_3d = kz[:, jnp.newaxis, jnp.newaxis]

    # Perpendicular magnitude
    k_perp = jnp.sqrt(kx_3d**2 + ky_3d**2)

    # Masks
    perp_mask = (k_perp >= kperp_min) & (k_perp <= kperp_max)
    kz_mask = kz_allowed[:, jnp.newaxis, jnp.newaxis]
    mask = perp_mask & kz_mask

    # White noise scaling
    scale = amplitude / jnp.sqrt(dt)
    noise = (real_part + 1j * imag_part) * scale

    forced_field = noise * mask.astype(noise.dtype)

    # Enforce rfft reality on kx=0 and kx=Nyquist planes
    # Note: For JIT compatibility, we always apply the Nyquist operation even when Nx_rfft==1
    # (in which case kx=0 and kx=Nyquist are the same, so the second operation is redundant but harmless)
    forced_field = forced_field.at[:, :, 0].set(forced_field[:, :, 0].real.astype(forced_field.dtype))
    Nx_rfft = forced_field.shape[2]
    nyquist_idx = Nx_rfft - 1
    forced_field = forced_field.at[:, :, nyquist_idx].set(
        forced_field[:, :, nyquist_idx].real.astype(forced_field.dtype)
    )

    # Zero DC
    forced_field = forced_field.at[0, 0, 0].set(0.0 + 0.0j)
    return forced_field


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
    # Note: For JIT compatibility, we always apply this operation even when Nx_rfft==1
    # (in which case kx=0 and kx=Nyquist are the same, so the operation is redundant but harmless)
    Nx_rfft = forced_field.shape[2]  # This is Nx//2+1
    nyquist_idx = Nx_rfft - 1
    forced_field = forced_field.at[:, :, nyquist_idx].set(
        forced_field[:, :, nyquist_idx].real.astype(forced_field.dtype)
    )

    return forced_field


@jax.jit
def _gandalf_forcing_fourier_jit(
    kx: Array,
    ky: Array,
    kz: Array,
    fampl: float,
    k_min: float,
    k_max: float,
    dt: float,
    random_amplitude: Array,
    random_phase: Array,
) -> Array:
    """
    JIT-compiled core function for original GANDALF forcing (forcing.cu).

    Original formula: amp = (1/|k_perp|) * sqrt((fampl/dt) * log(random))

    Args:
        kx, ky, kz: Wavenumber arrays (broadcast-compatible shapes)
        fampl: GANDALF forcing amplitude parameter (thesis uses 1.0)
        k_min: Minimum wavenumber for forcing band
        k_max: Maximum wavenumber for forcing band
        dt: Timestep
        random_amplitude: Random uniform samples in (0,1] for amplitude [Nz, Ny, Nx//2+1]
        random_phase: Random uniform samples in [0, 2π] for phase [Nz, Ny, Nx//2+1]

    Returns:
        Complex Fourier field with GANDALF forcing at k ∈ [k_min, k_max]
    """
    # Broadcast wavenumbers to 3D grids
    kx_3d = kx[jnp.newaxis, jnp.newaxis, :]
    ky_3d = ky[jnp.newaxis, :, jnp.newaxis]
    kz_3d = kz[:, jnp.newaxis, jnp.newaxis]

    # Compute perpendicular wavenumber magnitude
    k_perp = jnp.sqrt(kx_3d**2 + ky_3d**2)

    # Compute total wavenumber for mask
    k_mag = jnp.sqrt(kx_3d**2 + ky_3d**2 + kz_3d**2)

    # Create spectral mask: force only modes with k ∈ [k_min, k_max]
    forcing_mask = (k_mag >= k_min) & (k_mag <= k_max)

    # Original GANDALF amplitude formula:
    # amp = (1/|k_perp|) * sqrt((fampl/dt) * |log(random)|)
    # Note: abs(log(random)) ensures positive values since random ∈ (0,1]
    # Clip k_perp to avoid division by zero (k_perp=0 modes get zero amplitude anyway)
    k_perp_safe = jnp.maximum(k_perp, 1e-10)

    # Scale factor from logarithmic random modulation
    # Use abs(log) since log of uniform(0,1) gives negative values
    log_factor = jnp.sqrt(jnp.abs((fampl / dt) * jnp.log(random_amplitude)))

    # Combined amplitude: (1/k_perp) * sqrt((fampl/dt) * |log(random)|)
    amplitude = (1.0 / k_perp_safe) * log_factor

    # Create complex forcing with random phases
    # Original GANDALF uses: phase = π * (2*random - 1) ∈ [-π, π]
    forced_field = amplitude * jnp.exp(1j * random_phase)

    # Apply spectral mask
    forced_field = forced_field * forcing_mask.astype(forced_field.dtype)

    # Zero out k=0 mode (no forcing of mean field)
    forced_field = forced_field.at[0, 0, 0].set(0.0 + 0.0j)

    # Enforce Hermitian symmetry for rfft format
    # Note: For JIT compatibility, we always apply the Nyquist operation even when Nx_rfft==1
    # (in which case kx=0 and kx=Nyquist are the same, so the operation is redundant but harmless)
    forced_field = forced_field.at[:, :, 0].set(forced_field[:, :, 0].real.astype(forced_field.dtype))

    Nx_rfft = forced_field.shape[2]
    nyquist_idx = Nx_rfft - 1
    forced_field = forced_field.at[:, :, nyquist_idx].set(
        forced_field[:, :, nyquist_idx].real.astype(forced_field.dtype)
    )

    return forced_field


def gandalf_forcing_fourier(
    grid: SpectralGrid3D,
    fampl: float,
    n_min: int,
    n_max: int,
    dt: float,
    key: Array,
) -> Tuple[Array, Array]:
    """
    Generate forcing using original GANDALF formula (forcing.cu).

    This implements the exact forcing scheme from the thesis/original GANDALF:
        amp = (1/|k_perp|) * sqrt((fampl/dt) * |log(random)|)

    Key differences from gaussian_white_noise_fourier:
    - Amplitude scales as 1/k_perp (larger scales get more energy)
    - Logarithmic random modulation instead of Gaussian
    - Effective amplitude ~ sqrt(fampl/dt), much stronger than linear scaling

    Args:
        grid: SpectralGrid3D defining the computational domain
        fampl: GANDALF forcing amplitude parameter (thesis uses 1.0 at 128³)
        n_min: Minimum mode number to force (integer)
        n_max: Maximum mode number to force (integer)
        dt: Current timestep (affects amplitude scaling)
        key: JAX random key for generating stochastic forcing

    Returns:
        (forced_field, new_key): Complex Fourier field and updated random key

    Example:
        >>> key = jax.random.PRNGKey(42)
        >>> # Original GANDALF parameters
        >>> forcing, key = gandalf_forcing_fourier(
        ...     grid, fampl=1.0, n_min=1, n_max=2, dt=0.005, key=key
        ... )

    Reference:
        - Original GANDALF forcing.cu
        - Gandalf.in: fampl=1.0, nkstir=6, alpha_hyper=3, nu_hyper=1.0
    """
    # Convert mode numbers to wavenumbers
    L_min = min(grid.Lx, grid.Ly, grid.Lz)
    k_min = _mode_to_wavenumber(n_min, L_min)
    k_max = _mode_to_wavenumber(n_max, L_min)

    # Generate random numbers for amplitude (uniform in (0,1]) and phase
    key, subkey1, subkey2 = jax.random.split(key, 3)

    # Random amplitude in (0,1] (avoid exact 0 for log)
    random_amplitude = jax.random.uniform(subkey1, shape=(grid.Nz, grid.Ny, grid.Nx//2+1),
                                         minval=1e-10, maxval=1.0)

    # Random phase in [0, 2π]
    random_phase = jax.random.uniform(subkey2, shape=(grid.Nz, grid.Ny, grid.Nx//2+1),
                                     minval=0.0, maxval=2.0*jnp.pi)

    # Apply GANDALF forcing formula
    forced_field = _gandalf_forcing_fourier_jit(
        grid.kx, grid.ky, grid.kz,
        fampl, k_min, k_max, dt,
        random_amplitude, random_phase
    )

    return forced_field, key


def gaussian_white_noise_fourier_perp_lowkz(
    grid: SpectralGrid3D,
    amplitude: float,
    n_min: int,
    n_max: int,
    max_nz: int,
    include_nz0: bool,
    dt: float,
    key: Array,
) -> Tuple[Array, Array]:
    """
    Generate Gaussian white noise forcing restricted to a perpendicular band
    n_perp ∈ [n_min, n_max] and low-|nz| planes (|nz| ≤ max_nz, optionally excluding nz=0).

    This respects RMHD ordering by avoiding injection at high k_parallel and
    is useful for robust Alfvénic cascade development.

    Args:
        grid: SpectralGrid3D
        amplitude: Forcing amplitude
        n_min, n_max: Perpendicular mode-number band (integers)
        max_nz: Maximum |nz| allowed (e.g., 1)
        include_nz0: Whether to include kz=0 plane
        dt: Timestep
        key: JAX PRNG key

    Returns:
        (forced_field, new_key)
    """
    if amplitude <= 0:
        raise ValueError(f"amplitude must be positive, got {amplitude}")
    if n_min <= 0 or n_max < n_min:
        raise ValueError(f"Invalid band: n_min={n_min}, n_max={n_max}")
    if max_nz < 0:
        raise ValueError(f"max_nz must be ≥ 0, got {max_nz}")

    # Perpendicular band limits
    kperp_min = _mode_to_wavenumber(n_min, grid.Lx)
    kperp_max = _mode_to_wavenumber(n_max, grid.Lx)

    # Allowed kz planes: |nz| ≤ max_nz and optionally nz=0 excluded
    k1 = 2.0 * jnp.pi / grid.Lz
    kz = grid.kz
    # Build boolean mask over kz by nearest-integer test on |kz|/k1
    nz_float = jnp.round(jnp.abs(kz) / k1)
    nz_int = nz_float.astype(jnp.int32)
    kz_allowed = nz_int <= max_nz
    if not include_nz0:
        kz_allowed = kz_allowed & (nz_int != 0)

    # Random fields
    key, k1a, k1b = jax.random.split(key, 3)
    real_part = jax.random.normal(k1a, shape=(grid.Nz, grid.Ny, grid.Nx // 2 + 1))
    imag_part = jax.random.normal(k1b, shape=(grid.Nz, grid.Ny, grid.Nx // 2 + 1))

    forced_field = _gaussian_white_noise_fourier_perp_lowkz_jit(
        grid.kx, grid.ky, grid.kz,
        amplitude,
        float(kperp_min), float(kperp_max),
        kz_allowed,
        float(dt),
        real_part, imag_part,
    )
    return forced_field, key


def force_alfven_modes_balanced(
    state: KRMHDState,
    amplitude: float,
    n_min: int,
    n_max: int,
    dt: float,
    key: Array,
    max_nz: int = 1,
    include_nz0: bool = False,
    correlation: float = 0.0,
) -> Tuple[KRMHDState, Array]:
    """
    Force Alfvén modes with independent z⁺/z⁻ forcing to sustain strong
    counter-propagating interactions. Restricts to low-|nz| to respect RMHD.

    Args:
        state: KRMHD state
        amplitude: Forcing amplitude (per field)
        n_min, n_max: Perpendicular band (mode numbers)
        dt: Timestep
        key: JAX key
        max_nz: Allowed |nz| (default 1)
        include_nz0: Include kz=0 plane (default False)
        correlation: Correlation coefficient between z⁺ and z⁻ forcing in [0,1)
                     0 → independent (recommended), 0.5 → partially correlated.

    Returns:
        (new_state, new_key)
    """
    if not (0.0 <= correlation < 1.0):
        raise ValueError(f"correlation must be in [0,1), got {correlation}")

    # Draw two independent fields
    key, kA, kB, kMix = jax.random.split(key, 4)
    Fp, kA = gaussian_white_noise_fourier_perp_lowkz(
        state.grid, amplitude, n_min, n_max, max_nz, include_nz0, dt, kA
    )
    Fm, kB = gaussian_white_noise_fourier_perp_lowkz(
        state.grid, amplitude, n_min, n_max, max_nz, include_nz0, dt, kB
    )

    if correlation > 0.0:
        # Mix fields to achieve desired correlation approximately
        # z− forcing ← sqrt(1-ρ²)·Fm + ρ·Fp
        rho = jnp.float32(correlation)
        alpha = jnp.sqrt(jnp.maximum(0.0, 1.0 - rho * rho))
        Fm = alpha * Fm + rho * Fp

    z_plus_new = state.z_plus + Fp
    z_minus_new = state.z_minus + Fm

    new_state = KRMHDState(
        z_plus=z_plus_new,
        z_minus=z_minus_new,
        B_parallel=state.B_parallel,
        g=state.g,
        M=state.M,
        beta_i=state.beta_i,
        v_th=state.v_th,
        nu=state.nu,
        Lambda=state.Lambda,
        time=state.time,
        grid=state.grid,
    )
    return new_state, key


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


def force_alfven_modes_gandalf(
    state: KRMHDState,
    fampl: float,
    n_min: int,
    n_max: int,
    dt: float,
    key: Array,
) -> Tuple[KRMHDState, Array]:
    """
    Apply original GANDALF forcing to Alfvén modes (Elsasser variables).

    This uses the exact forcing formula from the thesis/original GANDALF code:
        amp = (1/|k_perp|) * sqrt((fampl/dt) * |log(random)|)

    Like force_alfven_modes(), this forces z⁺ and z⁻ IDENTICALLY to drive
    velocity fluctuations only (φ forcing) without artificial magnetic reconnection.

    Args:
        state: Current KRMHD state with Elsasser variables
        fampl: GANDALF forcing amplitude parameter (thesis uses 1.0 at 128³)
        n_min: Minimum mode number for forcing band (original: 1)
        n_max: Maximum mode number for forcing band (original: 2)
        dt: Timestep size
        key: JAX random key

    Returns:
        new_state: State with GANDALF forcing applied to z⁺ and z⁻
        new_key: Updated random key

    Example:
        >>> key = jax.random.PRNGKey(42)
        >>> # Original GANDALF thesis parameters
        >>> state_forced, key = force_alfven_modes_gandalf(
        ...     state, fampl=1.0, n_min=1, n_max=2, dt=0.005, key=key
        ... )

    Reference:
        - Original GANDALF forcing.cu
        - Gandalf.in: fampl=1.0, nkstir=6 modes
    """
    # Input validation
    if fampl <= 0:
        raise ValueError(f"fampl must be positive, got {fampl}")

    # Generate forcing using GANDALF formula
    forcing, key = gandalf_forcing_fourier(
        state.grid, fampl, n_min, n_max, dt, key
    )

    # Apply IDENTICAL forcing to both Elsasser variables
    z_plus_new = state.z_plus + forcing
    z_minus_new = state.z_minus + forcing

    # Create new state with forced Elsasser variables
    new_state = KRMHDState(
        z_plus=z_plus_new,
        z_minus=z_minus_new,
        B_parallel=state.B_parallel,
        g=state.g,
        M=state.M,
        beta_i=state.beta_i,
        v_th=state.v_th,
        nu=state.nu,
        Lambda=state.Lambda,
        time=state.time,
        grid=state.grid,
    )

    return new_state, key


def _mode_triplet_to_indices(
    nx: int, ny: int, nz: int,
    Nx: int, Ny: int, Nz: int
) -> Tuple[int, int, int]:
    """
    Convert mode number triplet (nx, ny, nz) to array indices for rfft format.

    Mode numbers can be negative (e.g., -1, 0, 1), but array indices must be
    non-negative. This function handles the periodic wraparound:
    - Positive modes: nx → nx (for kx >= 0)
    - Negative modes: ny = -1 → Ny-1, nz = -1 → Nz-1

    Args:
        nx, ny, nz: Mode numbers (can be negative)
        Nx, Ny, Nz: Grid dimensions

    Returns:
        (ix, iy, iz): Array indices in [0, N-1]

    Example:
        >>> # Mode (-1, 0, 1) on 64³ grid
        >>> ix, iy, iz = _mode_triplet_to_indices(-1, 0, 1, 64, 64, 64)
        >>> # ix = -1 (signals need for Hermitian conjugate)
        >>> # iy = 0, iz = 1
    """
    # For kx: keep as-is (can be negative, signals conjugate mode)
    ix = nx

    # For ky, kz: wrap negative indices
    iy = ny % Ny
    iz = nz % Nz

    return ix, iy, iz


def _get_rfft_compatible_modes(
    mode_triplets: list
) -> list:
    """
    Convert mode triplets to rfft-compatible format using Hermitian symmetry.

    In rfft format, only kx >= 0 is stored. For modes with kx < 0, we use
    the Hermitian symmetry f(-kx, -ky, -kz) = f*(kx, ky, kz) to find the
    stored conjugate partner.

    Args:
        mode_triplets: List of (nx, ny, nz) integer mode number triplets

    Returns:
        List of (nx_stored, ny_stored, nz_stored, conjugate) tuples
        where conjugate=True means to use complex conjugate

    Example:
        >>> # Original GANDALF modes
        >>> modes = [(1, 0, 1), (0, 1, 1), (-1, 0, 1),
        ...          (1, 1, -1), (0, 1, -1), (-1, 1, -1)]
        >>> compatible = _get_rfft_compatible_modes(modes)
        >>> # (-1, 0, 1) → stored as conj(1, 0, -1)
        >>> # (-1, 1, -1) → stored as conj(1, -1, 1)
    """
    rfft_modes = []

    for nx, ny, nz in mode_triplets:
        if nx >= 0:
            # kx >= 0: mode is directly stored
            rfft_modes.append((nx, ny, nz, False))
        else:
            # kx < 0: use Hermitian symmetry f(-kx,-ky,-kz) = f*(kx,ky,kz)
            # So mode (nx, ny, nz) with nx<0 is stored as conj(-nx, -ny, -nz)
            rfft_modes.append((-nx, -ny, -nz, True))

    return rfft_modes


@partial(jax.jit, static_argnums=(0,))
def _gandalf_forcing_specific_jit(
    shape: Tuple[int, int, int],
    mode_indices: Array,  # [N_modes, 3] array of (ix, iy, iz)
    conjugate_flags: Array,  # [N_modes] boolean array
    amplitudes: Array,  # [N_modes] complex amplitudes
) -> Array:
    """
    JIT-compiled kernel for forcing specific mode triplets.

    This creates a Fourier field with forcing applied only at the specified
    mode triplets, preserving the original GANDALF approach of forcing exactly
    6 specific (kx, ky, kz) modes with low k_z.

    Args:
        shape: Output array shape [Nz, Ny, Nx//2+1] (static argument)
        mode_indices: Integer array [N_modes, 3] of (ix, iy, iz) indices
        conjugate_flags: Boolean array [N_modes] indicating if conjugate needed
        amplitudes: Complex array [N_modes] of forcing amplitudes

    Returns:
        Complex Fourier field [Nz, Ny, Nx//2+1] with forcing at specified modes
    """
    Nz, Ny, Nx_rfft = shape

    # Initialize empty field
    forced_field = jnp.zeros((Nz, Ny, Nx_rfft), dtype=jnp.complex64)

    # Apply forcing to each mode
    for i in range(mode_indices.shape[0]):
        iz, iy, ix = mode_indices[i]  # Note: reversed order from storage
        amplitude = amplitudes[i]
        use_conjugate = conjugate_flags[i]

        # Apply conjugate if needed (for negative kx modes)
        # Use jnp.where for JAX compatibility (no Python control flow on tracers)
        amplitude = jnp.where(use_conjugate, jnp.conj(amplitude), amplitude)

        # Set the mode (JAX requires .at syntax for in-place updates)
        forced_field = forced_field.at[iz, iy, ix].set(amplitude)

    return forced_field


def force_alfven_modes_specific(
    state: KRMHDState,
    mode_triplets: list,
    fampl: float,
    dt: float,
    key: Array,
) -> Tuple[KRMHDState, Array]:
    """
    Force specific (kx, ky, kz) mode triplets matching original GANDALF.

    This implements the exact forcing scheme from the original GANDALF code,
    which forces only 6 specific mode triplets with low k_z (±1):

    Original GANDALF modes:
        - (1, 0, 1), (0, 1, 1), (-1, 0, 1)   # k_z = +1
        - (1, 1, -1), (0, 1, -1), (-1, 1, -1)  # k_z = -1

    Key difference from band forcing:
    - Band forcing: All ~50-100 modes with k_perp ∈ [n_min, n_max]
    - Specific forcing: Only specified modes (typically 6)
    - Result: 10-20× less energy injection, respects RMHD ordering k⊥ >> k∥

    Each mode is forced with GANDALF amplitude formula:
        amp = (1/|k_perp|) * sqrt((fampl/dt) * |log(random)|)

    Args:
        state: Current KRMHD state
        mode_triplets: List of (nx, ny, nz) integer mode number triplets to force
        fampl: GANDALF forcing amplitude parameter (thesis uses 1.0)
        dt: Timestep
        key: JAX random key

    Returns:
        new_state: State with forcing applied to z⁺ and z⁻
        new_key: Updated random key

    Example:
        >>> # Original GANDALF configuration
        >>> GANDALF_MODES = [
        ...     (1, 0, 1), (0, 1, 1), (-1, 0, 1),
        ...     (1, 1, -1), (0, 1, -1), (-1, 1, -1)
        ... ]
        >>> state, key = force_alfven_modes_specific(
        ...     state, GANDALF_MODES, fampl=1.0, dt=0.005, key=key
        ... )

    Physics:
        By forcing only low-k_z modes (|nz| = 1), we respect the RMHD ordering:
        k_perp >> k_parallel

        This prevents the exponential instability seen with band forcing, which
        inadvertently forces high-k_z modes (up to k_z ~ 30 at 64³ resolution).

    Reference:
        - Original GANDALF Gandalf.in: nkstir=6 specific modes
        - forcing.cu: Mode list hardcoded in stirring kernel
    """
    # Input validation
    if fampl <= 0:
        raise ValueError(f"fampl must be positive, got {fampl}")
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
    if not mode_triplets:
        raise ValueError("mode_triplets cannot be empty")

    grid = state.grid
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz

    # Convert mode triplets to rfft-compatible format
    rfft_modes = _get_rfft_compatible_modes(mode_triplets)
    n_modes = len(rfft_modes)

    # Generate random numbers for each mode
    key, subkey1, subkey2 = jax.random.split(key, 3)

    # Random amplitude in (0,1] for log factor
    random_amp = jax.random.uniform(subkey1, shape=(n_modes,),
                                    minval=1e-10, maxval=1.0)

    # Random phase in [0, 2π]
    random_phase = jax.random.uniform(subkey2, shape=(n_modes,),
                                     minval=0.0, maxval=2.0*jnp.pi)

    # Compute amplitude and phase for each mode
    mode_indices = []
    conjugate_flags = []
    amplitudes = []

    for i, (nx_stored, ny_stored, nz_stored, conjugate) in enumerate(rfft_modes):
        # Convert to array indices
        ix, iy, iz = _mode_triplet_to_indices(nx_stored, ny_stored, nz_stored,
                                              Nx, Ny, Nz)

        # Compute k_perp for this mode (using stored coordinates)
        kx = grid.kx[ix] if ix < len(grid.kx) else 0.0
        ky = grid.ky[iy] if iy < len(grid.ky) else 0.0
        k_perp = jnp.sqrt(kx**2 + ky**2)
        k_perp_safe = jnp.maximum(k_perp, 1e-10)

        # GANDALF amplitude formula
        log_factor = jnp.sqrt(jnp.abs((fampl / dt) * jnp.log(random_amp[i])))
        amplitude_mag = (1.0 / k_perp_safe) * log_factor

        # Complex amplitude with random phase
        amplitude = amplitude_mag * jnp.exp(1j * random_phase[i])

        mode_indices.append([iz, iy, ix])  # Note: [Nz, Ny, Nx] order
        conjugate_flags.append(conjugate)
        amplitudes.append(amplitude)

    # Convert to JAX arrays
    mode_indices = jnp.array(mode_indices, dtype=jnp.int32)
    conjugate_flags = jnp.array(conjugate_flags, dtype=jnp.bool_)
    amplitudes = jnp.array(amplitudes, dtype=jnp.complex64)

    # Apply forcing via JIT kernel
    shape = (Nz, Ny, Nx // 2 + 1)
    forcing = _gandalf_forcing_specific_jit(shape, mode_indices,
                                            conjugate_flags, amplitudes)

    # Apply IDENTICAL forcing to both Elsasser variables (drives φ only)
    z_plus_new = state.z_plus + forcing
    z_minus_new = state.z_minus + forcing

    # Create new state
    new_state = KRMHDState(
        z_plus=z_plus_new,
        z_minus=z_minus_new,
        B_parallel=state.B_parallel,
        g=state.g,
        M=state.M,
        beta_i=state.beta_i,
        v_th=state.v_th,
        nu=state.nu,
        Lambda=state.Lambda,
        time=state.time,
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


def force_hermite_moments(
    state: KRMHDState,
    amplitude: float,
    n_min: int,
    n_max: int,
    dt: float,
    key: Array,
    forced_moments: Tuple[int, ...] = (0,),
) -> Tuple[KRMHDState, Array]:
    """
    Apply Gaussian white noise forcing to Hermite moments (kinetic distribution).

    **CRITICAL FOR KINETIC FDT VALIDATION**: This function forces the Hermite
    moments g_m directly, which is required for proper fluctuation-dissipation
    theorem (FDT) validation in velocity space (Thesis Chapter 3, Eq 3.26).

    Unlike force_alfven_modes() which forces z± and excites g_m indirectly through
    nonlinear coupling, this function adds stochastic forcing directly to the
    specified Hermite moments. For FDT validation, typically only g₀ (density) is
    forced, matching the kinetic Langevin equation:

        ∂g₀/∂t + ∂(g₁/√2)/∂z = χ(t)

    where χ(t) is Gaussian white noise forcing in velocity space.

    The forcing is additive: g_m → g_m + F for each m ∈ forced_moments.

    Args:
        state: Current KRMHD state with Hermite moments
        amplitude: Forcing amplitude (energy injection ~ amplitude²)
        n_min: Minimum mode number for forcing band (typically n=1 or n=2)
        n_max: Maximum mode number for forcing band (typically n=2 to n=5)
        dt: Timestep size (for white noise scaling)
        key: JAX random key
        forced_moments: Tuple of Hermite moment indices to force (default: (0,) for g₀ only)
            Examples:
            - (0,): Force density g₀ only (standard FDT validation)
            - (0, 1): Force both g₀ and g₁ (density + parallel velocity)
            - (0, 1, 2): Force g₀, g₁, g₂ (including parallel pressure)

    Returns:
        new_state: State with forcing applied to specified Hermite moments
        new_key: Updated random key

    Example:
        >>> key = jax.random.PRNGKey(42)
        >>> # Force g₀ at mode n=1 (fundamental mode) for FDT validation
        >>> state_forced, key = force_hermite_moments(
        ...     state, amplitude=0.15, n_min=1, n_max=1, dt=0.01, key=key,
        ...     forced_moments=(0,)
        ... )

    Physics:
        For kinetic FDT validation (Thesis §3.2.1):
        - Drive a single k-mode with white noise: χ(k,t)
        - Measure time-averaged Hermite spectrum: ⟨|g_m(k)|²⟩
        - Compare with analytical prediction from fluctuation-dissipation theorem

        The steady-state spectrum decays with m due to:
        - Landau damping: Parallel streaming drives energy to higher moments
        - Collisional damping: Lenard-Bernstein operator damps high-m modes
        - m_crit ~ k∥v_th/ν: Collisional cutoff where damping dominates

        Mode number convention (same as force_alfven_modes):
        - n=1: Fundamental mode (largest wavelength λ = L)
        - n=2: Second harmonic (λ = L/2), etc.

    References:
        - Thesis Chapter 3: "Fluctuation-dissipation relations for a kinetic Langevin equation"
        - Thesis Eq 3.26: Forcing term in g₀ equation
        - Thesis Eq 3.37: Analytical phase mixing spectrum
        - Kanekar et al. (2015) J. Plasma Phys. 81: Kinetic FDT validation
    """
    from typing import Tuple as TupleType  # Import for type hint

    # Input validation (gaussian_white_noise_fourier validates n_min, n_max, dt)
    if amplitude <= 0:
        raise ValueError(f"amplitude must be positive, got {amplitude}")
    if not forced_moments:
        raise ValueError("forced_moments cannot be empty")
    if not isinstance(forced_moments, (tuple, list)):
        raise TypeError(f"forced_moments must be tuple or list, got {type(forced_moments)}")

    # Validate moment indices
    for m in forced_moments:
        if not isinstance(m, int):
            raise TypeError(f"Moment indices must be integers, got {type(m).__name__} for m={m}")
        if m < 0:
            raise ValueError(f"Moment indices must be non-negative, got m={m}")
        if m > state.M:
            raise ValueError(f"Moment index m={m} exceeds M={state.M}")

    # Generate forcing field (same noise for all specified moments)
    forcing, key = gaussian_white_noise_fourier(
        state.grid, amplitude, n_min, n_max, dt, key
    )

    # Apply forcing to specified Hermite moments
    # g has shape [Nz, Ny, Nx//2+1, M+1], forcing has shape [Nz, Ny, Nx//2+1]
    g_new = jnp.array(state.g)  # Create mutable copy

    for m in forced_moments:
        g_new = g_new.at[:, :, :, m].add(forcing)

    # Create new state
    new_state = KRMHDState(
        z_plus=state.z_plus,  # Alfvén modes unchanged
        z_minus=state.z_minus,
        B_parallel=state.B_parallel,  # Slow modes unchanged
        g=g_new,  # Forced Hermite moments
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
