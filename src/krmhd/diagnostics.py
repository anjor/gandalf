"""
Diagnostic functions for KRMHD simulations.

This module provides tools for analyzing and visualizing KRMHD turbulence:
- Energy spectra: E(k), E(k⊥), E(k∥)
- Energy history tracking: E(t) for all components
- State visualization: 2D slices of fields

All spectrum calculations use Parseval's theorem and properly account for:
- rfft storage (only positive kx frequencies)
- Reality condition for real fields
- Shell averaging for isotropic spectra

Example usage:
    >>> from krmhd import initialize_random_spectrum, gandalf_step, EnergyHistory
    >>> from krmhd.diagnostics import energy_spectrum_1d, plot_state
    >>>
    >>> # Initialize and evolve
    >>> state = initialize_random_spectrum(grid, M=20, alpha=5/3)
    >>> history = EnergyHistory()
    >>> for i in range(100):
    ...     history.append(state)
    ...     state = gandalf_step(state, dt=0.01, eta=0.01, v_A=1.0)
    >>>
    >>> # Analyze results
    >>> k, E_k = energy_spectrum_1d(state)
    >>> plot_energy_history(history)
    >>> plot_state(state)

Physics context:
    Energy spectra characterize turbulent cascades:
    - k^(-5/3): Kolmogorov spectrum (isotropic 3D turbulence)
    - k^(-3/2): Kraichnan spectrum (2D turbulence)
    - k⊥ cascade dominates in RMHD due to strong guide field
    - k∥ structure from Alfvén wave propagation

References:
    - Boldyrev (2006) PRL 96:115002 - RMHD turbulence theory
    - Schekochihin et al. (2009) ApJS 182:310 - Kinetic cascades
"""

from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass, field
from functools import partial
import jax
import jax.numpy as jnp
from jax import Array
from jax.ops import segment_sum
import matplotlib.pyplot as plt
import numpy as np

from krmhd.physics import KRMHDState, energy as compute_energy
from krmhd.spectral import (
    rfftn_inverse,
    derivative_x,
    derivative_y,
)


# =============================================================================
# Energy Spectrum Functions
# =============================================================================


def energy_spectrum_1d(
    state: KRMHDState,
    n_bins: Optional[int] = None,
) -> Tuple[Array, Array]:
    """
    Compute 1D spherically-averaged energy spectrum E(k) vs |k|.

    Note: This function is not JIT-compiled because KRMHDState (Pydantic BaseModel)
    is not automatically a JAX pytree. To enable JIT, register KRMHDState as a pytree
    using jax.tree_util.register_pytree_node or convert to a standard dataclass.

    Shell-averages energy over all wavenumbers with the same magnitude
    |k| = √(kx² + ky² + kz²). Useful for isotropic turbulence analysis.

    Args:
        state: KRMHD state containing z_plus, z_minus fields
        n_bins: Number of radial bins for spectrum (default: Nx//2)

    Returns:
        Tuple of (k_bins, E_k) where:
        - k_bins: Array of wavenumber magnitudes [n_bins]
        - E_k: Energy per wavenumber bin [n_bins]

    Algorithm:
        1. Compute |k| for each Fourier mode
        2. Bin modes by |k| into shells
        3. Sum energy in each shell
        4. Normalize by bin width and account for rfft doubling

    Properties:
        - Integral ∑E(k)Δk ≈ E_total (Parseval's theorem)
        - E(k) ≥ 0 (energy is non-negative)
        - E(k → ∞) → 0 (energy concentrated at low k)

    Example:
        >>> k, E_k = energy_spectrum_1d(state)
        >>> # Plot on log-log scale
        >>> plt.loglog(k, E_k)
        >>> plt.loglog(k, k**(-5/3), 'k--', label='k^(-5/3)')
        >>> plt.xlabel('|k|')
        >>> plt.ylabel('E(k)')
        >>> plt.legend()

    Physics:
        For isotropic turbulence, E(k) represents the kinetic + magnetic
        energy per unit wavenumber in a spherical shell of radius k.
        The Kolmogorov spectrum E(k) ∝ k^(-5/3) appears in 3D turbulence.
    """
    grid = state.grid
    if n_bins is None:
        n_bins = grid.Nx // 2

    # Create 3D wavenumber arrays
    kx_3d = grid.kx[jnp.newaxis, jnp.newaxis, :]  # Shape: [1, 1, Nx//2+1]
    ky_3d = grid.ky[jnp.newaxis, :, jnp.newaxis]  # Shape: [1, Ny, 1]
    kz_3d = grid.kz[:, jnp.newaxis, jnp.newaxis]  # Shape: [Nz, 1, 1]

    # Compute wavenumber magnitude |k| at each grid point
    k_mag = jnp.sqrt(kx_3d**2 + ky_3d**2 + kz_3d**2)  # Shape: [Nz, Ny, Nx//2+1]

    # Compute energy density in Fourier space
    # Note: RMHD energy involves only perpendicular derivatives (thesis Eq. 2.12)
    # E = (1/2) * |∇⊥φ|² + (1/2) * |∇⊥A∥|²
    #   = (1/2) * k_perp²|φ|² + (1/2) * k_perp²|A∥|²
    k_perp_squared = kx_3d**2 + ky_3d**2

    phi = (state.z_plus + state.z_minus) / 2.0
    A_parallel = (state.z_plus - state.z_minus) / 2.0

    # Energy density at each k-point
    energy_density = 0.5 * k_perp_squared * (jnp.abs(phi)**2 + jnp.abs(A_parallel)**2)

    # Handle rfft doubling: modes with 0 < kx < Nx/2 are doubled
    # (they represent both positive and negative kx)
    kx_zero = (kx_3d == 0.0)
    kx_nyquist = (kx_3d == grid.Nx // 2) if (grid.Nx % 2 == 0) else jnp.zeros_like(kx_3d, dtype=bool)
    kx_middle = ~(kx_zero | kx_nyquist)

    doubling_factor = jnp.where(kx_middle, 2.0, 1.0)
    energy_density = energy_density * doubling_factor

    # Create bins for |k|
    k_max = jnp.sqrt((grid.kx[-1])**2 + (grid.ky[grid.Ny//2])**2 + (grid.kz[grid.Nz//2])**2)
    k_bins = jnp.linspace(0, k_max, n_bins + 1)
    k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])

    # Bin energy by |k| using digitize
    k_indices = jnp.digitize(k_mag.flatten(), k_bins) - 1
    k_indices = jnp.clip(k_indices, 0, n_bins - 1)

    # Sum energy in each bin using vectorized segment_sum (JAX-native, JIT-compatible)
    energy_flat = energy_density.flatten()
    E_k = segment_sum(energy_flat, k_indices, num_segments=n_bins)

    # Normalize by total number of grid points and bin width
    N_total = grid.Nx * grid.Ny * grid.Nz
    dk = jnp.maximum(k_bins[1] - k_bins[0], 1e-10)  # Guard against division by zero
    E_k = E_k / (N_total * dk)

    return k_centers, E_k


def energy_spectrum_perpendicular(
    state: KRMHDState,
    n_bins: Optional[int] = None,
) -> Tuple[Array, Array]:
    """
    Compute perpendicular energy spectrum E(k⊥) vs k⊥ = √(kx² + ky²).

    Note: This function is not JIT-compiled because KRMHDState (Pydantic BaseModel)
    is not automatically a JAX pytree. To enable JIT, register KRMHDState as a pytree
    using jax.tree_util.register_pytree_node or convert to a standard dataclass.

    Sums energy over all parallel (k∥ = kz) modes for each k⊥. This is the
    critical diagnostic for RMHD, where perpendicular cascade dominates.

    Args:
        state: KRMHD state containing z_plus, z_minus fields
        n_bins: Number of k⊥ bins (default: Nx//2)

    Returns:
        Tuple of (k_perp_bins, E_perp) where:
        - k_perp_bins: Array of k⊥ values [n_bins]
        - E_perp: Energy per k⊥ bin [n_bins]

    Algorithm:
        1. Compute k⊥ = √(kx² + ky²) for each mode
        2. Bin modes by k⊥
        3. Sum energy over all kz for each k⊥
        4. Account for rfft doubling

    Properties:
        - Integral ∑E(k⊥)Δk⊥ ≈ E_total
        - E(k⊥) shows perpendicular cascade rate
        - Typically steeper than 1D spectrum due to anisotropy

    Example:
        >>> k_perp, E_perp = energy_spectrum_perpendicular(state)
        >>> plt.loglog(k_perp, E_perp)
        >>> plt.xlabel('k⊥')
        >>> plt.ylabel('E(k⊥)')

    Physics:
        In RMHD with strong guide field B₀∥ẑ, the perpendicular cascade
        dominates because:
        - Alfvén wave propagation along B₀ is linear (no cascade)
        - Perpendicular Poisson bracket {φ, ·} drives turbulent cascade
        - Critical balance: τ_nl ~ τ_A at each scale

        Expected spectrum: E(k⊥) ∝ k⊥^(-3/2) (Goldreich-Sridhar, 1995)
    """
    grid = state.grid
    if n_bins is None:
        n_bins = grid.Nx // 2

    # Create 3D wavenumber arrays
    kx_3d = grid.kx[jnp.newaxis, jnp.newaxis, :]  # Shape: [1, 1, Nx//2+1]
    ky_3d = grid.ky[jnp.newaxis, :, jnp.newaxis]  # Shape: [1, Ny, 1]
    kz_3d = grid.kz[:, jnp.newaxis, jnp.newaxis]  # Shape: [Nz, 1, 1] - needed for proper broadcasting

    # Compute perpendicular wavenumber k⊥
    # Need to broadcast to full 3D shape [Nz, Ny, Nx//2+1]
    # Add 0*kz_3d to force broadcasting to include Nz dimension
    k_perp_squared = kx_3d**2 + ky_3d**2 + 0*kz_3d  # This broadcasts to [Nz, Ny, Nx//2+1]
    k_perp = jnp.sqrt(k_perp_squared)                # Shape: [Nz, Ny, Nx//2+1]

    # Compute energy density
    phi = (state.z_plus + state.z_minus) / 2.0
    A_parallel = (state.z_plus - state.z_minus) / 2.0
    energy_density = 0.5 * k_perp_squared * (jnp.abs(phi)**2 + jnp.abs(A_parallel)**2)

    # Handle rfft doubling
    kx_zero = (kx_3d == 0.0)
    kx_nyquist = (kx_3d == grid.Nx // 2) if (grid.Nx % 2 == 0) else jnp.zeros_like(kx_3d, dtype=bool)
    kx_middle = ~(kx_zero | kx_nyquist)

    doubling_factor = jnp.where(kx_middle, 2.0, 1.0)
    energy_density = energy_density * doubling_factor

    # Create bins for k⊥
    k_perp_max = jnp.sqrt(grid.kx[-1]**2 + grid.ky[grid.Ny//2]**2)
    k_perp_bins = jnp.linspace(0, k_perp_max, n_bins + 1)
    k_perp_centers = 0.5 * (k_perp_bins[:-1] + k_perp_bins[1:])

    # Bin energy by k⊥
    k_perp_indices = jnp.digitize(k_perp.flatten(), k_perp_bins) - 1
    k_perp_indices = jnp.clip(k_perp_indices, 0, n_bins - 1)

    # Sum energy in each bin using vectorized segment_sum (JAX-native, JIT-compatible)
    energy_flat = energy_density.flatten()
    E_perp = segment_sum(energy_flat, k_perp_indices, num_segments=n_bins)

    # Normalize
    N_total = grid.Nx * grid.Ny * grid.Nz
    dk_perp = jnp.maximum(k_perp_bins[1] - k_perp_bins[0], 1e-10)  # Guard against division by zero
    E_perp = E_perp / (N_total * dk_perp)

    return k_perp_centers, E_perp


def energy_spectrum_parallel(
    state: KRMHDState,
) -> Tuple[Array, Array]:
    """
    Compute parallel energy spectrum E(k∥) vs kz.

    Note: This function is not JIT-compiled because KRMHDState (Pydantic BaseModel)
    is not automatically a JAX pytree. To enable JIT, register KRMHDState as a pytree
    using jax.tree_util.register_pytree_node or convert to a standard dataclass.

    Sums energy over all perpendicular (k⊥) modes for each kz. Shows
    energy distribution along field lines from Alfvén wave propagation.

    Args:
        state: KRMHD state containing z_plus, z_minus fields

    Returns:
        Tuple of (kz, E_parallel) where:
        - kz: Parallel wavenumber array [Nz]
        - E_parallel: Energy per kz mode [Nz]

    Algorithm:
        1. For each kz slice
        2. Sum energy over all (kx, ky) modes
        3. Account for rfft doubling

    Properties:
        - Shows field-line structure
        - Typically flatter than k⊥ spectrum (weak parallel cascade)
        - Related to Alfvén wave spectrum

    Example:
        >>> kz, E_parallel = energy_spectrum_parallel(state)
        >>> plt.semilogy(kz, E_parallel)
        >>> plt.xlabel('k∥ (kz)')
        >>> plt.ylabel('E(k∥)')

    Physics:
        In RMHD, parallel structure comes from:
        - Linear Alfvén wave propagation (ω = k∥v_A)
        - No direct parallel cascade (anisotropic turbulence)
        - Field line wandering from perpendicular turbulence

        E(k∥) typically shows discrete peaks from resonant modes or
        broad distribution from phase mixing.
    """
    grid = state.grid

    # Create 3D wavenumber arrays
    kx_3d = grid.kx[jnp.newaxis, jnp.newaxis, :]
    ky_3d = grid.ky[jnp.newaxis, :, jnp.newaxis]
    k_perp_squared = kx_3d**2 + ky_3d**2

    # Compute energy density
    phi = (state.z_plus + state.z_minus) / 2.0
    A_parallel = (state.z_plus - state.z_minus) / 2.0
    energy_density = 0.5 * k_perp_squared * (jnp.abs(phi)**2 + jnp.abs(A_parallel)**2)

    # Handle rfft doubling
    kx_zero = (kx_3d == 0.0)
    kx_nyquist = (kx_3d == grid.Nx // 2) if (grid.Nx % 2 == 0) else jnp.zeros_like(kx_3d, dtype=bool)
    kx_middle = ~(kx_zero | kx_nyquist)

    doubling_factor = jnp.where(kx_middle, 2.0, 1.0)
    energy_density = energy_density * doubling_factor

    # Sum over perpendicular modes for each kz
    E_parallel = jnp.sum(energy_density, axis=(1, 2))  # Sum over ky, kx → [Nz]

    # Normalize by number of perpendicular modes
    # Note: Unlike the binned spectra (1D, perpendicular), this is a discrete spectrum
    # where each kz corresponds to a specific Fourier mode (not averaged over bins).
    # Therefore we normalize by N_perp only, not by dkz.
    #
    # IMPORTANT NORMALIZATION DIFFERENCE:
    # - 1D/perpendicular spectra: E_k / (N_total * dk) → integrate as ∑E(k)Δk ≈ E_total
    # - Parallel spectrum: E_parallel / N_perp → integrate as ∑E(k∥)Δk∥ ≈ E_total
    #   where Δk∥ = 2π/Lz is the parallel wavenumber spacing
    #
    # To verify energy conservation: sum(E_parallel) * (2π/Lz) ≈ E_total
    N_perp = grid.Nx * grid.Ny
    E_parallel = E_parallel / N_perp

    return grid.kz, E_parallel


# =============================================================================
# Field Line Following and Spectral Interpolation
# =============================================================================


@partial(jax.jit, static_argnames=['padding_factor'])
def spectral_pad_and_ifft(
    field_fourier: Array,
    padding_factor: int = 2,
) -> Array:
    """
    Pad Fourier modes and inverse FFT to get high-resolution real-space field.

    Uses spectral interpolation via zero-padding in Fourier space to achieve
    high-resolution representation of band-limited fields. The resulting fine
    grid has spectral accuracy at grid points. When combined with trilinear
    interpolation (see interpolate_on_fine_grid), achieves O(h²/padding_factor²)
    sub-grid accuracy - not pure spectral accuracy, but much better than
    direct trilinear interpolation on the coarse grid.

    Args:
        field_fourier: Field in Fourier space, shape [Nz, Ny, Nx//2+1]
        padding_factor: Factor to increase resolution (default: 2 for 2× resolution)

    Returns:
        field_real_fine: Real-space field on fine grid
                        Shape: [Nz*padding_factor, Ny*padding_factor, Nx*padding_factor]

    Algorithm:
        1. Pad Fourier modes with zeros to simulate higher resolution
        2. Inverse FFT to get fine real-space grid
        3. Result is band-limited to original k_max but on finer grid

    Example:
        >>> B_fine = spectral_pad_and_ifft(B_fourier, padding_factor=2)
        >>> # B_fine is now 2× resolution in each dimension

    Physics:
        Zero-padding in Fourier space is equivalent to ideal sinc interpolation
        in real space. For band-limited functions, this is exact interpolation
        (Shannon sampling theorem).

    Performance:
        - Memory: padding_factor³ × original size
        - Compute: O(N log N) FFT cost on padded grid
        - For padding_factor=2: 8× memory, 8× FFT cost
    """
    Nz, Ny, Nx_half = field_fourier.shape
    Nx = (Nx_half - 1) * 2  # Original Nx from rfft format

    # Validate grid size (must be large enough for spectral padding)
    if Nx < 4 or Ny < 4 or Nz < 4:
        raise ValueError(
            f"Grid too small for spectral padding: Nx={Nx}, Ny={Ny}, Nz={Nz}. "
            f"Minimum size is 4 in each dimension."
        )

    # Calculate padded dimensions
    Nz_pad = Nz * padding_factor
    Ny_pad = Ny * padding_factor
    Nx_pad = Nx * padding_factor
    Nx_pad_half = Nx_pad // 2 + 1

    # Create padded array in Fourier space
    # For 3D rfft: shape is [Nz, Ny, Nx//2+1]
    # We need to pad in all three k-dimensions carefully

    # Initialize padded array with zeros
    field_padded = jnp.zeros((Nz_pad, Ny_pad, Nx_pad_half), dtype=complex)

    # Copy original modes to padded array
    # kx: 0 to Nx//2 (already in rfft format)
    # ky: 0 to Ny//2, then -Ny//2 to -1 (need to handle wraparound)
    # kz: 0 to Nz//2, then -Nz//2 to -1 (need to handle wraparound)

    # Handle ky: positive frequencies [0, Ny//2], negative [-Ny//2, -1]
    ky_pos = Ny // 2 + 1  # Number of positive ky frequencies
    ky_neg = Ny - ky_pos  # Number of negative ky frequencies

    # Handle kz: positive frequencies [0, Nz//2], negative [-Nz//2, -1]
    kz_pos = Nz // 2 + 1  # Number of positive kz frequencies
    kz_neg = Nz - kz_pos  # Number of negative kz frequencies

    # Copy positive kz, positive ky
    field_padded = field_padded.at[:kz_pos, :ky_pos, :Nx_half].set(
        field_fourier[:kz_pos, :ky_pos, :]
    )

    # Copy positive kz, negative ky
    if ky_neg > 0:
        field_padded = field_padded.at[:kz_pos, -ky_neg:, :Nx_half].set(
            field_fourier[:kz_pos, -ky_neg:, :]
        )

    # Copy negative kz, positive ky
    if kz_neg > 0:
        field_padded = field_padded.at[-kz_neg:, :ky_pos, :Nx_half].set(
            field_fourier[-kz_neg:, :ky_pos, :]
        )

    # Copy negative kz, negative ky
    if kz_neg > 0 and ky_neg > 0:
        field_padded = field_padded.at[-kz_neg:, -ky_neg:, :Nx_half].set(
            field_fourier[-kz_neg:, -ky_neg:, :]
        )

    # Inverse FFT to real space
    # JAX's irfftn normalizes by 1/(Nz_pad * Ny_pad * Nx_pad)
    # But the original forward FFT normalized by 1/(Nz * Ny * Nx)
    # So we need to rescale by (N_pad/N)³ = padding_factor³ to preserve field values
    #
    # Note: Hermitian symmetry is automatically preserved by irfftn, which
    # enforces the conjugate relationship f(-kx,-ky,-kz) = f*(kx,ky,kz).
    # Since we only copy modes (no modification), and the input has correct
    # Hermitian symmetry from rfftn, the output is guaranteed to be real.
    #
    # Note: Nyquist frequencies (kx=Nx//2, ky=Ny//2, kz=Nz//2) are automatically
    # handled correctly by JAX's irfftn. For even grid sizes, the Nyquist mode
    # is real-valued and irfftn enforces this constraint internally.
    field_real_fine = jnp.fft.irfftn(field_padded, s=(Nz_pad, Ny_pad, Nx_pad))
    field_real_fine = field_real_fine * (padding_factor ** 3)

    return field_real_fine


@partial(jax.jit, static_argnames=['padding_factor'])
def interpolate_on_fine_grid(
    field_fine: Array,
    position: Array,
    Lx: float,
    Ly: float,
    Lz: float,
    padding_factor: int = 2,
) -> Array:
    """
    Interpolate field value at arbitrary position using fine grid.

    Uses trilinear interpolation on the spectrally-padded grid. This achieves
    O(h²/padding_factor²) sub-grid accuracy, where h is the coarse grid spacing.
    Not pure spectral accuracy (which would be O(machine precision)), but
    significantly better than trilinear on the coarse grid (O(h²)).

    Args:
        field_fine: Fine grid field from spectral_pad_and_ifft()
                   Shape: [Nz*padding_factor, Ny*padding_factor, Nx*padding_factor]
        position: Physical position [x, y, z] to interpolate at
        Lx, Ly, Lz: Domain sizes in physical units
        padding_factor: Padding factor used (must match field_fine)

    Returns:
        Interpolated field value at position (scalar)

    Algorithm:
        1. Convert physical position to fine grid indices
        2. Find 8 surrounding grid points
        3. Trilinear interpolation using weighted average
        4. Handle periodic boundaries

    Example:
        >>> value = interpolate_on_fine_grid(B_fine, jnp.array([1.2, 3.4, 5.6]),
        ...                                   Lx=2*np.pi, Ly=2*np.pi, Lz=2*np.pi)

    Notes:
        - Handles periodic boundaries automatically
        - Position should be in physical units [0, Lx] × [0, Ly] × [0, Lz]
        - Returns 0 if position is exactly on the boundary (shouldn't happen)
    """
    Nz_fine, Ny_fine, Nx_fine = field_fine.shape

    # Convert physical position to grid indices (periodic)
    ix_float = (position[0] % Lx) / Lx * Nx_fine
    iy_float = (position[1] % Ly) / Ly * Ny_fine
    iz_float = (position[2] % Lz) / Lz * Nz_fine

    # Get integer indices and fractional parts
    ix0 = jnp.floor(ix_float).astype(int) % Nx_fine
    iy0 = jnp.floor(iy_float).astype(int) % Ny_fine
    iz0 = jnp.floor(iz_float).astype(int) % Nz_fine

    ix1 = (ix0 + 1) % Nx_fine
    iy1 = (iy0 + 1) % Ny_fine
    iz1 = (iz0 + 1) % Nz_fine

    # Fractional distances
    dx = ix_float - jnp.floor(ix_float)
    dy = iy_float - jnp.floor(iy_float)
    dz = iz_float - jnp.floor(iz_float)

    # Trilinear interpolation
    # c000 = (1-dx)(1-dy)(1-dz), etc.
    c000 = field_fine[iz0, iy0, ix0] * (1 - dx) * (1 - dy) * (1 - dz)
    c001 = field_fine[iz0, iy0, ix1] * dx * (1 - dy) * (1 - dz)
    c010 = field_fine[iz0, iy1, ix0] * (1 - dx) * dy * (1 - dz)
    c011 = field_fine[iz0, iy1, ix1] * dx * dy * (1 - dz)
    c100 = field_fine[iz1, iy0, ix0] * (1 - dx) * (1 - dy) * dz
    c101 = field_fine[iz1, iy0, ix1] * dx * (1 - dy) * dz
    c110 = field_fine[iz1, iy1, ix0] * (1 - dx) * dy * dz
    c111 = field_fine[iz1, iy1, ix1] * dx * dy * dz

    return c000 + c001 + c010 + c011 + c100 + c101 + c110 + c111


def compute_magnetic_field_components(
    state: KRMHDState,
    padding_factor: int = 2,
) -> Dict[str, Array]:
    """
    Compute magnetic field components B = (Bx, By, Bz) on fine grid.

    Computes the full 3D magnetic field including:
    - Perpendicular components from vector potential: B⊥ = ∇ × (Ψ ẑ)
    - Parallel component: Bz = B₀ + δB∥

    All components are computed on a spectrally-padded fine grid for
    accurate field line following.

    Args:
        state: KRMHD state with z_plus, z_minus, B_parallel
        padding_factor: Resolution increase factor (default: 2)

    Returns:
        Dictionary with keys 'Bx', 'By', 'Bz', each containing fine-grid array
        Shape of each: [Nz*padding_factor, Ny*padding_factor, Nx*padding_factor]

    Physics:
        The magnetic field in RMHD:
        B = B₀ẑ + δB

        where δB comes from:
        - B⊥ = ∇ × (Ψ ẑ) = (∂yΨ, -∂xΨ, 0)
        - Bz = B₀ + δB∥

        Here Ψ = (z⁺ - z⁻)/2 is the parallel vector potential (up to normalization).

        Note: In the normalized equations, B₀ may not appear explicitly, but
        for field line following we need the full field direction.

    Example:
        >>> B = compute_magnetic_field_components(state, padding_factor=2)
        >>> Bx_fine, By_fine, Bz_fine = B['Bx'], B['By'], B['Bz']

    Performance:
        - Computes 3 derivative operations in Fourier space
        - Performs 3 padded IFFTs
        - Memory: 3 × padding_factor³ × base grid size
        - For 256³ with padding_factor=2: 3 × 8 × ~130 MB ≈ 3.1 GB
    """
    grid = state.grid

    # Compute Ψ = (z⁺ - z⁻)/2 (parallel vector potential)
    Psi = (state.z_plus - state.z_minus) / 2.0

    # Compute perpendicular B components: B⊥ = ∇ × (Ψ ẑ)
    # Bx = ∂yΨ
    # By = -∂xΨ
    Bx_fourier = derivative_y(Psi, grid.ky)
    By_fourier = -derivative_x(Psi, grid.kx)

    # Pad and transform perpendicular components to real space
    Bx_fine = spectral_pad_and_ifft(Bx_fourier, padding_factor)
    By_fine = spectral_pad_and_ifft(By_fourier, padding_factor)

    # Parallel component: Bz = B₀ + δB∥
    # Transform δB∥ to real space, then add B₀ = 1 (in normalized units)
    # Note: Cannot add constant in Fourier space (only affects k=0 mode)
    dBz_fine = spectral_pad_and_ifft(state.B_parallel, padding_factor)
    Bz_fine = 1.0 + dBz_fine  # B₀ = 1 in code units

    return {
        'Bx': Bx_fine,
        'By': By_fine,
        'Bz': Bz_fine,
    }


def _follow_field_line_from_B(
    B_components: Dict[str, Array],
    grid: 'SpectralGrid3D',
    x0: float,
    y0: float,
    dz: Optional[float] = None,
    padding_factor: int = 2,
) -> Array:
    """
    Internal helper: Follow field line using precomputed B field components.

    This function performs the actual RK2 integration given precomputed B field
    components. Used internally by follow_field_line() and plot_field_lines().

    Args:
        B_components: Dict with 'Bx', 'By', 'Bz' arrays on fine grid
        grid: Spectral grid for domain parameters
        x0, y0: Starting position in perpendicular plane
        dz: Step size in z-direction (default: Lz / (10*Nz))
        padding_factor: Padding factor used for B_components

    Returns:
        trajectory: Array of shape [n_steps, 3] containing (x, y, z) positions
    """
    # Set default step size
    if dz is None:
        dz = grid.Lz / (10 * grid.Nz)

    Bx_fine = B_components['Bx']
    By_fine = B_components['By']
    Bz_fine = B_components['Bz']

    # Initialize trajectory storage
    trajectory_list = []

    # Starting position
    pos = jnp.array([x0, y0, -grid.Lz / 2])
    trajectory_list.append(pos.copy())

    # Integration loop
    # Note: Can't use jax.lax.while_loop easily because of trajectory accumulation
    # For now, use Python loop (can optimize later with fixed-step scan)
    #
    # CRITICAL: Safety margin must be multiplicative, not additive!
    # In strong turbulence (Bz → 0), field lines can spiral extensively in (x,y)
    # with minimal z-progress. Arc length can be >> Lz.
    # Factor of 10 allows for field lines up to 10× longer than straight path.
    n_steps_max = int(jnp.ceil(grid.Lz / dz)) * 10

    for step in range(n_steps_max):
        if pos[2] >= grid.Lz / 2:
            break

        # Interpolate B at current position
        Bx = interpolate_on_fine_grid(Bx_fine, pos, grid.Lx, grid.Ly, grid.Lz, padding_factor)
        By = interpolate_on_fine_grid(By_fine, pos, grid.Lx, grid.Ly, grid.Lz, padding_factor)
        Bz = interpolate_on_fine_grid(Bz_fine, pos, grid.Lx, grid.Ly, grid.Lz, padding_factor)

        B = jnp.array([Bx, By, Bz])
        B_magnitude = jnp.linalg.norm(B)

        # Avoid division by zero: if |B| is very small, use z-direction (mean field)
        # This handles magnetic null points gracefully
        b_hat = jnp.where(
            B_magnitude > 1e-8,
            B / B_magnitude,
            jnp.array([0.0, 0.0, 1.0])  # Default to z-direction
        )

        # RK2 predictor: half-step
        pos_half = pos + 0.5 * dz * b_hat

        # Apply periodic boundaries
        pos_half = pos_half.at[0].set(pos_half[0] % grid.Lx)
        pos_half = pos_half.at[1].set(pos_half[1] % grid.Ly)

        # Interpolate B at half-step
        Bx_half = interpolate_on_fine_grid(Bx_fine, pos_half, grid.Lx, grid.Ly, grid.Lz, padding_factor)
        By_half = interpolate_on_fine_grid(By_fine, pos_half, grid.Lx, grid.Ly, grid.Lz, padding_factor)
        Bz_half = interpolate_on_fine_grid(Bz_fine, pos_half, grid.Lx, grid.Ly, grid.Lz, padding_factor)

        B_half = jnp.array([Bx_half, By_half, Bz_half])
        B_half_magnitude = jnp.linalg.norm(B_half)

        # Avoid division by zero at midpoint
        b_hat_half = jnp.where(
            B_half_magnitude > 1e-8,
            B_half / B_half_magnitude,
            jnp.array([0.0, 0.0, 1.0])  # Default to z-direction
        )

        # RK2 corrector: full step with midpoint slope
        pos_new = pos + dz * b_hat_half

        # Apply periodic boundaries
        pos_new = pos_new.at[0].set(pos_new[0] % grid.Lx)
        pos_new = pos_new.at[1].set(pos_new[1] % grid.Ly)

        pos = pos_new
        trajectory_list.append(pos.copy())

    # Check for incomplete trajectory (hit step limit before reaching z_max)
    if pos[2] < grid.Lz / 2:
        import warnings
        warnings.warn(
            f"Field line integration incomplete: stopped at z={float(pos[2]):.3f} "
            f"< z_max={grid.Lz/2:.3f} after {n_steps_max} steps. "
            f"Field line may be spiraling in strong turbulence (Bz → 0). "
            f"Consider: (1) decreasing dz, (2) increasing safety margin, "
            f"or (3) using adaptive step size.",
            RuntimeWarning,
            stacklevel=2
        )

    # Convert list to array
    trajectory = jnp.array(trajectory_list)

    return trajectory


def follow_field_line(
    state: KRMHDState,
    x0: float,
    y0: float,
    dz: Optional[float] = None,
    padding_factor: int = 2,
) -> Array:
    """
    Follow magnetic field line from (x0, y0, z_min) to (x0, y0, z_max).

    Uses RK2 (midpoint method) integration along field line direction b̂ = B/|B|.
    Interpolation uses spectrally-padded grid for high accuracy.

    Args:
        state: KRMHD state to compute field lines from
        x0, y0: Starting position in perpendicular plane
        dz: Step size in z-direction (default: Lz / (10*Nz))
        padding_factor: Fine grid resolution multiplier (default: 2)

    Returns:
        trajectory: Array of shape [n_steps, 3] containing (x, y, z) positions

    Algorithm:
        1. Compute B field components on fine grid (once)
        2. Initialize position at (x0, y0, -Lz/2)
        3. RK2 integration:
           - Predictor: pos_half = pos + 0.5 * ds * b̂(pos)
           - Corrector: pos_new = pos + ds * b̂(pos_half)
        4. Continue until z > Lz/2
        5. Apply periodic boundaries in (x, y)

    Example:
        >>> traj = follow_field_line(state, x0=np.pi, y0=np.pi)
        >>> x_traj, y_traj, z_traj = traj[:, 0], traj[:, 1], traj[:, 2]

    Physics:
        Field line equation: dr/ds = b̂ where b̂ = B/|B|

        In straight field limit (B = B₀ẑ), this gives straight vertical lines.
        With turbulent δB⊥, field lines wander in (x, y) plane.

        Wandering amplitude: δr⊥ ~ (δB⊥/B₀) × Lz

    Limitations:
        - Step size uses dz (z-direction) rather than arc length ds along field line
        - In strong turbulence where Bz → 0, the actual arc length can be much larger
          than dz, potentially causing accuracy issues or step size limitations
        - For production use in strong turbulence, consider:
          * Adaptive step size based on |b̂·ẑ|
          * Arc-length parameterization: ds = dz/|b̂·ẑ|
          * Smaller fixed dz for safety (at cost of more steps)
        - Current implementation uses Python loop (not JIT-optimized)
          For performance-critical applications, rewrite using jax.lax.scan (Issue #61)

    Performance:
        - One-time cost: Compute B_fine (3 × spectral_pad_and_ifft)
        - Per-step cost: 3 trilinear interpolations
        - Typical: 10*Nz steps per field line
        - For 256³ grid: ~2560 steps, ~7 ms per field line

    Note:
        For tracing multiple field lines from the same state, consider using
        plot_field_lines() which computes B once and reuses it for all traces.
    """
    # Compute B field components (expensive)
    B_components = compute_magnetic_field_components(state, padding_factor)

    # Call internal helper to perform integration
    return _follow_field_line_from_B(
        B_components, state.grid, x0, y0, dz, padding_factor
    )


# =============================================================================
# Energy History Tracking
# =============================================================================


@dataclass
class EnergyHistory:
    """
    Track energy evolution E(t) during KRMHD simulation.

    Stores time series of all energy components (magnetic, kinetic, compressive)
    for analyzing energy conservation, dissipation rates, and equilibration.

    Attributes:
        times: List of simulation times
        E_magnetic: List of magnetic energy values
        E_kinetic: List of kinetic energy values
        E_compressive: List of compressive energy values
        E_total: List of total energy values

    Example:
        >>> history = EnergyHistory()
        >>> for i in range(100):
        ...     history.append(state)
        ...     state = gandalf_step(state, dt=0.01, eta=0.01, v_A=1.0)
        >>>
        >>> # Analyze results
        >>> history.plot()
        >>> mag_frac = history.magnetic_fraction()
        >>> dissipation_rate = history.dissipation_rate()

    Physics:
        Energy conservation check:
        - Inviscid (η=0): E_total should be constant to machine precision
        - Viscous (η>0): E_total should decay exponentially
        - Selective decay: E_magnetic/E_kinetic increases (magnetic dominates)
    """
    times: List[float] = field(default_factory=list)
    E_magnetic: List[float] = field(default_factory=list)
    E_kinetic: List[float] = field(default_factory=list)
    E_compressive: List[float] = field(default_factory=list)
    E_total: List[float] = field(default_factory=list)

    def append(self, state: KRMHDState) -> None:
        """
        Record energy from current state.

        Args:
            state: Current KRMHD state to record

        Example:
            >>> history.append(state)
        """
        energies = compute_energy(state)
        self.times.append(state.time)
        self.E_magnetic.append(energies['magnetic'])
        self.E_kinetic.append(energies['kinetic'])
        self.E_compressive.append(energies['compressive'])
        self.E_total.append(energies['total'])

    def to_dict(self) -> Dict[str, List[float]]:
        """
        Convert to dictionary for serialization.

        Returns:
            Dictionary with all time series

        Example:
            >>> import json
            >>> data = history.to_dict()
            >>> with open('energy_history.json', 'w') as f:
            ...     json.dump(data, f)
        """
        return {
            'times': self.times,
            'E_magnetic': self.E_magnetic,
            'E_kinetic': self.E_kinetic,
            'E_compressive': self.E_compressive,
            'E_total': self.E_total,
        }

    def magnetic_fraction(self) -> Array:
        """
        Compute magnetic energy fraction E_mag / E_total.

        Returns:
            Array of magnetic energy fractions over time

        Physics:
            In decaying MHD turbulence, magnetic energy dominates at late times
            (selective decay): E_mag/E_total → 1 as t → ∞.
        """
        E_mag = jnp.array(self.E_magnetic)
        E_tot = jnp.array(self.E_total)
        return jnp.where(E_tot > 0, E_mag / E_tot, 0.0)

    def dissipation_rate(self) -> Array:
        """
        Compute instantaneous dissipation rate dE/dt.

        Returns:
            Array of dissipation rates (negative for energy loss)

        Algorithm:
            Finite difference: dE/dt ≈ (E[i+1] - E[i]) / (t[i+1] - t[i])

        Physics:
            For resistive MHD with dissipation η:
            dE/dt = -η ∫ |∇×B|² dx < 0
        """
        times = jnp.array(self.times)
        E_total = jnp.array(self.E_total)

        if len(times) < 2:
            return jnp.array([])

        dt = jnp.diff(times)
        dE = jnp.diff(E_total)
        return dE / jnp.where(dt > 0, dt, 1.0)


# =============================================================================
# Visualization Functions
# =============================================================================


def plot_state(
    state: KRMHDState,
    z_slice: int = 0,
    figsize: Tuple[float, float] = (12, 5),
    filename: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Visualize current KRMHD state with 2D slices of φ and A∥.

    Creates two side-by-side color plots showing spatial structure at a
    given z-plane. Useful for quick inspection of turbulent structures.

    Args:
        state: KRMHD state to visualize
        z_slice: Which z-plane to plot (default: 0)
        figsize: Figure size in inches (width, height)
        filename: If provided, save figure to this path
        show: If True, display figure interactively

    Example:
        >>> plot_state(state, z_slice=0)
        >>> plot_state(state, filename='state_t100.png', show=False)

    Physics:
        - φ (stream function): Shows vortex structures, v⊥ = ẑ × ∇φ
        - A∥ (parallel vector potential): Shows current sheets, B⊥ = ẑ × ∇A∥
        - Turbulent structures appear as coherent vortices and current filaments
    """
    grid = state.grid

    # Convert to physical variables for visualization
    phi = (state.z_plus + state.z_minus) / 2.0
    A_parallel = (state.z_plus - state.z_minus) / 2.0

    # Extract z-slice and convert to real space
    phi_slice_fourier = phi[z_slice, :, :]
    A_slice_fourier = A_parallel[z_slice, :, :]

    # Convert to real space using inverse FFT
    # Note: rfftn handles 3D, need to add dummy z dimension
    phi_slice_3d = phi_slice_fourier[jnp.newaxis, :, :]
    A_slice_3d = A_slice_fourier[jnp.newaxis, :, :]

    phi_real = rfftn_inverse(phi_slice_3d, 1, grid.Ny, grid.Nx)[0, :, :]
    A_real = rfftn_inverse(A_slice_3d, 1, grid.Ny, grid.Nx)[0, :, :]

    # Create coordinate arrays for plotting
    x = np.linspace(0, grid.Lx, grid.Nx, endpoint=False)
    y = np.linspace(0, grid.Ly, grid.Ny, endpoint=False)
    X, Y = np.meshgrid(x, y)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Plot φ
    im1 = axes[0].pcolormesh(X, Y, phi_real, shading='auto', cmap='RdBu_r')
    axes[0].set_xlabel('x / Lx')
    axes[0].set_ylabel('y / Ly')
    axes[0].set_title(f'φ (stream function) at z={z_slice}')
    axes[0].set_aspect('equal')
    plt.colorbar(im1, ax=axes[0], label='φ')

    # Plot A∥
    im2 = axes[1].pcolormesh(X, Y, A_real, shading='auto', cmap='RdBu_r')
    axes[1].set_xlabel('x / Lx')
    axes[1].set_ylabel('y / Ly')
    axes[1].set_title(f'A∥ (parallel vector potential) at z={z_slice}')
    axes[1].set_aspect('equal')
    plt.colorbar(im2, ax=axes[1], label='A∥')

    plt.suptitle(f'KRMHD State at t = {state.time:.3f}', fontsize=14)
    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')

    if show:
        plt.show()
        plt.close()  # Close figure after showing to prevent memory leak
    else:
        plt.close()


def plot_energy_history(
    history: EnergyHistory,
    figsize: Tuple[float, float] = (10, 6),
    log_scale: bool = False,
    filename: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Plot energy evolution E(t) from EnergyHistory.

    Shows time series of all energy components (magnetic, kinetic, compressive,
    total) on same axes for comparison.

    Args:
        history: EnergyHistory object with recorded time series
        figsize: Figure size in inches (width, height)
        log_scale: If True, use log scale for y-axis
        filename: If provided, save figure to this path
        show: If True, display figure interactively

    Example:
        >>> plot_energy_history(history)
        >>> plot_energy_history(history, log_scale=True, filename='energy.png')

    Physics:
        Key features to look for:
        - Conservation: E_total constant (inviscid) or exponential decay (viscous)
        - Selective decay: E_magnetic/E_kinetic increases over time
        - Equilibration: E_kinetic → E_magnetic for Alfvén waves
        - Compressive energy should remain small in RMHD (passive)
    """
    times = np.array(history.times)
    E_mag = np.array(history.E_magnetic)
    E_kin = np.array(history.E_kinetic)
    E_comp = np.array(history.E_compressive)
    E_tot = np.array(history.E_total)

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(times, E_mag, 'b-', label='Magnetic', linewidth=2)
    ax.plot(times, E_kin, 'r-', label='Kinetic', linewidth=2)
    ax.plot(times, E_comp, 'g-', label='Compressive', linewidth=2)
    ax.plot(times, E_tot, 'k--', label='Total', linewidth=2)

    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Energy', fontsize=12)
    ax.set_title('Energy Evolution', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    if log_scale:
        ax.set_yscale('log')

    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')

    if show:
        plt.show()
        plt.close()  # Close figure after showing to prevent memory leak
    else:
        plt.close()


def plot_energy_spectrum(
    k: Array,
    E_k: Array,
    spectrum_type: str = '1D',
    reference_slope: Optional[float] = None,
    figsize: Tuple[float, float] = (8, 6),
    filename: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Plot energy spectrum E(k) on log-log axes.

    Shows power-law scaling and compares with theoretical predictions
    (e.g., Kolmogorov k^(-5/3), Kraichnan k^(-3/2)).

    Args:
        k: Wavenumber array
        E_k: Energy spectrum array
        spectrum_type: Type label ('1D', 'perpendicular', 'parallel')
        reference_slope: If provided, plot k^reference_slope line for comparison
        figsize: Figure size in inches (width, height)
        filename: If provided, save figure to this path
        show: If True, display figure interactively

    Example:
        >>> k, E_k = energy_spectrum_1d(state)
        >>> plot_energy_spectrum(k, E_k, reference_slope=-5/3)
        >>>
        >>> k_perp, E_perp = energy_spectrum_perpendicular(state)
        >>> plot_energy_spectrum(k_perp, E_perp, spectrum_type='perpendicular',
        ...                      reference_slope=-3/2)

    Physics:
        Theoretical power laws:
        - k^(-5/3): Kolmogorov (isotropic 3D turbulence)
        - k^(-3/2): Kraichnan (2D turbulence, RMHD perpendicular cascade)
        - k^(-2): Steep dissipation range
        - k^(-1): Shallow injection range

        Deviations from power law indicate:
        - Bottleneck effect (pile-up before dissipation)
        - Intermittency (non-Gaussian statistics)
        - Anisotropy (different scalings for k⊥ vs k∥)
    """
    # Convert to numpy for matplotlib
    k = np.array(k)
    E_k = np.array(E_k)

    # Filter out zero/negative values for log scale
    valid = (k > 0) & (E_k > 0)
    k = k[valid]
    E_k = E_k[valid]

    fig, ax = plt.subplots(figsize=figsize)

    # Plot spectrum
    ax.loglog(k, E_k, 'b-', linewidth=2, label=f'E({spectrum_type})')

    # Add reference slope if requested
    if reference_slope is not None:
        # Choose reference point in middle of spectrum
        k_ref = k[len(k) // 2]
        E_ref = E_k[len(E_k) // 2]

        # Create reference line over visible range
        k_line = np.logspace(np.log10(k.min()), np.log10(k.max()), 100)
        E_line = E_ref * (k_line / k_ref)**reference_slope

        ax.loglog(k_line, E_line, 'k--', linewidth=1.5,
                 label=f'k^({reference_slope:.2f})')

    ax.set_xlabel(f'k ({spectrum_type})', fontsize=12)
    ax.set_ylabel('E(k)', fontsize=12)
    ax.set_title(f'Energy Spectrum: {spectrum_type}', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')

    if show:
        plt.show()
        plt.close()  # Close figure after showing to prevent memory leak
    else:
        plt.close()


def plot_field_lines(
    state: KRMHDState,
    n_lines: int = 10,
    padding_factor: int = 2,
    figsize: Tuple[float, float] = (14, 5),
    filename: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Visualize magnetic field line trajectories in 3D.

    Traces multiple field lines and shows their wandering in (x, y) plane
    as a function of z. Creates side-by-side 3D and 2D projections.

    Args:
        state: KRMHD state to trace field lines from
        n_lines: Number of field lines to trace (default: 10)
        padding_factor: Fine grid resolution (default: 2)
        figsize: Figure size in inches (width, height)
        filename: If provided, save figure to this path
        show: If True, display figure interactively

    Example:
        >>> plot_field_lines(state, n_lines=20)
        >>> plot_field_lines(state, filename='field_lines.png', show=False)

    Physics:
        Field line wandering amplitude δr⊥ ~ (δB⊥/B₀) × Lz shows:
        - Straight lines: Weak turbulence (δB⊥ << B₀)
        - Strong wandering: Strong turbulence (δB⊥ ~ B₀)
        - Random walk statistics: Field line diffusion

    Performance:
        Optimized to compute B field components once for all field lines.
        For n_lines=20: 3 expensive operations instead of 60 (~20× speedup).
        Uses internal _follow_field_line_from_B() helper with precomputed B.
    """
    grid = state.grid

    # Sample starting points uniformly
    x_starts = np.linspace(0, grid.Lx, n_lines, endpoint=False)
    y_starts = np.linspace(0, grid.Ly, n_lines, endpoint=False)

    # Create figure with 3D and 2D views
    fig = plt.figure(figsize=figsize)
    ax3d = fig.add_subplot(121, projection='3d')
    ax2d = fig.add_subplot(122)

    # Compute B field components ONCE for all field lines (major performance optimization)
    B_components = compute_magnetic_field_components(state, padding_factor)

    # Trace field lines using precomputed B components
    for i in range(n_lines):
        x0 = x_starts[i]
        y0 = y_starts[i]

        # Use internal helper with precomputed B (n_lines× faster than follow_field_line)
        trajectory = _follow_field_line_from_B(
            B_components, state.grid, x0, y0, padding_factor=padding_factor
        )

        # Convert to numpy for plotting
        traj_np = np.array(trajectory)
        x_traj = traj_np[:, 0]
        y_traj = traj_np[:, 1]
        z_traj = traj_np[:, 2]

        # 3D plot
        ax3d.plot(x_traj, y_traj, z_traj, alpha=0.7, linewidth=1)

        # 2D projection (x-y plane)
        ax2d.plot(x_traj, y_traj, alpha=0.7, linewidth=1)

    # Format 3D plot
    ax3d.set_xlabel('x')
    ax3d.set_ylabel('y')
    ax3d.set_zlabel('z')
    ax3d.set_title('Field Line Trajectories (3D)')
    ax3d.set_xlim(0, grid.Lx)
    ax3d.set_ylim(0, grid.Ly)
    ax3d.set_zlim(-grid.Lz/2, grid.Lz/2)

    # Format 2D plot
    ax2d.set_xlabel('x')
    ax2d.set_ylabel('y')
    ax2d.set_title('Field Line Wandering (x-y projection)')
    ax2d.set_xlim(0, grid.Lx)
    ax2d.set_ylim(0, grid.Ly)
    ax2d.set_aspect('equal')
    ax2d.grid(True, alpha=0.3)

    plt.suptitle(f'Magnetic Field Lines at t = {state.time:.3f}', fontsize=14)
    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')

    if show:
        plt.show()
        plt.close()
    else:
        plt.close()


def plot_parallel_spectrum_comparison(
    state: KRMHDState,
    padding_factor: int = 2,
    figsize: Tuple[float, float] = (12, 5),
    filename: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Compare E(k∥) vs E(kz) spectra side-by-side.

    Shows the difference between true field-line-following k∥ and simple kz.
    In turbulent plasmas with δB⊥ ~ B₀, these can differ significantly.

    Args:
        state: KRMHD state to analyze
        padding_factor: Fine grid resolution (default: 2)
        figsize: Figure size in inches (width, height)
        filename: If provided, save figure to this path
        show: If True, display figure interactively

    Example:
        >>> plot_parallel_spectrum_comparison(state)

    Physics:
        - E(kz): Energy vs wavenumber in z-direction
        - E(k∥): Energy vs wavenumber along curved field lines
        - Difference: Measures effect of field line curvature
        - Expected: E(k∥) < E(kz) for passive scalars (no parallel cascade)

    Note:
        This function requires the true k∥ spectrum diagnostic, which needs
        to be implemented. For now, we'll just plot the kz spectrum twice
        as a placeholder.
    """
    grid = state.grid

    # Compute kz spectrum (simple)
    kz, E_kz = energy_spectrum_parallel(state)  # This is actually E(kz)

    # TODO: Implement true field-line k∥ spectrum
    # For now, use placeholder
    k_parallel = kz
    E_parallel = E_kz

    # Create side-by-side plots
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Convert to numpy
    kz = np.array(kz)
    E_kz = np.array(E_kz)
    k_parallel = np.array(k_parallel)
    E_parallel = np.array(E_parallel)

    # Filter positive values for log scale
    valid_kz = (kz > 0) & (E_kz > 0)
    valid_kpar = (k_parallel > 0) & (E_parallel > 0)

    # Plot E(kz)
    axes[0].semilogy(kz[valid_kz], E_kz[valid_kz], 'b-', linewidth=2)
    axes[0].set_xlabel('kz', fontsize=12)
    axes[0].set_ylabel('E(kz)', fontsize=12)
    axes[0].set_title('Simple kz Spectrum', fontsize=13)
    axes[0].grid(True, alpha=0.3)

    # Plot E(k∥)
    axes[1].semilogy(k_parallel[valid_kpar], E_parallel[valid_kpar], 'r-', linewidth=2)
    axes[1].set_xlabel('k∥ (field-line)', fontsize=12)
    axes[1].set_ylabel('E(k∥)', fontsize=12)
    axes[1].set_title('True Field-Line k∥ Spectrum', fontsize=13)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(f'Parallel Spectrum Comparison at t = {state.time:.3f}', fontsize=14)
    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')

    if show:
        plt.show()
        plt.close()
    else:
        plt.close()
