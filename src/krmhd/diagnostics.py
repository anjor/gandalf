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
import jax
import jax.numpy as jnp
from jax import Array
from jax.ops import segment_sum
import matplotlib.pyplot as plt
import numpy as np

from krmhd.physics import KRMHDState, energy as compute_energy
from krmhd.spectral import rfftn_inverse


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
