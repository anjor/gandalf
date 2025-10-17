"""
Validation utilities for KRMHD implementation.

This module provides infrastructure for validating the kinetic KRMHD implementation
against analytical predictions from linear Vlasov theory.

Main components:
- Analytical theory functions (FDT predictions)
- Forced single-mode simulation runner
- Spectrum comparison and plotting utilities

Physics context:
    The Fluctuation-Dissipation Theorem (FDT) predicts the steady-state Hermite
    moment spectrum |g_m|² when a single k-mode is driven with stochastic forcing.
    This is a critical benchmark for validating the kinetic implementation.

References:
    - Thesis §2.6.1 - FDT for kinetic turbulence
    - Thesis Chapter 3 - Analytical theory and numerical validation
    - Issue #27 - Implementation details
"""

from typing import Dict, Tuple, Any
import numpy as np
import jax
import jax.numpy as jnp

from krmhd import (
    SpectralGrid3D,
    initialize_alfven_wave,
    gandalf_step,
    compute_cfl_timestep,
    force_alfven_modes,
)
from krmhd.diagnostics import (
    hermite_moment_energy,
    EnergyHistory,
)


# ============================================================================
# Constants
# ============================================================================

# Forcing band configuration (relative to target wavenumber)
FORCING_BAND_MIN_ABSOLUTE = 0.1  # Minimum k_min to avoid k=0 mode
FORCING_BAND_LOWER_FACTOR = 0.9  # k_min = k_target * 0.9
FORCING_BAND_UPPER_FACTOR = 1.1  # k_max = k_target * 1.1

# Numerical thresholds
K_PARALLEL_ZERO_THRESHOLD = 1e-6  # Consider k∥ ≈ 0 below this value
COLLISION_FREQ_ZERO_THRESHOLD = 1e-6  # Consider ν ≈ 0 below this value (collisionless limit)
SPECTRUM_NORMALIZATION_THRESHOLD = 1e-15  # Minimum |g_0|² for safe normalization
STEADY_STATE_FLUCTUATION_THRESHOLD = 0.1  # 10% energy fluctuation criterion
COLLISIONLESS_M_CRIT = 1000.0  # Effective m_crit for collisionless limit (ν → 0)


# ============================================================================
# Analytical Theory Functions (Thesis Chapter 3)
# ============================================================================


def analytical_phase_mixing_spectrum(
    m_array: np.ndarray,
    k_parallel: float,
    k_perp: float,
    v_th: float,
    nu: float,
    Lambda: float,
    amplitude: float = 1.0,
) -> np.ndarray:
    """
    Analytical prediction for phase-mixing spectrum (Thesis Eq 3.37).

    In the phase mixing regime (large k∥), particles with different v∥
    dephase due to free streaming. This creates fine-scale structure in
    velocity space, causing energy to cascade to higher Hermite moments.

    With collisions, the cascade is arrested at m ~ m_crit where collisional
    damping balances the phase mixing rate.

    Args:
        m_array: Array of Hermite moment indices [0, 1, ..., M]
        k_parallel: Parallel wavenumber k∥
        k_perp: Perpendicular wavenumber k⊥
        v_th: Thermal velocity
        nu: Collision frequency
        Lambda: Kinetic parameter (1 - 1/Λ factor in g1 coupling)
        amplitude: Overall amplitude normalization

    Returns:
        |g_m|² spectrum vs m (analytical prediction)

    Physics:
        Phase mixing cascade rate: γ_mix ~ k∥ v_th
        Collision damping rate: γ_coll ~ ν m
        Critical moment: m_crit ~ k∥ v_th / ν

        Expected form:
        |g_m|² ~ amplitude * m^(-α) * exp(-m/m_crit)

        where α ≈ 1-2 depends on the forcing and response function.

    Note:
        This is a simplified analytical model. The exact expressions from
        Thesis Eq 3.37 involve Bessel functions and plasma dispersion functions.
        For production validation, these should be implemented from the thesis.
    """
    # Phase mixing scale: critical moment where collisions balance mixing
    # In collisionless limit (nu → 0), cascade extends to high m (set large m_crit)
    if nu < COLLISION_FREQ_ZERO_THRESHOLD:
        m_crit = COLLISIONLESS_M_CRIT  # Effectively infinite for collisionless case
    else:
        m_crit = k_parallel * v_th / nu

    # Power-law exponent (simplified - thesis has exact formula)
    # Typically α ~ 1-2 for phase mixing cascade
    # Note: This is an empirical approximation. Exact value depends on:
    #   - Linear response function (plasma dispersion function)
    #   - Forcing spectrum shape
    #   - Landau resonance structure
    # See Thesis §3.2 for derivation from kinetic response theory
    alpha = 1.5

    # k⊥ dependence (perpendicular structure affects spectrum normalization)
    # Explicit handling for k_parallel ≈ 0 case
    if abs(k_parallel) < K_PARALLEL_ZERO_THRESHOLD:
        # Pure perpendicular mode: phase mixing suppressed
        k_perp_factor = 1.0
    else:
        k_perp_factor = 1.0 / (1.0 + (k_perp / k_parallel)**2)

    # Analytical spectrum: power law × exponential cutoff
    spectrum = amplitude * k_perp_factor * (m_array + 1.0)**(-alpha) * np.exp(-m_array / m_crit)

    # Normalize by m=0 value for relative comparison
    if spectrum[0] < SPECTRUM_NORMALIZATION_THRESHOLD:
        raise ValueError(
            f"Phase mixing spectrum m=0 too small for normalization: {spectrum[0]} "
            f"< {SPECTRUM_NORMALIZATION_THRESHOLD}"
        )
    spectrum = spectrum / spectrum[0]

    return spectrum


def analytical_phase_unmixing_spectrum(
    m_array: np.ndarray,
    k_parallel: float,
    k_perp: float,
    v_th: float,
    nu: float,
    Lambda: float,
    amplitude: float = 1.0,
) -> np.ndarray:
    """
    Analytical prediction for phase-unmixing spectrum (Thesis Eq 3.58).

    In the phase unmixing regime (small k∥, large k⊥), nonlinear perpendicular
    advection can transfer energy back from higher to lower moments. This
    leads to a different spectral shape than pure phase mixing.

    Args:
        m_array: Array of Hermite moment indices [0, 1, ..., M]
        k_parallel: Parallel wavenumber k∥
        k_perp: Perpendicular wavenumber k⊥
        v_th: Thermal velocity
        nu: Collision frequency
        Lambda: Kinetic parameter
        amplitude: Overall amplitude normalization

    Returns:
        |g_m|² spectrum vs m (analytical prediction)

    Physics:
        Phase unmixing is driven by perpendicular nonlinearity.
        The spectrum typically has shallower slope than phase mixing.

    Note:
        Simplified model - thesis has exact expressions with plasma dispersion
        functions. The key difference from phase mixing is the slope and
        k∥/k⊥ dependence.
    """
    # Phase unmixing is weaker than phase mixing
    # In collisionless limit (nu → 0), unmixing is weak (set large m_crit)
    if nu < COLLISION_FREQ_ZERO_THRESHOLD:
        m_crit = COLLISIONLESS_M_CRIT  # Effectively infinite for collisionless case
    else:
        m_crit = k_perp * v_th / nu

    # Shallower power law for phase unmixing
    # Note: This is an empirical approximation. Phase unmixing has weaker
    # velocity-space cascade than phase mixing due to different nonlinear
    # coupling structure. Exact exponent requires solution of kinetic
    # equations with perpendicular advection.
    # See Thesis §3.3 for discussion of phase unmixing regime
    alpha = 0.5

    # k∥ dependence (small k∥ enhances phase unmixing)
    # Explicit handling for k_parallel ≈ 0 case
    if abs(k_parallel) < K_PARALLEL_ZERO_THRESHOLD:
        # Pure perpendicular mode: phase unmixing saturates
        k_ratio_factor = np.sqrt(k_perp * COLLISIONLESS_M_CRIT)
    else:
        k_ratio_factor = np.sqrt(k_perp / k_parallel)

    # Analytical spectrum
    spectrum = amplitude * k_ratio_factor * (m_array + 1.0)**(-alpha) * np.exp(-m_array / m_crit)

    # Normalize
    if spectrum[0] < SPECTRUM_NORMALIZATION_THRESHOLD:
        raise ValueError(
            f"Phase unmixing spectrum m=0 too small for normalization: {spectrum[0]} "
            f"< {SPECTRUM_NORMALIZATION_THRESHOLD}"
        )
    spectrum = spectrum / spectrum[0]

    return spectrum


def analytical_total_spectrum(
    m_array: np.ndarray,
    k_parallel: float,
    k_perp: float,
    v_th: float,
    nu: float,
    Lambda: float,
    amplitude: float = 1.0,
    mixing_weight: float = 0.7,
) -> np.ndarray:
    """
    Total analytical spectrum: weighted sum of mixing/unmixing contributions.

    In a driven system, both phase mixing and unmixing occur. The total
    spectrum is a weighted combination determined by the relative importance
    of free streaming (k∥) vs perpendicular advection (k⊥).

    Args:
        m_array: Array of Hermite moment indices
        k_parallel, k_perp, v_th, nu, Lambda: Physics parameters
        amplitude: Overall normalization
        mixing_weight: Weight of phase mixing (0-1), unmixing is (1 - mixing_weight)

    Returns:
        Total |g_m|² spectrum

    Physics:
        For large k∥/k⊥: phase mixing dominates (mixing_weight → 1)
        For small k∥/k⊥: phase unmixing matters (mixing_weight → 0.5)
    """
    # Compute individual contributions
    spec_mixing = analytical_phase_mixing_spectrum(
        m_array, k_parallel, k_perp, v_th, nu, Lambda, amplitude
    )
    spec_unmixing = analytical_phase_unmixing_spectrum(
        m_array, k_parallel, k_perp, v_th, nu, Lambda, amplitude
    )

    # Weighted combination
    total_spectrum = mixing_weight * spec_mixing + (1.0 - mixing_weight) * spec_unmixing

    return total_spectrum


# ============================================================================
# Simulation Infrastructure
# ============================================================================


def run_forced_single_mode(
    kx_mode: float,
    ky_mode: float,
    kz_mode: float,
    M: int,
    forcing_amplitude: float,
    eta: float,
    nu: float,
    v_th: float = 1.0,
    beta_i: float = 1.0,
    Lambda: float = 1.0,
    n_steps: int = 500,
    n_warmup: int = 300,
    steady_state_window: int = 50,
    grid_size: Tuple[int, int, int] = (32, 32, 16),
    cfl_safety: float = 0.3,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Run forced single-mode simulation to steady state and measure spectrum.

    This is the core function for FDT validation:
    1. Initialize single k-mode
    2. Apply narrow-band forcing at that mode
    3. Evolve to steady state (ε_inj ≈ ε_diss)
    4. Time-average |g_m|² spectrum over steady-state period

    Args:
        kx_mode, ky_mode, kz_mode: Wavenumber of driven mode
        M: Number of Hermite moments
        forcing_amplitude: Forcing strength
        eta: Resistivity
        nu: Collision frequency
        v_th: Thermal velocity
        beta_i: Ion plasma beta
        Lambda: Kinetic parameter
        n_steps: Total number of timesteps
        n_warmup: Warmup steps before steady state
        steady_state_window: Number of steps to average over
        grid_size: Grid dimensions (Nx, Ny, Nz)
        cfl_safety: CFL safety factor
        seed: Random seed for forcing

    Returns:
        Dictionary containing:
            - 'spectrum': Time-averaged |g_m|² spectrum [M+1]
            - 'energy_history': Total energy vs time
            - 'k_parallel': k∥ for this mode
            - 'k_perp': k⊥ for this mode
            - 'steady_state_reached': Boolean
            - 'relative_fluctuation': Energy fluctuation in steady state
    """
    Nx, Ny, Nz = grid_size
    Lx = Ly = Lz = 2 * np.pi

    # Create grid
    grid = SpectralGrid3D.create(Nx=Nx, Ny=Ny, Nz=Nz, Lx=Lx, Ly=Ly, Lz=Lz)

    # Initialize single k-mode
    state = initialize_alfven_wave(
        grid=grid,
        M=M,
        kx_mode=kx_mode,
        ky_mode=ky_mode,
        kz_mode=kz_mode,
        amplitude=0.01,  # Small initial amplitude
        v_th=v_th,
        beta_i=beta_i,
        nu=nu,
        Lambda=Lambda,
    )

    # Compute k∥ and k⊥ for this mode
    k_parallel = kz_mode  # In this geometry, k∥ = kz
    k_perp = np.sqrt(kx_mode**2 + ky_mode**2)

    # Initialize forcing random key
    key = jax.random.PRNGKey(seed)

    # Energy history
    history = EnergyHistory()
    history.append(state)

    # Spectrum accumulator for time-averaging
    spectrum_sum = np.zeros(M + 1)
    n_samples = 0

    # Physics parameters
    v_A = 1.0
    dt = compute_cfl_timestep(state, v_A=v_A, cfl_safety=cfl_safety)

    # Define narrow forcing band around target mode
    k_target = np.sqrt(kx_mode**2 + ky_mode**2 + kz_mode**2)
    k_min_force = max(FORCING_BAND_MIN_ABSOLUTE, k_target * FORCING_BAND_LOWER_FACTOR)
    k_max_force = k_target * FORCING_BAND_UPPER_FACTOR

    # Main evolution loop
    for i in range(n_steps):
        # Apply forcing
        state, key = force_alfven_modes(
            state,
            amplitude=forcing_amplitude,
            k_min=k_min_force,
            k_max=k_max_force,
            dt=dt,
            key=key,
        )

        # Evolve dynamics
        state = gandalf_step(state, dt=dt, eta=eta, v_A=v_A)

        # Check for NaN/Inf
        if not jnp.all(jnp.isfinite(state.z_plus)):
            raise ValueError(f"NaN/Inf in z_plus at step {i}")
        if not jnp.all(jnp.isfinite(state.z_minus)):
            raise ValueError(f"NaN/Inf in z_minus at step {i}")
        if not jnp.all(jnp.isfinite(state.g)):
            raise ValueError(f"NaN/Inf in Hermite moments at step {i}")

        # Record energy
        history.append(state)

        # After warmup, start accumulating spectrum
        if i >= n_warmup:
            spectrum = hermite_moment_energy(state, account_for_rfft=True)
            if not np.all(np.isfinite(spectrum)):
                raise ValueError(f"NaN/Inf in spectrum at step {i}: {spectrum}")
            spectrum_sum += np.array(spectrum)
            n_samples += 1

    # Compute time-averaged spectrum
    if n_samples == 0:
        raise ValueError(
            f"No samples collected for time-averaging! n_warmup={n_warmup} >= n_steps={n_steps}. "
            f"Increase n_steps or decrease n_warmup."
        )
    spectrum_avg = spectrum_sum / n_samples

    # Check if steady state was reached
    if len(history.E_total) >= steady_state_window:
        energy_window = history.E_total[-steady_state_window:]
        mean_energy = np.mean(energy_window)
        std_energy = np.std(energy_window)

        # Warn if energy is suspiciously small (possible dissipation-dominated regime)
        if mean_energy < SPECTRUM_NORMALIZATION_THRESHOLD:
            import warnings
            warnings.warn(
                f"Mean energy extremely small ({mean_energy:.2e}). "
                f"Possible dissipation-dominated regime or insufficient forcing.",
                RuntimeWarning
            )

        relative_fluctuation = std_energy / (mean_energy + SPECTRUM_NORMALIZATION_THRESHOLD)
        steady_state_reached = relative_fluctuation < STEADY_STATE_FLUCTUATION_THRESHOLD
    else:
        relative_fluctuation = 1.0
        steady_state_reached = False

    return {
        'spectrum': spectrum_avg,
        'energy_history': np.array(history.E_total),
        'k_parallel': k_parallel,
        'k_perp': k_perp,
        'steady_state_reached': steady_state_reached,
        'relative_fluctuation': relative_fluctuation,
        'M': M,
        'nu': nu,
        'v_th': v_th,
        'Lambda': Lambda,
    }


# ============================================================================
# Visualization Utilities
# ============================================================================


def plot_fdt_comparison(
    result: Dict[str, Any],
    analytical_spectrum: np.ndarray,
    title: str = ""
) -> Any:  # matplotlib.figure.Figure, but avoid hard dependency in type hint
    """
    Plot numerical vs analytical spectrum comparison.

    Args:
        result: Dictionary from run_forced_single_mode()
        analytical_spectrum: Analytical prediction array
        title: Plot title
    """
    import matplotlib.pyplot as plt

    M = result['M']
    m_array = np.arange(M + 1)

    # Normalize both spectra
    if result['spectrum'][0] < SPECTRUM_NORMALIZATION_THRESHOLD:
        raise ValueError(
            f"Numerical spectrum m=0 too small for normalization: {result['spectrum'][0]} "
            f"< {SPECTRUM_NORMALIZATION_THRESHOLD}"
        )
    if analytical_spectrum[0] < SPECTRUM_NORMALIZATION_THRESHOLD:
        raise ValueError(
            f"Analytical spectrum m=0 too small for normalization: {analytical_spectrum[0]} "
            f"< {SPECTRUM_NORMALIZATION_THRESHOLD}"
        )

    spec_num = result['spectrum'] / result['spectrum'][0]
    spec_ana = analytical_spectrum / analytical_spectrum[0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Spectrum comparison
    ax1.semilogy(m_array, spec_num, 'o-', label='Numerical', markersize=6)
    ax1.semilogy(m_array, spec_ana, '--', label='Analytical', linewidth=2)
    ax1.set_xlabel('Hermite moment m', fontsize=12)
    ax1.set_ylabel('$|g_m|^2$ (normalized)', fontsize=12)
    ax1.set_title(f'Hermite Moment Spectrum\n{title}', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Energy history
    ax2.plot(result['energy_history'])
    ax2.set_xlabel('Timestep', fontsize=12)
    ax2.set_ylabel('Total Energy', fontsize=12)
    ax2.set_title('Energy Evolution to Steady State', fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
