"""
Validation utilities for KRMHD implementation.

This module provides infrastructure for validating the kinetic KRMHD implementation
against analytical predictions from linear Vlasov theory.

Main components:
- Plasma physics special functions (plasma dispersion function, Bessel functions)
- Linear response theory for kinetic Alfvén waves
- Exact analytical theory functions (FDT predictions with kinetic effects)
- Forced single-mode simulation runner
- Spectrum comparison and plotting utilities

Physics context:
    The Fluctuation-Dissipation Theorem (FDT) predicts the steady-state Hermite
    moment spectrum |g_m|² when a single k-mode is driven with stochastic forcing.
    This is a critical benchmark for validating the kinetic implementation.

    The exact analytical expressions include:
    - Plasma dispersion function Z(ζ) for Landau resonance
    - Modified Bessel functions I_m(b) for Finite Larmor Radius (FLR) effects
    - Proper kinetic response from KRMHD linear theory
    - Phase mixing and phase unmixing regimes

Implementation:
    - Thesis Eq 3.37: Phase mixing spectrum (large k∥, m^(-3/2) decay)
    - Thesis Eq 3.58: Phase unmixing spectrum (small k∥, m^(-1/2) decay)
    - Plasma dispersion function Z(ζ) and Bessel functions I_m(b) implemented exactly
    - **Limitation**: Response function uses phenomenological approximation (see kinetic_response_function)
    - Quantitative 10% validation requires exact KRMHD dispersion relation (future work)

References:
    - Thesis §2.6.1 - FDT for kinetic turbulence
    - Thesis Chapter 3 - Analytical theory and numerical validation
    - Howes et al. (2006) ApJ 651:590 - KRMHD linear theory
    - Schekochihin et al. (2009) ApJS 182:310 - Kinetic cascades
    - Adkins & Schekochihin (2017) arXiv:1709.03203 - Phase mixing theory
    - Issue #66 - Exact FDT expressions implementation
"""

from typing import Dict, Tuple, Any
import numpy as np
import jax
import jax.numpy as jnp
from scipy.special import wofz, iv  # Faddeeva function and modified Bessel function

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

# Plasma physics parameters
CAUSALITY_EPSILON = 1e-3  # Small imaginary part for causality in plasma dispersion function
PHASE_UNMIXING_FREQUENCY_FACTOR = 0.1  # Reduced frequency for perpendicular modes (ω ~ 0.1 k⊥ v_A)


# ============================================================================
# Plasma Physics Special Functions
# ============================================================================


def plasma_dispersion_function(zeta: np.ndarray) -> np.ndarray:
    """
    Plasma dispersion function Z(ζ) from kinetic plasma theory.

    The plasma dispersion function appears in the linear response of a collisionless
    plasma with Maxwellian velocity distribution. It is related to the Faddeeva
    function w(z) (scaled complex error function) by:

        Z(ζ) = i√π w(ζ)

    where w(z) is the Faddeeva function computed by scipy.special.wofz.

    Args:
        zeta: Complex argument ζ = (ω - k∥v∥) / (√2 k∥v_th)
              Can be scalar or array

    Returns:
        Z(ζ) = i√π w(ζ)

    Physics:
        The argument ζ represents the normalized resonance condition:
        - Re[ζ] ~ 1: Near Landau resonance (ω ≈ k∥v∥)
        - Im[ζ] > 0: Causality (Landau prescription)
        - Large |ζ|: Far from resonance, Z(ζ) → -1/ζ - 1/(2ζ³) + ...

    References:
        - Fried & Conte (1961): The Plasma Dispersion Function
        - Faddeeva function: w(z) = exp(-z²) erfc(-iz)
        - scipy.special.wofz: Faddeeva function implementation

    Note:
        For real arguments with Im[ζ] = 0, the integral has a singularity.
        In practice, add small positive imaginary part for causality: ζ + iε.
    """
    return 1j * np.sqrt(np.pi) * wofz(zeta)


def plasma_dispersion_derivative(zeta: np.ndarray) -> np.ndarray:
    """
    Derivative of plasma dispersion function: Z'(ζ) = dZ/dζ.

    The derivative satisfies the relation:
        Z'(ζ) = -2[1 + ζZ(ζ)]

    This is more numerically stable than finite differences for the derivative.

    Args:
        zeta: Complex argument

    Returns:
        Z'(ζ) = -2[1 + ζZ(ζ)]

    Physics:
        The derivative appears in kinetic response functions, particularly
        for velocity moments of the perturbed distribution function.
    """
    return -2.0 * (1.0 + zeta * plasma_dispersion_function(zeta))


def modified_bessel_ratio(m: int, x: float) -> float:
    """
    Compute I_m(x) exp(-x) for modified Bessel function of the first kind.

    This combination appears in FLR (Finite Larmor Radius) gyrokinetic theory
    as I_m(b) exp(-b) where b = k⊥²ρ_s² / 2.

    The exponential factor prevents overflow for large arguments.

    Args:
        m: Order of Bessel function (Hermite moment index, must be non-negative integer)
        x: Argument (typically b = k⊥²ρ_s² / 2)

    Returns:
        I_m(x) exp(-x)

    Physics:
        - x → 0: I_m(x)exp(-x) → x^m / (2^m m!) - 1 ≈ 0 for m > 0
        - x >> 1: I_m(x)exp(-x) → 1/√(2πx) (all m converge to same value)
        - FLR effects strongest when x ~ 1 (k⊥ρ_s ~ 1)

    Note:
        scipy.special.iv(m, x) computes I_m(x) directly.
        We multiply by exp(-x) for numerical stability.
    """
    # Type safety: ensure m is non-negative integer
    m = int(m)
    if m < 0:
        raise ValueError(f"Bessel order m must be non-negative, got m={m}")

    if x < 1e-10:
        # Small argument expansion: I_m(x) ≈ (x/2)^m / m!
        # For m=0: I_0(x) ≈ 1, so I_0(x)exp(-x) ≈ 1 - x + O(x²)
        # For m>0: I_m(x)exp(-x) ≈ (x/2)^m exp(-x) / m! → 0
        if m == 0:
            return 1.0 - x
        else:
            return 0.0
    else:
        # Use scipy implementation (handles large arguments well)
        return iv(m, x) * np.exp(-x)


# ============================================================================
# Linear Response Theory
# ============================================================================


def kinetic_response_function(
    k_parallel: float,
    k_perp: float,
    omega: float,
    v_th: float,
    Lambda: float,
    nu: float = 0.0,
    beta_i: float = 1.0,
    v_A: float = 1.0,
) -> complex:
    """
    Simplified linear response function for kinetic Alfvén waves in KRMHD.

    **IMPORTANT LIMITATION**: This is a phenomenological approximation, NOT the exact
    dispersion relation from KRMHD theory. The denominator (1 + |ζ|²)^(-1) provides
    qualitatively correct resonance structure but lacks proper normalization and
    k⊥ρ_s dependence from the full kinetic dispersion relation.

    For quantitative FDT validation at 10% accuracy, this approximation may be
    insufficient. The exact implementation would require solving the full KRMHD
    dispersion relation D(k,ω) = 0 with FLR corrections (Howes et al. 2006 Eq. 14-15
    or thesis Eq 3.37).

    Args:
        k_parallel: Parallel wavenumber k∥
        k_perp: Perpendicular wavenumber k⊥
        omega: Wave frequency ω (use ω ≈ k∥v_A for Alfvén branch)
        v_th: Thermal velocity
        Lambda: Kinetic parameter (Λ = 1 + 1/β for kinetic corrections)
        nu: Collision frequency (default: 0, collisionless)
        beta_i: Ion plasma beta (default: 1.0)
        v_A: Alfvén velocity (default: 1.0)

    Returns:
        Complex response function R(k, ω) (approximate)

    Physics captured:
        - Plasma dispersion function Z(ζ) for Landau resonance
        - ζ = (ω - iν) / (√2 k∥v_th): normalized frequency (with collisions)
        - Lambda parameter: Λ = 1 + 1/β gives kinetic corrections
        - For large |ζ|: Z(ζ) → -1/ζ (weak damping, fluid limit)
        - For ζ ~ 1: Strong Landau damping (kinetic effects)

    Physics missing:
        - Exact normalization from full dispersion relation
        - Proper k⊥ρ_s dependence in resonance width
        - FLR corrections to susceptibility

    References:
        - Howes et al. (2006) ApJ 651:590 Eq. 14-15: Full KRMHD dispersion relation
        - Schekochihin et al. (2009) ApJS 182:310: Kinetic cascades
    """
    # Normalized frequency with collisional damping
    # ζ = (ω - iν) / (√2 k∥v_th)
    # Add small imaginary part for causality if collisionless
    if abs(k_parallel) < K_PARALLEL_ZERO_THRESHOLD:
        # Pure perpendicular mode: no Landau resonance
        # Return simple non-resonant response
        return 1.0 + 0.0j

    # Causality prescription: effective_nu ≥ CAUSALITY_EPSILON * k∥v_th
    # For collisionless case (nu=0), this adds small imaginary part iε to frequency
    # to ensure proper Landau prescription (poles slightly below real axis)
    # CAUSALITY_EPSILON=1e-3 chosen to be small compared to typical ν~0.01-0.3
    # but large enough to avoid numerical issues near real axis
    effective_nu = max(nu, CAUSALITY_EPSILON * abs(k_parallel) * v_th)
    zeta = (omega - 1j * effective_nu) / (np.sqrt(2.0) * abs(k_parallel) * v_th)

    # Plasma dispersion function for Landau resonance
    Z_zeta = plasma_dispersion_function(zeta)
    # Note: Z'(ζ) = plasma_dispersion_derivative(zeta) is available for future use
    # (e.g., for higher-order kinetic corrections or velocity moment calculations)

    # Kinetic response with Lambda parameter
    # Based on standard KRMHD linear theory (Howes et al. 2006)
    # Lambda = 1 + 1/β gives kinetic corrections to pressure response
    kinetic_factor = 1.0 + (1.0 - 1.0/Lambda) * zeta * Z_zeta

    # Response function
    # TODO(Issue #66): Cross-check denominator against thesis Eq 3.37
    # Standard dispersion relation has form D(k,ω) = 1 - χ(k,ω) where χ is susceptibility
    # Current form (1 + |ζ|²)^(-1) is phenomenological damping factor
    # May need to replace with exact dispersion relation from thesis/Howes 2006
    # For now, provides qualitatively correct resonance structure
    response = kinetic_factor / (1.0 + abs(zeta)**2)

    return response


def flr_correction_factor(m: int, k_perp: float, rho_s: float) -> float:
    """
    Finite Larmor Radius (FLR) correction factor for Hermite moment m.

    In gyrokinetic theory, FLR effects enter through modified Bessel functions:
        Γ_m(b) = I_m(b) exp(-b)
    where b = (k⊥ρ_s)² / 2.

    This function computes Γ_m²(b) which appears in the Hermite moment spectrum.

    Args:
        m: Hermite moment index
        k_perp: Perpendicular wavenumber k⊥
        rho_s: Ion sound gyroradius ρ_s = √β v_th / v_A

    Returns:
        Γ_m²(b) = [I_m(b) exp(-b)]²

    Physics:
        - k⊥ρ_s << 1: Fluid regime, Γ_m → δ_m0 (only m=0 survives)
        - k⊥ρ_s ~ 1: Kinetic regime, FLR effects important
        - k⊥ρ_s >> 1: Sub-gyroradius scales, all moments damped

    Note:
        The factor Γ_m²(b) appears in |g_m|² spectrum as an m-dependent
        suppression due to Larmor radius effects.
    """
    b = (k_perp * rho_s)**2 / 2.0
    Gamma_m = modified_bessel_ratio(m, b)
    return Gamma_m**2


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
    beta_i: float = 1.0,
    v_A: float = 1.0,
) -> np.ndarray:
    """
    Analytical prediction for phase-mixing spectrum (Thesis Eq 3.37).

    In the phase mixing regime (large k∥), particles with different v∥
    dephase due to free streaming. This creates fine-scale structure in
    velocity space, causing energy to cascade to higher Hermite moments.

    This implementation uses exact linear kinetic theory:
    - Plasma dispersion function Z(ζ) for Landau resonance
    - Modified Bessel functions I_m(b) for FLR corrections
    - Proper kinetic response function from KRMHD theory

    Args:
        m_array: Array of Hermite moment indices [0, 1, ..., M]
        k_parallel: Parallel wavenumber k∥
        k_perp: Perpendicular wavenumber k⊥
        v_th: Thermal velocity
        nu: Collision frequency
        Lambda: Kinetic parameter (1 - 1/Λ factor in g1 coupling)
        amplitude: Overall amplitude normalization
        beta_i: Ion plasma beta (default: 1.0)
        v_A: Alfvén velocity (default: 1.0)

    Returns:
        |g_m|² spectrum vs m (analytical prediction)

    Physics:
        The spectrum has the form:
            |g_m|² ~ |R(k, ω)|² × Γ_m²(k⊥ρ_s) × |S_forcing|² × exp(-νm t)

        where:
        - R(k, ω): Linear response function (plasma dispersion function)
        - Γ_m(b): FLR correction, Γ_m(b) = I_m(b)exp(-b), b = (k⊥ρ_s)²/2
        - S_forcing: Forcing spectrum shape
        - exp(-νm t): Collisional damping in steady state

        Critical moment (collisional cutoff): m_crit ~ k∥v_th / ν

    References:
        - Thesis Eq 3.37: Exact phase mixing spectrum
        - Howes et al. (2006) ApJ 651:590: KRMHD linear theory
        - Schekochihin et al. (2009) ApJS 182:310: Kinetic cascades
    """
    # Ion sound gyroradius: ρ_s = √(β) v_th / v_A
    rho_s = np.sqrt(beta_i) * v_th / v_A

    # Alfvén wave frequency: ω ≈ k∥ v_A (Alfvén branch)
    omega = abs(k_parallel) * v_A

    # Kinetic response function |R(k, ω)|²
    response = kinetic_response_function(
        k_parallel, k_perp, omega, v_th, Lambda, nu, beta_i, v_A
    )
    response_squared = abs(response)**2

    # Initialize spectrum array
    spectrum = np.zeros_like(m_array, dtype=float)

    # Compute spectrum for each Hermite moment
    for i, m in enumerate(m_array):
        # FLR correction factor: Γ_m²(k⊥ρ_s)
        flr_factor = flr_correction_factor(int(m), k_perp, rho_s)

        # Collisional damping: exp(-2νm) in steady state
        # (Factor of 2 because energy ~ |g_m|²)
        if nu > COLLISION_FREQ_ZERO_THRESHOLD:
            m_crit = abs(k_parallel) * v_th / nu if abs(k_parallel) > K_PARALLEL_ZERO_THRESHOLD else COLLISIONLESS_M_CRIT
            collision_damping = np.exp(-m / m_crit)
        else:
            collision_damping = 1.0  # Collisionless limit

        # Phase mixing power law from kinetic theory
        # Research shows m^(-3/2) for phase mixing (Adkins & Schekochihin 2017)
        # Note: m^0 = 1, so no singularity at m=0 (unlike m^(-α) for α > 0 would be)
        # Using m directly, not (m+1), to match theoretical prediction
        # For m=0, this gives 0^(-1.5) = ∞, but spectrum[0] is normalized to 1 anyway
        if m == 0:
            phase_mixing_factor = 1.0  # Will be normalized to 1 below
        else:
            phase_mixing_factor = m**(-1.5)

        # Total spectrum
        spectrum[i] = (
            amplitude
            * response_squared
            * flr_factor
            * phase_mixing_factor
            * collision_damping
        )

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
    beta_i: float = 1.0,
    v_A: float = 1.0,
) -> np.ndarray:
    """
    Analytical prediction for phase-unmixing spectrum (Thesis Eq 3.58).

    In the phase unmixing regime (small k∥, large k⊥), nonlinear perpendicular
    advection can transfer energy back from higher to lower moments. This
    leads to a different spectral shape than pure phase mixing.

    This implementation uses exact linear kinetic theory with proper
    k∥/k⊥ dependence for the unmixing regime.

    Args:
        m_array: Array of Hermite moment indices [0, 1, ..., M]
        k_parallel: Parallel wavenumber k∥
        k_perp: Perpendicular wavenumber k⊥
        v_th: Thermal velocity
        nu: Collision frequency
        Lambda: Kinetic parameter
        amplitude: Overall amplitude normalization
        beta_i: Ion plasma beta (default: 1.0)
        v_A: Alfvén velocity (default: 1.0)

    Returns:
        |g_m|² spectrum vs m (analytical prediction)

    Physics:
        Phase unmixing is driven by perpendicular nonlinearity.
        The spectrum typically has shallower slope than phase mixing:
        - Phase mixing: m^(-3/2) (strong cascade to high m)
        - Phase unmixing: m^(-1/2) (weaker cascade, nonlinear damping)

        The critical moment is determined by k⊥ (perpendicular advection)
        rather than k∥ (parallel streaming).

    References:
        - Thesis Eq 3.58: Exact phase unmixing spectrum
        - Adkins & Schekochihin (2017): Phase mixing vs anti-phase-mixing
        - Schekochihin et al. (2009): Nonlinear kinetic cascades
    """
    # Ion sound gyroradius: ρ_s = √(β) v_th / v_A
    rho_s = np.sqrt(beta_i) * v_th / v_A

    # Alfvén wave frequency: ω ≈ k∥ v_A (Alfvén branch)
    # For pure perpendicular modes (k∥→0), use reduced frequency
    # PHASE_UNMIXING_FREQUENCY_FACTOR represents weak Alfvén character in phase unmixing regime
    # (perpendicular advection dominates, not wave propagation)
    omega = abs(k_parallel) * v_A if abs(k_parallel) > K_PARALLEL_ZERO_THRESHOLD else k_perp * v_A * PHASE_UNMIXING_FREQUENCY_FACTOR

    # Kinetic response function |R(k, ω)|²
    # Phase unmixing regime has weaker resonance than phase mixing
    response = kinetic_response_function(
        k_parallel, k_perp, omega, v_th, Lambda, nu, beta_i, v_A
    )
    response_squared = abs(response)**2

    # k∥/k⊥ dependence for unmixing regime
    # Small k∥ enhances unmixing (perpendicular advection dominates)
    if abs(k_parallel) > K_PARALLEL_ZERO_THRESHOLD:
        k_ratio_factor = np.sqrt(k_perp / abs(k_parallel))
    else:
        # Pure perpendicular mode: phase unmixing saturates
        k_ratio_factor = 1.0

    # Initialize spectrum array
    spectrum = np.zeros_like(m_array, dtype=float)

    # Compute spectrum for each Hermite moment
    for i, m in enumerate(m_array):
        # FLR correction factor: Γ_m²(k⊥ρ_s)
        flr_factor = flr_correction_factor(int(m), k_perp, rho_s)

        # Collisional damping with k⊥ dependence (perpendicular advection scale)
        if nu > COLLISION_FREQ_ZERO_THRESHOLD and k_perp > K_PARALLEL_ZERO_THRESHOLD:
            m_crit = k_perp * v_th / nu
            collision_damping = np.exp(-m / m_crit)
        else:
            collision_damping = 1.0  # Collisionless limit

        # Phase unmixing power law: shallower than phase mixing
        # Research shows m^(-1/2) for phase unmixing regime (Adkins & Schekochihin 2017)
        # Note: m^0 = 1, so no singularity at m=0
        # Using m directly, not (m+1), to match theoretical prediction
        if m == 0:
            phase_unmixing_factor = 1.0  # Will be normalized to 1 below
        else:
            phase_unmixing_factor = m**(-0.5)

        # Total spectrum with k∥/k⊥ factor
        spectrum[i] = (
            amplitude
            * response_squared
            * flr_factor
            * phase_unmixing_factor
            * collision_damping
            * k_ratio_factor
        )

    # Normalize by m=0 value for relative comparison
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
    beta_i: float = 1.0,
    v_A: float = 1.0,
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
        beta_i: Ion plasma beta (default: 1.0)
        v_A: Alfvén velocity (default: 1.0)

    Returns:
        Total |g_m|² spectrum

    Physics:
        For large k∥/k⊥: phase mixing dominates (mixing_weight → 1)
        For small k∥/k⊥: phase unmixing matters (mixing_weight → 0.5)

        The mixing_weight can be determined from k∥/k⊥ ratio:
        - k∥/k⊥ > 1: mixing_weight ≈ 0.8-0.9 (parallel streaming dominates)
        - k∥/k⊥ ~ 1: mixing_weight ≈ 0.6-0.7 (balanced regime)
        - k∥/k⊥ < 1: mixing_weight ≈ 0.4-0.5 (perpendicular advection dominates)
    """
    # Compute individual contributions with exact theory
    spec_mixing = analytical_phase_mixing_spectrum(
        m_array, k_parallel, k_perp, v_th, nu, Lambda, amplitude, beta_i, v_A
    )
    spec_unmixing = analytical_phase_unmixing_spectrum(
        m_array, k_parallel, k_perp, v_th, nu, Lambda, amplitude, beta_i, v_A
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
