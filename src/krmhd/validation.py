"""
Parameter Validation and Suggestion Module

This module provides automated parameter checking and suggestion functions
to help users avoid common pitfalls and numerical instabilities.

Key features:
- Overflow safety checks (η·dt < 50, ν·dt < 50)
- CFL condition validation
- Forcing stability checks (Issue #82)
- Resolution-dependent parameter suggestions
- Config file validation

References:
- CLAUDE.md: Hyper-dissipation section
- docs/ISSUE82_SUMMARY.md: Forced turbulence stability constraints
- docs/recommended_parameters.md: Parameter selection guide
"""

from dataclasses import dataclass
from typing import Optional, Dict, List
import warnings

import jax.numpy as jnp


# Validation thresholds (constants)
OVERFLOW_SAFETY_THRESHOLD = 50.0  # Maximum safe value for η·dt and ν·dt
WARNING_THRESHOLD_RATIO = 0.8  # Warn when approaching threshold (80%)
INJECTION_DISSIPATION_RATIO_LIMIT = 10.0  # Max ratio before warning
DEFAULT_CFL_LIMIT = 1.0  # Default CFL stability limit
CFL_WARNING_RATIO = 0.8  # Warn when CFL > 80% of limit

# Resolution-dependent amplitude limits
HIGH_RES_AMPLITUDE_LIMIT_128 = 0.4  # Max amplitude for 128³
HIGH_RES_AMPLITUDE_LIMIT_256 = 0.3  # Max amplitude for 256³


@dataclass
class ValidationResult:
    """Result of parameter validation."""

    valid: bool
    warnings: List[str]
    errors: List[str]
    suggestions: List[str]

    def print_report(self):
        """Print human-readable validation report."""
        if self.valid and not self.warnings:
            print("✓ All parameters valid")
            return

        if self.errors:
            print("\n❌ ERRORS (must fix):")
            for err in self.errors:
                print(f"  • {err}")

        if self.warnings:
            print("\n⚠️  WARNINGS (recommended fixes):")
            for warn in self.warnings:
                print(f"  • {warn}")

        if self.suggestions:
            print("\n💡 SUGGESTIONS:")
            for sug in self.suggestions:
                print(f"  • {sug}")


def validate_overflow_safety(
    eta: float,
    dt: float,
    nu: Optional[float] = None,
    hyper_r: int = 1,
    hyper_n: int = 1,
    threshold: float = OVERFLOW_SAFETY_THRESHOLD
) -> ValidationResult:
    """
    Check overflow safety for hyper-dissipation operators.

    The normalized dissipation formulation requires:
    - η·dt < 50 (resistivity overflow safety)
    - ν·dt < 50 (collision overflow safety)

    These constraints are INDEPENDENT of resolution and hyper-order!

    Parameters
    ----------
    eta : float
        Resistivity coefficient
    dt : float
        Timestep
    nu : float, optional
        Collision frequency (for Hermite moments)
    hyper_r : int, default=1
        Spatial hyper-dissipation order
    hyper_n : int, default=1
        Moment hyper-collision order
    threshold : float, default=50.0
        Safety threshold for overflow

    Returns
    -------
    ValidationResult
        Validation result with any errors/warnings

    References
    ----------
    - CLAUDE.md: Hyper-Dissipation section
    - Original GANDALF: damping_kernel.cu:50, timestep.cu:111
    """
    errors = []
    warnings_list = []
    suggestions = []

    # Check resistivity overflow
    eta_dt = eta * dt
    if eta_dt >= threshold:
        errors.append(
            f"Resistivity overflow: η·dt = {eta_dt:.2f} ≥ {threshold} "
            f"(CRITICAL: will cause exponential overflow)"
        )
        suggestions.append(
            f"Reduce η to < {threshold/dt:.4f} or reduce dt to < {threshold/eta:.4f}"
        )
    elif eta_dt > WARNING_THRESHOLD_RATIO * threshold:
        warnings_list.append(
            f"Resistivity near overflow: η·dt = {eta_dt:.2f} > {WARNING_THRESHOLD_RATIO*threshold:.1f} "
            f"(close to safety limit)"
        )
        suggestions.append(f"Consider reducing η or dt for better safety margin")

    # Check collision overflow
    if nu is not None:
        nu_dt = nu * dt
        if nu_dt >= threshold:
            errors.append(
                f"Collision overflow: ν·dt = {nu_dt:.2f} ≥ {threshold} "
                f"(CRITICAL: will cause exponential overflow)"
            )
            suggestions.append(
                f"Reduce ν to < {threshold/dt:.4f} or reduce dt to < {threshold/nu:.4f}"
            )
        elif nu_dt > WARNING_THRESHOLD_RATIO * threshold:
            warnings_list.append(
                f"Collision near overflow: ν·dt = {nu_dt:.2f} > {WARNING_THRESHOLD_RATIO*threshold:.1f}"
            )
            suggestions.append(f"Consider reducing ν or dt for better safety margin")

    # Note about hyper-orders
    if hyper_r > 2 or hyper_n > 2:
        suggestions.append(
            f"Using high hyper-orders (r={hyper_r}, n={hyper_n}): "
            f"Overflow safety is guaranteed, but numerical stability may require "
            f"additional tuning (see Issue #82)"
        )

    valid = len(errors) == 0
    return ValidationResult(valid, warnings_list, errors, suggestions)


def validate_cfl_condition(
    dt: float,
    dx: float,
    v_A: float = 1.0,
    cfl_limit: float = DEFAULT_CFL_LIMIT
) -> ValidationResult:
    """
    Check CFL (Courant-Friedrichs-Lewy) condition for numerical stability.

    The CFL condition requires: dt < CFL_limit * dx / v_A

    Typical values:
    - CFL_limit = 0.3-0.5 for explicit schemes (recommended)
    - CFL_limit = 1.0 for marginal stability

    Parameters
    ----------
    dt : float
        Timestep
    dx : float
        Grid spacing (minimum of dx, dy, dz)
    v_A : float, default=1.0
        Alfvén velocity
    cfl_limit : float, default=1.0
        CFL stability limit

    Returns
    -------
    ValidationResult
        Validation result with any errors/warnings
    """
    errors = []
    warnings_list = []
    suggestions = []

    # Handle edge cases
    if dx <= 0:
        errors.append(
            f"Invalid grid spacing: dx = {dx} (must be > 0)"
        )
        return ValidationResult(False, warnings_list, errors, suggestions)

    if v_A <= 0:
        errors.append(
            f"Invalid Alfvén velocity: v_A = {v_A} (must be > 0)"
        )
        return ValidationResult(False, warnings_list, errors, suggestions)

    cfl_actual = dt * v_A / dx

    if cfl_actual > cfl_limit:
        errors.append(
            f"CFL condition violated: CFL = {cfl_actual:.3f} > {cfl_limit} "
            f"(CRITICAL: numerical instability likely)"
        )
        dt_max = cfl_limit * dx / v_A
        suggestions.append(f"Reduce dt to < {dt_max:.4f} for CFL = {cfl_limit}")
    elif cfl_actual > CFL_WARNING_RATIO * cfl_limit:
        warnings_list.append(
            f"CFL near limit: CFL = {cfl_actual:.3f} > {CFL_WARNING_RATIO*cfl_limit:.2f}"
        )
        suggestions.append(
            f"Consider using dt with safety factor 0.3-0.5 for robustness"
        )
    else:
        suggestions.append(
            f"CFL = {cfl_actual:.3f} is safe (< {cfl_limit})"
        )

    valid = len(errors) == 0
    return ValidationResult(valid, warnings_list, errors, suggestions)


def validate_forcing_stability(
    force_amplitude: float,
    eta: float,
    resolution: int,
    hyper_r: int = 1,
    n_force_max: int = 2
) -> ValidationResult:
    """
    Check forcing stability based on Issue #82 investigation.

    Key findings from Issue #82:
    - High forcing amplitudes can cause exponential energy growth
    - Energy injection must balance dissipation
    - Higher resolutions need stronger dissipation or weaker forcing
    - High hyper-orders (r≥4) may have additional stability constraints

    Parameters
    ----------
    force_amplitude : float
        Forcing amplitude (ε_inj ∝ amplitude²)
    eta : float
        Resistivity coefficient
    resolution : int
        Grid resolution (Nx or Ny, whichever is larger)
    hyper_r : int, default=1
        Hyper-dissipation order
    n_force_max : int, default=2
        Maximum forcing mode number

    Returns
    -------
    ValidationResult
        Validation result with any errors/warnings

    References
    ----------
    - docs/ISSUE82_SUMMARY.md: Forced turbulence stability analysis
    """
    warnings_list = []
    suggestions = []

    # Empirical stability constraints from Issue #82

    # 1. Check forcing amplitude vs resolution
    if resolution >= 128:
        if force_amplitude > HIGH_RES_AMPLITUDE_LIMIT_128:
            warnings_list.append(
                f"High forcing amplitude ({force_amplitude:.2f}) at {resolution}³ resolution: "
                f"May cause exponential energy growth (Issue #82)"
            )
            suggestions.append(
                f"Recommended: amplitude < {HIGH_RES_AMPLITUDE_LIMIT_128} for {resolution}³ resolution"
            )

    if resolution >= 256:
        if force_amplitude > HIGH_RES_AMPLITUDE_LIMIT_256:
            warnings_list.append(
                f"Forcing amplitude ({force_amplitude:.2f}) may be too strong for {resolution}³"
            )
            suggestions.append(f"Try reducing to < {HIGH_RES_AMPLITUDE_LIMIT_256} for stability")

    # 2. Check forcing vs dissipation balance
    # Rough estimate: dissipation ~ η k² ~ η (2π n_max)²
    k_force = 2 * jnp.pi * n_force_max
    dissipation_estimate = eta * k_force**2
    injection_estimate = force_amplitude**2  # ε_inj ∝ amplitude²

    if injection_estimate > INJECTION_DISSIPATION_RATIO_LIMIT * dissipation_estimate:
        warnings_list.append(
            f"Energy injection >> dissipation: may not reach steady state"
        )
        suggestions.append(
            f"Increase η (currently {eta:.3f}) or reduce forcing amplitude"
        )

    # 3. High hyper-order warning
    if hyper_r >= 4:
        warnings_list.append(
            f"High hyper-order (r={hyper_r}) with forcing: "
            f"May have additional stability constraints (Issue #82)"
        )
        suggestions.append(
            "Monitor max(|z±|) during evolution - exponential growth indicates instability"
        )
        suggestions.append(
            "Consider using r=2 for forced turbulence until r=4 stability is better understood"
        )

    # Always valid (warnings only)
    return ValidationResult(True, warnings_list, [], suggestions)


def suggest_parameters(
    Nx: int,
    Ny: int,
    Nz: int,
    simulation_type: str = "forced",
    target_cfl: float = 0.3
) -> Dict[str, float]:
    """
    Suggest safe parameter values for given resolution.

    Parameters
    ----------
    Nx, Ny, Nz : int
        Grid resolution
    simulation_type : str, default="forced"
        Type of simulation: "forced", "decaying", or "benchmark"
    target_cfl : float, default=0.3
        Target CFL safety factor

    Returns
    -------
    dict
        Suggested parameters: dt, eta, nu, force_amplitude, etc.

    References
    ----------
    - docs/recommended_parameters.md: Parameter selection guide
    """
    # Grid spacing (assuming unit box)
    dx = 1.0 / Nx
    dy = 1.0 / Ny
    dz = 1.0 / Nz
    dx_min = min(dx, dy, dz)

    # CFL-limited timestep (v_A = 1.0)
    dt = target_cfl * dx_min / 1.0

    # Resolution-dependent dissipation
    # Scale with resolution to maintain similar dissipation range
    if Nx <= 64:
        eta_base = 0.02
        nu_base = 0.02
    elif Nx <= 128:
        eta_base = 0.01
        nu_base = 0.01
    else:
        eta_base = 0.005
        nu_base = 0.005

    # Simulation-specific parameters
    if simulation_type == "forced":
        # Forced turbulence: moderate dissipation, moderate forcing
        eta = eta_base
        nu = nu_base
        force_amplitude = 0.3 if Nx <= 128 else 0.2
        n_force_min = 1
        n_force_max = 2
    elif simulation_type == "decaying":
        # Decaying turbulence: lower dissipation for slower decay
        eta = eta_base * 0.5
        nu = nu_base * 0.5
        force_amplitude = 0.0  # No forcing
        n_force_min = 0
        n_force_max = 0
    else:  # benchmark
        # Benchmark: parameters from published results
        eta = 0.01
        nu = 0.01
        force_amplitude = 0.2
        n_force_min = 1
        n_force_max = 2

    params = {
        "dt": dt,
        "eta": eta,
        "nu": nu,
        "v_A": 1.0,
        "beta_i": 1.0,
        "force_amplitude": force_amplitude,
        "n_force_min": n_force_min,
        "n_force_max": n_force_max,
        "cfl_safety": target_cfl,
    }

    return params


def validate_parameters(
    dt: float,
    eta: float,
    nu: float,
    Nx: int,
    Ny: int,
    Nz: int,
    v_A: float = 1.0,
    force_amplitude: Optional[float] = None,
    hyper_r: int = 1,
    hyper_n: int = 1,
    n_force_max: int = 2
) -> ValidationResult:
    """
    Comprehensive parameter validation.

    Checks:
    1. Overflow safety (η·dt < 50, ν·dt < 50)
    2. CFL condition
    3. Forcing stability (if forcing enabled)

    Parameters
    ----------
    dt : float
        Timestep
    eta : float
        Resistivity
    nu : float
        Collision frequency
    Nx, Ny, Nz : int
        Grid resolution
    v_A : float, default=1.0
        Alfvén velocity
    force_amplitude : float, optional
        Forcing amplitude (if using forcing)
    hyper_r : int, default=1
        Spatial hyper-dissipation order
    hyper_n : int, default=1
        Moment hyper-collision order
    n_force_max : int, default=2
        Maximum forcing mode number

    Returns
    -------
    ValidationResult
        Combined validation result
    """
    all_warnings = []
    all_errors = []
    all_suggestions = []

    # 1. Overflow safety
    result_overflow = validate_overflow_safety(eta, dt, nu, hyper_r, hyper_n)
    all_warnings.extend(result_overflow.warnings)
    all_errors.extend(result_overflow.errors)
    all_suggestions.extend(result_overflow.suggestions)

    # 2. CFL condition
    dx_min = min(1.0/Nx, 1.0/Ny, 1.0/Nz)  # Assuming unit box
    result_cfl = validate_cfl_condition(dt, dx_min, v_A)
    all_warnings.extend(result_cfl.warnings)
    all_errors.extend(result_cfl.errors)
    all_suggestions.extend(result_cfl.suggestions)

    # 3. Forcing stability (if applicable)
    if force_amplitude is not None and force_amplitude > 0:
        resolution = max(Nx, Ny)
        result_forcing = validate_forcing_stability(
            force_amplitude, eta, resolution, hyper_r, n_force_max
        )
        all_warnings.extend(result_forcing.warnings)
        all_errors.extend(result_forcing.errors)
        all_suggestions.extend(result_forcing.suggestions)

    valid = len(all_errors) == 0
    return ValidationResult(valid, all_warnings, all_errors, all_suggestions)


def validate_config_dict(config: Dict) -> ValidationResult:
    """
    Validate parameters from a config dictionary.

    Parameters
    ----------
    config : dict
        Configuration dictionary with simulation parameters

    Returns
    -------
    ValidationResult
        Validation result

    Example
    -------
    >>> config = {
    ...     "dt": 0.01,
    ...     "eta": 0.5,
    ...     "nu": 0.5,
    ...     "Nx": 64,
    ...     "Ny": 64,
    ...     "Nz": 32,
    ...     "force_amplitude": 0.3,
    ... }
    >>> result = validate_config_dict(config)
    >>> result.print_report()
    """
    # Extract parameters with defaults
    dt = config.get("dt", 0.01)
    eta = config.get("eta", 0.01)
    nu = config.get("nu", 0.01)
    Nx = config.get("Nx", config.get("grid", {}).get("Nx", 64))
    Ny = config.get("Ny", config.get("grid", {}).get("Ny", 64))
    Nz = config.get("Nz", config.get("grid", {}).get("Nz", 32))
    v_A = config.get("v_A", 1.0)
    force_amplitude = config.get("force_amplitude", config.get("forcing", {}).get("amplitude"))
    hyper_r = config.get("hyper_r", config.get("physics", {}).get("hyper_r", 1))
    hyper_n = config.get("hyper_n", config.get("physics", {}).get("hyper_n", 1))
    n_force_max = config.get("n_force_max", config.get("forcing", {}).get("n_max", 2))

    return validate_parameters(
        dt, eta, nu, Nx, Ny, Nz, v_A,
        force_amplitude, hyper_r, hyper_n, n_force_max
    )
