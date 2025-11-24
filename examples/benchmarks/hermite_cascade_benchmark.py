#!/usr/bin/env python3
"""
Hermite Moment Cascade Benchmark (Thesis Chapter 3)

Reproduces velocity-space cascade with m^(-1/2) power law spectrum.
Validates phase mixing physics and collision damping in Hermite moment hierarchy.

Physics:
- Force g₀ and g₁ (density and momentum) directly with white noise
- Parallel streaming drives energy to higher moments via ∇∥ coupling
- Collisions damp high-m modes: C[gm] = -νm·gm
- Steady state spectrum: E_m ~ m^(-1/2) (phase mixing/unmixing balance)
- Zero Alfvén fields (z± = 0) to isolate kinetic physics

Expected outputs:
- Time series: E_total(t) reaching plateau
- Hermite spectrum: log-log plot of E_m vs m
- Power law fit: slope ~ -0.5 in inertial range (m ~ 2-M/2)
- Comparison with thesis Figure 3.3

Recommended parameter combinations:

**Strategy for achieving m^(-1/2) spectrum:**

The key is balancing energy injection (forcing) with energy removal (collisions).
Current parameter sets produce too-steep spectra (m^-2.7 to m^-13) indicating
collisional damping dominates phase mixing.

**RECOMMENDED STARTING POINTS:**

1. **Conservative (most likely to work)**:
   ```bash
   --resolution 32 --hermite-moments 32 \
   --nu 0.3 --force-amplitude 0.2 \
   --total-time 100 --averaging-start 80
   ```
   Physics: Lower collisions + higher forcing for better balance
   Expected: Flatter spectrum, slower approach to steady state

2. **Moderate (balanced)**:
   ```bash
   --resolution 32 --hermite-moments 32 \
   --nu 0.5 --force-amplitude 0.15 \
   --lambda-param -2.0 \
   --total-time 100 --averaging-start 80
   ```
   Physics: Moderate parameters with stronger kinetic coupling (Lambda=-2.0)

3. **High-forcing (rapid energy buildup)**:
   ```bash
   --resolution 32 --hermite-moments 48 \
   --nu 0.5 --force-amplitude 0.3 \
   --total-time 150 --averaging-start 120
   ```
   Physics: Strong forcing may help establish cascade before dissipation

**Force both g₀ and g₁** (default): Forcing only g₀ (--force-g0-only) may limit
cascade development. Try default first (forces both density and momentum).

**Monitor energy balance diagnostics**: Script now shows injection/dissipation ratio.
Target: ratio ≈ 1.0 at steady state. If ratio > 1.5, reduce amplitude or increase nu.
If ratio < 0.67, increase amplitude or decrease nu.

**Use checkpoints**: Long runs benefit from --resume-from for parameter adjustments.

**Iterative tuning workflow**:
1. Start with conservative parameters above
2. Run short test (50 τ_A), watch energy balance ratio
3. Adjust nu or amplitude based on ratio
4. Resume with adjusted parameters until steady state achieved

**Thesis comparison needed**: Original thesis achieved m^(-1/2). Parameters likely differ
from current defaults. Consider reviewing thesis Chapter 3 for exact values.

Typical parameters (for reference, may need adjustment):
- Spatial resolution: 32³ or 64³ (velocity-space physics doesn't need high resolution)
- Hermite moments: M=32-48 (wide power law range)
- Collision parameter: ν ~ 0.3-0.5 (REDUCED from 1.0)
- Forcing: amplitude ~ 0.15-0.3 (INCREASED from 0.01), modes n=1-2, both g₀ and g₁
- Runtime: ~100 τ_A to reach steady state (LONGER than 20)

Expected runtime:
- 32³, M=32: ~5-10 minutes
- 64³, M=32: ~15-30 minutes

Resume functionality:
- Checkpoints saved every 500 steps to output directory
- Resume with: --resume-from checkpoint_file.h5
- Continues from saved time until --total-time
- Can adjust nu, lambda-param when resuming
"""

import argparse
import sys
import time
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from scipy import stats

from krmhd import (
    SpectralGrid3D,
    KRMHDState,
    initialize_random_spectrum,
    gandalf_step,
    compute_cfl_timestep,
    energy as compute_energy,
    force_hermite_moments,
    force_hermite_moments_specific,
    compute_energy_injection_rate,
)
from krmhd.diagnostics import (
    EnergyHistory,
    hermite_moment_energy,
)
from krmhd.io import save_checkpoint, load_checkpoint


def detect_steady_state(energy_history, window=100, threshold=0.05, n_smooth=None):
    """
    Detect if system has reached steady state (energy plateau) - DIAGNOSTIC ONLY.

    Checks if energy has stopped growing by looking at the trend over
    a long window. True steady state requires energy to plateau, not
    just have small fluctuations.

    Args:
        energy_history: List of total energy values
        window: Number of recent points to check (default 100)
        threshold: Relative energy change threshold (default 5%)
        n_smooth: Number of points to average for smoothing (default: window//10, min 5)

    Returns:
        True if steady state detected (energy plateau), False otherwise
    """
    if len(energy_history) < window:
        return False

    recent = energy_history[-window:]
    if n_smooth is None:
        n_smooth = max(5, window // 10)

    E_start = np.mean(recent[:n_smooth])
    E_end = np.mean(recent[-n_smooth:])

    if E_start == 0:
        return False

    relative_change = abs(E_end - E_start) / E_start
    return relative_change < threshold


def fit_power_law(m_values, E_m, m_min=2, m_max=16):
    """
    Fit power law E_m ~ m^α to Hermite spectrum.

    Args:
        m_values: Array of Hermite moment indices
        E_m: Array of Hermite moment energies
        m_min: Minimum moment for fit range (default: 2, skip m=0,1)
        m_max: Maximum moment for fit range

    Returns:
        slope: Power law exponent α
        intercept: Log10 of normalization constant
        r_squared: Coefficient of determination (quality of fit)
        m_fit: Moment values in fit range
        E_fit_predicted: Fitted power law values
    """
    # Select fit range
    mask = (m_values >= m_min) & (m_values <= m_max) & (E_m > 0)
    m_fit = m_values[mask]
    E_fit = E_m[mask]

    if len(m_fit) < 3:
        # Not enough points for meaningful fit
        return np.nan, np.nan, np.nan, m_fit, np.full_like(m_fit, np.nan)

    # Log-log linear regression
    log_m = np.log10(m_fit)
    log_E = np.log10(E_fit)

    slope, intercept, r_value, _, _ = stats.linregress(log_m, log_E)
    r_squared = r_value**2

    # Predicted values for plotting
    E_fit_predicted = 10**(intercept + slope * log_m)

    return slope, intercept, r_squared, m_fit, E_fit_predicted


def compute_collision_dissipation_rate(state, nu):
    """
    Compute energy dissipation rate from collisions: dE/dt|_coll = -Σ_m ν·m·E_m.

    The collision operator is C[gm] = -νm·gm, which removes energy at rate
    proportional to both the collision frequency ν, moment index m, and energy E_m.

    Args:
        state: KRMHDState with Hermite moments
        nu: Collision frequency

    Returns:
        dissipation_rate: Total collisional dissipation rate (positive = energy removal)
    """
    E_m = hermite_moment_energy(state)  # Energy in each moment
    M = state.M
    m_values = jnp.arange(M + 1)

    # Dissipation rate: Σ_m ν·m·E_m
    # (m=0,1 exempt in actual timestepping, but compute total here for diagnostics)
    dissipation_rate = float(jnp.sum(nu * m_values * E_m))

    return dissipation_rate


def main():
    """Run Hermite moment cascade benchmark to steady state."""

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Hermite Moment Cascade Benchmark')
    parser.add_argument('--resolution', type=int, default=32,
                        help='Spatial grid resolution (default: 32, use 8 or 16 for single-k-mode)')
    parser.add_argument('--hermite-moments', type=int, default=32,
                        help='Maximum Hermite moment M (default: 32)')
    parser.add_argument('--nu', type=float, default=1.0,
                        help='Collision parameter (default: 1.0)')
    parser.add_argument('--hyper-n', type=int, default=1,
                        help='Hyper-collision exponent (default: 1, typical: 1-4)')
    parser.add_argument('--force-amplitude', type=float, default=0.01,
                        help='Forcing amplitude (default: 0.01, reduced from 0.1)')
    parser.add_argument('--lambda-param', type=float, default=-1.0,
                        help='Kinetic parameter Lambda in g1_rhs (default: -1.0 matching thesis)')
    parser.add_argument('--force-g0-only', action='store_true',
                        help='Force only g₀ (density), not g₁ (momentum)')
    parser.add_argument('--single-k-mode', action='store_true',
                        help='Force single k-mode (0,0,1) instead of shell forcing')
    parser.add_argument('--total-time', type=float, default=50.0,
                        help='Total simulation time in Alfvén times (default: 50)')
    parser.add_argument('--averaging-start', type=float, default=25.0,
                        help='When to start averaging in Alfvén times (default: 25)')
    parser.add_argument('--output-dir', type=str, default='examples/output',
                        help='Output directory for plots (default: examples/output)')
    parser.add_argument('--save-diagnostics', action='store_true',
                        help='Save detailed diagnostics to HDF5')
    parser.add_argument('--diagnostic-interval', type=int, default=5,
                        help='Save diagnostics every N steps (default: 5)')
    parser.add_argument('--resume-from', type=str, default=None,
                        help='Resume from checkpoint file (HDF5)')

    args = parser.parse_args()

    print("=" * 70)
    print("Hermite Moment Cascade Benchmark (Thesis Chapter 3)")
    print("=" * 70)

    # ==========================================================================
    # PARAMETERS
    # ==========================================================================

    # Grid (velocity-space physics doesn't need high spatial resolution)
    Nx = Ny = Nz = args.resolution
    Lx = Ly = Lz = 1.0  # Unit box

    # Hermite moments (high M for wide power law range)
    M = args.hermite_moments

    # Physics parameters
    v_A = 1.0  # Alfvén velocity
    beta_i = 1.0  # Ion plasma beta
    v_th = 1.0  # Thermal velocity

    # Dissipation (minimal on Alfvén fields since z± = 0)
    eta = 0.5
    hyper_r = 2

    # Collisions (critical for high-m damping)
    nu = args.nu
    hyper_n = args.hyper_n

    # Forcing parameters (inject energy at large scales)
    force_amplitude = args.force_amplitude
    n_force_min = 1  # Fundamental mode
    n_force_max = 2  # Narrow injection range
    # Configure which moments to force based on command-line flag
    if args.force_g0_only:
        forced_moments = (0,)  # Force only density (g₀)
    else:
        forced_moments = (0, 1)  # Force both density and momentum

    # Time integration
    tau_A = Lz / v_A
    total_time = args.total_time * tau_A
    averaging_start = args.averaging_start * tau_A
    cfl_safety = 0.3
    save_interval = 10

    # Diagnostic intervals
    steady_state_check_interval = 50
    progress_print_interval = 50
    steady_state_window = 100
    steady_state_threshold = 0.05  # 5% relative change
    checkpoint_interval = 500  # Save checkpoint every N steps

    # Validate averaging window
    if averaging_start >= total_time:
        print(f"\n" + "!" * 70)
        print(f"ERROR: Averaging start time ({averaging_start/tau_A:.1f} τ_A) >= total time ({total_time/tau_A:.1f} τ_A)")
        print(f"       Averaging will never start!")
        print("!" * 70)
        sys.exit(1)

    print(f"\nGrid: {Nx} × {Ny} × {Nz}")
    print(f"Hermite moments: M={M}")
    print(f"Domain: [{Lx:.1f}, {Ly:.1f}, {Lz:.1f}]")
    print(f"Physics: v_A={v_A}, v_th={v_th}, β_i={beta_i}, Λ={args.lambda_param:.1f}")
    print(f"Dissipation: η={eta:.1e}, ν={nu:.1e}")
    print(f"Hyper-dissipation: r={hyper_r}, n={hyper_n}")
    if args.single_k_mode:
        print(f"Forcing: amplitude={force_amplitude}, single k-mode (0,0,1)")
    else:
        print(f"Forcing: amplitude={force_amplitude}, shell modes n ∈ [{n_force_min}, {n_force_max}]")
    if args.force_g0_only:
        print(f"         Forced moments: g₀ only (density)")
    else:
        print(f"         Forced moments: g₀ and g₁ (density + momentum)")
    print(f"Evolution: Run for {args.total_time:.1f} τ_A total, average last {args.total_time - args.averaging_start:.1f} τ_A")
    print(f"CFL safety: {cfl_safety}")

    if args.single_k_mode and args.resolution <= 16:
        print(f"\nEstimated runtime: <1 minute (single-mode, small grid)")
    elif args.resolution == 32:
        print(f"\nEstimated runtime: ~5-10 minutes")
    elif args.resolution == 64:
        print(f"\nEstimated runtime: ~15-30 minutes")
    else:
        print(f"\nEstimated runtime: varies with resolution")

    # ==========================================================================
    # Initialize or Resume Grid and State
    # ==========================================================================

    print("\n" + "-" * 70)

    if args.resume_from:
        print(f"Resuming from checkpoint: {args.resume_from}")
        try:
            state, grid, metadata = load_checkpoint(args.resume_from)

            # Validate grid parameters match
            if grid.Nx != Nx or grid.Ny != Ny or grid.Nz != Nz:
                print(f"\n" + "!" * 70)
                print(f"WARNING: Checkpoint grid ({grid.Nx}×{grid.Ny}×{grid.Nz}) differs from requested ({Nx}×{Ny}×{Nz})")
                print(f"         Using checkpoint grid parameters.")
                print("!" * 70)
                Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz

            # Validate Hermite moments match
            M_checkpoint = state.M
            if M_checkpoint != M:
                print(f"\n" + "!" * 70)
                print(f"WARNING: Checkpoint has M={M_checkpoint}, but --hermite-moments={M} was specified")
                print(f"         Using checkpoint value M={M_checkpoint}")
                print("!" * 70)
                M = M_checkpoint

            # Update state parameters that may differ (Pydantic model_copy)
            state = state.model_copy(update={
                'nu': nu,
                'Lambda': args.lambda_param,
            })

            E_m_init = hermite_moment_energy(state)
            E0 = float(jnp.sum(E_m_init))
            print(f"✓ Loaded checkpoint from t={state.time:.2f} τ_A")
            print(f"  Grid: {grid.Nx}×{grid.Ny}×{grid.Nz}, M={M}")
            print(f"  Hermite energy: E_total = {E0:.6e}")
            print(f"  Will continue until t={total_time:.1f} τ_A")

        except Exception as e:
            print(f"\n" + "!" * 70)
            print(f"ERROR: Failed to load checkpoint: {e}")
            print("!" * 70)
            sys.exit(1)
    else:
        print("Initializing new simulation...")

        grid = SpectralGrid3D.create(Nx=Nx, Ny=Ny, Nz=Nz, Lx=Lx, Ly=Ly, Lz=Lz)
        print(f"✓ Created {Nx}×{Ny}×{Nz} spectral grid")

        # Initialize with ZERO Alfvén fields to isolate kinetic physics
        # Only g₀ gets a small random perturbation as seed
        state = KRMHDState(
            z_plus=jnp.zeros((Nz, Ny, Nx//2+1), dtype=complex),
            z_minus=jnp.zeros((Nz, Ny, Nx//2+1), dtype=complex),
            B_parallel=jnp.zeros((Nz, Ny, Nx//2+1), dtype=complex),
            g=jnp.zeros((Nz, Ny, Nx//2+1, M+1), dtype=complex),
            grid=grid,
            time=0.0,
            M=M,
            v_th=v_th,
            beta_i=beta_i,
            nu=nu,
            Lambda=args.lambda_param,  # Use Lambda from command line (default -1.0 for thesis)
        )

        # Start from truly zero state - let forcing build up energy
        # This is simpler and avoids any initialization artifacts
        E_m_init = hermite_moment_energy(state)
        E0 = float(jnp.sum(E_m_init))
        print(f"✓ Initialized with zero Alfvén fields and zero Hermite moments")
        print(f"  Initial Hermite energy: E_total = {E0:.6e}")
        print(f"  Hermite moments: M={M}")
        print(f"  Forcing will build up energy from zero initial condition")

    # ==========================================================================
    # Compute Timestep
    # ==========================================================================

    dt = compute_cfl_timestep(state=state, v_A=v_A, cfl_safety=cfl_safety)

    print(f"\n  Using dt = {dt:.4f} (CFL-limited)")
    print(f"  Total runtime: {total_time:.1f} time units = {int(total_time/dt)} timesteps")
    print(f"  Averaging starts at: {averaging_start:.1f} time units ({args.averaging_start:.1f} τ_A)")
    print(f"  Averaging duration: {total_time - averaging_start:.1f} time units ({args.total_time - args.averaging_start:.1f} τ_A)")

    print(f"\n  Collision rate at M: ν·dt = {nu * dt:.4f}")
    print(f"  Damping factor at M: exp(-ν·1^{hyper_n}·dt) = {np.exp(-nu * dt):.6f}")

    # ==========================================================================
    # Time Evolution with Forcing
    # ==========================================================================

    print("\n" + "-" * 70)
    print("Running forced evolution...")

    # Prepare output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Energy history tracking
    history = EnergyHistory()
    energy_values = []

    # Time-averaged Hermite spectra (accumulated during averaging window)
    hermite_spectrum_list = []
    averaging_started = False
    averaging_start_array_idx = None

    # Random key for forcing
    key = jax.random.PRNGKey(42)

    start_time = time.time()
    step = 0

    # Main loop
    while state.time < total_time:
        # Compute Hermite moment energy (NOT Alfvén energy, since z± = 0)
        E_m_instant = hermite_moment_energy(state)
        E_total = float(jnp.sum(E_m_instant))
        energy_values.append(E_total)

        if step % save_interval == 0:
            history.append(state)

        # Save checkpoint periodically
        if step > 0 and step % checkpoint_interval == 0:
            checkpoint_filename = output_dir / f"hermite_cascade_checkpoint_step{step}_t{state.time:.1f}.h5"
            save_checkpoint(state, str(checkpoint_filename))
            print(f"  💾 Saved checkpoint: {checkpoint_filename.name}")

        if step % save_interval == 0:
            # Start averaging when we reach averaging_start time
            if not averaging_started and state.time >= averaging_start:
                averaging_started = True
                averaging_start_array_idx = len(energy_values)
                print(f"\n  *** AVERAGING STARTED at step {step}, t={state.time:.2f} τ_A ***\n")

            # Collect Hermite spectra during averaging window
            if averaging_started:
                E_m = hermite_moment_energy(state)
                hermite_spectrum_list.append(np.array(E_m))

                # Periodically check steady-state status
                if step % steady_state_check_interval == 0 and len(energy_values) >= averaging_start_array_idx + steady_state_window:
                    recent_energy = energy_values[averaging_start_array_idx:]
                    is_steady = detect_steady_state(
                        energy_values,
                        window=min(steady_state_window, len(recent_energy)),
                        threshold=steady_state_threshold
                    )
                    energy_variation = (max(recent_energy) - min(recent_energy)) / np.mean(recent_energy) * 100
                    status_symbol = "✓" if is_steady else "✗"
                    print(f"  {status_symbol} Steady-state check: ΔE/⟨E⟩ = {energy_variation:.1f}% ({'PASS' if is_steady else 'FAIL'})")

                # Energy balance diagnostics (injection vs dissipation)
                if step % (steady_state_check_interval * 2) == 0 and step > 0:
                    # Compute collision dissipation rate
                    diss_rate = compute_collision_dissipation_rate(state, nu)

                    # Estimate injection rate from forcing parameters
                    # For white noise forcing: ε_inj ~ amplitude² × n_modes / dt
                    # This is approximate - actual rate varies with state
                    if args.single_k_mode:
                        n_forced_modes = 1
                    else:
                        # Estimate number of modes in shell [n_min, n_max]
                        n_forced_modes = len(forced_moments) * (n_force_max - n_force_min + 1) ** 3
                    inj_rate_est = (force_amplitude ** 2) * n_forced_modes / dt

                    # Balance ratio: >1 means energy builds up, <1 means decays
                    if diss_rate > 0:
                        balance_ratio = inj_rate_est / diss_rate
                        print(f"  ⚖️  Energy balance: Injection ≈ {inj_rate_est:.2e}, Dissipation = {diss_rate:.2e}, Ratio = {balance_ratio:.2f}")
                        if balance_ratio > 1.5:
                            print(f"      ⚠️  Injection >> Dissipation: Energy accumulating (reduce amplitude or increase nu)")
                        elif balance_ratio < 0.67:
                            print(f"      ⚠️  Dissipation >> Injection: Energy decaying (increase amplitude or reduce nu)")

            # Print progress
            if step % progress_print_interval == 0:
                # Check for NaN/Inf
                if not np.isfinite(E_total):
                    print(f"\n  ERROR: NaN/Inf detected at step {step}, t={state.time:.2f} τ_A")
                    print(f"         E_total = {E_total}")
                    print(f"         Terminating simulation.")
                    sys.exit(1)

                # Verify z± stays zero (critical for linear physics)
                max_z_plus = float(jnp.max(jnp.abs(state.z_plus)))
                max_z_minus = float(jnp.max(jnp.abs(state.z_minus)))

                phase = "[AVERAGING]" if averaging_started else "[SPIN-UP]"
                n_spectra = len(hermite_spectrum_list)
                print(f"  Step {step:5d}: t={state.time:.2f} τ_A, E={E_total:.4e}, max|z±|={max(max_z_plus, max_z_minus):.2e} {phase} (spectra: {n_spectra})")

        # Apply forcing to Hermite moments
        key, subkey = jax.random.split(key)
        if args.single_k_mode:
            # Force single k-mode (0,0,1) for clean linear physics test
            state, key = force_hermite_moments_specific(
                state,
                mode_triplets=[(0, 0, 1)],  # Fundamental k_z mode only
                amplitude=force_amplitude,
                dt=dt,
                key=subkey,
                forced_moments=forced_moments,
            )
        else:
            # Shell forcing (default)
            state, key = force_hermite_moments(
                state,
                amplitude=force_amplitude,
                n_min=n_force_min,
                n_max=n_force_max,
                dt=dt,
                key=subkey,
                forced_moments=forced_moments,
            )

        # Time step with collisions
        state = gandalf_step(
            state,
            dt=dt,
            eta=eta,
            nu=nu,
            v_A=v_A,
            hyper_r=hyper_r,
            hyper_n=hyper_n,
        )

        step += 1

    # Final summary
    if averaging_started and averaging_start_array_idx is not None:
        recent_energy = energy_values[averaging_start_array_idx:]
        if len(recent_energy) > 0:
            energy_variation = (max(recent_energy) - min(recent_energy)) / np.mean(recent_energy) * 100
            print(f"\n  *** EVOLUTION COMPLETE: {len(hermite_spectrum_list)} spectra collected ***")
            print(f"  *** Energy range: [{min(recent_energy):.2e}, {max(recent_energy):.2e}] ***")
            print(f"  *** Relative variation: {energy_variation:.1f}% (target: <10%) ***\n")

    elapsed = time.time() - start_time
    print(f"\n✓ Completed evolution")
    print(f"  Runtime: {elapsed:.1f} seconds ({elapsed/60:.2f} minutes)")

    # ==========================================================================
    # Compute Time-Averaged Hermite Spectrum
    # ==========================================================================

    print("\n" + "-" * 70)
    print("Computing time-averaged Hermite spectrum...")

    if len(hermite_spectrum_list) == 0:
        print("ERROR: No spectra collected! Steady state not reached.")
        return

    # Time-average
    E_m_avg = np.mean(hermite_spectrum_list, axis=0)
    E_m_std = np.std(hermite_spectrum_list, axis=0)

    averaging_duration = total_time - averaging_start
    print(f"✓ Averaged {len(hermite_spectrum_list)} spectra over {averaging_duration:.1f} τ_A")

    # Debug: print spectrum statistics
    print(f"  DEBUG: hermite_spectrum_list shape: {np.array(hermite_spectrum_list).shape}")
    print(f"  DEBUG: E_m_avg shape: {E_m_avg.shape}")
    print(f"  DEBUG: E_m_avg sum: {np.sum(E_m_avg)}")
    print(f"  DEBUG: E_m_avg has positive values: {np.any(E_m_avg > 0)}")
    print(f"  DEBUG: E_m_avg first 10 values: {E_m_avg[:10]}")

    # ==========================================================================
    # Fit Power Law
    # ==========================================================================

    print("\n" + "-" * 70)
    print("Fitting power law...")

    m_values = np.arange(M+1)

    # Fit range: skip m=0,1 (forced modes), fit inertial range
    m_fit_min = 2
    m_fit_max = min(16, M // 2)

    slope, intercept, r_squared, m_fit, E_fit_predicted = fit_power_law(
        m_values, E_m_avg, m_min=m_fit_min, m_max=m_fit_max
    )

    print(f"\n  Power law fit (m ∈ [{m_fit_min}, {m_fit_max}]):")
    print(f"    E_m ~ m^({slope:.3f})  (expected: -0.5)")
    print(f"    R² = {r_squared:.4f}  (quality of fit)")
    print(f"    Relative error: {abs(slope + 0.5) / 0.5 * 100:.1f}%")

    # Validation
    if abs(slope + 0.5) / 0.5 < 0.1:  # Within 10% of -0.5
        print(f"\n  ✓ VALIDATION PASSED: m^(-1/2) spectrum reproduced!")
    else:
        print(f"\n  ✗ VALIDATION MARGINAL: Power law slope deviates from -0.5")

    # ==========================================================================
    # Visualization
    # ==========================================================================

    print("\n" + "-" * 70)
    print("Creating visualizations...")

    fig = plt.figure(figsize=(14, 5))

    # -------------------------------------------------------------------------
    # Panel 1: Energy Evolution
    # -------------------------------------------------------------------------
    ax1 = plt.subplot(121)
    ax1.set_yscale('log')
    # Use Hermite energy, not Alfvén energy (z± = 0 so history.E_total is all zeros)
    times = np.array(history.times)
    energies = np.array(energy_values[:len(history.times)])  # Match history length
    ax1.plot(times, energies, 'k-', linewidth=2)

    # Mark averaging window
    if averaging_started:
        ax1.axvline(averaging_start, color='r', linestyle='--', linewidth=1.5,
                   label=f'Averaging starts (t={averaging_start:.1f})')
        ax1.axvspan(averaging_start, total_time,
                   color='red', alpha=0.1, label=f'Averaging window ({total_time - averaging_start:.1f} τ_A)')

    ax1.set_xlabel('Time [τ_A]', fontsize=12)
    ax1.set_ylabel('Total Energy', fontsize=12)
    ax1.set_title('Energy Evolution', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)

    # -------------------------------------------------------------------------
    # Panel 2: Time-Averaged Hermite Spectrum
    # -------------------------------------------------------------------------
    ax2 = plt.subplot(122)

    # Filter out zero values for log-log plot
    mask_nonzero = E_m_avg > 0
    m_plot = m_values[mask_nonzero]
    E_m_plot = E_m_avg[mask_nonzero]
    E_m_std_plot = E_m_std[mask_nonzero]

    # Hermite spectrum with error bars
    ax2.loglog(m_plot, E_m_plot, 'bo-', linewidth=2, markersize=6, label='Simulation')
    ax2.fill_between(m_plot, E_m_plot - E_m_std_plot, E_m_plot + E_m_std_plot,
                     color='b', alpha=0.2)

    # Reference slope m^(-1/2)
    m_ref = np.array([2.0, float(m_fit_max)])
    # Normalize to E_m at m=3 (index 2 in fit range, away from forcing)
    if len(m_fit) > 2 and m_fit[2] >= 3:
        idx_ref = np.where(m_fit == 3)[0]
        if len(idx_ref) > 0:
            E_ref = m_ref**(-0.5) * E_m_avg[3] / (3.0**(-0.5))
        else:
            E_ref = m_ref**(-0.5) * E_m_avg[m_fit[2]] / (m_fit[2]**(-0.5))
    else:
        E_ref = m_ref**(-0.5)
    ax2.loglog(m_ref, E_ref, 'k--', linewidth=2, label='m^(-1/2) reference')

    # Fitted power law
    if not np.isnan(slope):
        m_fit_plot = np.logspace(np.log10(m_fit_min), np.log10(m_fit_max), 50)
        E_fit_plot = 10**(intercept) * m_fit_plot**slope
        ax2.loglog(m_fit_plot, E_fit_plot, 'r-', linewidth=1.5, alpha=0.7,
                  label=f'Fit: m^({slope:.2f}), R²={r_squared:.3f}')

    # Highlight forcing range
    ax2.axvspan(0.5, 1.5, color='green', alpha=0.15, label='Forced: g₀, g₁')

    # Highlight fit range
    ax2.axvspan(m_fit_min, m_fit_max, color='orange', alpha=0.1, label=f'Fit range [{m_fit_min},{m_fit_max}]')

    ax2.set_xlabel('Hermite moment m', fontsize=12)
    ax2.set_ylabel('Energy E_m', fontsize=12)
    ax2.set_title('Hermite Moment Spectrum (Time-Averaged)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.legend(fontsize=9, loc='best')
    ax2.set_xlim(0.8, M)
    ax2.set_ylim(bottom=max(1e-5, np.min(E_m_plot[E_m_plot > 0]) * 0.1))

    plt.tight_layout()

    filename = f"hermite_cascade_{args.resolution}cubed_M{M}.png"
    filepath = output_dir / filename
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"✓ Saved figure: {filepath}")

    print("\n" + "=" * 70)
    print("Benchmark complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
