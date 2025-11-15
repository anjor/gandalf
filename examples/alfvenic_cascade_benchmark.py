#!/usr/bin/env python3
"""
Alfvénic Turbulent Cascade Benchmark (Thesis Section 2.6.3, Figure 2.2)

Reproduces the thesis benchmark showing k⊥^(-5/3) critical-balance spectrum
for kinetic and magnetic energy. Runs to steady state and time-averages
spectra over the final window.

Thesis parameters:
- Resolutions: 64³ and 128³
- Hyper-diffusion: r=4 and r=8 (thesis)
- This implementation:
  - 32³: r=2 (stable and practical for turbulence studies)
  - 64³: r=2 (r=4 yields instability despite no overflow: η·dt=0.1<<50)
  - 128³: r=2 (r=4 also unstable, not overflow-limited)
- Run to saturation (steady state)
- Time-averaged spectra over final window (default: 30-50 τ_A)

Expected runtime (default 50 τ_A):
- 32³:  ~5-10 minutes
- 64³:  ~15-25 minutes
- 128³: ~1-2 hours

Steady-State Considerations:
- True steady state requires energy injection = dissipation (plateau in E(t))
- Default runtime (50 τ_A) should achieve steady state for 32³ and 64³
- For 128³ or publication quality: use --total-time 100 for cleaner results
- Check "Steady-state check" output during run: target ΔE/⟨E⟩ < 2%
- Averaging over final 20 τ_A (30-50) provides better statistics than shorter windows

Acceptable Energy Variation During Averaging:
- Ideal: ΔE/⟨E⟩ < 2% (truly steady state, recommended for publication)
- Good: ΔE/⟨E⟩ < 5% (weak growth acceptable, spectral slopes reliable)
- Marginal: ΔE/⟨E⟩ < 10% (spectral shape qualitatively correct, quantitative error ~5-10%)
- Unacceptable: ΔE/⟨E⟩ > 10% (non-stationary, biased spectra, extend runtime)
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

from krmhd import (
    SpectralGrid3D,
    initialize_random_spectrum,
    gandalf_step,
    compute_cfl_timestep,
    energy as compute_energy,
    force_alfven_modes,
    force_alfven_modes_gandalf,
    force_alfven_modes_specific,
    compute_energy_injection_rate,
)
from krmhd.diagnostics import (
    EnergyHistory,
    energy_spectrum_perpendicular_kinetic,
    energy_spectrum_perpendicular_magnetic,
    plot_energy_history,
    compute_turbulence_diagnostics,
)
from krmhd.io import save_turbulence_diagnostics, save_checkpoint, load_checkpoint
from krmhd.forcing import force_alfven_modes_balanced


# Original GANDALF mode triplets (6 modes with low k_z = ±1)
# From original GANDALF Gandalf.in and forcing.cu
GANDALF_ORIGINAL_MODES = [
    (1, 0, 1),   # k_perp = 1, k_z = +1
    (0, 1, 1),   # k_perp = 1, k_z = +1
    (-1, 0, 1),  # k_perp = 1, k_z = +1
    (1, 1, -1),  # k_perp = sqrt(2), k_z = -1
    (0, 1, -1),  # k_perp = 1, k_z = -1
    (-1, 1, -1), # k_perp = sqrt(2), k_z = -1
]


def save_snapshot(snapshot_dir, step, state, k_perp, E_kin_perp, E_mag_perp,
                  energy_history, n_force_min, n_force_max, Lx):
    """
    Save intermediate spectrum snapshot (PNG + CSV).

    Args:
        snapshot_dir: Directory to save snapshots
        step: Current step number
        state: Current KRMHDState
        k_perp: Perpendicular wavenumber bins
        E_kin_perp: Kinetic energy spectrum
        E_mag_perp: Magnetic energy spectrum
        energy_history: EnergyHistory object with time series
        n_force_min, n_force_max: Forcing mode range
        Lx: Domain size (for mode number conversion)
    """
    # Convert k⊥ to mode numbers
    n_perp = k_perp * Lx / (2 * np.pi)
    time_tau_A = state.time

    # Save CSV files
    kinetic_csv = snapshot_dir / f"kinetic_t{time_tau_A:.1f}.csv"
    magnetic_csv = snapshot_dir / f"magnetic_t{time_tau_A:.1f}.csv"

    header = f"Alfvenic Cascade Snapshot - t={time_tau_A:.2f} tau_A, step={step}\nk_perp,E_spectrum,mode_number"

    np.savetxt(kinetic_csv, np.column_stack([k_perp, E_kin_perp, n_perp]),
               delimiter=',', header=header, comments='')
    np.savetxt(magnetic_csv, np.column_stack([k_perp, E_mag_perp, n_perp]),
               delimiter=',', header=header, comments='')

    # Create 3-panel snapshot plot
    fig = plt.figure(figsize=(15, 5))

    # Panel 1: Energy Evolution (up to current time)
    ax1 = plt.subplot(131)
    ax1.set_yscale('log')
    times = np.array(energy_history.times)
    energies = np.array(energy_history.E_total)
    ax1.plot(times, energies, 'k-', linewidth=2)
    ax1.set_xlabel('Time [τ_A]', fontsize=12)
    ax1.set_ylabel('Total Energy', fontsize=12)
    ax1.set_title(f'Energy Evolution (t={time_tau_A:.1f} τ_A)', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Panel 2: Kinetic Spectrum
    ax2 = plt.subplot(132)
    ax2.loglog(n_perp, E_kin_perp, 'b-', linewidth=2, label=f't={time_tau_A:.1f} τ_A')
    # Reference k⊥^(-5/3) line
    k_ref = np.array([2.0, 10.0])
    E_ref = k_ref**(-5/3) * E_kin_perp[1] / (n_perp[1]**(-5/3))
    ax2.loglog(k_ref, E_ref, 'k--', linewidth=1.5, label='n^(-5/3)')
    # Highlight forcing range
    ax2.axvspan(n_force_min, n_force_max, alpha=0.2, color='green', label=f'Forcing modes {n_force_min}-{n_force_max}')
    ax2.set_xlabel('Mode number n', fontsize=12)
    ax2.set_ylabel('E_kin(n)', fontsize=12)
    ax2.set_title('Kinetic Energy Spectrum', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Magnetic Spectrum
    ax3 = plt.subplot(133)
    ax3.loglog(n_perp, E_mag_perp, 'r-', linewidth=2, label=f't={time_tau_A:.1f} τ_A')
    # Reference k⊥^(-5/3) line
    E_ref_mag = k_ref**(-5/3) * E_mag_perp[1] / (n_perp[1]**(-5/3))
    ax3.loglog(k_ref, E_ref_mag, 'k--', linewidth=1.5, label='n^(-5/3)')
    # Highlight forcing range
    ax3.axvspan(n_force_min, n_force_max, alpha=0.2, color='green', label=f'Forcing modes {n_force_min}-{n_force_max}')
    ax3.set_xlabel('Mode number n', fontsize=12)
    ax3.set_ylabel('E_mag(n)', fontsize=12)
    ax3.set_title('Magnetic Energy Spectrum', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    snapshot_png = snapshot_dir / f"spectrum_t{time_tau_A:.1f}.png"
    plt.savefig(snapshot_png, dpi=100, bbox_inches='tight')
    plt.close(fig)


def detect_steady_state(energy_history, window=100, threshold=0.02, n_smooth=None):
    """
    Detect if system has reached steady state (energy plateau) - DIAGNOSTIC ONLY.

    This function is used for logging/monitoring purposes only and does NOT
    control simulation runtime. The simulation always runs for the fixed
    total_time specified by the user, regardless of steady-state status.

    Checks if energy has stopped growing by looking at the trend over
    a long window. True steady state requires energy to plateau, not
    just have small fluctuations.

    Args:
        energy_history: List of total energy values
        window: Number of recent points to check (default 100)
        threshold: Relative energy change threshold (default 2%)
        n_smooth: Number of points to average for smoothing (default: window//10, min 5)

    Returns:
        True if steady state detected (energy plateau), False otherwise

    Note:
        This is used only for informational logging during the run. To ensure
        steady state is achieved, users should increase --total-time or monitor
        the ΔE/⟨E⟩ values printed during averaging.
    """
    if len(energy_history) < window:
        return False

    recent = energy_history[-window:]
    # Average over n_smooth points to smooth out high-frequency fluctuations
    # while preserving low-frequency trends. Default: use 10% of window size
    # to adapt to different window lengths, with a minimum of 5 points.
    if n_smooth is None:
        n_smooth = max(5, window // 10)

    E_start = np.mean(recent[:n_smooth])   # Average of first n_smooth points in window
    E_end = np.mean(recent[-n_smooth:])    # Average of last n_smooth points in window

    if E_start == 0:
        return False

    # Check if energy has stopped growing (plateau)
    relative_change = abs(E_end - E_start) / E_start
    return relative_change < threshold


def main():
    """Run Alfvénic cascade benchmark to steady state."""

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Alfvénic Turbulent Cascade Benchmark')
    parser.add_argument('--resolution', type=int, default=64, choices=[32, 64, 128],
                        help='Grid resolution (32, 64, or 128)')
    parser.add_argument('--total-time', type=float, default=50.0,
                        help='Total simulation time in Alfvén times (default: 50)')
    parser.add_argument('--averaging-start', type=float, default=30.0,
                        help='When to start averaging in Alfvén times (default: 30)')
    parser.add_argument('--output-dir', type=str, default='examples/output',
                        help='Output directory for plots (default: examples/output)')
    parser.add_argument('--save-diagnostics', action='store_true',
                        help='Save detailed turbulence diagnostics for Issue #82 investigation')
    parser.add_argument('--diagnostic-interval', type=int, default=5,
                        help='Compute diagnostics every N steps (default: 5, lower = more frequent)')
    parser.add_argument('--save-spectra', action='store_true',
                        help='Save spectral data for detailed analysis and custom plotting')
    parser.add_argument('--snapshot-interval', type=int, default=0,
                        help='Save spectrum snapshots every N steps during averaging (0=disabled, default)')
    parser.add_argument('--eta', type=float, default=None,
                        help='Hyper-resistivity coefficient (overrides resolution default)')
    parser.add_argument('--force-amplitude', type=float, default=None,
                        help='Forcing amplitude (overrides resolution default)')
    parser.add_argument('--hyper-r', type=int, default=None,
                        help='Hyper-dissipation order (overrides resolution default)')
    parser.add_argument('--hyper-n', type=int, default=None, choices=[1, 2, 4],
                        help='Hyper-collision order (overrides resolution default)')
    parser.add_argument('--use-gandalf-forcing', action='store_true',
                        help='Use original GANDALF forcing formula (1/k_perp weighting with log-random modulation)')
    parser.add_argument('--use-specific-modes', action='store_true',
                        help='Force only 6 specific mode triplets matching original GANDALF (respects RMHD ordering)')
    parser.add_argument('--balanced-elsasser', action='store_true',
                        help='Force z⁺ and z⁻ independently in a low-|nz| perpendicular band (recommended for robust cascade)')
    parser.add_argument('--max-nz', type=int, default=1,
                        help='For balanced forcing: allow |nz| ≤ max_nz (default 1)')
    parser.add_argument('--include-nz0', action='store_true',
                        help='For balanced forcing: include kz=0 plane (default: exclude)')
    parser.add_argument('--correlation', type=float, default=0.0,
                        help='For balanced forcing: correlation between z⁺ and z⁻ forcing in [0,1). 0=independent')

    # Checkpoint configuration
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                        help='Directory for checkpoint files (default: output-dir/checkpoints)')
    parser.add_argument('--checkpoint-interval-time', type=float, default=None,
                        help='Save checkpoint every N Alfvén times (e.g., 20.0)')
    parser.add_argument('--checkpoint-interval-steps', type=int, default=None,
                        help='Save checkpoint every N steps (e.g., 5000)')
    parser.add_argument('--checkpoint-on-issues', action='store_true', default=True,
                        help='Auto-save checkpoint on CFL violation or energy spike (default: enabled)')
    parser.add_argument('--no-checkpoint-on-issues', dest='checkpoint_on_issues', action='store_false',
                        help='Disable auto-save on issues')
    parser.add_argument('--checkpoint-final', action='store_true', default=True,
                        help='Save final checkpoint at end of run (default: enabled)')
    parser.add_argument('--no-checkpoint-final', dest='checkpoint_final', action='store_false',
                        help='Disable final checkpoint')

    # Resume from checkpoint
    parser.add_argument('--resume-from', type=str, default=None,
                        help='Resume from checkpoint file (e.g., checkpoints/checkpoint_t050.0.h5)')

    args = parser.parse_args()

    print("=" * 70)
    print("Alfvénic Turbulent Cascade Benchmark (Thesis Section 2.6.3)")
    print("=" * 70)

    # ==========================================================================
    # CHECKPOINT SETUP
    # ==========================================================================

    # Determine checkpoint directory
    checkpoint_dir = None
    if args.checkpoint_dir is not None:
        checkpoint_dir = Path(args.checkpoint_dir)
    elif args.checkpoint_interval_time or args.checkpoint_interval_steps or args.checkpoint_final:
        # Use default: output_dir/checkpoints
        checkpoint_dir = Path(args.output_dir) / "checkpoints"

    # Create checkpoint directory if needed
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint if resuming
    resumed_state = None
    resumed_grid = None
    resumed_metadata = None
    initial_time = 0.0
    initial_step = 0

    if args.resume_from is not None:
        print("\n" + "=" * 70)
        print(f"RESUMING FROM CHECKPOINT: {args.resume_from}")
        print("=" * 70)

        resumed_state, resumed_grid, resumed_metadata = load_checkpoint(args.resume_from)
        initial_time = float(resumed_state.time)
        initial_step = int(resumed_metadata.get('step', 0))

        print(f"✓ Loaded checkpoint from t={initial_time:.2f} τ_A (step {initial_step})")
        print(f"  Grid: {resumed_grid.Nx}×{resumed_grid.Ny}×{resumed_grid.Nz}")
        print(f"  Checkpoint saved: {resumed_metadata.get('timestamp', 'unknown')}")

        # Print original parameters from checkpoint
        print(f"\n  Original parameters from checkpoint:")
        for key in ['eta', 'nu', 'hyper_r', 'hyper_n', 'force_amplitude', 'v_A']:
            if key in resumed_metadata:
                print(f"    {key}: {resumed_metadata[key]}")

    # ==========================================================================
    # PARAMETERS
    # ==========================================================================

    # Resolution-dependent parameters
    # If resuming, use grid from checkpoint; otherwise use command-line args
    if resumed_grid is not None:
        Nx = resumed_grid.Nx
        Ny = resumed_grid.Ny
        Nz = resumed_grid.Nz
        if args.resolution != Nx:
            print(f"\n  ⚠️  WARNING: --resolution={args.resolution} ignored (using checkpoint grid {Nx}×{Ny}×{Nz})")
    else:
        Nx = Ny = args.resolution
        Nz = args.resolution  # Cubic grid (matching original GANDALF)

    # With NORMALIZED hyper-dissipation (matching original GANDALF):
    # - Constraint is eta·dt < 50 (independent of resolution!)
    # - Using r=2 for practical stability with clean inertial range
    # - r=2 provides moderate dissipation range (broader than r=4, narrower than r=1)
    if args.resolution == 32:
        eta = 1.0         # Moderate dissipation for r=2
        hyper_r = 2       # Practical choice: stable with clean inertial range
        hyper_n = 2
    elif args.resolution == 64:
        # 64³ ANOMALY (Issue #82): Requires unusually strong dissipation OR very weak forcing
        # After fixing Issue #97, we use weak forcing (0.005) with moderate dissipation
        # Alternative: eta=20.0 with amplitude=0.01 (documented in CLAUDE.md)
        eta = 2.0         # Moderate dissipation (2× stronger than 32³)
        hyper_r = 2       # Stable and practical for turbulence studies
        hyper_n = 2       # Fourth-order dissipation: exp(-η(∇²)^2)
    else:  # 128³
        eta = 2.0         # Stronger for high resolution
        hyper_r = 2       # Maintains stability at high resolution
        hyper_n = 2

    # Collision frequency: Set to 0 since Hermite moments (slow modes) are passive
    # and don't affect Alfvén dynamics in the RMHD limit
    nu = 0.0

    Lx = Ly = Lz = 1.0    # Unit box (thesis convention)

    # Physics parameters
    v_A = 1.0             # Alfvén velocity
    beta_i = 1.0          # Ion plasma beta

    # Forcing parameters (inject energy at large scales)
    # Force narrow range at large scales for clean inertial range development
    # With r=2 hyper-dissipation, need gentle forcing to avoid numerical instability
    # NOTE: After fixing Issue #97, forcing now works correctly! Old amplitudes were too strong
    # because the forcing mask was empty (no modes satisfied k_min=2.0 for L=1.0 domain).
    # Now that modes n=1-2 are actually forced, we need much weaker amplitudes.
    if args.resolution == 32:
        force_amplitude = 0.01   # Weak forcing for stability (5× weaker than before Issue #97)
    elif args.resolution == 64:
        force_amplitude = 0.005  # Very weak for 64³ anomaly (Issue #82)
    else:  # 128³
        force_amplitude = 0.01   # Weak forcing for high resolution

    # NARROW FORCING RANGE: modes n ∈ [1, 2] for clean scale separation
    # Mode numbers are integers (n=1 is fundamental, n=2 is second harmonic, etc.)
    # See https://github.com/anjor/gandalf/issues/97 (now resolved!)
    # Narrowed from n=1-3 to n=1-2 for better stability
    n_force_min = 1  # Largest scale (fundamental mode)
    n_force_max = 2  # Narrow injection range (was 3, reduced for stability)

    # Override parameters with command-line arguments if provided
    # Track which parameters were overridden (for resume info)
    overridden_params = {}
    if args.eta is not None:
        if resumed_metadata and 'eta' in resumed_metadata and resumed_metadata['eta'] != args.eta:
            overridden_params['eta'] = (resumed_metadata['eta'], args.eta)
        eta = args.eta
    if args.force_amplitude is not None:
        if resumed_metadata and 'force_amplitude' in resumed_metadata and resumed_metadata['force_amplitude'] != args.force_amplitude:
            overridden_params['force_amplitude'] = (resumed_metadata['force_amplitude'], args.force_amplitude)
        force_amplitude = args.force_amplitude
    if args.hyper_r is not None:
        if resumed_metadata and 'hyper_r' in resumed_metadata and resumed_metadata['hyper_r'] != args.hyper_r:
            overridden_params['hyper_r'] = (resumed_metadata['hyper_r'], args.hyper_r)
        hyper_r = args.hyper_r
    if args.hyper_n is not None:
        if resumed_metadata and 'hyper_n' in resumed_metadata and resumed_metadata['hyper_n'] != args.hyper_n:
            overridden_params['hyper_n'] = (resumed_metadata['hyper_n'], args.hyper_n)
        hyper_n = args.hyper_n

    # Print parameter overrides if resuming
    if resumed_metadata and overridden_params:
        print(f"\n  Parameter overrides:")
        for param, (old_val, new_val) in overridden_params.items():
            print(f"    {param}: {old_val} → {new_val}")

    # Initial condition (weak, let forcing drive the turbulence)
    alpha = 5.0 / 3.0     # k^(-5/3) initial spectrum
    amplitude = 0.05      # Weak initial amplitude
    k_min = 1.0
    k_max = 15.0

    # Time integration
    tau_A = Lz / v_A  # Alfvén crossing time (= 1.0 for unit box)
    total_time = args.total_time * tau_A  # Total simulation time
    averaging_start = args.averaging_start * tau_A  # When to start averaging
    cfl_safety = 0.3
    save_interval = 10

    # Diagnostic intervals (for logging/monitoring only, doesn't affect physics)
    steady_state_check_interval = 50  # Check every N steps during averaging
    progress_print_interval = 50      # Print progress every N steps (includes NaN detection)
    steady_state_window = 100
    steady_state_threshold = 0.02  # 2% relative change

    # Warn user if runtime may be insufficient for steady state
    if args.total_time < 30.0:
        print("\n" + "!" * 70)
        print("⚠️  WARNING: Runtime may be insufficient for true steady state!")
        print(f"    Current: {args.total_time} τ_A")
        print("    Recommended: ≥50 τ_A (default) or ≥100 τ_A for publication quality")
        print("    Use --total-time 50 or higher for reliable results")
        print("    Monitor 'Steady-state check' output during run (target: ΔE/⟨E⟩ < 2%)")
        print("!" * 70)

    print(f"\nGrid: {Nx} × {Ny} × {Nz}")
    print(f"Domain: [{Lx:.1f}, {Ly:.1f}, {Lz:.1f}]")
    print(f"Physics: v_A={v_A}, η={eta:.1e} (ν=0: collisionless)")
    print(f"Hyper-dissipation: r={hyper_r}, n={hyper_n}")

    if args.use_specific_modes:
        forcing_mode = "Specific modes (original GANDALF)"
        print(f"Forcing: {forcing_mode}, amplitude={force_amplitude}")
        print(f"  Forced modes: {len(GANDALF_ORIGINAL_MODES)} triplets with |k_z|=1 (respects RMHD ordering)")
        print(f"  {GANDALF_ORIGINAL_MODES}")
    else:
        forcing_mode = "GANDALF (1/k_perp + log-random)" if args.use_gandalf_forcing else "Gaussian white noise"
        print(f"Forcing: {forcing_mode}, amplitude={force_amplitude}, modes n ∈ [{n_force_min}, {n_force_max}]")

    print(f"Evolution: Run for {args.total_time:.1f} τ_A total, average last {args.total_time - args.averaging_start:.1f} τ_A")
    print(f"CFL safety: {cfl_safety}")

    # Diagnostic: Print k_perp_max (dt-dependent diagnostics moved below)
    # This helps debug resolution-dependent instabilities (Issue #82)
    print(f"\n--- Normalized Hyper-Dissipation Diagnostics ---")
    # k_perp_max is computed at 2/3 dealiasing boundary: (N-1)//3
    idx_max_x = (Nx - 1) // 3
    idx_max_y = (Ny - 1) // 3
    k_perp_max_x = (2 * np.pi / Lx) * idx_max_x
    k_perp_max_y = (2 * np.pi / Ly) * idx_max_y
    k_perp_max = np.sqrt(k_perp_max_x**2 + k_perp_max_y**2)
    print(f"k_perp_max (2/3 boundary): {k_perp_max:.2f} (idx={idx_max_x}, {idx_max_y})")

    if args.resolution == 32:
        print(f"\nEstimated runtime: ~2-5 minutes")
    elif args.resolution == 64:
        print(f"\nEstimated runtime: ~5-10 minutes")
    else:
        print(f"\nEstimated runtime: ~30-60 minutes")

    # ==========================================================================
    # Initialize Grid and State
    # ==========================================================================

    print("\n" + "-" * 70)
    if resumed_state is not None:
        print("Using state from checkpoint...")
        grid = resumed_grid
        state = resumed_state
        E0 = compute_energy(state)['total']
        print(f"✓ Loaded state from checkpoint")
        print(f"  Time: t = {state.time:.2f} τ_A")
        print(f"  Energy: E_total = {E0:.6e}")
        print(f"  Grid: {Nx}×{Ny}×{Nz}")
    else:
        print("Initializing...")
        grid = SpectralGrid3D.create(Nx=Nx, Ny=Ny, Nz=Nz, Lx=Lx, Ly=Ly, Lz=Lz)
        print(f"✓ Created {Nx}×{Ny}×{Nz} spectral grid")

        # Weak initial spectrum
        M = 10  # Number of Hermite moments
        state = initialize_random_spectrum(
            grid,
            M=M,
            alpha=alpha,
            amplitude=amplitude,
            k_min=k_min,
            k_max=k_max,
            v_th=1.0,
            beta_i=beta_i,
            seed=42
        )

        E0 = compute_energy(state)['total']
        print(f"✓ Initialized weak k^(-{alpha:.2f}) spectrum")
        print(f"  Initial energy: E_total = {E0:.6e}")

    # ==========================================================================
    # Compute Timestep
    # ==========================================================================

    dt = compute_cfl_timestep(
        state=state,
        v_A=v_A,
        cfl_safety=cfl_safety
    )

    print(f"\n  Using dt = {dt:.4f} (CFL-limited)")
    print(f"  Total runtime: {total_time:.1f} time units = {int(total_time/dt)} timesteps")
    print(f"  Averaging starts at: {averaging_start:.1f} time units ({args.averaging_start:.1f} τ_A)")
    print(f"  Averaging duration: {total_time - averaging_start:.1f} time units ({args.total_time - args.averaging_start:.1f} τ_A)")

    # Print dt-dependent dissipation diagnostics (now that dt is computed)
    print(f"\n  Dissipation rate at k_max: η·dt = {eta * dt:.4f} (constraint: < 50)")
    print(f"  Normalized damping factor: exp(-η·1^{hyper_r}·dt) = {np.exp(-eta * dt):.6f} (at k_max)")
    print(f"  ------------------------------------------------")

    # ==========================================================================
    # Time Evolution with Forcing
    # ==========================================================================

    print("\n" + "-" * 70)
    print("Running forced evolution...")

    # Prepare output/snapshot directories BEFORE the loop so snapshots can be saved
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir = None
    if args.snapshot_interval > 0:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_dir = output_dir / f"snapshots_{args.resolution}cubed_{timestamp}"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        print(f"Snapshot directory: {snapshot_dir}")
        print(f"Snapshots will be saved every {args.snapshot_interval} steps during averaging")

    # Energy history tracking
    history = EnergyHistory()
    energy_values = []  # For steady-state logging

    # Store energy injection rates
    injection_rates = []

    # Turbulence diagnostics for Issue #82 investigation
    diagnostics_list = [] if args.save_diagnostics else None
    if args.save_diagnostics:
        print("\n" + "=" * 70)
        print("⚠️  DIAGNOSTIC MODE ENABLED (Issue #82 Investigation)")
        print("=" * 70)
        print(f"  Computing turbulence diagnostics every {args.diagnostic_interval} steps")
        print("  Tracking: max_velocity, CFL, max_nonlinear, energy_highk, critical_balance")
        print("  Output will be saved to: turbulence_diagnostics_{resolution}.h5")
        print("  This will increase runtime by ~10-15%")
        print("=" * 70 + "\n")

    # Time-averaged spectra (accumulated during averaging window)
    spectrum_kinetic_list = []
    spectrum_magnetic_list = []
    averaging_started = False
    averaging_start_step = None

    # Random key for forcing
    key = jax.random.PRNGKey(42)

    start_time = time.time()
    step = initial_step  # Start from checkpoint step if resuming

    # Track last checkpoint times for periodic saving
    last_checkpoint_time = initial_time
    last_checkpoint_step = initial_step

    # Main loop: run until we reach total_time
    while state.time < total_time:
        # Compute total energy (needed every step for steady-state tracking)
        E_dict = compute_energy(state)
        E_total = E_dict['total']

        # Track history (needed for steady-state detection)
        energy_values.append(E_total)

        # Compute turbulence diagnostics for Issue #82 investigation or checkpoint-on-issues
        # Compute diagnostics if either saving them OR need them for checkpoint-on-issues
        compute_diagnostics = (args.save_diagnostics or args.checkpoint_on_issues) and step % args.diagnostic_interval == 0
        if compute_diagnostics:
            diag = compute_turbulence_diagnostics(state, dt=dt, v_A=v_A)

            # Save diagnostics if requested
            if args.save_diagnostics:
                diagnostics_list.append(diag)

            # Print warning if CFL > 1.0 or max_velocity is very large
            cfl_violation = diag.cfl_number > 1.0
            high_velocity = diag.max_velocity > 100.0
            if cfl_violation:
                print(f"  ⚠️  CFL VIOLATION at step {step}, t={state.time:.2f}: CFL = {diag.cfl_number:.3f}")
            if high_velocity:
                print(f"  ⚠️  HIGH VELOCITY at step {step}, t={state.time:.2f}: max_vel = {diag.max_velocity:.2e}")

            # Checkpoint on issues if enabled
            if args.checkpoint_on_issues and checkpoint_dir is not None:
                if cfl_violation or high_velocity:
                    issue_type = "CFL_VIOLATION" if cfl_violation else "HIGH_VELOCITY"
                    checkpoint_filename = f"checkpoint_t{state.time:06.1f}_{issue_type}.h5"
                    checkpoint_path = checkpoint_dir / checkpoint_filename
                    checkpoint_metadata = {
                        'step': int(step),
                        'eta': float(eta),
                        'nu': float(nu),
                        'hyper_r': int(hyper_r),
                        'hyper_n': int(hyper_n),
                        'force_amplitude': float(force_amplitude),
                        'v_A': float(v_A),
                        'dt': float(dt),
                        'issue_type': issue_type,
                        'cfl_number': float(diag.cfl_number),
                        'max_velocity': float(diag.max_velocity),
                        'description': f'Auto-saved on {issue_type} at t={state.time:.2f} τ_A'
                    }
                    save_checkpoint(state, str(checkpoint_path), metadata=checkpoint_metadata, overwrite=True)
                    print(f"  💾 Auto-saved checkpoint: {checkpoint_filename}")

        # Periodic checkpoint saving (by time)
        if (checkpoint_dir is not None and
            args.checkpoint_interval_time is not None and
            state.time - last_checkpoint_time >= args.checkpoint_interval_time):
            checkpoint_filename = f"checkpoint_t{state.time:06.1f}.h5"
            checkpoint_path = checkpoint_dir / checkpoint_filename
            checkpoint_metadata = {
                'step': int(step),
                'eta': float(eta),
                'nu': float(nu),
                'hyper_r': int(hyper_r),
                'hyper_n': int(hyper_n),
                'force_amplitude': float(force_amplitude),
                'v_A': float(v_A),
                'dt': float(dt),
                'description': f'Periodic checkpoint at t={state.time:.2f} τ_A'
            }
            save_checkpoint(state, str(checkpoint_path), metadata=checkpoint_metadata, overwrite=True)
            last_checkpoint_time = state.time
            print(f"  💾 Saved periodic checkpoint (by time): {checkpoint_filename}")

        # Periodic checkpoint saving (by steps)
        if (checkpoint_dir is not None and
            args.checkpoint_interval_steps is not None and
            step - last_checkpoint_step >= args.checkpoint_interval_steps):
            checkpoint_filename = f"checkpoint_step{step:07d}.h5"
            checkpoint_path = checkpoint_dir / checkpoint_filename
            checkpoint_metadata = {
                'step': int(step),
                'eta': float(eta),
                'nu': float(nu),
                'hyper_r': int(hyper_r),
                'hyper_n': int(hyper_n),
                'force_amplitude': float(force_amplitude),
                'v_A': float(v_A),
                'dt': float(dt),
                'description': f'Periodic checkpoint at step {step}'
            }
            save_checkpoint(state, str(checkpoint_path), metadata=checkpoint_metadata, overwrite=True)
            last_checkpoint_step = step
            print(f"  💾 Saved periodic checkpoint (by steps): {checkpoint_filename}")

        if step % save_interval == 0:
            history.append(state)

            # Start averaging when we reach averaging_start time
            if not averaging_started and state.time >= averaging_start:
                averaging_started = True
                averaging_start_step = step
                print(f"\n  *** AVERAGING STARTED at step {step}, t={state.time:.2f} τ_A ***\n")

            # Collect spectra during averaging window
            if averaging_started:
                k_perp, E_kin_perp = energy_spectrum_perpendicular_kinetic(state)
                k_perp, E_mag_perp = energy_spectrum_perpendicular_magnetic(state)
                spectrum_kinetic_list.append(E_kin_perp)
                spectrum_magnetic_list.append(E_mag_perp)

                # Save snapshot if enabled and at snapshot interval
                if snapshot_dir is not None and step % args.snapshot_interval == 0:
                    save_snapshot(snapshot_dir, step, state, k_perp, E_kin_perp, E_mag_perp,
                                  history, n_force_min, n_force_max, Lx)
                    print(f"  📸 Snapshot saved: t={state.time:.1f} τ_A")

                # Periodically check and log steady-state status
                if step % steady_state_check_interval == 0 and len(energy_values) >= averaging_start_step + steady_state_window:
                    recent_energy = energy_values[averaging_start_step:]
                    is_steady = detect_steady_state(
                        energy_values,
                        window=min(steady_state_window, len(recent_energy)),
                        threshold=steady_state_threshold
                    )
                    energy_variation = (max(recent_energy) - min(recent_energy)) / np.mean(recent_energy) * 100
                    status_symbol = "✓" if is_steady else "✗"
                    print(f"  {status_symbol} Steady-state check: ΔE/⟨E⟩ = {energy_variation:.1f}% ({'PASS' if is_steady else 'FAIL'})")

            # Print progress (also check for NaN/Inf at this interval)
            if step % progress_print_interval == 0:
                # Check for NaN/Inf (robustness check)
                if not np.isfinite(E_total):
                    # Extract breakdown for error message
                    E_kin = E_dict['kinetic']
                    E_mag = E_dict['magnetic']
                    print(f"\n  ERROR: NaN/Inf detected at step {step}, t={state.time:.2f} τ_A")
                    print(f"         E_total = {E_total}, E_kin = {E_kin}, E_mag = {E_mag}")
                    print(f"         Terminating simulation immediately.")
                    sys.exit(1)  # Exit immediately on NaN to prevent wasted computation

                # Extract energy breakdown for progress display (only when printing)
                E_mag = E_dict['magnetic']
                mag_frac = E_mag / E_total if E_total > 0 else 0
                avg_inj = np.mean(injection_rates[-50:]) if len(injection_rates) >= 50 else 0
                phase = "[AVERAGING]" if averaging_started else "[SPIN-UP]"
                n_spectra = len(spectrum_kinetic_list)
                # Compute cross-helicity proxy σ_c from Elsasser gradient energies
                # E_total = 0.25 ∫ (|∇z+|² + |∇z-|²) dx
                grid = state.grid
                kx_3d = grid.kx[jnp.newaxis, jnp.newaxis, :]
                ky_3d = grid.ky[jnp.newaxis, :, jnp.newaxis]
                k_perp_sq = kx_3d**2 + ky_3d**2
                kx_zero = (kx_3d == 0.0)
                kx_nyq = (kx_3d == grid.Nx // 2) if (grid.Nx % 2 == 0) else jnp.zeros_like(kx_3d, dtype=bool)
                kx_mid = ~(kx_zero | kx_nyq)
                dbl = jnp.where(kx_mid, 2.0, 1.0)
                N_perp = grid.Nx * grid.Ny
                Ezp = 0.25 * (1.0 / N_perp) * jnp.sum(dbl * k_perp_sq * jnp.abs(state.z_plus)**2).real
                Ezm = 0.25 * (1.0 / N_perp) * jnp.sum(dbl * k_perp_sq * jnp.abs(state.z_minus)**2).real
                sigma_c = float((Ezp - Ezm) / (Ezp + Ezm + 1e-30))

                print(f"  Step {step:5d}: t={state.time:.2f} τ_A, "
                      f"E={E_total:.4e}, f_mag={mag_frac:.3f}, σ_c={sigma_c:+.3f}, "
                      f"⟨ε_inj⟩={avg_inj:.2e} {phase} (spectra: {n_spectra})")

        # Apply forcing
        state_before = state
        key, subkey = jax.random.split(key)
        if args.balanced_elsasser:
            # Force independent z⁺/z⁻ in a perpendicular band, restrict to |nz| ≤ max_nz
            state, key = force_alfven_modes_balanced(
                state,
                amplitude=force_amplitude,
                n_min=n_force_min,
                n_max=n_force_max,
                dt=dt,
                key=subkey,
                max_nz=args.max_nz,
                include_nz0=args.include_nz0,
                correlation=args.correlation,
            )
        elif args.use_specific_modes:
            # Force only 6 specific mode triplets (original GANDALF)
            # This respects RMHD ordering k⊥ >> k∥ by forcing only low-k_z modes
            state, key = force_alfven_modes_specific(
                state,
                mode_triplets=GANDALF_ORIGINAL_MODES,
                fampl=force_amplitude,  # Interpreted as fampl for specific forcing
                dt=dt,
                key=subkey
            )
        elif args.use_gandalf_forcing:
            # Use original GANDALF forcing formula (1/k_perp weighting with log-random modulation)
            # Forces all modes in band [n_min, n_max] - can include high k_z
            state, key = force_alfven_modes_gandalf(
                state,
                fampl=force_amplitude,  # Interpreted as fampl for GANDALF forcing
                n_min=n_force_min,
                n_max=n_force_max,
                dt=dt,
                key=subkey
            )
        else:
            # Use Gaussian white noise forcing (default)
            state, key = force_alfven_modes(
                state,
                amplitude=force_amplitude,
                n_min=n_force_min,
                n_max=n_force_max,
                dt=dt,
                key=subkey
            )

        # Compute energy injection rate
        eps_inj = compute_energy_injection_rate(state_before, state, dt)
        injection_rates.append(float(eps_inj))

        # Time step with hyper-dissipation
        state = gandalf_step(
            state,
            dt=dt,
            eta=eta,
            nu=nu,
            v_A=v_A,
            hyper_r=hyper_r,
            hyper_n=hyper_n
        )

        step += 1

    # ==========================================================================
    # Save Final Checkpoint
    # ==========================================================================

    if args.checkpoint_final and checkpoint_dir is not None:
        checkpoint_filename = f"checkpoint_final_t{state.time:06.1f}.h5"
        checkpoint_path = checkpoint_dir / checkpoint_filename
        checkpoint_metadata = {
            'step': int(step),
            'eta': float(eta),
            'nu': float(nu),
            'hyper_r': int(hyper_r),
            'hyper_n': int(hyper_n),
            'force_amplitude': float(force_amplitude),
            'v_A': float(v_A),
            'dt': float(dt),
            'total_time': float(total_time),
            'description': f'Final checkpoint at t={state.time:.2f} τ_A (end of run)'
        }
        save_checkpoint(state, str(checkpoint_path), metadata=checkpoint_metadata, overwrite=True)
        print(f"\n💾 Saved final checkpoint: {checkpoint_filename}")

    # Final summary
    if averaging_started:
        recent_energy = energy_values[averaging_start_step:]
        energy_variation = (max(recent_energy) - min(recent_energy)) / np.mean(recent_energy) * 100
        print(f"\n  *** EVOLUTION COMPLETE: {len(spectrum_kinetic_list)} spectra collected ***")
        print(f"  *** Energy range during averaging: [{min(recent_energy):.2e}, {max(recent_energy):.2e}] ***")
        print(f"  *** Relative variation: {energy_variation:.1f}% (target: <10% for good statistics) ***\n")

    elapsed = time.time() - start_time

    print(f"\n✓ Completed evolution")
    print(f"  Runtime: {elapsed:.1f} seconds ({elapsed/60:.2f} minutes)")

    # ==========================================================================
    # Compute Time-Averaged Spectra
    # ==========================================================================

    print("\n" + "-" * 70)
    print("Computing time-averaged spectra...")

    if len(spectrum_kinetic_list) == 0:
        print("ERROR: No spectra collected! Steady state not reached.")
        return

    # Time-average
    E_kin_avg = np.mean(spectrum_kinetic_list, axis=0)
    E_mag_avg = np.mean(spectrum_magnetic_list, axis=0)

    # Standard deviation for error bars
    E_kin_std = np.std(spectrum_kinetic_list, axis=0)
    E_mag_std = np.std(spectrum_magnetic_list, axis=0)

    averaging_duration = total_time - averaging_start
    print(f"✓ Averaged {len(spectrum_kinetic_list)} spectra over {averaging_duration:.1f} τ_A")

    # ==========================================================================
    # Final Diagnostics
    # ==========================================================================

    print("\n" + "-" * 70)
    print("Final diagnostics...")

    E_final_dict = compute_energy(state)
    E_final = E_final_dict['total']
    E_mag_final = E_final_dict['magnetic']
    E_kin_final = E_final_dict['kinetic']

    total_injection = np.trapz(injection_rates, dx=dt)
    avg_injection = np.mean(injection_rates[len(injection_rates)//2:])  # Second half

    print(f"  Final energy: E_total = {E_final:.6e}")
    print(f"  Energy change: ΔE = {E_final - E0:.6e}")
    print(f"  Total injection: ∫ε_inj dt = {total_injection:.6e}")
    print(f"  Kinetic fraction: {E_kin_final / E_final:.3f}")
    print(f"  Magnetic fraction: {E_mag_final / E_final:.3f}")
    print(f"  Average injection rate (late time): ⟨ε_inj⟩ = {avg_injection:.3e}")

    # Analyze spectrum slope
    # Fit k⊥^(-5/3) over a MODE-NUMBER inertial range n⊥ ∈ [n_min, n_max]
    # This avoids confusion with 2π factors and matches thesis convention (L=1 ⇒ k=2πn).
    n_perp = k_perp * Lx / (2 * np.pi)
    if args.resolution == 32:
        n_fit_min, n_fit_max = 3.0, 8.0
    elif args.resolution == 64:
        n_fit_min, n_fit_max = 3.0, 12.0
    else:
        n_fit_min, n_fit_max = 3.0, 20.0

    n_mask = (n_perp >= n_fit_min) & (n_perp <= n_fit_max)
    if np.sum(n_mask) > 5:
        k_fit = k_perp[n_mask]
        E_kin_fit = E_kin_avg[n_mask]
        E_mag_fit = E_mag_avg[n_mask]

        # Log-log linear fit (use relative floor for numerical stability)
        log_k = np.log10(k_fit)
        floor_kin = 1e-12 * np.max(E_kin_fit)
        floor_mag = 1e-12 * np.max(E_mag_fit)
        log_E_kin = np.log10(np.maximum(E_kin_fit, floor_kin))
        log_E_mag = np.log10(np.maximum(E_mag_fit, floor_mag))

        slope_kin = np.polyfit(log_k, log_E_kin, 1)[0]
        slope_mag = np.polyfit(log_k, log_E_mag, 1)[0]

        print(f"\n  Inertial range slopes (n⊥ ∈ [{n_fit_min:.0f}, {n_fit_max:.0f}]):")
        print(f"    Kinetic:  k⊥^({slope_kin:.2f})  (expected: -5/3 = -1.67)")
        print(f"    Magnetic: k⊥^({slope_mag:.2f})  (expected: -5/3 = -1.67)")

    # ==========================================================================
    # Visualization
    # ==========================================================================

    print("\n" + "-" * 70)
    print("Creating visualizations...")

    # Output directory prepared earlier; re-announce for clarity
    print(f"Output directory: {output_dir}")
    if snapshot_dir is not None:
        print(f"Snapshot directory: {snapshot_dir}")
        print(f"Snapshots will be saved every {args.snapshot_interval} steps during averaging")

    # Convert k⊥ to mode numbers: n = k⊥ L / (2π)
    n_perp = k_perp * Lx / (2 * np.pi)
    # n_force_min and n_force_max already defined as integers above

    # Create figure with three panels
    fig = plt.figure(figsize=(15, 5))

    # -------------------------------------------------------------------------
    # Panel 1: Energy Evolution
    # -------------------------------------------------------------------------
    ax1 = plt.subplot(131)
    ax1.set_yscale('log')  # Logarithmic y-axis to clearly show exponential vs linear growth
    times = np.array(history.times)
    energies = np.array(history.E_total)
    ax1.plot(times, energies, 'k-', linewidth=2)

    # Mark averaging window
    if averaging_started:
        ax1.axvline(averaging_start, color='r', linestyle='--', linewidth=1.5,
                   label=f'Averaging starts (t={averaging_start:.1f})')
        ax1.axvspan(averaging_start, total_time,
                   color='red', alpha=0.1, label=f'Averaging window ({total_time - averaging_start:.1f} time units)')

    ax1.set_xlabel('Time [τ_A]', fontsize=12)
    ax1.set_ylabel('Total Energy', fontsize=12)
    ax1.set_title('Energy Evolution', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)

    # -------------------------------------------------------------------------
    # Panel 2: Time-Averaged Kinetic Spectrum
    # -------------------------------------------------------------------------
    ax2 = plt.subplot(132)

    # Kinetic spectrum with error bars
    ax2.loglog(n_perp, E_kin_avg, 'b-', linewidth=2, label='Kinetic')
    ax2.fill_between(n_perp, E_kin_avg - E_kin_std, E_kin_avg + E_kin_std,
                     color='b', alpha=0.2)

    # Reference slope n^(-5/3) starting at inertial-range minimum
    if args.resolution == 32:
        n_ref = np.array([2.0, 8.0])
    elif args.resolution == 64:
        n_ref = np.array([3.0, 12.0])
    else:
        n_ref = np.array([3.0, 20.0])
    E_ref = 0.5 * n_ref**(-5.0/3.0)
    ax2.loglog(n_ref, E_ref, 'k--', linewidth=1.5, label='n^(-5/3)')

    # Forcing range
    ax2.axvspan(n_force_min, n_force_max, color='green', alpha=0.1,
               label='Forcing modes 1-2')

    ax2.set_xlabel('Mode number n', fontsize=12)
    ax2.set_ylabel('E_kin(n)', fontsize=12)
    ax2.set_title('Kinetic Energy Spectrum (Time-Averaged)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.legend(fontsize=10)
    ax2.set_xlim(0.5, n_perp[-1])
    ax2.set_ylim(1e-3, None)

    # -------------------------------------------------------------------------
    # Panel 3: Time-Averaged Magnetic Spectrum
    # -------------------------------------------------------------------------
    ax3 = plt.subplot(133)

    # Magnetic spectrum with error bars
    ax3.loglog(n_perp, E_mag_avg, 'r-', linewidth=2, label='Magnetic')
    ax3.fill_between(n_perp, E_mag_avg - E_mag_std, E_mag_avg + E_mag_std,
                     color='r', alpha=0.2)

    # Reference slope n^(-5/3)
    ax3.loglog(n_ref, E_ref, 'k--', linewidth=1.5, label='n^(-5/3)')

    # Forcing range
    ax3.axvspan(n_force_min, n_force_max, color='green', alpha=0.1,
               label='Forcing modes 1-2')

    ax3.set_xlabel('Mode number n', fontsize=12)
    ax3.set_ylabel('E_mag(n)', fontsize=12)
    ax3.set_title('Magnetic Energy Spectrum (Time-Averaged)', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, which='both')
    ax3.legend(fontsize=10)
    ax3.set_xlim(0.5, n_perp[-1])
    ax3.set_ylim(1e-3, None)

    plt.tight_layout()

    filename = f"alfvenic_cascade_{args.resolution}cubed.png"
    filepath = output_dir / filename
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"✓ Saved figure: {filepath}")

    # -------------------------------------------------------------------------
    # Combined Kinetic + Magnetic Plot (like thesis Figure 2.2)
    # -------------------------------------------------------------------------
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 5))

    # Kinetic
    ax_left.loglog(n_perp, E_kin_avg, 'b-', linewidth=2.5,
                   label=f'{args.resolution}³, r={hyper_r}')
    ax_left.fill_between(n_perp, E_kin_avg - E_kin_std, E_kin_avg + E_kin_std,
                        color='b', alpha=0.2)
    ax_left.loglog(n_ref, E_ref, 'k--', linewidth=2, label='k⊥^(-5/3)')
    ax_left.set_xlabel('k⊥', fontsize=14)
    ax_left.set_ylabel('Kinetic Energy', fontsize=14)
    ax_left.set_title('Kinetic Energy Spectrum', fontsize=16, fontweight='bold')
    ax_left.grid(True, alpha=0.3, which='both')
    ax_left.legend(fontsize=12)
    ax_left.set_xlim(0.5, n_perp[-1])
    ax_left.set_ylim(1e-3, None)

    # Magnetic
    ax_right.loglog(n_perp, E_mag_avg, 'r-', linewidth=2.5,
                    label=f'{args.resolution}³, r={hyper_r}')
    ax_right.fill_between(n_perp, E_mag_avg - E_mag_std, E_mag_avg + E_mag_std,
                         color='r', alpha=0.2)
    ax_right.loglog(n_ref, E_ref, 'k--', linewidth=2, label='k⊥^(-5/3)')
    ax_right.set_xlabel('k⊥', fontsize=14)
    ax_right.set_ylabel('Magnetic Energy', fontsize=14)
    ax_right.set_title('Magnetic Energy Spectrum', fontsize=16, fontweight='bold')
    ax_right.grid(True, alpha=0.3, which='both')
    ax_right.legend(fontsize=12)
    ax_right.set_xlim(0.5, n_perp[-1])
    ax_right.set_ylim(1e-3, None)

    plt.tight_layout()

    filename_thesis = f"alfvenic_cascade_thesis_style_{args.resolution}cubed.png"
    filepath_thesis = output_dir / filename_thesis
    plt.savefig(filepath_thesis, dpi=150, bbox_inches='tight')
    print(f"✓ Saved thesis-style figure: {filepath_thesis}")

    # ==========================================================================
    # Save Turbulence Diagnostics (Issue #82 Investigation)
    # ==========================================================================

    if args.save_diagnostics and diagnostics_list:
        print("\n" + "=" * 70)
        print("Saving turbulence diagnostics (Issue #82 Investigation)")
        print("=" * 70)

        diag_filename = f"turbulence_diagnostics_{args.resolution}cubed.h5"
        diag_filepath = output_dir / diag_filename

        # Prepare metadata
        metadata = {
            'resolution': args.resolution,
            'eta': float(eta),
            'nu': float(nu),
            'hyper_r': int(hyper_r),
            'hyper_n': int(hyper_n),
            'force_amplitude': float(force_amplitude),
            'dt': float(dt),
            'total_time': float(total_time),
            'n_steps': int(step),
            'description': f'Turbulence diagnostics for {args.resolution}³ resolution (Issue #82)'
        }

        save_turbulence_diagnostics(diagnostics_list, str(diag_filepath), metadata=metadata)
        print(f"✓ Saved {len(diagnostics_list)} diagnostic samples to: {diag_filepath}")
        print(f"  Time range: t={diagnostics_list[0].time:.2f} to {diagnostics_list[-1].time:.2f} τ_A")
        print(f"  Sampling interval: every {args.diagnostic_interval} steps")

        # Print summary statistics
        max_velocities = [d.max_velocity for d in diagnostics_list]
        cfl_numbers = [d.cfl_number for d in diagnostics_list]
        energy_highk = [d.energy_highk for d in diagnostics_list]

        print(f"\n  Summary Statistics:")
        print(f"    max_velocity:  min={min(max_velocities):.3f}, max={max(max_velocities):.3f}, mean={np.mean(max_velocities):.3f}")
        print(f"    CFL number:    min={min(cfl_numbers):.3f}, max={max(cfl_numbers):.3f}, mean={np.mean(cfl_numbers):.3f}")
        print(f"    High-k energy: min={min(energy_highk):.4f}, max={max(energy_highk):.4f}, mean={np.mean(energy_highk):.4f}")

        if max(cfl_numbers) > 1.0:
            print(f"\n  ⚠️  WARNING: CFL number exceeded 1.0 (max={max(cfl_numbers):.3f})")
        if max(max_velocities) > 100.0:
            print(f"  ⚠️  WARNING: Very high velocities detected (max={max(max_velocities):.2e})")

    # ==========================================================================
    # Save Spectral Data (for detailed analysis)
    # ==========================================================================

    if args.save_spectra:
        import h5py

        print("\n" + "=" * 70)
        print("Saving spectral data")
        print("=" * 70)

        spectra_filename = f"spectral_data_{args.resolution}cubed.h5"
        spectra_filepath = output_dir / spectra_filename

        # Convert lists to arrays for saving
        spectrum_kinetic_array = np.array(spectrum_kinetic_list)
        spectrum_magnetic_array = np.array(spectrum_magnetic_list)

        with h5py.File(spectra_filepath, 'w') as f:
            # Metadata
            f.attrs['resolution'] = args.resolution
            f.attrs['domain_size'] = (Lx, Ly, Lz)
            f.attrs['v_A'] = float(v_A)
            f.attrs['eta'] = float(eta)
            f.attrs['nu'] = float(nu)
            f.attrs['hyper_r'] = int(hyper_r)
            f.attrs['hyper_n'] = int(hyper_n)
            f.attrs['force_amplitude'] = float(force_amplitude)
            f.attrs['force_n_min'] = int(n_force_min)
            f.attrs['force_n_max'] = int(n_force_max)
            f.attrs['dt'] = float(dt)
            f.attrs['total_time'] = float(total_time)
            f.attrs['averaging_start'] = float(averaging_start)
            f.attrs['averaging_duration'] = float(total_time - averaging_start)
            f.attrs['n_spectra'] = len(spectrum_kinetic_list)
            f.attrs['description'] = f'Turbulent cascade spectral data N={args.resolution}³'

            # Wavenumber and mode-number grids
            f.create_dataset('k_perp', data=k_perp)
            # Save mode-number axis to avoid any 2π conversions in analysis/plots
            n_perp = k_perp * Lx / (2 * np.pi)
            f.create_dataset('n_perp', data=n_perp)

            # Time-averaged spectra
            f.create_dataset('E_kinetic_avg', data=E_kin_avg)
            f.create_dataset('E_magnetic_avg', data=E_mag_avg)
            f.create_dataset('E_kinetic_std', data=E_kin_std)
            f.create_dataset('E_magnetic_std', data=E_mag_std)

            # Full time series of spectra (for evolution plots)
            f.create_dataset('E_kinetic_all', data=spectrum_kinetic_array)
            f.create_dataset('E_magnetic_all', data=spectrum_magnetic_array)

            # Time series data for steady-state verification
            f.create_dataset('times', data=np.array(history.times))
            f.create_dataset('energy_total', data=np.array(history.E_total))
            f.create_dataset('injection_rate', data=np.array(injection_rates))

            # Save averaging window info for analysis
            averaging_mask = np.array(history.times) >= averaging_start
            f.create_dataset('averaging_times', data=np.array(history.times)[averaging_mask])
            f.create_dataset('averaging_energies', data=np.array(history.E_total)[averaging_mask])

        print(f"✓ Saved spectral data to: {spectra_filepath}")
        print(f"  - Time-averaged spectra: {len(k_perp)} k-bins")
        print(f"  - Individual spectra: {spectrum_kinetic_array.shape}")
        print(f"  - Time series: {len(history.times)} points")

    print("\n" + "=" * 70)
    print("Benchmark complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
