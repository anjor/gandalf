#!/usr/bin/env python3
"""
Hermite Spectrum Evolution - Short Run Diagnostic

Runs a short forced Hermite cascade simulation (10-50 steps) and plots
the spectrum evolution at each step. Useful for debugging parameter choices
and understanding how the cascade develops.

Usage:
    python hermite_spectrum_evolution.py --steps 10
    python hermite_spectrum_evolution.py --steps 50 --nu 0.3 --amplitude 0.2
"""

import argparse
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from pathlib import Path

from krmhd import (
    SpectralGrid3D,
    KRMHDState,
    gandalf_step,
    compute_cfl_timestep,
    force_hermite_moments,
)
from krmhd.diagnostics import hermite_moment_energy


def main():
    parser = argparse.ArgumentParser(description='Short Hermite spectrum evolution diagnostic')
    parser.add_argument('--steps', type=int, default=10,
                        help='Number of timesteps to run (default: 10)')
    parser.add_argument('--resolution', type=int, default=16,
                        help='Spatial resolution (default: 16)')
    parser.add_argument('--hermite-moments', type=int, default=32,
                        help='Max Hermite moment M (default: 32)')
    parser.add_argument('--nu', type=float, default=0.5,
                        help='Collision frequency (default: 0.5)')
    parser.add_argument('--hyper-n', type=int, default=2,
                        help='Hyper-collision exponent (default: 2)')
    parser.add_argument('--amplitude', type=float, default=0.1,
                        help='Forcing amplitude (default: 0.1)')
    parser.add_argument('--lambda-param', type=float, default=-1.0,
                        help='Kinetic parameter Lambda (default: -1.0)')
    parser.add_argument('--output', type=str, default='examples/output/hermite_evolution.png',
                        help='Output figure path')

    args = parser.parse_args()

    print("=" * 70)
    print("Hermite Spectrum Evolution Diagnostic")
    print("=" * 70)
    print(f"\nRunning {args.steps} steps with:")
    print(f"  Grid: {args.resolution}³")
    print(f"  Hermite moments: M = {args.hermite_moments}")
    print(f"  Collision: ν = {args.nu}, hyper_n = {args.hyper_n}")
    print(f"  Forcing: amplitude = {args.amplitude}")
    print(f"  Lambda: Λ = {args.lambda_param}")

    # Initialize grid
    Nx = Ny = Nz = args.resolution
    Lx = Ly = Lz = 1.0
    M = args.hermite_moments

    grid = SpectralGrid3D.create(Nx=Nx, Ny=Ny, Nz=Nz, Lx=Lx, Ly=Ly, Lz=Lz)

    # Initialize state
    state = KRMHDState(
        z_plus=jnp.zeros((Nz, Ny, Nx//2+1), dtype=complex),
        z_minus=jnp.zeros((Nz, Ny, Nx//2+1), dtype=complex),
        B_parallel=jnp.zeros((Nz, Ny, Nx//2+1), dtype=complex),
        g=jnp.zeros((Nz, Ny, Nx//2+1, M+1), dtype=complex),
        grid=grid,
        time=0.0,
        M=M,
        v_th=1.0,
        beta_i=1.0,
        nu=args.nu,
        Lambda=args.lambda_param,
    )

    # Compute timestep
    v_A = 1.0
    dt = compute_cfl_timestep(state=state, v_A=v_A, cfl_safety=0.3)

    print(f"\n  Timestep: dt = {dt:.4f}")
    print(f"  Total time: {args.steps * dt:.4f} τ_A")

    # Storage for spectra
    spectra_list = []
    times = []

    # Random key for forcing
    key = jax.random.PRNGKey(42)

    print(f"\nRunning evolution...")

    # Main loop
    for step in range(args.steps + 1):
        # Compute and store spectrum
        E_m = np.array(hermite_moment_energy(state))
        spectra_list.append(E_m)
        times.append(state.time)

        E_total = np.sum(E_m)
        print(f"  Step {step:3d}: t={state.time:.4f} τ_A, E_total={E_total:.4e}")

        if step == args.steps:
            break

        # Apply forcing
        key, subkey = jax.random.split(key)
        state, key = force_hermite_moments(
            state,
            amplitude=args.amplitude,
            n_min=1,
            n_max=2,
            dt=dt,
            key=subkey,
            forced_moments=(0, 1),  # Force both g0 and g1
        )

        # Time step
        state = gandalf_step(
            state,
            dt=dt,
            eta=0.5,
            nu=args.nu,
            v_A=v_A,
            hyper_r=2,
            hyper_n=args.hyper_n,
        )

    print(f"\n✓ Evolution complete")

    # Create figure with spectrum evolution
    print(f"\nCreating visualization...")

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Spectrum evolution (all timesteps)
    m_values = np.arange(M + 1)
    spectra_array = np.array(spectra_list)

    # Plot every spectrum with color gradient
    cmap = plt.cm.viridis
    for i, (E_m, t) in enumerate(zip(spectra_list, times)):
        color = cmap(i / len(spectra_list))
        mask = E_m > 0
        if i == 0:
            label = f't={t:.3f} (initial)'
            alpha = 0.3
            lw = 1
        elif i == len(spectra_list) - 1:
            label = f't={t:.3f} (final)'
            alpha = 1.0
            lw = 2
        else:
            label = None
            alpha = 0.5
            lw = 1

        ax1.loglog(m_values[mask], E_m[mask], '-', color=color,
                   alpha=alpha, linewidth=lw, label=label)

    # Reference line m^(-1/2)
    m_ref = np.array([2.0, min(16, M//2)])
    E_ref = m_ref**(-0.5)
    # Normalize to final spectrum at m=3
    if spectra_list[-1][3] > 0:
        E_ref *= spectra_list[-1][3] / (3.0**(-0.5))
    ax1.loglog(m_ref, E_ref, 'k--', linewidth=2, label='m$^{-1/2}$ reference')

    ax1.set_xlabel('Hermite moment m', fontsize=12)
    ax1.set_ylabel('Energy E$_m$', fontsize=12)
    ax1.set_title('Spectrum Evolution', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, which='both')
    ax1.legend(fontsize=10)
    ax1.set_xlim(0.8, M)
    ax1.set_ylim(bottom=1e-8)

    # Panel 2: Energy growth
    total_energies = np.sum(spectra_array, axis=1)
    ax2.semilogy(times, total_energies, 'b-', linewidth=2, marker='o', markersize=4)
    ax2.set_xlabel('Time [τ$_A$]', fontsize=12)
    ax2.set_ylabel('Total Energy', fontsize=12)
    ax2.set_title('Energy Growth', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Add text with parameters
    param_text = f"ν={args.nu}, n={args.hyper_n}, A={args.amplitude}, Λ={args.lambda_param}"
    ax2.text(0.05, 0.95, param_text, transform=ax2.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Panel 3: Final spectrum with fixed y-axis
    E_m_final = spectra_list[-1]
    mask_final = E_m_final > 0

    # Plot final spectrum
    ax3.loglog(m_values[mask_final], E_m_final[mask_final], 'bo-',
               linewidth=2, markersize=6, label=f't={times[-1]:.3f} (final)')

    # Reference line m^(-1/2)
    if E_m_final[3] > 0:
        E_ref_final = m_ref**(-0.5) * E_m_final[3] / (3.0**(-0.5))
    else:
        E_ref_final = m_ref**(-0.5)
    ax3.loglog(m_ref, E_ref_final, 'k--', linewidth=2, label='m$^{-1/2}$ reference')

    # Fit power law to final spectrum
    from scipy import stats
    m_fit_min = 2
    m_fit_max = min(16, M // 2)
    mask_fit = (m_values >= m_fit_min) & (m_values <= m_fit_max) & (E_m_final > 0)
    m_fit = m_values[mask_fit]
    E_fit = E_m_final[mask_fit]

    if len(m_fit) >= 3:
        log_m = np.log10(m_fit)
        log_E = np.log10(E_fit)
        slope, intercept, r_value, _, _ = stats.linregress(log_m, log_E)
        r_squared = r_value**2

        # Plot fitted line
        m_fit_plot = np.logspace(np.log10(m_fit_min), np.log10(m_fit_max), 50)
        E_fit_plot = 10**(intercept) * m_fit_plot**slope
        ax3.loglog(m_fit_plot, E_fit_plot, 'r-', linewidth=1.5, alpha=0.7,
                   label=f'Fit: m$^{{{slope:.2f}}}$, R²={r_squared:.3f}')

    # Highlight regions
    ax3.axvspan(0.5, 1.5, color='green', alpha=0.15, label='Forced: g₀, g₁')
    ax3.axvspan(m_fit_min, m_fit_max, color='orange', alpha=0.1,
                label=f'Fit range [{m_fit_min},{m_fit_max}]')

    ax3.set_xlabel('Hermite moment m', fontsize=12)
    ax3.set_ylabel('Energy E$_m$', fontsize=12)
    ax3.set_title('Final Spectrum (Fixed Scale)', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, which='both')
    ax3.legend(fontsize=9, loc='best')
    ax3.set_xlim(0.8, M)
    ax3.set_ylim(bottom=1e-5, top=max(E_m_final[E_m_final > 0]) * 10)

    plt.tight_layout()

    # Save figure
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")

    plt.show()

    print("\n" + "=" * 70)
    print("Diagnostic complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
