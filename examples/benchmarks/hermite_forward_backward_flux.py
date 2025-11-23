#!/usr/bin/env python3
"""
Hermite Forward/Backward Flux Diagnostic (Thesis Section 3.4.5)

Computes and visualizes the decomposition of Hermite moments into:
- C+ (forward flux, phase mixing modes): Energy flows m → m+1
- C- (backward flux, phase unmixing modes): Energy flows m+1 → m

Reproduces the physics from thesis Eq. (3.49-3.56), showing that:
- True cascade should have C+ >> C- (forward flux dominates)
- C- ≈ 0 indicates clean phase mixing without unphysical unmixing

Usage:
    python hermite_forward_backward_flux.py --steps 60
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


def compute_forward_backward_spectra(state: KRMHDState):
    """
    Compute forward (C+) and backward (C-) Hermite spectra.

    Following thesis Eq. (3.49-3.52) and original GANDALF implementation.

    The thesis works with transformed variables g̃_{m,k} = (i·sgn(k_z))^m · g_{m,k}.

    In terms of raw moments g (matching original GANDALF):

    g̃+_{m,k} = (i·sgn(k_z))^m · (g_m + i·sgn(k_z)·g_{m+1}) / 2
    g̃-_{m,k} = (-1)^m · (i·sgn(k_z))^m · (g_m - i·sgn(k_z)·g_{m+1}) / 2

    C±_{m,k} = ⟨|g̃±_{m,k}|²⟩

    Args:
        state: KRMHD state with Hermite moments

    Returns:
        C_plus: Forward flux spectrum [M] (summed over k)
        C_minus: Backward flux spectrum [M] (summed over k)
        C_total: Total spectrum [M] (should equal hermite_moment_energy)
    """
    g = state.g  # Shape: [Nz, Ny, Nx//2+1, M+1]
    M = state.M
    grid = state.grid

    # Get k_z wavenumber array
    kz = grid.kz  # Shape: [Nz]

    # Compute sgn(k_z) for each z-mode
    # For k_z = 0, set sign to 1 (arbitrary but consistent)
    kz_sign = jnp.sign(kz)
    kz_sign = jnp.where(kz == 0, 1.0, kz_sign)

    # Reshape for broadcasting: [Nz, 1, 1]
    kz_sign_3d = kz_sign[:, jnp.newaxis, jnp.newaxis]

    # Get kx array for rfft accounting
    kx = grid.kx  # Shape: [Nx//2+1]
    Nx = grid.Nx

    # Compute forward and backward modes for each m
    C_plus = []
    C_minus = []

    for m in range(M):  # m = 0 to M-1
        g_m = g[:, :, :, m]
        g_m_plus_1 = g[:, :, :, m + 1]

        # Phase factor: (i·sgn(k_z))^m
        phase_m = (1j * kz_sign_3d) ** m

        # Forward mode: (i·sgn(k_z))^m · (g_m + i·sgn(k_z)·g_{m+1}) / 2
        g_plus = phase_m * (g_m + 1j * kz_sign_3d * g_m_plus_1) / 2.0

        # Backward mode: (-1)^m · (i·sgn(k_z))^m · (g_m - i·sgn(k_z)·g_{m+1}) / 2
        sign_m = (-1.0) ** m
        g_minus = sign_m * phase_m * (g_m - 1j * kz_sign_3d * g_m_plus_1) / 2.0

        # Compute energy: |g_plus|² and |g_minus|²
        E_plus_kspace = jnp.abs(g_plus) ** 2
        E_minus_kspace = jnp.abs(g_minus) ** 2

        # Account for rfft format:
        # - kx = 0: Count once
        # - kx = Nyquist (if Nx even): Count once
        # - kx > 0 and kx < Nyquist: Count twice (complex conjugate pairs)

        # Create weight array
        weight = jnp.ones_like(kx)
        # kx=0 plane (idx=0): weight = 1
        # kx=Nyquist plane (idx=Nx//2 if Nx even): weight = 1
        # All other kx: weight = 2
        weight = weight.at[1:].set(2.0)  # Set all to 2 except kx=0
        if Nx % 2 == 0:  # If Nx is even, Nyquist mode exists
            weight = weight.at[-1].set(1.0)  # Set Nyquist to 1

        # Apply weights (broadcast over z, y dimensions)
        weight_3d = weight[jnp.newaxis, jnp.newaxis, :]  # Shape: [1, 1, Nx//2+1]

        energy_plus = float(jnp.sum(E_plus_kspace * weight_3d))
        energy_minus = float(jnp.sum(E_minus_kspace * weight_3d))

        C_plus.append(energy_plus)
        C_minus.append(energy_minus)

    C_plus = np.array(C_plus)
    C_minus = np.array(C_minus)
    C_total = C_plus + C_minus

    return C_plus, C_minus, C_total


def main():
    parser = argparse.ArgumentParser(
        description='Forward/backward flux diagnostic (thesis Section 3.4.5)'
    )
    parser.add_argument('--steps', type=int, default=60,
                        help='Number of timesteps (default: 60)')
    parser.add_argument('--resolution', type=int, default=16,
                        help='Spatial resolution (default: 16)')
    parser.add_argument('--hermite-moments', type=int, default=32,
                        help='Max Hermite moment M (default: 32)')
    parser.add_argument('--nu', type=float, default=0.1,
                        help='Collision frequency (default: 0.1)')
    parser.add_argument('--hyper-n', type=int, default=3,
                        help='Hyper-collision exponent (default: 3)')
    parser.add_argument('--amplitude', type=float, default=0.05,
                        help='Forcing amplitude (default: 0.05)')
    parser.add_argument('--lambda-param', type=float, default=-1.0,
                        help='Kinetic parameter Lambda (default: -1.0)')
    parser.add_argument('--output', type=str,
                        default='examples/output/hermite_forward_backward.png',
                        help='Output figure path')
    parser.add_argument('--thesis-style', action='store_true',
                        help='Use thesis-style single-panel plot (Figure 3.3)')

    args = parser.parse_args()

    print("=" * 70)
    print("Hermite Forward/Backward Flux Diagnostic (Thesis §3.4.5)")
    print("=" * 70)
    print(f"\nParameters:")
    print(f"  Grid: {args.resolution}³, M = {args.hermite_moments}")
    print(f"  Collision: ν = {args.nu}, hyper_n = {args.hyper_n}")
    print(f"  Forcing: amplitude = {args.amplitude}")
    print(f"  Steps: {args.steps}")

    # Initialize grid and state
    Nx = Ny = Nz = args.resolution
    Lx = Ly = Lz = 1.0
    M = args.hermite_moments

    grid = SpectralGrid3D.create(Nx=Nx, Ny=Ny, Nz=Nz, Lx=Lx, Ly=Ly, Lz=Lz)

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

    # Random key for forcing
    key = jax.random.PRNGKey(42)

    # Run evolution
    print(f"\nRunning evolution...")
    for step in range(args.steps):
        if step % 10 == 0:
            E_m = hermite_moment_energy(state)
            E_total = np.sum(E_m)
            print(f"  Step {step:3d}: t={state.time:.4f} τ_A, E_total={E_total:.4e}")

        # Apply forcing
        key, subkey = jax.random.split(key)
        state, key = force_hermite_moments(
            state,
            amplitude=args.amplitude,
            n_min=1,
            n_max=2,
            dt=dt,
            key=subkey,
            forced_moments=(0, 1),
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

    print(f"\n✓ Evolution complete at t = {state.time:.4f} τ_A")

    # Compute forward/backward decomposition
    print(f"\nComputing forward/backward decomposition...")
    C_plus, C_minus, C_total = compute_forward_backward_spectra(state)
    E_m_direct = np.array(hermite_moment_energy(state))

    # Check consistency
    print(f"\n  Verification:")
    print(f"    C+ + C- vs E_m: max error = {np.max(np.abs(C_total - E_m_direct[:-1])):.2e}")

    # Compute forward fraction
    forward_fraction = C_plus / (C_plus + C_minus + 1e-20)

    # Create visualization
    print(f"\nCreating visualization...")

    m_values = np.arange(M)

    if args.thesis_style:
        # Thesis-style single-panel plot (Figure 3.3)
        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot C+ (solid black line)
        mask_plus = C_plus > 0
        ax.loglog(m_values[mask_plus], C_plus[mask_plus], 'k-',
                  linewidth=2, label='$C_m^+$')

        # Plot C- (dotted red line)
        mask_minus = C_minus > 0
        ax.loglog(m_values[mask_minus], C_minus[mask_minus], 'r:',
                  linewidth=2, label='$C_m^-$')

        # Reference lines (full x-range)
        m_ref = np.array([1.0, 20.0])

        # m^(-1/2) reference (normalize to C+ at m=3)
        if C_plus[3] > 0:
            E_ref_half = m_ref**(-0.5) * C_plus[3] / (3.0**(-0.5))
            line_half, = ax.loglog(m_ref, E_ref_half, 'k--', linewidth=1, alpha=0.7)
            # Add text label on the line
            ax.text(10, E_ref_half[0] * (10**(-0.5)) / (m_ref[0]**(-0.5)),
                   '$m^{-1/2}$', fontsize=11, ha='center', va='bottom')

        # m^(-3/2) reference (normalize to C- at m=3 if exists)
        if mask_minus.sum() > 3 and C_minus[3] > 0:
            E_ref_three_half = m_ref**(-1.5) * C_minus[3] / (3.0**(-1.5))
            line_three_half, = ax.loglog(m_ref, E_ref_three_half, 'k--', linewidth=1, alpha=0.7)
            # Add text label on the line
            ax.text(10, E_ref_three_half[0] * (10**(-1.5)) / (m_ref[0]**(-1.5)),
                   '$m^{-3/2}$', fontsize=11, ha='center', va='top')

        ax.set_xlabel('$m$', fontsize=14)
        ax.set_ylabel('$C_m^\\pm$', fontsize=14)
        ax.set_xlim(1, 20)
        ax.set_ylim(1e-3, 2)
        ax.legend(fontsize=12, loc='upper right', framealpha=0.9)
        ax.grid(True, alpha=0.2, which='both')

        # Add parameter text
        param_text = f"$\\nu={args.nu}$, $n={args.hyper_n}$"
        ax.text(0.05, 0.05, param_text, transform=ax.transAxes,
                fontsize=11, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()

    else:
        # Default 2x2 panel plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

        # Panel 1: C+ (forward flux spectrum)
        mask_plus = C_plus > 0
        ax1.loglog(m_values[mask_plus], C_plus[mask_plus], 'b-o',
                   linewidth=2, markersize=4, label='C$^+$ (forward flux)')

        # Reference m^(-1/2)
        m_ref = np.array([2.0, min(16, M//2)])
        if C_plus[3] > 0:
            E_ref = m_ref**(-0.5) * C_plus[3] / (3.0**(-0.5))
            ax1.loglog(m_ref, E_ref, 'k--', linewidth=2, label='m$^{-1/2}$ reference')

        ax1.set_xlabel('Moment m', fontsize=11)
        ax1.set_ylabel('C$^+_m$ (forward)', fontsize=11)
        ax1.set_title('Forward Flux Spectrum (Phase Mixing)', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, which='both')
        ax1.legend(fontsize=9)
        ax1.set_xlim(1, 20)
        ax1.set_ylim(1e-3, 20)

        # Panel 2: C- (backward flux spectrum)
        mask_minus = C_minus > 0
        ax2.loglog(m_values[mask_minus], C_minus[mask_minus], 'r-o',
                   linewidth=2, markersize=4, label='C$^-$ (backward flux)')

        ax2.set_xlabel('Moment m', fontsize=11)
        ax2.set_ylabel('C$^-_m$ (backward)', fontsize=11)
        ax2.set_title('Backward Flux Spectrum (Phase Unmixing)', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, which='both')
        ax2.legend(fontsize=9)
        ax2.set_xlim(1, 20)
        ax2.set_ylim(1e-3, 20)

        # Panel 3: C+ vs C- comparison
        ax3.loglog(m_values[mask_plus], C_plus[mask_plus], 'b-o',
                   linewidth=2, markersize=4, label='C$^+$ (forward)')
        ax3.loglog(m_values[mask_minus], C_minus[mask_minus], 'r-o',
                   linewidth=2, markersize=4, label='C$^-$ (backward)')

        if C_plus[3] > 0:
            ax3.loglog(m_ref, E_ref, 'k--', linewidth=2, label='m$^{-1/2}$')

        ax3.set_xlabel('Moment m', fontsize=11)
        ax3.set_ylabel('Energy', fontsize=11)
        ax3.set_title('Forward vs Backward Comparison', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, which='both')
        ax3.legend(fontsize=9)
        ax3.set_xlim(1, 20)
        ax3.set_ylim(1e-3, 20)

        # Panel 4: Forward fraction C+/(C+ + C-)
        ax4.semilogx(m_values, forward_fraction * 100, 'g-o',
                     linewidth=2, markersize=4)
        ax4.axhline(100, color='k', linestyle='--', linewidth=1, alpha=0.5,
                    label='100% (pure forward)')
        ax4.axhline(50, color='k', linestyle=':', linewidth=1, alpha=0.5,
                    label='50% (equal mix)')

        ax4.set_xlabel('Moment m', fontsize=11)
        ax4.set_ylabel('Forward fraction [%]', fontsize=11)
        ax4.set_title('Phase Mixing Dominance', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.legend(fontsize=9)
        ax4.set_xlim(1, 20)
        ax4.set_ylim(0, 105)

        # Add parameter text
        param_text = f"t={state.time:.2f} τ$_A$, ν={args.nu}, n={args.hyper_n}, A={args.amplitude}"
        fig.suptitle(f'Hermite Forward/Backward Flux Decomposition\n{param_text}',
                     fontsize=13, fontweight='bold')

        plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save figure
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")

    plt.show()

    # Print diagnostic summary
    print("\n" + "=" * 70)
    print("Diagnostic Summary:")
    print("=" * 70)
    print(f"\nForward/Backward ratio at key moments:")
    for m_test in [2, 5, 10, 15]:
        if m_test < M:
            ratio = C_plus[m_test] / (C_minus[m_test] + 1e-20)
            frac = forward_fraction[m_test] * 100
            print(f"  m={m_test:2d}: C+/C- = {ratio:8.2f}, forward% = {frac:5.1f}%")

    print(f"\nExpected: C- ≈ 0 (backward flux negligible)")
    print(f"          Forward% ≈ 100% (pure phase mixing)")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
