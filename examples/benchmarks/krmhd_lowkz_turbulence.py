#!/usr/bin/env python3
"""
Low-k_z Forced KRMHD Turbulence

Demonstrates RMHD-compatible forcing for kinetic turbulence studies using
``force_alfven_modes_balanced()`` with low-|k_z| restriction.

Why low-k_z forcing?
    The full-|k| shell forcing (``force_alfven_modes_gandalf``) injects energy
    at all k_z, including high-k_z modes. In kinetic runs this drives unphysical
    parallel phase mixing because high-k_z energy cascades directly into Hermite
    moments rather than through the physical perpendicular cascade. Restricting
    to |nz| <= 1 respects RMHD critical balance and gives a physically sensible
    drive.

This example also demonstrates proper kinetic energy accounting:
    - ``energy()`` returns **fluid-only** energy (excludes Hermite moments)
    - ``hermite_moment_energy()`` gives the velocity-space energy distribution
    - The tail metric R_tail checks Hermite truncation adequacy

The default M=8 is deliberately small for fast runtime. The tail metric
will flag this as under-resolved — that is expected and demonstrates the
diagnostic working correctly. For a resolved run, increase M to 16+ and
lower nu if needed.

Runtime: ~3-5 minutes on M1 Pro for 32^3 with M=8
"""

import time
import jax
import jax.numpy as jnp
import numpy as np

from krmhd import (
    SpectralGrid3D,
    initialize_alfven_wave,
    gandalf_step,
    force_alfven_modes_balanced,
    energy as compute_energy,
)
from krmhd.diagnostics import hermite_moment_energy


def compute_tail_metric(state, n_tail=4):
    """Fraction of Hermite energy in the top n_tail moments."""
    E_m = hermite_moment_energy(state)
    return float(E_m[-n_tail:].sum() / jnp.maximum(E_m.sum(), 1e-30))


def main():
    print("=" * 70)
    print("KRMHD Low-k_z Forced Turbulence")
    print("=" * 70)

    # Grid and kinetic parameters
    N = 32
    M = 8           # Hermite moments (velocity-space resolution)
    Lx = Ly = Lz = 1.0

    # Physics
    v_A = 1.0
    eta = 1.0       # Normalized hyper-resistivity
    nu = 0.25       # Normalized collision frequency (thesis value)
    hyper_r = 2     # Spatial hyper-dissipation order
    hyper_n = 6     # Hermite hyper-collision order
    beta_i = 1.0
    dt = 0.005

    # Forcing: low-k_z balanced Elsasser
    amplitude = 0.01
    n_min, n_max = 1, 2
    max_nz = 1

    # Time integration
    total_time = 50.0   # Alfven crossing times
    n_steps = int(total_time / dt)
    diag_interval = int(5.0 / dt)  # Print every 5 tau_A

    print(f"\nGrid: {N}^3, M={M} Hermite moments")
    print(f"Forcing: balanced Elsasser, |nz| <= {max_nz}, amplitude={amplitude}")
    print(f"Dissipation: eta={eta} (r={hyper_r}), nu={nu} (n={hyper_n})")
    print(f"Total time: {total_time} tau_A ({n_steps} steps)")

    # Initialize
    grid = SpectralGrid3D.create(Nx=N, Ny=N, Nz=N, Lx=Lx, Ly=Ly, Lz=Lz)
    state = initialize_alfven_wave(
        grid, M=M, kz_mode=1, amplitude=0.001, beta_i=beta_i
    )
    key = jax.random.PRNGKey(42)

    E0 = compute_energy(state)
    print(f"\nInitial fluid energy: {E0['total']:.4e}")
    print(f"  (magnetic={E0['magnetic']:.4e}, kinetic={E0['kinetic']:.4e})")

    print(f"\n{'t/tau_A':>8s}  {'E_fluid':>10s}  {'E_mag':>10s}  {'E_kin':>10s}  "
          f"{'E_hermite':>10s}  {'R_tail':>8s}")
    print("-" * 70)

    t0 = time.time()

    for step in range(n_steps):
        # Force
        state, key = force_alfven_modes_balanced(
            state, amplitude=amplitude, n_min=n_min, n_max=n_max,
            dt=dt, key=key, max_nz=max_nz,
        )

        # Evolve
        state = gandalf_step(
            state, dt=dt, eta=eta, v_A=v_A,
            hyper_r=hyper_r, hyper_n=hyper_n,
        )

        # Diagnostics
        if (step + 1) % diag_interval == 0 or step == n_steps - 1:
            t = (step + 1) * dt
            E = compute_energy(state)
            E_herm = float(hermite_moment_energy(state).sum())
            R_tail = compute_tail_metric(state)
            print(f"{t:8.1f}  {E['total']:10.4e}  {E['magnetic']:10.4e}  "
                  f"{E['kinetic']:10.4e}  {E_herm:10.4e}  {R_tail:8.4f}")

    wall_time = time.time() - t0
    print(f"\nWall time: {wall_time:.1f}s")

    # Final Hermite spectrum
    E_m = hermite_moment_energy(state)
    R_tail = compute_tail_metric(state)
    print(f"\nHermite moment spectrum (E_m):")
    for m in range(min(M + 1, 12)):
        bar = "#" * max(1, int(40 * float(E_m[m]) / float(E_m.max())))
        print(f"  m={m:2d}: {float(E_m[m]):10.4e}  {bar}")
    if M >= 12:
        print(f"  ... (M={M})")
        print(f"  m={M}: {float(E_m[M]):10.4e}")

    print(f"\nTail metric R_tail = {R_tail:.4f}", end="")
    if R_tail < 0.05:
        print("  (GOOD: well-resolved truncation)")
    elif R_tail < 0.10:
        print("  (MARGINAL: consider increasing M or nu)")
    else:
        print("  (WARNING: under-resolved, increase M or nu)")


if __name__ == "__main__":
    main()
