#!/usr/bin/env python3
"""
Hermite Convergence Demonstration

This example demonstrates how to use the Hermite closure diagnostics to:
1. Check convergence of the Hermite moment hierarchy
2. Compare different truncation levels (M values)
3. Compare zero vs. symmetric closures
4. Monitor convergence during time evolution

This is a pedagogical example showing best practices for production runs.

Usage:
    python examples/hermite_convergence.py

Requirements:
    - JAX with Metal/CUDA support
    - matplotlib for visualization
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from krmhd.spectral import SpectralGrid3D
from krmhd.physics import (
    initialize_hermite_moments,
    g0_rhs,
    g1_rhs,
    gm_rhs,
)
from krmhd.hermite import (
    closure_zero,
    closure_symmetric,
    check_hermite_convergence,
)


def demonstrate_convergence_check():
    """Demonstrate basic convergence checking."""
    print("=" * 70)
    print("1. Basic Convergence Check")
    print("=" * 70)

    grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)

    # Test different M values
    M_values = [5, 10, 15, 20]

    print("\nInitializing Hermite moments with perturbations...")
    print(f"Grid: {grid.Nx}×{grid.Ny}×{grid.Nz}")
    print()

    results = []

    for M in M_values:
        # Initialize with perturbations
        g = initialize_hermite_moments(
            grid=grid,
            M=M,
            v_th=1.0,
            perturbation_amplitude=0.1,
            seed=42,
        )

        # Check convergence
        result = check_hermite_convergence(g, threshold=1e-3)
        results.append(result)

        # Print results
        status = "✓ CONVERGED" if result["is_converged"] else "✗ NOT CONVERGED"
        print(f"M = {M:2d}: {status}")
        print(f"  Energy fraction: {result['energy_fraction']:.6f}")
        print(f"  E_highest: {result['energy_highest_moment']:.6e}")
        print(f"  E_total:   {result['energy_total']:.6e}")
        print()

    return results


def compare_closures():
    """Compare zero vs. symmetric closures."""
    print("=" * 70)
    print("2. Comparing Closure Schemes")
    print("=" * 70)

    grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)
    M = 10

    # Initialize moments
    g = initialize_hermite_moments(
        grid=grid, M=M, v_th=1.0, perturbation_amplitude=0.1, seed=42
    )

    print(f"\nComparing closures for M = {M}:")
    print(f"Current moment array shape: {g.shape}")
    print()

    # Apply different closures
    g_M_plus_1_zero = closure_zero(g, M)
    g_M_plus_1_sym = closure_symmetric(g, M)

    print("Zero closure (gₘ₊₁ = 0):")
    print(f"  Result shape: {g_M_plus_1_zero.shape}")
    print(f"  Max value: {jnp.max(jnp.abs(g_M_plus_1_zero)):.6e}")
    print(f"  All zeros: {jnp.all(g_M_plus_1_zero == 0.0)}")
    print()

    print("Symmetric closure (gₘ₊₁ = gₘ₋₁):")
    print(f"  Result shape: {g_M_plus_1_sym.shape}")
    print(f"  Max value: {jnp.max(jnp.abs(g_M_plus_1_sym)):.6e}")
    print(f"  Matches g[M-1]: {jnp.allclose(g_M_plus_1_sym, g[:, :, :, M - 1])}")
    print()

    # Show that they differ
    diff = jnp.max(jnp.abs(g_M_plus_1_zero - g_M_plus_1_sym))
    print(f"Difference between closures: {diff:.6e}")
    print(
        "Note: With finite collisions (ν > 0) and sufficient M,\n"
        "      results should be independent of closure choice."
    )
    print()


def monitor_convergence_during_evolution():
    """Monitor convergence as moments evolve with collision damping."""
    print("=" * 70)
    print("3. Monitoring Convergence During Time Evolution")
    print("=" * 70)

    grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)
    M = 15

    # Initialize
    g = initialize_hermite_moments(
        grid=grid, M=M, v_th=1.0, perturbation_amplitude=0.1, seed=42
    )

    # Collision parameters
    nu = 0.1  # Strong collisions
    dt = 0.01
    n_steps = 20

    print(f"\nEvolution parameters:")
    print(f"  M = {M}")
    print(f"  Collision frequency: ν = {nu}")
    print(f"  Timestep: dt = {dt}")
    print(f"  Number of steps: {n_steps}")
    print()

    # Track convergence over time
    times = []
    energy_fractions = []
    convergence_status = []

    # Initial state
    result = check_hermite_convergence(g, threshold=1e-3)
    times.append(0.0)
    energy_fractions.append(result["energy_fraction"])
    convergence_status.append(result["is_converged"])

    print(f"{'Time':>8s} {'E_M/E_total':>12s} {'Status':>12s}")
    print("-" * 35)
    print(
        f"{0.0:8.3f} {result['energy_fraction']:12.6e} "
        f"{'CONVERGED' if result['is_converged'] else 'NOT CONV'}"
    )

    # Evolve with collision damping
    g_evolved = g
    for step in range(n_steps):
        # Apply collision damping to moments m ≥ 2
        # (simplified evolution for demonstration)
        m_indices = jnp.arange(2, M + 1)
        damping_factors = jnp.exp(-nu * m_indices * dt)
        g_evolved = g_evolved.at[:, :, :, 2:].multiply(
            damping_factors[None, None, None, :]
        )

        # Check convergence
        result = check_hermite_convergence(g_evolved, threshold=1e-3)

        time = (step + 1) * dt
        times.append(time)
        energy_fractions.append(result["energy_fraction"])
        convergence_status.append(result["is_converged"])

        # Print every 5 steps
        if (step + 1) % 5 == 0:
            print(
                f"{time:8.3f} {result['energy_fraction']:12.6e} "
                f"{'CONVERGED' if result['is_converged'] else 'NOT CONV'}"
            )

    print()
    print("Observations:")
    print("  → High moments are damped by collisions: gₘ ~ exp(-νmt)")
    print("  → Energy fraction E_M/E_total decreases over time")
    print("  → System becomes better converged as t increases")
    print()

    return times, energy_fractions, convergence_status


def visualize_moment_decay():
    """Visualize how moment energy decays with m."""
    print("=" * 70)
    print("4. Visualizing Moment Energy Distribution")
    print("=" * 70)

    grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)
    M = 20

    # Initialize
    g = initialize_hermite_moments(
        grid=grid, M=M, v_th=1.0, perturbation_amplitude=0.1, seed=42
    )

    print(f"\nComputing energy in each moment (M = {M})...")

    # Compute energy per moment (accounting for rfft)
    g_squared = jnp.abs(g) ** 2
    energy_kx0 = jnp.sum(g_squared[:, :, 0, :], axis=(0, 1))
    energy_kx_pos = jnp.sum(g_squared[:, :, 1:, :], axis=(0, 1, 2))
    energies = energy_kx0 + 2.0 * energy_kx_pos

    # Normalize
    energies = energies / jnp.sum(energies)

    print(f"  Total energy: {jnp.sum(energies):.6f} (normalized)")
    print(f"  Energy in g_0: {energies[0]:.6e}")
    print(f"  Energy in g_M: {energies[M]:.6e}")
    print()

    # Create visualization
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Linear scale
        ax1.bar(range(M + 1), energies, alpha=0.7, edgecolor="black")
        ax1.set_xlabel("Moment index m", fontsize=12)
        ax1.set_ylabel("Energy fraction", fontsize=12)
        ax1.set_title("Moment Energy Distribution (Linear Scale)", fontsize=13)
        ax1.grid(True, alpha=0.3)

        # Log scale
        ax2.semilogy(
            range(M + 1),
            energies,
            "o-",
            markersize=6,
            linewidth=2,
            label="Moment energy",
        )
        ax2.axhline(1e-3, color="r", linestyle="--", label="1e-3 threshold")
        ax2.set_xlabel("Moment index m", fontsize=12)
        ax2.set_ylabel("Energy fraction (log scale)", fontsize=12)
        ax2.set_title("Moment Energy Decay (Log Scale)", fontsize=13)
        ax2.grid(True, alpha=0.3, which="both")
        ax2.legend()

        plt.tight_layout()
        plt.savefig("hermite_convergence_energy.png", dpi=150, bbox_inches="tight")
        print("  → Saved figure: hermite_convergence_energy.png")
        print()
    except Exception as e:
        print(f"  → Could not create figure: {e}")
        print()


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 70)
    print("HERMITE CLOSURE CONVERGENCE DEMONSTRATION")
    print("=" * 70)
    print()
    print("This example demonstrates best practices for using Hermite")
    print("closure diagnostics in production KRMHD simulations.")
    print()

    # 1. Basic convergence check
    demonstrate_convergence_check()

    # 2. Compare closures
    compare_closures()

    # 3. Monitor during evolution
    monitor_convergence_during_evolution()

    # 4. Visualize moment decay
    visualize_moment_decay()

    # Summary
    print("=" * 70)
    print("SUMMARY AND RECOMMENDATIONS")
    print("=" * 70)
    print()
    print("Best practices for production runs:")
    print()
    print("1. **Choose M based on convergence**:")
    print("   - Start with M ~ 15-20 for initial tests")
    print("   - Use check_hermite_convergence() to verify E_M/E_total < 1e-3")
    print("   - Increase M if not converged")
    print()
    print("2. **Use finite collisions**:")
    print("   - Set ν > 0 to damp high moments")
    print("   - Typical values: ν ~ 0.01-0.1")
    print("   - Results should be closure-independent when converged")
    print()
    print("3. **Monitor convergence during runs**:")
    print("   - Check periodically (e.g., every 10-100 timesteps)")
    print("   - Verify E_M/E_total remains small")
    print("   - Watch for sudden increases (may indicate instability)")
    print()
    print("4. **Compare closures for validation**:")
    print("   - Run same setup with closure_zero() and closure_symmetric()")
    print("   - Results should agree if properly converged")
    print("   - Large differences indicate insufficient M or ν")
    print()
    print("5. **Visualization helps**:")
    print("   - Plot moment energy distribution")
    print("   - Should see exponential decay E_m ~ exp(-αm)")
    print("   - Highest moments should have negligible energy")
    print()
    print("Questions? See Issue #24 and Thesis §2.4")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
