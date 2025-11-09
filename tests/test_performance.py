#!/usr/bin/env python3
"""
Performance benchmarks for KRMHD spectral operations.

Measures throughput and scaling of the Poisson bracket operator, which is the
computational bottleneck in KRMHD simulations.

Run with: pytest tests/test_performance.py -v -s

Baseline performance (M1 Pro, macOS 13.5, JAX 0.4.20 with Metal):

2D Poisson Bracket:
- 64²:  ~0.11 ms/call, ~9100 calls/sec
- 128²: ~0.21 ms/call, ~4700 calls/sec
- 256²: ~0.85 ms/call, ~1170 calls/sec
- 512²: ~3.15 ms/call, ~317 calls/sec

3D Poisson Bracket (primary):
- 32³:  ~0.57 ms/call, ~1770 calls/sec
- 64³:  ~3.46 ms/call, ~289 calls/sec
- 128³: ~28.2 ms/call, ~35.5 calls/sec
- 256³: ~257 ms/call, ~3.9 calls/sec

Realistic workload (128³, sustained throughput):
- ~31.6 ms/call, ~31.6 calls/sec
- ~190 ms/timestep (assuming 6 brackets/timestep)
- ~5.3 timesteps/second
- ~5.3 hours for 100K timesteps

Use -s flag to see detailed timing output during test runs.
"""

import time
import pytest
import jax
import jax.numpy as jnp
from typing import Tuple

from krmhd.spectral import SpectralGrid2D, SpectralGrid3D, rfft2_forward, rfftn_forward
from krmhd.physics import poisson_bracket_2d, poisson_bracket_3d


# ============================================================================
# Utility Functions
# ============================================================================


def create_random_field_2d(grid: SpectralGrid2D, seed: int = 42) -> jnp.ndarray:
    """
    Create random 2D field in Fourier space with proper reality condition.

    Args:
        grid: SpectralGrid2D instance
        seed: Random seed for reproducibility

    Returns:
        Complex array [Ny, Nx//2+1] satisfying f(-k) = f*(k)
    """
    key = jax.random.PRNGKey(seed)

    # Create random real-space field
    field_real = jax.random.normal(key, shape=(grid.Ny, grid.Nx))

    # Transform to Fourier space (automatically satisfies reality condition)
    field_fourier = rfft2_forward(field_real)

    return field_fourier


def create_random_field_3d(grid: SpectralGrid3D, seed: int = 42) -> jnp.ndarray:
    """
    Create random 3D field in Fourier space with proper reality condition.

    Args:
        grid: SpectralGrid3D instance
        seed: Random seed for reproducibility

    Returns:
        Complex array [Nz, Ny, Nx//2+1] satisfying f(-k) = f*(k)
    """
    key = jax.random.PRNGKey(seed)

    # Create random real-space field
    field_real = jax.random.normal(key, shape=(grid.Nz, grid.Ny, grid.Nx))

    # Transform to Fourier space (automatically satisfies reality condition)
    field_fourier = rfftn_forward(field_real)

    return field_fourier


def warmup_jit(func, *args, **kwargs) -> float:
    """
    Warm up JIT compilation and return compilation time.

    Args:
        func: JAX function to compile
        *args, **kwargs: Arguments to pass to function

    Returns:
        Compilation time in seconds
    """
    t0 = time.time()
    _ = func(*args, **kwargs).block_until_ready()  # Ensure completion
    t_compile = time.time() - t0
    return t_compile


def time_function(func, *args, n_calls: int = 10, **kwargs) -> Tuple[float, float]:
    """
    Time a JAX function over multiple calls (after warmup).

    Args:
        func: JAX function to time
        *args, **kwargs: Arguments to pass to function
        n_calls: Number of calls to average over

    Returns:
        (time_per_call, throughput_per_second)
    """
    # Timing loop
    t0 = time.time()
    for _ in range(n_calls):
        _ = func(*args, **kwargs).block_until_ready()
    t_total = time.time() - t0

    time_per_call = t_total / n_calls
    throughput = n_calls / t_total

    return time_per_call, throughput


def format_timing_results(
    resolution: str,
    compilation_time: float,
    time_per_call: float,
    throughput: float,
    memory_mb: float
) -> str:
    """Format timing results for display."""
    return (
        f"\n{'='*70}\n"
        f"Resolution: {resolution}\n"
        f"{'='*70}\n"
        f"Compilation time:  {compilation_time*1000:8.2f} ms (first call only)\n"
        f"Time per call:     {time_per_call*1000:8.2f} ms (after warmup)\n"
        f"Throughput:        {throughput:8.2f} calls/sec\n"
        f"Memory per field:  {memory_mb:8.1f} MB\n"
        f"{'='*70}\n"
    )


# ============================================================================
# 2D Poisson Bracket Benchmarks
# ============================================================================


class TestPoissonBracket2DPerformance:
    """Performance benchmarks for 2D Poisson bracket operator."""

    @pytest.mark.benchmark
    @pytest.mark.parametrize("N", [64, 128, 256, 512])
    def test_poisson_bracket_2d_scaling(self, N: int):
        """
        Benchmark 2D Poisson bracket at multiple resolutions.

        Tests: 64², 128², 256², 512²

        Measures:
        - JIT compilation time (first call)
        - Runtime per call (after warmup)
        - Throughput (brackets/second)
        - Memory usage per field
        """
        # Create grid
        grid = SpectralGrid2D.create(Nx=N, Ny=N, Lx=2*jnp.pi, Ly=2*jnp.pi)

        # Create random test fields
        f_fourier = create_random_field_2d(grid, seed=42)
        g_fourier = create_random_field_2d(grid, seed=43)

        # Memory estimate: two input fields only (minimum, excludes intermediate arrays)
        memory_mb = 2 * f_fourier.nbytes / 1e6

        # Warmup: trigger JIT compilation
        t_compile = warmup_jit(
            poisson_bracket_2d,
            f_fourier,
            g_fourier,
            grid.kx,
            grid.ky,
            grid.Ny,
            grid.Nx,
            grid.dealias_mask
        )

        # Time compiled calls
        time_per_call, throughput = time_function(
            poisson_bracket_2d,
            f_fourier,
            g_fourier,
            grid.kx,
            grid.ky,
            grid.Ny,
            grid.Nx,
            grid.dealias_mask,
            n_calls=10
        )

        # Print results (visible with pytest -s)
        results = format_timing_results(
            f"{N}² (2D)",
            t_compile,
            time_per_call,
            throughput,
            memory_mb
        )
        print(results)

        # Basic sanity checks (not strict performance requirements)
        # Thresholds are intentionally permissive (~100× baseline) for cross-platform compatibility
        assert t_compile < 30.0, f"Compilation took too long: {t_compile:.2f}s"
        assert time_per_call < 10.0, f"Runtime too slow: {time_per_call:.2f}s"
        assert throughput > 0.1, f"Throughput too low: {throughput:.2f} calls/sec"


# ============================================================================
# 3D Poisson Bracket Benchmarks (Primary Focus)
# ============================================================================


class TestPoissonBracket3DPerformance:
    """Performance benchmarks for 3D Poisson bracket operator."""

    @pytest.mark.benchmark
    @pytest.mark.parametrize("N", [32, 64, 128, 256])
    def test_poisson_bracket_3d_scaling(self, N: int):
        """
        Benchmark 3D Poisson bracket at multiple resolutions.

        Tests: 32³, 64³, 128³, 256³

        Measures:
        - JIT compilation time (first call)
        - Runtime per call (after warmup)
        - Throughput (brackets/second)
        - Memory usage per field

        This is the CRITICAL benchmark as 3D Poisson bracket is the computational
        bottleneck in KRMHD simulations. Each timestepping call requires multiple
        bracket evaluations.
        """
        # Create grid (unit box Lx=Ly=Lz=1.0, typical for KRMHD)
        grid = SpectralGrid3D.create(Nx=N, Ny=N, Nz=N, Lx=1.0, Ly=1.0, Lz=1.0)

        # Create random test fields
        f_fourier = create_random_field_3d(grid, seed=42)
        g_fourier = create_random_field_3d(grid, seed=43)

        # Memory estimate: two input fields only (minimum, excludes intermediate arrays)
        memory_mb = 2 * f_fourier.nbytes / 1e6

        # Warmup: trigger JIT compilation
        t_compile = warmup_jit(
            poisson_bracket_3d,
            f_fourier,
            g_fourier,
            grid.kx,
            grid.ky,
            grid.Nz,
            grid.Ny,
            grid.Nx,
            grid.dealias_mask
        )

        # Time compiled calls
        time_per_call, throughput = time_function(
            poisson_bracket_3d,
            f_fourier,
            g_fourier,
            grid.kx,
            grid.ky,
            grid.Nz,
            grid.Ny,
            grid.Nx,
            grid.dealias_mask,
            n_calls=10
        )

        # Print results (visible with pytest -s)
        results = format_timing_results(
            f"{N}³ (3D)",
            t_compile,
            time_per_call,
            throughput,
            memory_mb
        )
        print(results)

        # Basic sanity checks (not strict performance requirements)
        # Thresholds are intentionally permissive (~1000× baseline) for cross-platform compatibility
        # M1 Pro baseline: 32³ ~0.6ms, 64³ ~3.5ms, 128³ ~28ms, 256³ ~257ms
        assert t_compile < 60.0, f"Compilation took too long: {t_compile:.2f}s"
        assert time_per_call < 30.0, f"Runtime too slow: {time_per_call:.2f}s"
        assert throughput > 0.03, f"Throughput too low: {throughput:.2f} calls/sec"

    @pytest.mark.benchmark
    def test_poisson_bracket_3d_realistic_workload(self):
        """
        Benchmark realistic turbulence simulation workload at 128³.

        Simulates typical usage pattern:
        - 128³ resolution (standard for development)
        - Multiple bracket calls per timestep (realistic scenario)
        - Measures sustained throughput over 100 calls

        This represents the actual performance users will experience during
        turbulence simulations.
        """
        N = 128
        grid = SpectralGrid3D.create(Nx=N, Ny=N, Nz=N, Lx=1.0, Ly=1.0, Lz=1.0)

        # Create fields
        f_fourier = create_random_field_3d(grid, seed=42)
        g_fourier = create_random_field_3d(grid, seed=43)

        memory_mb = 2 * f_fourier.nbytes / 1e6

        # Warmup
        t_compile = warmup_jit(
            poisson_bracket_3d,
            f_fourier,
            g_fourier,
            grid.kx,
            grid.ky,
            grid.Nz,
            grid.Ny,
            grid.Nx,
            grid.dealias_mask
        )

        # Time 100 calls (realistic workload)
        time_per_call, throughput = time_function(
            poisson_bracket_3d,
            f_fourier,
            g_fourier,
            grid.kx,
            grid.ky,
            grid.Nz,
            grid.Ny,
            grid.Nx,
            grid.dealias_mask,
            n_calls=100
        )

        # Calculate simulation estimates
        # Typical simulation: ~6 bracket calls per timestep (RK2 with 2 Elsasser variables + Hermite coupling)
        brackets_per_step = 6
        time_per_step = time_per_call * brackets_per_step
        steps_per_sec = 1.0 / time_per_step

        # Print results
        results = format_timing_results(
            f"{N}³ (realistic workload)",
            t_compile,
            time_per_call,
            throughput,
            memory_mb
        )
        print(results)
        print(f"Simulation estimates (assuming {brackets_per_step} brackets/timestep):")
        print(f"  Time per timestep:  {time_per_step*1000:8.2f} ms")
        print(f"  Timesteps/second:   {steps_per_sec:8.2f}")
        print(f"  Time for 1000 steps: {time_per_step*1000/60:8.1f} min")
        print(f"  Time for 100K steps: {time_per_step*100000/3600:8.1f} hours")
        print(f"{'='*70}\n")

        # Sanity checks (permissive for cross-platform compatibility)
        # M1 Pro baseline: ~28ms/call, ~35 calls/sec
        assert time_per_call < 5.0, f"128³ runtime too slow: {time_per_call:.2f}s (expected ~0.03s)"
        assert throughput > 0.2, f"128³ throughput too low: {throughput:.2f} calls/sec (expected ~35)"


# ============================================================================
# Scaling Analysis
# ============================================================================


class TestScalingAnalysis:
    """Analyze computational scaling with resolution."""

    @pytest.mark.benchmark
    @pytest.mark.slow
    def test_3d_scaling_law(self):
        """
        Measure how 3D Poisson bracket runtime scales with resolution.

        Theory: FFT-based algorithm should scale as O(N³ log N)

        Tests resolutions from 32³ to 256³ and fits scaling exponent.
        This is marked as slow since it tests all resolutions.
        """
        resolutions = [32, 64, 128]  # 256 omitted for faster testing
        times = []

        print("\n" + "="*70)
        print("3D Scaling Analysis")
        print("="*70)
        print(f"{'Resolution':<12} {'Time (ms)':<12} {'Throughput':<15} {'Scaling'}")
        print("-"*70)

        for N in resolutions:
            grid = SpectralGrid3D.create(Nx=N, Ny=N, Nz=N)
            f_fourier = create_random_field_3d(grid, seed=42)
            g_fourier = create_random_field_3d(grid, seed=43)

            # Warmup
            _ = warmup_jit(
                poisson_bracket_3d,
                f_fourier, g_fourier, grid.kx, grid.ky,
                grid.Nz, grid.Ny, grid.Nx, grid.dealias_mask
            )

            # Time
            time_per_call, throughput = time_function(
                poisson_bracket_3d,
                f_fourier, g_fourier, grid.kx, grid.ky,
                grid.Nz, grid.Ny, grid.Nx, grid.dealias_mask,
                n_calls=10
            )

            times.append(time_per_call)

            # Compute scaling relative to smallest grid
            if N == resolutions[0]:
                scaling = 1.0
            else:
                expected_scaling = (N / resolutions[0])**3 * jnp.log2(N) / jnp.log2(resolutions[0])
                actual_scaling = time_per_call / times[0]
                scaling = actual_scaling / expected_scaling

            print(f"{N}³{'':<8} {time_per_call*1000:8.2f}     "
                  f"{throughput:8.2f} calls/sec  {scaling:8.2f}x")

        print("="*70)
        print("Note: Scaling = (actual speedup) / (theoretical N³logN speedup)")
        print("      Values near 1.0 indicate good scaling behavior")
        print("="*70 + "\n")


# ============================================================================
# Main Entry Point (for direct execution)
# ============================================================================


if __name__ == "__main__":
    """
    Run benchmarks directly (without pytest).

    Usage: python tests/test_performance.py
    """
    print("\n" + "="*70)
    print("KRMHD Poisson Bracket Performance Benchmarks")
    print("="*70)
    print("\nRunning all benchmarks...\n")

    # Run 2D benchmarks
    test_2d = TestPoissonBracket2DPerformance()
    for N in [64, 128, 256]:
        test_2d.test_poisson_bracket_2d_scaling(N)

    # Run 3D benchmarks
    test_3d = TestPoissonBracket3DPerformance()
    for N in [32, 64, 128]:
        test_3d.test_poisson_bracket_3d_scaling(N)

    # Run realistic workload
    test_3d.test_poisson_bracket_3d_realistic_workload()

    # Run scaling analysis
    test_scaling = TestScalingAnalysis()
    test_scaling.test_3d_scaling_law()

    print("\nBenchmarks complete!")
