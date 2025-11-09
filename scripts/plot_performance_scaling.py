#!/usr/bin/env python3
"""
Generate performance scaling plots from benchmark data.

Creates publication-quality plots showing:
1. Time per call vs resolution (2D and 3D)
2. Throughput vs resolution
3. Theoretical vs actual scaling

Run after benchmarks to visualize performance characteristics.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# REFERENCE BASELINE (M1 Pro, macOS 13.5, JAX 0.4.20 with Metal)
# DO NOT update unless establishing new baseline - run benchmarks manually
# To generate new baseline: pytest tests/test_performance.py -v -s
# These values are used for comparison and should only change deliberately

BENCHMARK_DATA_2D = {
    'N': np.array([64, 128, 256, 512]),
    'time_ms': np.array([0.11, 0.21, 0.85, 3.15]),
    'throughput': np.array([9100, 4700, 1170, 317]),
    'memory_mb': np.array([0.03, 0.13, 0.52, 2.10]),
}

BENCHMARK_DATA_3D = {
    'N': np.array([32, 64, 128, 256]),
    'time_ms': np.array([0.57, 3.46, 28.2, 257]),
    'throughput': np.array([1770, 289, 35.5, 3.9]),
    'memory_mb': np.array([0.26, 2.15, 17.0, 135.3]),
}


def plot_scaling_analysis(output_dir: Path = Path('docs/figures')):
    """
    Generate comprehensive scaling analysis plots.

    Args:
        output_dir: Directory to save plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create figure with 2x2 subplots
    fig = plt.figure(figsize=(12, 10))

    # ========================================================================
    # Panel 1: Time per call vs Resolution (log-log)
    # ========================================================================
    ax1 = plt.subplot(2, 2, 1)

    # Plot 2D data
    N_2d = BENCHMARK_DATA_2D['N']
    time_2d = BENCHMARK_DATA_2D['time_ms']
    ax1.loglog(N_2d, time_2d, 'o-', linewidth=2, markersize=8,
               color='#1f77b4', label='2D (measured)')

    # Theoretical 2D scaling: O(N² log N)
    N_fine = np.logspace(np.log10(N_2d[0]), np.log10(N_2d[-1]), 50)
    scale_2d = time_2d[0] / (N_2d[0]**2 * np.log2(N_2d[0]))
    theory_2d = scale_2d * N_fine**2 * np.log2(N_fine)
    ax1.loglog(N_fine, theory_2d, '--', color='#1f77b4', alpha=0.5,
               label='2D theory (N² log N)')

    # Plot 3D data
    N_3d = BENCHMARK_DATA_3D['N']
    time_3d = BENCHMARK_DATA_3D['time_ms']
    ax1.loglog(N_3d, time_3d, 's-', linewidth=2, markersize=8,
               color='#ff7f0e', label='3D (measured)')

    # Theoretical 3D scaling: O(N³ log N)
    scale_3d = time_3d[0] / (N_3d[0]**3 * np.log2(N_3d[0]))
    theory_3d = scale_3d * N_fine**3 * np.log2(N_fine)
    ax1.loglog(N_fine, theory_3d, '--', color='#ff7f0e', alpha=0.5,
               label='3D theory (N³ log N)')

    ax1.set_xlabel('Resolution N (per dimension)', fontsize=11)
    ax1.set_ylabel('Time per call (ms)', fontsize=11)
    ax1.set_title('Poisson Bracket Performance Scaling', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9, loc='upper left')
    ax1.grid(True, alpha=0.3, which='both')

    # Add annotation for 128³ (common resolution)
    try:
        idx_128 = np.where(N_3d == 128)[0][0]
        ax1.annotate(f'128³: {time_3d[idx_128]:.1f} ms',
                    xy=(128, time_3d[idx_128]), xytext=(128, time_3d[idx_128]*0.3),
                    arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                    fontsize=10, color='red', fontweight='bold')
    except IndexError:
        pass  # Skip annotation if 128³ not in dataset

    # ========================================================================
    # Panel 2: Throughput vs Resolution (log-log)
    # ========================================================================
    ax2 = plt.subplot(2, 2, 2)

    throughput_2d = BENCHMARK_DATA_2D['throughput']
    throughput_3d = BENCHMARK_DATA_3D['throughput']

    ax2.loglog(N_2d, throughput_2d, 'o-', linewidth=2, markersize=8,
               color='#1f77b4', label='2D')
    ax2.loglog(N_3d, throughput_3d, 's-', linewidth=2, markersize=8,
               color='#ff7f0e', label='3D')

    ax2.set_xlabel('Resolution N (per dimension)', fontsize=11)
    ax2.set_ylabel('Throughput (calls/sec)', fontsize=11)
    ax2.set_title('Computational Throughput', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, which='both')

    # Add shaded region for practical working range
    ax2.axvspan(64, 128, alpha=0.1, color='green', label='Practical range')
    ax2.text(85, throughput_3d.max()*0.5, 'Practical\nworking range',
             fontsize=9, ha='center', va='center', color='green', fontweight='bold')

    # ========================================================================
    # Panel 3: Scaling Efficiency (measured / theoretical)
    # ========================================================================
    ax3 = plt.subplot(2, 2, 3)

    # Compute scaling efficiency
    def compute_efficiency(N, time):
        """Scaling efficiency: measured_time / expected_time relative to N[0]"""
        efficiency = []
        for i in range(len(N)):
            expected = (N[i]/N[0])**3 * np.log2(N[i])/np.log2(N[0])
            measured = time[i] / time[0]
            efficiency.append(measured / expected)
        return np.array(efficiency)

    eff_3d = compute_efficiency(N_3d, time_3d)

    ax3.plot(N_3d, eff_3d, 's-', linewidth=2, markersize=8, color='#ff7f0e')
    ax3.axhline(1.0, color='black', linestyle='--', linewidth=1, alpha=0.5,
                label='Ideal (N³ log N)')
    ax3.fill_between(N_3d, 0.5, 1.0, alpha=0.1, color='green')

    ax3.set_xlabel('Resolution N (per dimension)', fontsize=11)
    ax3.set_ylabel('Scaling Efficiency\n(measured / theoretical)', fontsize=11)
    ax3.set_title('3D Scaling Efficiency', fontsize=12, fontweight='bold')
    ax3.set_ylim([0, 1.5])
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=10)

    # Add text annotation
    ax3.text(0.95, 0.05,
             'Values < 1.0 indicate\nbetter than theoretical scaling',
             transform=ax3.transAxes, fontsize=9, ha='right', va='bottom',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # ========================================================================
    # Panel 4: Memory Usage vs Resolution
    # ========================================================================
    ax4 = plt.subplot(2, 2, 4)

    memory_2d = BENCHMARK_DATA_2D['memory_mb']
    memory_3d = BENCHMARK_DATA_3D['memory_mb']

    ax4.semilogy(N_2d, memory_2d, 'o-', linewidth=2, markersize=8,
                 color='#1f77b4', label='2D (per field)')
    ax4.semilogy(N_3d, memory_3d, 's-', linewidth=2, markersize=8,
                 color='#ff7f0e', label='3D (per field)')

    # Add theoretical scaling lines
    mem_theory_2d = memory_2d[0] * (N_fine / N_2d[0])**2
    mem_theory_3d = memory_3d[0] * (N_fine / N_3d[0])**3
    ax4.semilogy(N_fine, mem_theory_2d, '--', color='#1f77b4', alpha=0.3)
    ax4.semilogy(N_fine, mem_theory_3d, '--', color='#ff7f0e', alpha=0.3)

    ax4.set_xlabel('Resolution N (per dimension)', fontsize=11)
    ax4.set_ylabel('Memory per field (MB)', fontsize=11)
    ax4.set_title('Memory Scaling (rfft format)', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    # Add horizontal lines for GPU memory limits
    ax4.axhline(1000, color='red', linestyle=':', linewidth=1, alpha=0.5)
    ax4.text(N_3d[-1]*1.1, 1000, '1 GB', fontsize=9, va='center', color='red')

    plt.tight_layout()

    # Save figure
    output_path = output_dir / 'performance_scaling.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    # Also save as PDF for publication quality
    output_path_pdf = output_dir / 'performance_scaling.pdf'
    plt.savefig(output_path_pdf, bbox_inches='tight')
    print(f"Saved: {output_path_pdf}")

    plt.close()


def print_summary_table():
    """Print formatted summary table of benchmark results."""
    print("\n" + "="*70)
    print("PERFORMANCE BENCHMARK SUMMARY (M1 Pro, JAX 0.4.20 with Metal)")
    print("="*70)

    print("\n3D Poisson Bracket (primary use case):")
    print("-" * 70)
    print(f"{'Resolution':<12} {'Time/call':<12} {'Throughput':<18} {'Memory/field'}")
    print("-" * 70)

    N_3d = BENCHMARK_DATA_3D['N']
    time_3d = BENCHMARK_DATA_3D['time_ms']
    throughput_3d = BENCHMARK_DATA_3D['throughput']
    memory_3d = BENCHMARK_DATA_3D['memory_mb']

    for i in range(len(N_3d)):
        res = f"{N_3d[i]}³"
        time_str = f"{time_3d[i]:.2f} ms"
        throughput_str = f"{throughput_3d[i]:.1f} calls/sec"
        memory_str = f"{memory_3d[i]:.1f} MB"
        print(f"{res:<12} {time_str:<12} {throughput_str:<18} {memory_str}")

    print("\n2D Poisson Bracket:")
    print("-" * 70)
    print(f"{'Resolution':<12} {'Time/call':<12} {'Throughput':<18} {'Memory/field'}")
    print("-" * 70)

    N_2d = BENCHMARK_DATA_2D['N']
    time_2d = BENCHMARK_DATA_2D['time_ms']
    throughput_2d = BENCHMARK_DATA_2D['throughput']
    memory_2d = BENCHMARK_DATA_2D['memory_mb']

    for i in range(len(N_2d)):
        res = f"{N_2d[i]}²"
        time_str = f"{time_2d[i]:.2f} ms"
        throughput_str = f"{throughput_2d[i]:.0f} calls/sec"
        memory_str = f"{memory_2d[i]:.2f} MB"
        print(f"{res:<12} {time_str:<12} {throughput_str:<18} {memory_str}")

    print("\nRealistic Workload Estimates (128³):")
    print("-" * 70)
    time_per_call = 31.6  # ms, sustained throughput
    brackets_per_step = 6
    time_per_step = time_per_call * brackets_per_step

    print(f"Time per bracket call:    {time_per_call:.1f} ms")
    print(f"Time per timestep:        {time_per_step:.1f} ms (6 brackets/step)")
    print(f"Timesteps per second:     {1000/time_per_step:.2f}")
    print(f"Time for 1K steps:        {time_per_step*1000/1000/60:.1f} min")
    print(f"Time for 100K steps:      {time_per_step*100000/1000/3600:.1f} hours")
    print("="*70 + "\n")


if __name__ == '__main__':
    print("Generating performance scaling plots...")
    print_summary_table()
    plot_scaling_analysis()
    print("\nDone! Plots saved to docs/figures/")
