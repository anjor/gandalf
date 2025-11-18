# KRMHD Utility Scripts

This directory contains utility scripts for post-processing, visualization, and analysis of KRMHD simulation data.

## Scripts

### `plot_checkpoint_spectrum.py`

Plot perpendicular energy spectra E(k⊥) from checkpoint files without rerunning simulations. Separates kinetic and magnetic contributions and compares to k⊥^(-5/3) Kolmogorov spectrum.

**Basic Usage:**

```bash
# Standard style (mode number axis)
uv run python scripts/plot_checkpoint_spectrum.py checkpoint_t0300.0.h5

# Thesis style (wavenumber axis, clean formatting)
uv run python scripts/plot_checkpoint_spectrum.py --thesis-style checkpoint.h5

# Custom output filename
uv run python scripts/plot_checkpoint_spectrum.py --output fig_spectrum.png checkpoint.h5

# Interactive display
uv run python scripts/plot_checkpoint_spectrum.py --show checkpoint.h5
```

**Output:**

Creates two-panel plot showing:
- **Left panel**: Total energy spectrum (or kinetic in thesis style) with k⊥^(-5/3) reference
- **Right panel**: Kinetic vs Magnetic comparison (or magnetic in thesis style)

Prints energy summary: total energy, magnetic fraction, simulation time.

**Interpretation:**
- k⊥^(-5/3) match in n ~ 3-10: Healthy turbulence cascade
- High magnetic fraction (f_mag > 0.5): Selective decay underway
- Exponential cutoff at high-k: Hyper-dissipation working correctly
- Flat/rising spectrum at high-k: Under-dissipated, increase η

### `field_line_visualization.py`

Visualize magnetic field line wandering and compare parallel energy spectra computed along field lines vs along the z-axis. Demonstrates spectral interpolation for field line following.

**Physical Context:**

In RMHD turbulence, magnetic field lines deviate from straight paths along B₀ due to perpendicular field perturbations δB⊥. The field line wandering distance scales as δr⊥ ~ ε Lz where ε ~ δB⊥/B₀ is the RMHD ordering parameter. For valid RMHD (ε << 1), wandering should be small compared to box size.

True k∥ spectra should be computed along curved field lines, not along the straight z-axis. This script demonstrates the difference between these two approaches and validates the spectral interpolation infrastructure for field line following.

**Basic Usage:**

```bash
# Run example simulation and create visualizations
uv run python scripts/field_line_visualization.py

# Use existing checkpoint
uv run python scripts/field_line_visualization.py --checkpoint checkpoint.h5

# Adjust number of field lines traced
uv run python scripts/field_line_visualization.py --num-lines 10

# Custom output directory
uv run python scripts/field_line_visualization.py --output-dir ./figures/
```

**Output:**

Creates two visualization files:
- **field_lines_3d.png**: 3D trajectories showing field line wandering in x-y plane as function of z
- **parallel_spectra_comparison.png**: E(k∥) computed along field lines (curved) vs z-axis (straight)

**Key Features:**
- **Spectral interpolation**: Uses FFT padding (2× resolution) for accurate field interpolation
- **RK2 integration**: Second-order Runge-Kutta for field line tracing
- **RMHD validity check**: Prints wandering amplitude to verify ε << 1 ordering

**Interpretation:**
- Small wandering (δr⊥ << Lz): RMHD ordering valid, field line corrections minimal
- Large wandering (δr⊥ ~ Lz): RMHD breakdown, need full MHD treatment
- Spectral differences: Shows how field line curvature affects parallel structure measurements

**Note:** This is a diagnostic tool demonstrating field line following infrastructure. Full k∥ spectra via FFT along curved field lines is deferred (infrastructure complete, see CLAUDE.md Issue #25).

### `plot_energy_evolution.py`

Creates publication-quality plots of energy evolution from Orszag-Tang vortex simulations.

**Basic Usage:**

```bash
# Run simulation and create plot in one step
uv run python scripts/plot_energy_evolution.py --run-simulation

# Just create plot from existing data
uv run python scripts/plot_energy_evolution.py
```

**Advanced Options:**

```bash
# Custom output filename
uv run python scripts/plot_energy_evolution.py --output my_plot.png

# Use absolute time instead of normalized t/τ_A
uv run python scripts/plot_energy_evolution.py --no-normalize-time

# Use different history file
uv run python scripts/plot_energy_evolution.py --history-file my_data.pkl
```

**Output:**

Creates a plot showing:
- **Total Energy** (solid black line) - should be approximately constant
- **Kinetic Energy** (dashed red line) - decreases over time
- **Magnetic Energy** (dotted green line) - increases over time (selective decay)

Time axis is normalized by Alfvén crossing time: τ_A = L/v_A

**Example Plot:**

The plot matches Figure 2.1 from the thesis, showing the characteristic energy exchange in Orszag-Tang vortex evolution:
- Kinetic energy decreases as flow structures develop
- Magnetic energy increases as current sheets form
- Total energy remains nearly constant (validates energy conservation)

## Data Format

Energy history is saved as a Python pickle file containing:
```python
{
    'times': np.array,        # Time snapshots
    'E_total': np.array,      # Total energy
    'E_magnetic': np.array,   # Magnetic energy
    'E_kinetic': np.array,    # Kinetic energy
    'E_compressive': np.array,# Compressive energy
    'v_A': float,             # Alfvén velocity
    'Lx': float,              # Domain size
}
```

## Adding New Scripts

When adding new utility scripts to this directory:
1. Use `#!/usr/bin/env python3` shebang
2. Add comprehensive docstrings
3. Support command-line arguments with `argparse`
4. Update this README with usage instructions
