# GANDALF Tutorial Notebooks

Interactive Jupyter notebooks for learning KRMHD turbulence simulations.

## Getting Started

### Prerequisites

```bash
# Install Jupyter (if not already installed)
uv pip install jupyter
```

### Launch Notebooks

```bash
# Start Jupyter from the project root
uv run jupyter notebook notebooks/

# Or launch a specific notebook
uv run jupyter notebook notebooks/01_getting_started.ipynb
```

## Available Tutorials

### 1. Getting Started (`01_getting_started.ipynb`)
**Duration:** ~10 minutes
**Runtime:** ~2 seconds on M1 Pro

Your first GANDALF simulation covering:
- Setting up a spectral grid
- Initializing turbulent states
- Running forced turbulence with energy injection
- Monitoring energy evolution
- Basic visualization

**Best for:** Newcomers to GANDALF, learning the basic workflow

---

### 2. Driven Turbulence (`02_driven_turbulence.ipynb`)
**Duration:** ~30 minutes
**Runtime:** ~20 seconds on M1 Pro

Comprehensive forced turbulence analysis covering:
- Computing energy spectra: E(k), E(k⊥), E(k∥)
- Identifying inertial range scaling (k⁻⁵/³)
- Understanding energy balance: injection vs dissipation
- Parameter scanning for different forcing amplitudes
- Steady-state detection

**Best for:** Understanding forced turbulence physics and spectral analysis

---

### 3. Analyzing Decay (`03_analyzing_decay.ipynb`)
**Duration:** ~20 minutes
**Runtime:** ~1-2 minutes on M1 Pro

Decaying turbulence analysis covering:
- Exponential energy decay: E(t) = E₀ exp(-γt)
- Selective decay: magnetic energy dominance
- Spectral slope evolution
- Measuring decay rates and characteristic timescales

**Best for:** Understanding decaying turbulence and energy conservation

---

## Tips for Using Notebooks

### Recommended Order
1. Start with **01_getting_started.ipynb** - builds foundation
2. Progress to **02_driven_turbulence.ipynb** - adds complexity
3. Finish with **03_analyzing_decay.ipynb** - contrasts with forced case

### Experimentation
- **Modify parameters** - notebooks encourage hands-on experimentation
- **Add cells** - extend analyses with custom diagnostics
- **Export plots** - all visualizations can be saved for publications

### Common Working Directories
Notebooks assume you're running from the project root. If you encounter `ModuleNotFoundError`, ensure you've:
```bash
# Activate the virtual environment
source .venv/bin/activate

# Or use uv run
uv run jupyter notebook
```

## Troubleshooting

### Import Errors
```python
ModuleNotFoundError: No module named 'krmhd'
```

**Solution:** Make sure you're in the project root and the virtual environment is activated:
```bash
cd /path/to/gandalf
source .venv/bin/activate
jupyter notebook
```

### JAX Device Issues
```python
RuntimeError: No GPU/TPU found
```

**Solution:** JAX will fall back to CPU automatically. For GPU acceleration:
- **macOS**: Install jax-metal with `uv sync --extra metal`
- **Linux**: Ensure CUDA drivers are installed

### Slow Execution
If notebooks run slowly:
- Check JAX device: `import jax; print(jax.devices())`
- Reduce resolution parameters (e.g., 32³ instead of 64³)
- Use smaller `n_steps` for initial exploration

## Exporting Results

### Save Figures from Notebooks
All plots can be saved with:
```python
plt.savefig('my_figure.png', dpi=150, bbox_inches='tight')
```

### Export Notebook to PDF/HTML
```bash
# Export to HTML
jupyter nbconvert --to html notebooks/01_getting_started.ipynb

# Export to PDF (requires LaTeX)
jupyter nbconvert --to pdf notebooks/01_getting_started.ipynb
```

## Contributing

Found an issue or want to add a new tutorial?
- **Report issues:** https://github.com/anjor/gandalf/issues
- **Contribute:** See `CONTRIBUTING.md` for guidelines

## See Also

- **Python examples:** `examples/benchmarks/` - Production-ready scripts
- **Documentation:** `docs/` - Detailed physics and numerical methods
- **Parameter validation:** `python -m krmhd validate --help`
