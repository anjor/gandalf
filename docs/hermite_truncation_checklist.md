# Hermite Truncation Checklist

How to verify that your Hermite moment truncation (M) is adequate for a kinetic turbulence run.

## Quick Checklist

1. **Inspect W(m) spectrum**: Should show power-law decay (e.g. m^(-1/2)), NOT a pile-up or flattening near m=M.

2. **Compute tail metric**:
   ```python
   from krmhd.diagnostics import hermite_moment_energy
   E_m = hermite_moment_energy(state)
   R_tail = float(E_m[-4:].sum() / E_m.sum())
   ```
   - R_tail < 0.05: Well-resolved truncation
   - R_tail 0.05–0.10: Marginal — consider increasing M or nu
   - R_tail > 0.10: Under-resolved — increase M or nu before trusting results

3. **Monitor collisional dissipation rate** eps_nu(t): Should be stationary in steady state, not growing. Growing eps_nu indicates energy piling up at the truncation boundary faster than collisions can remove it.

4. **Check at multiple times**: A run that looks resolved at early times may develop truncation issues as the cascade fills in. Check R_tail at several points during the averaging window.

## Reference: Resolved Cascade

The Hermite cascade investigation (see `HERMITE_CASCADE_INVESTIGATION.md`) established a known-good reference case:

```bash
uv run python examples/benchmarks/hermite_forward_backward_flux.py \
  --steps 90 --nu 0.25 --hyper-n 6 --amplitude 0.0035 \
  --hermite-moments 128 --resolution 32 --thesis-style
```

**Expected results:**
- Hermite spectrum: m^(-1/2) power law from m=2 to m~20, then sharp exponential cutoff from hyper-collisions (n=6)
- R_tail < 0.01 (well-resolved — energy drops many orders of magnitude before truncation)
- Forward flux C+ >> backward flux C- (healthy forward cascade in velocity space)

**Visual signatures of "resolved" vs "under-resolved":**

| Feature | Resolved | Under-resolved |
|---------|----------|----------------|
| Spectrum near m=M | Sharp exponential drop | Flat or rising |
| R_tail | < 0.05 | > 0.10 |
| eps_nu(t) | Stationary | Growing |
| Spectrum slope | Clean power law | Steepening or flattening at high m |

## What To Do When Under-resolved

In order of preference:

1. **Increase M** (more Hermite moments): Most direct fix. Double M and re-run. Cost: linear increase in memory and compute per timestep.

2. **Increase nu** (collision frequency): Strengthens the collisional cutoff, dissipating energy before it reaches the truncation boundary. Trade-off: reduces the extent of the inertial range in Hermite space.

3. **Increase hyper_n** (collision order): Sharpens the cutoff (concentrates damping near m=M), extending the inertial range for a given nu. Values used in practice: n=2 (moderate), n=6 (thesis value).

4. **Reduce forcing amplitude**: Less energy injected means less energy to cascade through Hermite space. Only appropriate if physics allows — may change the turbulence regime.

## Relationship to Fluid Energy

The `energy()` function in `physics.py` computes **fluid energy only** (magnetic + kinetic + compressive). It does NOT include Hermite moment energy. For kinetic runs:

- Steady-state in fluid energy does NOT guarantee steady-state in Hermite energy
- Always check `hermite_moment_energy()` alongside `energy()` to get the full picture
- The example `krmhd_lowkz_turbulence.py` demonstrates both diagnostics side by side

## Further Reading

- [HERMITE_CASCADE_INVESTIGATION.md](HERMITE_CASCADE_INVESTIGATION.md) — Detailed cascade reproduction study
- [Numerical Methods: Hermite Moment Expansion](numerical_methods.md#hermite-moment-expansion) — Implementation details
- [Numerical Methods: Collision Operator](numerical_methods.md#collisions-and-normalized-collision-operator) — Normalized collision formulation
