---
name: check-a-run
description: Use when running a GANDALF simulation or judging whether one is healthy or valid. Signs of numerical instability, and the linear and nonlinear benchmarks a change should pass.
---

# Checking a GANDALF run

## Detecting numerical instabilities

When running turbulence simulations, watch for these warning signs:
1. **Monitor max field amplitude**: Track `max(|z±|)` during evolution
   - Healthy: Saturates to O(1-10) values in forced turbulence
   - Warning: Exponential growth beyond O(100) indicates instability
2. **Plot E(t) on log scale**: Detect exponential growth
   - Healthy: Steady plateau or slow linear growth in forced runs
   - Warning: Straight line on log-scale = exponential blow-up
3. **Check for NaN/Inf**: Added in alfvenic_cascade_benchmark.py:350-360
   - Terminates with diagnostic message if detected
4. **Spectrum pile-up**: Look for energy accumulating at k_max
   - Healthy: Sharp exponential cutoff from hyper-dissipation
   - Warning: Flat spectrum extending to k_max = dealiasing failure

Full details: `compute_turbulence_diagnostics()` in `src/krmhd/diagnostics.py`.

## Validation Suite

### Linear Physics Tests
1. **Alfvén wave dispersion**: ω² = k∥²v_A²
2. **Slow mode damping**: Verify passive advection without back-reaction
3. **Landau damping**: Emerges from parallel streaming in Hermite moments

### Nonlinear Benchmarks
1. **Orszag-Tang vortex**: Standard MHD test (fluid limit)
2. **Decaying turbulence**: Check k⁻⁵/³ spectrum
3. **Alfvénic cascade** (Thesis Section 2.6.3, Figure 2.2): Forced turbulence with k⊥⁻⁵/³ critical-balance spectrum
4. **Energy partition**: Verify equipartition in steady state
5. **Selective decay**: Magnetic energy should dominate at late times

The critical checks (passive slow modes, phase mixing, CFL) stay in the root CLAUDE.md.
