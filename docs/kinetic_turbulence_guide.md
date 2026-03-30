# Getting to a Credible Kinetic-Turbulence Run

A progressive workflow for going from "code works" to "kinetic turbulence results I can trust." Each step builds on the previous one. Do not skip steps — issues that are invisible on a dev grid can become catastrophic at production resolution.

## Step 1: Fluid-only benchmark (~5 min)

**Goal:** Confirm the code works and establish a baseline with known-good physics.

```bash
uv run python examples/benchmarks/alfvenic_cascade_benchmark.py --resolution 32 --total-time 50
```

**What to check:**
- Energy reaches steady state (ΔE/⟨E⟩ < 10%)
- Perpendicular spectrum shows k⊥^(-5/3) inertial range
- No NaN/Inf or exponential energy growth

**Success:** Steady-state turbulence with recognizable inertial range. This is a pure RMHD run (M=0), so no kinetic effects yet.

**Common failure:** Energy blows up → reduce `force_amplitude` or increase `eta`. See `examples/benchmarks/README.md` for parameter guidance.

## Step 2: Low-drive coupled dev-grid run (~10 min)

**Goal:** Verify kinetic coupling works and Hermite truncation is adequate at low resolution.

```bash
uv run python examples/benchmarks/krmhd_lowkz_turbulence.py
```

This runs 32³ with M=8 Hermite moments, using low-k_z balanced Elsasser forcing.

**What to check:**
- Fluid energy reaches approximate steady state
- Hermite energy (printed separately) does not grow without bound
- Hermite spectrum shows decay toward high m (not pile-up)
- Tail metric R_tail < 0.05

**Success:** Both fluid and Hermite sectors reach steady state. R_tail confirms truncation is resolved.

**Common failure:**
- R_tail > 0.10 → increase M (try M=16) or increase nu
- Hermite energy grows indefinitely → forcing too strong for this M, reduce amplitude

## Step 3: Production-grid smoke test (~30 min)

**Goal:** Confirm that behavior at dev resolution carries over to production resolution. Higher resolution resolves more of the cascade, which can reveal truncation issues invisible at 32³.

Modify the example or run directly:
```python
# In krmhd_lowkz_turbulence.py, change:
N = 64
M = 16
```

Or create a quick script:
```bash
# Copy and modify
cp examples/benchmarks/krmhd_lowkz_turbulence.py /tmp/smoke_test.py
# Edit N=64, M=16, then run:
uv run python /tmp/smoke_test.py
```

**What to check (same as Step 2, plus):**
- Compare R_tail at 32³ vs 64³ — should be similar or better
- Energy level should be in the same ballpark (not orders of magnitude different)
- Hermite spectrum shape should be qualitatively similar

**Warning sign:** R_tail much worse at 64³ than 32³. This means the stronger cascade at higher resolution is overwhelming the truncation. Fix before proceeding: increase M or nu.

## Step 4: Collisionality scan

**Goal:** Map out the dependence on collision frequency nu to find the physically interesting regime.

Vary nu across {0.1, 0.25, 0.5, 1.0} at fixed resolution (64³, M=16 or higher). For each run:

**What to check:**
- Hermite spectrum shape: lower nu → longer inertial range in Hermite space, but higher risk of under-resolution
- Collisional cutoff location: should move to higher m as nu decreases
- R_tail: must remain < 0.05 for all runs. If R_tail > 0.05 at low nu, increase M before reducing nu further
- Fluid energy: should be approximately independent of nu (collisions mainly affect Hermite moments, not fluid)

**Expected behavior:**
| nu | Inertial range | R_tail | Notes |
|----|---------------|--------|-------|
| 1.0 | Short (strong damping) | Very small | Safe but over-dissipated |
| 0.5 | Moderate | Small | Good starting point |
| 0.25 | Extended | Check carefully | Thesis value |
| 0.1 | Long | May exceed 0.05 | May need larger M |

**Batch workflow:** See [Parameter Scans](parameter_scans.md) for systematic scan infrastructure.

## Key Gotchas

1. **`energy()` is fluid-only.** It excludes Hermite moment energy. Always check `hermite_moment_energy()` in kinetic runs. See [Hermite Truncation Checklist](hermite_truncation_checklist.md).

2. **Full-|k| forcing vs low-k_z forcing.** `force_alfven_modes_gandalf()` forces all k_z modes. For kinetic runs, prefer `force_alfven_modes_balanced(max_nz=1)` to avoid driving unphysical parallel phase mixing. See [Numerical Methods: Forcing Schemes](numerical_methods.md#forcing-schemes).

3. **Collision operator is normalized.** The damping rate is `nu * (m/M)^n`, not `nu * m^n`. The maximum damping rate at m=M is simply nu, independent of M. See [Numerical Methods: Collision Operator](numerical_methods.md#collisions-and-normalized-collision-operator).

4. **Resolution can unmask truncation issues.** A run that looks calm at 32³ may show a much stronger Hermite cascade at 64³ because more perpendicular modes are resolved, driving more energy into velocity space. Always smoke-test at production resolution before committing to long scans.
