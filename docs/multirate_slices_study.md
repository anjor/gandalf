# Multirate slice stepping

Issue #191, item 1. `lawson_rk4` advances every phase-velocity slice of the Hermite hierarchy with one step, sized for the fastest slice. `scheme="lawson_rk4_multirate"` gives each slice its own step. This page describes the scheme and measures its stability, order, conservation and cost against `lawson_rk4`.

## Result

- Safe as an opt-in for runs with hyper-resistivity. Every forced and decaying P1-like run stayed bounded, at 32³ and 64³, M = 16 and 64, for up to 20 τ_A, under the Courant rule and under P1's exact rule. In a frozen flow each slice is stable exactly when its own substep is inside the RK4 limit h_j·ρ_j ≤ 2√2.
- Not for η = ν = 0 at a large macro step. z± take midpoint RK2 steps at their own Courant limit, as in `imex_rk222`. At 0.8 of that limit an ideal run blew up within 0.76 τ_A. At 0.3, E_A grew by 1.4e-3 per τ_A, against 6.7e-5 for `lawson_rk4` at its own step.
- Second order without dissipation, with the same errors as `lawson_rk4` at the same macro step, to three digits. With P1's dissipation both are first order, because the damping is applied after the step. The cost of the speed-up is the larger z± step: at each scheme's own Courant step the multirate error is 10 to 55 times larger without dissipation and 3 to 8 times with it.
- Time averages move with the z± step. ⟨W⟩ agrees with `lawson_rk4` within the seed scatter at M = 16 and sits 4 to 7% higher at M = 64. ⟨E_A⟩ moves by up to 16%, as much as `lawson_rk4` moves between its Courant step and P1's step.
- Cost at M = 64: 2.0 to 2.2 times less wall time per τ_A than `lawson_rk4` on the CPU, under either rule, from 1.8 to 2.0 times fewer slice updates. The bound from the slice speeds is 2.2. At M = 16: 1.1 to 1.7 times. The gain grows with M.

## The scheme

Slice j is a passive scalar advected by Φ + √β_i λ_j Ψ, with λ_j an eigenvalue of the streaming matrix T. The largest |λ_j| grows like √(2M), so most slices move much slower than the one that sets dt. Between the collision updates the slices are independent.

One macro step H of `lawson_rk4_multirate` (`krmhd.multirate`):

1. z± advance by H with `lawson_rk4`'s Lawson midpoint RK2. This gives z± at t, at t + H/2 (the Euler predictor) and at t + H.
2. Slice j takes n_j Lawson-RK4 substeps of H/n_j, with n_j = ceil(H·v_j / (C·Δ)). Here v_j = max|∇⊥(Φ + √β_i λ_j Ψ)| at time t, Δ = min(Δx, Δy, Δz), and C is the substep Courant number (`substep_cfl`, default 0.3). This is `compute_cfl_timestep`'s Courant rule applied slice by slice. The slices with the same n_j form a rate class. Each substep is `lawson_rk4`'s Hermite step: the streaming factor is exact and diagonal, and the advection is RK4 in the interaction picture.
3. Collisions, resistive and k_z damping act once per macro step, as in `lawson_rk4`. Collisions are folded into the back-projection from the slices to the moments.

`compute_cfl_timestep(mode="multirate")` and `CflDt` under this scheme give H from the Elsasser and v_A terms of the Courant estimate. The Hermite term is left out because each slice substeps to its own limit.

The Courant rule is loose for the slices. In the turbulent states below, each slice's exact radius ρ_j (`slice_advection_radius`) is 1.38 to 1.73 times v_j/Δ, within ±6% across the slices of one state, against the worst case k⊥,max·Δ = 2.96. `compute_cfl_timestep(mode="exact")` uses that for `lawson_rk4`, where it gives 1.7 to 2.0 times the Courant dt. The multirate counterpart is `exact_substep_cfl(state, s)`. It returns one Courant number per slice, C_j = s·2√2·v_j/(ρ_j·Δ), which puts every slice at s of its own RK4 limit on that state. Passed as `substep_cfl`, it keeps the shape factor ρ_j/v_j of the state it was computed on, while n_j still follows v_j step by step.

### Rate classes

The classes are integers, not powers of two. Powers of two nest the stage times, so that classes can share evaluations of Φ and Ψ. Interpolation in time makes the stage times free, so nesting buys nothing, while rounding a slice up to the next power of two costs about a factor 1.44 in substeps on average. The kernel sorts the slices by n_j and advances one slice at a time (`hermite_chunk=1`) in a loop whose trip count is n_j, on the device. A larger chunk batches several slices, which then take the substeps of the fastest one.

### Φ and Ψ at the substep stage times

`lawson_rk4` evaluates the Hermite stages with z± at t, at the predictor and at t + H. The predictor is off by O(H²), so each Hermite step makes an O(H³) error, and the coupled scheme is second order, like z± itself.

The multirate step takes Φ and Ψ at t + θH from the quadratic through those three states. The interpolation acts on the real-space gradients of Φ and Ψ, a linear map of z±, so it costs no FFTs.

- Order. The interpolant is off by the predictor's O(H²), weighted by 4θ(1 − θ) ≤ 1, plus its own O(H³). The Hermite error per step stays O(H³), and the scheme stays second order. In a flow that does not change in time the interpolant is exact, and each slice is integrated by RK4 at fourth order.
- One substep. At θ = 0, 1/2 and 1 the Lagrange weights are exactly 0 and 1. With n_j = 1 for every slice the step is `lawson_rk4`'s to round-off.
- Why not linear. Interpolating between t and t + H also keeps second order, but it replaces the predictor at the midpoint, so it does not contain `lawson_rk4`. A cubic would need a fourth state and cannot beat the second order of z±.
- The Alfvén phase. Interpolating z± in the lab frame also interpolates exp(±i k_z t). The relative error is below 0.008·(k_z H)³. Within the v_A Courant limit, k_z,max·H ≤ 2πC/3, which bounds it by 0.2% at k_z,max for C = 0.3. Interpolating in the interaction picture would remove it at 4 inverse FFTs per stage time.

### Stability

Each interpolated Φ + √β_i λ_j Ψ is a real band-limited field, so each substep's advection operator is skew-adjoint and RK4 is stable for h_j·ρ_j ≤ 2√2 (`compute_cfl_timestep`, Issue #189). The Courant rule bounds ρ_j by k⊥,max·v_j, and k⊥,max·Δ ≤ 2√2·π/3 on the 2/3-dealiased grid. So C ≤ 0.955 keeps every substep inside the RK4 limit in a frozen flow.

z± still take the macro step H with midpoint RK2, whose |R(iy)|² = 1 + y⁴/4 has no stability interval. In `lawson_rk4` the fastest slice keeps dt several times below the Elsasser limit. In the multirate step z± run at their own Courant limit and rely on dissipation there, as they do in `imex_rk222`.

## Setup

The physics is P1's `forced_beta1` at P1's test parameters, as in `docs/precision_float32_study.md`: Λ = √5, β_i = 1, zero closure, hyper-resistivity η = 200 (r = 2), hyper-collisions ν = 30 (n = 6), and white-noise forcing of Φ and g_0 on k⊥L/2π ∈ [1, 2], |n_z| = 1, at ε_A = ε_g = 0.3. Everything runs in float64 on the CPU, through `DeviceLoop`. GPU is not measured.

Four states:

| state | how it was made |
|---|---|
| 32³, M = 16 | 6 τ_A of forced `lawson_rk4` from rest |
| 32³, M = 64 | the 32³, M = 16 state with g padded to M = 64, then 3 τ_A of forced `lawson_rk4` |
| 64³, M = 16 | the 32³, M = 16 state padded in Fourier space, then 1 τ_A of forced `lawson_rk4` |
| 64³, M = 64 | the 64³, M = 16 state with g padded to M = 64 (cost only depends on z±) |

Two step rules, each applied to both schemes:

- Courant: `CflDt(0.3, every=10)`. `lawson_rk4` takes the Courant estimate with the Hermite term. The multirate step takes it without (`mode="multirate"`), and its substeps use the same Courant number 0.3 slice by slice.
- Exact, P1's rule: a fixed dt of 0.8 times the exact limit at safety 1, recomputed every 0.25 τ_A. For `lawson_rk4` that is `compute_cfl_timestep(mode="exact")`, set by the fastest slice or by z±. The multirate step takes 0.8 times the Elsasser and v_A limit, and `exact_substep_cfl(state, 0.8)` puts every slice at 0.8 of its own RK4 limit.

## Stability

### Frozen flow, slice by slice

In the shear flow z± = (1 ± 1/2)Φ(x), z± are steady, the interpolation is exact, and slice j is advected by (1 + c_j/2)Φ, so the slices evolve independently and ρ_j/v_j is one constant κ. With 16 × 16 × 8 points, M = 16, η = ν = 0 and a macro step of 5·2√2/ρ_max, the substep Courant number is set so that the largest h_j·ρ_j is 0.95·2√2 or 1.15·2√2. The norm of each slice after 12 macro steps (`frozen`):

| setting | n_j | slices with h_j·ρ_j ≤ 0.95·2√2 | slices with h_j·ρ_j ≥ 1.05·2√2 |
|---|---|---|---|
| C·κ = 0.95·2√2 | 1 to 6 | all 17, norm ratio ≤ 1 − 4e-6 | none |
| C·κ = 1.15·2√2 | 1 to 5 | 9, norm ratio ≤ 1 + 9e-6 | 3: ×1.2e3, ×8.1e5, ×7.6e12 |

Each slice is stable exactly when its own substep is inside the RK4 limit, whatever the other slices do. The 9e-6 is round-off of the growing slices leaking through the projections. Slices between 1.00 and 1.05 of the limit grow slowly (×2.6 at 1.012, ×144 at 1.045). `tests/test_multirate.py` checks the same threshold.

### Turbulence with P1's dissipation

Every forced and decaying run stayed bounded:

| state | rule | forced, multirate / `lawson_rk4` | decaying |
|---|---|---|---|
| 32³, M = 16 | Courant | 20 τ_A / 20 τ_A | 5 τ_A / 5 τ_A |
| 32³, M = 16 | exact | 20 τ_A / 20 τ_A | 5 τ_A / 5 τ_A |
| 32³, M = 16 | Courant 0.5 and 0.8 for H, 0.3 for the substeps | 5 τ_A each | |
| 32³, M = 64 | exact | 10 τ_A / 10 τ_A | |
| 32³, M = 64 | Courant | 2 τ_A / 2 τ_A | |
| 64³, M = 16 | exact | 2 τ_A / 2 τ_A | |
| 64³, M = 16 | Courant | 0.5 τ_A / 0.5 τ_A | |
| 64³, M = 64 | exact | 0.3 τ_A / 0.3 τ_A | |
| 64³, M = 64 | Courant | 0.06 τ_A / 0.05 τ_A | |

The decaying runs agree at equal times: E_A(5 τ_A)/E_A(0) is 0.2300 for both schemes under the Courant rule, and 0.2306 against 0.2302 under the exact rule. W(5 τ_A)/W(0) is 0.0092 for all four.

Time averages after the first 2 τ_A of the forced runs, with a second forcing seed as the yardstick:

| state, rule | z± step, multirate / `lawson_rk4` | ⟨E_A⟩ multirate, seeds 0 and 1 | ⟨E_A⟩ `lawson_rk4` | ⟨W⟩ multirate | ⟨W⟩ `lawson_rk4` |
|---|---|---|---|---|---|
| 32³, M = 16, Courant, 20 τ_A | 2.3e-3 / 7.3e-4 | 0.551, 0.567 | 0.498, 0.523 | 0.215, 0.221 | 0.217, 0.213 |
| 32³, M = 16, exact, 20 τ_A | 6.1e-3 / 3.7e-3 | 0.519, 0.536 | 0.579, 0.555 | 0.228, 0.224 | 0.214, 0.238 |
| 32³, M = 64, exact, 10 τ_A | 5.6e-3 / 1.5e-3 | 0.486, 0.545 | 0.581, 0.616 | 0.331, 0.335 | 0.318, 0.313 |

A seed change moves ⟨E_A⟩ by 0.016 to 0.060 and ⟨W⟩ by 0.004 to 0.025. ⟨W⟩ agrees within that, except at M = 64, where the multirate runs sit 4 to 7% higher in both seeds. ⟨E_A⟩ differs by up to 16%, in both seeds, with a sign that follows the z± step rather than the scheme: `lawson_rk4` itself moves ⟨E_A⟩ by +0.03 and +0.08 between the Courant step and P1's 5 times larger step. The accuracy runs below show that at a given z± step the slice substeps add no error of their own, so these shifts are the dependence of the statistics on the z± step, which the multirate step changes.

### Without dissipation

η = ν = 0, unforced, from the 32³, M = 16 state, 1 τ_A:

| rule | scheme | ΔE_A/E_A | ΔW/W |
|---|---|---|---|
| Courant | `lawson_rk4` | +6.7e-5 | −8.4e-9 |
| Courant | multirate | +1.4e-3 | −7.7e-7 |
| exact | `lawson_rk4` | +8.8e-3 | −5.8e-5 |
| exact | multirate | blew up at 0.76 τ_A | −4.9e-5 by 0.53 τ_A, before z± ran away |

E_A grows because z± take midpoint RK2 steps, whose amplification |R(iy)|² = 1 + y⁴/4 exceeds 1 at every step size, and nothing damps the grid scale. The multirate step runs z± at their own Courant limit: 3.2 times `lawson_rk4`'s step under the Courant rule, and 0.8 of the Elsasser limit under the exact rule, where the grid-scale growth runs away within one τ_A. W stays passive until z± blow up. `imex_rk222` has the same z± step and the same dependence on dissipation.

## Accuracy

Unforced runs of T = 8H from the 32³ states, against `lawson_rk4` at H/64 (`accuracy`). H is the multirate macro step at Courant 0.3: 2.29e-3 at M = 16 and 2.10e-3 at M = 64. Relative L2 errors of g:

| M | dissipation | multirate, H and C halved together (n_j fixed) | multirate, C = 0.3 | `lawson_rk4` at the same H |
|---|---|---|---|---|
| 16 | none | 1.10e-5, 2.76e-6, 6.91e-7, 1.71e-7 | the same to 3 digits | 1.10e-5, 2.76e-6, 6.91e-7, 1.71e-7 |
| 16 | P1 | 6.87e-4, 3.38e-4, 1.63e-4, 7.62e-5 | the same | the same |
| 64 | none | 9.33e-6, 2.35e-6, 5.87e-7, 1.45e-7 | the same | the same |
| 64 | P1 | 3.20e-4, 1.58e-4, 7.62e-5, 3.56e-5 | the same | the same |

- Without dissipation the observed order is 2.00 ± 0.01, like z±. The errors of z± are identical in all three columns (3.0e-5 at H), and g follows them: the slice substeps add nothing visible at the three digits shown.
- With P1's dissipation both schemes are first order (1.02 to 1.10). The resistive and collisional factors are applied after the step, a Lie splitting whose error is O(H) per unit time and dominates at η = 200, ν = 30. That is a property of `lawson_rk4` that the multirate step inherits.
- At the same macro step the multirate step is as accurate as `lawson_rk4`, while its fast slices take substeps inside their limit.

The price of the speed-up is the larger z± step. At each scheme's own Courant step the multirate error of g is 10.5 times `lawson_rk4`'s at M = 16 and 55 times at M = 64 without dissipation (the square of the step ratio), and 3.4 and 8.2 times with P1's dissipation (the step ratio).

## Conservation

In the ideal accuracy runs, W changes over T = 8H by −3.1e-10 (multirate) against −2.9e-12 (`lawson_rk4` at its own dt) at M = 16, and −1.5e-10 against +1.4e-13 at M = 64. At the same macro step `lawson_rk4` loses more, −1.1e-9 and −5.8e-10, because its fastest slices then take the whole step H. Over 1 τ_A (the table above) the multirate step loses 7.7e-7 of W under the Courant rule: 90 times `lawson_rk4`, and 8e-7 of the injection ε_g·τ_A of the forced runs. Under the exact rule `lawson_rk4` loses 5.8e-5 of W per τ_A. The loss is RK4's damping of the top modes, |R(iy)|² = 1 − y⁶/72 + ..., which the multirate step applies to every slice near its own limit, not only to the fastest.

## Cost

Wall time per τ_A on the CPU, forced, from each state, `DeviceLoop`, 5 alternating repetitions of each scheme (median and range), plus the slice updates per τ_A of the same window (`cost`):

| state | rule | `lawson_rk4` dt | multirate H | `lawson_rk4` s/τ_A | multirate s/τ_A | wall speed-up | slice updates, `lawson_rk4` / multirate |
|---|---|---|---|---|---|---|---|
| 32³, M = 16 | Courant | 7.3e-4 | 2.3e-3 | 42.7 [42.5, 43.0] | 29.7 [29.7, 30.4] | 1.43 | 1.51 |
| 32³, M = 16 | exact | 3.7e-3 | 6.1e-3 | 9.0 [9.0, 9.0] | 8.0 [8.0, 8.2] | 1.12 | 1.26 |
| 32³, M = 64 | Courant | 2.9e-4 | 2.1e-3 | 408 [408, 409] | 199 [199, 199] | 2.05 | 1.92 |
| 32³, M = 64 | exact | 1.5e-3 | 5.6e-3 | 90.7 [90.6, 90.8] | 44.7 [44.6, 44.9] | 2.03 | 1.98 |
| 64³, M = 16 | Courant | 3.2e-4 | 1.2e-3 | 902 [900, 937] | 530 [528, 533] | 1.70 | 1.54 |
| 64³, M = 16 | exact | 1.4e-3 | 3.1e-3 | 198 [198, 198] | 141 [140, 142] | 1.41 | 1.32 |
| 64³, M = 64 | Courant | 1.5e-4 | 1.2e-3 | 7050 [7032, 7065] | 3166 [3116, 3241] | 2.23 | 1.91 |
| 64³, M = 64 | exact | 6.7e-4 | 3.1e-3 | 1554 [1553, 1558] | 776 [772, 779] | 2.00 | 1.76 |

GPU is not measured. The machine was shared; every timed run held a lock that other heavy jobs also took.

What limits the speed-up:

- The bound. The mean of ρ_j/ρ_max over the slices is 0.497 at M = 16 and 0.454 at M = 64 (`radii`), close to the issue's 0.44, so a scheme that steps each slice at exactly its own limit would make 2.0 and 2.2 times fewer slice updates. The Courant speeds v_j give the same ratios to 3%.
- Whole substeps. n_j is an integer, so each slice steps a little below its limit, and slices slower than z± still take one substep of H. With at most 4 substeps at M = 16 this costs a factor 1.3, with 8 at M = 64 a factor 1.15. Under the exact rule H/dt is 1.6 to 4.7, against 3.2 to 7.9 under the Courant rule, so n_j is smaller and the rounding costs more.
- A slice update costs about the same in both schemes, 1.7 to 2.0 ms at 32³ and 14 to 16 ms at 64³: three FFTs per stage. At 64³, M = 64 the multirate update, one slice at a time, is 12% cheaper than one slice of `lawson_rk4`'s batch of 65. The z± part, 9 to 13% of a `lawson_rk4` step at M = 16 and 3% at M = 64, runs once per macro step. Both make the wall speed-up exceed the update ratio at M = 64.
- The speed-up grows with M. The issue's production runs at M = 768 have λ_max = 38.7 against 10.6 at M = 64, so H/dt and n_max grow about 3.7 times and the rounding loss shrinks; the bound stays near 2.3. That is extrapolated, not measured.

## Recommendation

Use `lawson_rk4_multirate` as an opt-in for production-like runs at M ≥ 64 with hyper-resistivity, where it halves the cost:

- Take H from `CflDt` or `compute_cfl_timestep(mode="multirate")` at a Courant number of 0.3 to 0.5 (0.8 ran stably with P1's dissipation, but it is the setting that blew up without dissipation).
- Keep `substep_cfl` at 0.3, or at most 0.95, where the Courant rule still bounds every substep inside the RK4 limit. Or pass `exact_substep_cfl(state, 0.8)` and recompute it every 0.25 τ_A, as P1 recomputes its dt.
- Check the time averages you need against a run at a smaller macro step, as for any change of dt: the z± step moves them at the 10% level in these 32³ windows.
- Keep `lawson_rk4` for η = 0, for budget-closure studies that need the z± error of the small step, and at M ≤ 16, where the gain under P1's rule is 1.1 to 1.4 times.

## Reproduce

The script is `scripts/multirate_study.py`. Each command writes to the folder given by `--out`. Times are for one run on an 8-core Apple M1 Pro CPU shared with other jobs.

```bash
S="uv run python scripts/multirate_study.py"
# States (spin-ups with lawson_rk4)
$S spinup --out DIR --N 32 --M 16 --t 6 --name spin32m16.npz                            # 4 min
$S spinup --out DIR --N 32 --M 64 --t 3 --start spin32m16.npz --name spin32m64.npz     # 20 min
$S spinup --out DIR --N 64 --M 16 --t 1 --start spin32m16.npz --name spin64m16.npz     # 13 min
$S spinup --out DIR --N 64 --M 64 --t 0 --start spin64m16.npz --name spin64m64.npz     # pads g
# Frozen-flow thresholds, radii and speed-up bounds, accuracy
$S frozen --out DIR                                                                       # 15 s
$S radii --out DIR --states spin32m16.npz spin32m64.npz spin64m16.npz spin64m64.npz --tag all   # 1 min
$S accuracy --out DIR --state spin32m16.npz --tag 32m16                                  # 2 min
$S accuracy --out DIR --state spin32m64.npz --tag 32m64                                  # 6 min
# Cost: Courant rule, then exact rule (window in τ_A per repetition)
$S cost --out DIR --state spin32m16.npz --window 0.3 --tag 32m16                         # 4 min
$S cost --out DIR --state spin32m16.npz --window 0.5 --rule exact --tag x32m16           # 2 min
$S cost --out DIR --state spin32m64.npz --window 0.1 --tag 32m64                         # 6 min
$S cost --out DIR --state spin32m64.npz --window 0.2 --rule exact --tag x32m64           # 3 min
$S cost --out DIR --state spin64m16.npz --window 0.05 --tag 64m16                        # 7 min
$S cost --out DIR --state spin64m16.npz --window 0.1 --rule exact --tag x64m16           # 4 min
$S cost --out DIR --state spin64m64.npz --window 0.01 --tag 64m64                        # 11 min
$S cost --out DIR --state spin64m64.npz --window 0.03 --rule exact --tag x64m64          # 8 min
# Runs: each for --scheme lawson_rk4_multirate and --scheme lawson_rk4 (tag suffix _mr, _sr)
$S run --out DIR --state spin32m16.npz --scheme lawson_rk4_multirate --t 20 --tag f32m16_mr        # 10 min
$S run --out DIR --state spin32m16.npz --scheme lawson_rk4_multirate --t 20 --rule exact --tag xf32m16_mr
$S run --out DIR --state spin32m16.npz --scheme lawson_rk4_multirate --t 5 --decaying --tag d32m16_mr
$S run --out DIR --state spin32m16.npz --scheme lawson_rk4_multirate --t 1 --decaying --ideal --chunk 10 --tag i32m16_mr
$S run --out DIR --state spin32m16.npz --scheme lawson_rk4_multirate --t 5 --cfl 0.8 --tag f32m16_mr_c08
$S run --out DIR --state spin32m64.npz --scheme lawson_rk4_multirate --t 10 --rule exact --tag xf32m64_mr  # 8 min
$S run --out DIR --state spin32m16.npz --scheme lawson_rk4 --t 20 --rule exact --seed 1 --tag xf32m16s1_sr
$S report --out DIR                                                                       # the tables above
```

The other runs of the stability table follow the same pattern. Single runs take from 15 s (ideal, 32³) to 15 min (`lawson_rk4`, 32³, M = 64, 10 τ_A, exact rule).
