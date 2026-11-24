# Float32 state with float64 budgets

Issue #191 asks whether a run can keep its state in float32, which halves memory traffic and makes consumer GPUs usable, and still close the budgets of the krmhd-research P1 runs. P1 runs Lawson-RK4 in float64 and closes its per-m free-energy budgets to about 1e-4 of the injection rate. This page measures what a float32 state does to those budgets.

## Result

- Keep float64 for budget-closure studies at P1's production time steps. A float32 state biases the energies by up to one float32 ulp per step, and the bias does not shrink with dt. At dt = 3.8e-4, inside the 2e-4 to 5e-4 range of P1's 128³ and 256³ runs, a float32 state moves the W budget by 7e-5 to 1.2e-4 of the injection and the per-m residuals by 4e-5 to 8e-5. Float64 closes the same W budget to 6e-12.
- At dt = 3.0e-3, the dt of this study's 32³ window, the same bias is 4e-6 to 7e-6 of the injection.
- The state's own rounding sets the error. Evaluating the budget sums in float32 adds an error of the same order. Float32 accumulators add 1e-8 when each `advance()` chunk sums in float32 and the host sums the chunks in float64.
- An adaptive dt averages the E_A part of the bias away, from −8.2e-5 to +6.9e-6 at dt ≈ 3.8e-4. The W part stays at −7.1e-5.
- Float32 suits spectra, fluxes and other time averages. Over 6 τ_A the float32 time averages match float64 to 5e-6, while another forcing seed moves them by 1.4% to 9.5%.

## Setup

The configuration is P1's `forced_beta1` physics at the parameters of P1's 32³ test (`forced_test32.yaml`), with M = 16:

- 32³, M = 16, Λ = √5, β_i = 1, zero closure, `lawson_rk4`.
- Hyper-resistivity η = 200 (r = 2) and hyper-collisions ν = 30 (n = 6).
- White-noise forcing of Φ and of g_0 on k⊥L/2π ∈ [1, 2], |n_z| = 1, at ε_A = ε_g = 0.3 (`WhiteNoiseBand(band="perp_lowkz")`).
- A 6 τ_A float64 spin-up from rest. Then a 6 τ_A window of 2000 steps at a fixed dt = 3.0e-3, which is 0.8 times the smallest exact Lawson-RK4 limit of the last spin-up τ_A (P1's rule, `compute_cfl_timestep(mode="exact")`).

Four variants run the window in lockstep from the same spun-up state, all under `jax_enable_x64`:

| variant | state | forcing |
|---|---|---|
| `f64` | complex128 | seed 0, the reference |
| `f32` | complex64 | seed 0: the same draws, rounded to float32 |
| `f64-rounded` | complex128, started from the float32-rounded state | seed 0 |
| `f64-seed1` | complex128 | seed 1 |

`f64-rounded` shows how far chaos carries a one-time float32-sized perturbation. `f64-seed1` sets the scale of statistical differences.

Every step, the accounting hook of `DeviceLoop` records P1's exact budget terms (krmhd-research `h1lib.make_accounting`), evaluated in float64 from the fields. The terms are the injection, the hyper-resistive and hyper-collisional losses per m, and the change of E_A and of each W_m = p_m⟨g_m²⟩ in the ideal part of the step. The hook also records the field-aligned Hermite flux Π_{m+1/2} = 2√β_i p_m T_{m,m+1}⟨g_m ∇∥g_{m+1}⟩ at the start and at the end of the ideal step. For `f32` it records the same terms evaluated in float32 arithmetic, and every tenth step the float64 step of the float32 step's input.

## Budget closure at dt = 3.0e-3

The ideal-step error is P1's energy-conservation gate: the change of E_A or W in the ideal part of the step, summed over the window, over the injection. The per-m residual is P1's per-m budget (Π_{m-1/2} − Π_{m+1/2}) + inj_m − D_m − dW_m/dt over ε_g, with the flux integrated by the trapezoid rule every step.

| variant | ideal-step error E_A / inj | ideal-step error W / inj | max over m of the per-m residual / ε_g |
|---|---|---|---|
| `f64` | +9.54e-04 | −2.16e-07 | 3.44e-03 |
| `f32` | +9.47e-04 | −3.84e-06 | 3.44e-03 |
| `f64-rounded` | +9.54e-04 | −2.16e-07 | 3.44e-03 |
| `f64-seed1` | +8.19e-04 | −1.80e-07 | 3.23e-03 |

The per-m residual of 3.4e-3 is the error of the flux quadrature, not of the budgets. At this dt the fastest slices turn by about a radian per step, so the trapezoid rule misses part of the transfer. P1's own sampling every 20 steps gives 3.5e-3 here. The precision effect is the change against `f64`:

| variant | ideal-step error E_A / inj | ideal-step error W / inj | max over m of the per-m residual change / ε_g |
|---|---|---|---|
| `f32` | −7.0e-6 | −3.6e-6 | 3.2e-6 |
| `f64-rounded` | −3.0e-11 | +7.7e-15 | 2.1e-10 |

Chaos alone moves the budgets by 1e-10 over this window, so the `f32` change is the precision effect.

## Where the float32 error comes from

### State rounding

The float64 step of the float32 step's input, taken every tenth step (200 samples), isolates the state's own rounding. Per step, in units of the per-step injection ε dt:

| quantity | mean (bias) | standard deviation | over the window: bias + std/√2000 |
|---|---|---|---|
| E_A | −6.1e-6 | 1.2e-5 | 6.4e-6 |
| W | −3.3e-6 | 6.0e-6 | 3.4e-6 |
| W_m, worst m (m = 0) | −2.9e-6 | 4.8e-6 | 3.0e-6 |

The bias predicts the measured change of the budgets, −7.0e-6 and −3.6e-6. The median relative state error per step is 5e-8 for z± and 1.3e-7 for g, one or two float32 ulp.

The step's multiplicative factors are one source of the bias. Those factors are the Elsasser phase exp(±ik_z dt), the hyper-resistive and hyper-collisional factors, and the Hermite streaming factor. Evaluated one at a time on the start state, the float32 rounding of each factor changes E_A or W by 0.5e-6 to 9e-6 of ε dt per step, with both signs. They do not add up to the measured bias, so no single factor dominates it.

### Evaluation and accumulation

Window totals of the `f32` run's budget terms, relative to the window injection. Column (b) evaluates the terms in float32 and sums them in float64. The (c) columns evaluate in float64 and sum in float32 three ways.

| term | (b) float32 evaluation | (c) one float32 sum over the window | (c) float32 Kahan sum | (c) float32 within 50-step chunks, float64 across |
|---|---|---|---|---|
| ideal-step error E_A | −1.6e-6 | +3.4e-10 | +1.7e-11 | +2.6e-11 |
| ideal-step error W | +2.5e-6 | +8.7e-7 | −1.1e-8 | −6.4e-9 |
| injection E_A | +1.2e-6 | +1.3e-7 | +1.3e-9 | −3.3e-8 |
| injection W | −5.8e-6 | +3.1e-7 | −3.8e-8 | −1.1e-8 |
| loss E_A | −2.8e-7 | −2.8e-7 | −2.0e-8 | −7.2e-9 |
| loss W | −3.5e-6 | +4.4e-7 | −6.3e-9 | +4.8e-9 |

Float32 evaluation costs as much as the state's rounding. It computes E_A and W as sums over 17,000 Fourier modes and takes differences of order ε dt between them. Float32 accumulation matters only when one float32 sum runs over thousands of steps. `DeviceLoop` starts each `advance()` from the accumulator's initial value, so a driver that advances in chunks and sums the chunks in float64 on the host gets the last column.

## The bias does not shrink with dt

A step's budget terms scale with dt, but the rounding of the step does not. The float64 twin of 200 float32 steps from the spun-up state, at four fixed dt, gives the bias per step in units of ε dt (± the standard error of the mean):

| dt | E_A | W | max over m of W_m | E_A bias per step / E_A |
|---|---|---|---|---|
| 3.01e-3 | −2.0e-6 ± 8e-7 | −6.6e-6 ± 6e-7 | 5.8e-6 | −3.2e-9 |
| 1.50e-3 | −6.2e-6 ± 2e-6 | −5.3e-5 ± 1e-6 | 2.8e-5 | −5.1e-9 |
| 7.52e-4 | +1.6e-4 ± 3e-6 | −7.1e-7 ± 3e-6 | 1.1e-5 | +6.4e-8 |
| 3.76e-4 | −9.4e-5 ± 7e-6 | −1.7e-4 ± 6e-6 | 1.1e-4 | −1.9e-8 |

The absolute bias per step is between 0.05 and 1 float32 ulp of the energy (the unit roundoff is 6e-8). Its size and sign change from one dt to the next, because each dt rounds the step's factors differently.

Two 1 τ_A runs of `f64` and `f32` in lockstep at dt ≈ 3.8e-4 measure the budget change directly. One holds dt fixed. The other recomputes dt every 25 steps with `CflDt(0.15, every=25)`, as P1 does with its own rule:

| dt | steps | distinct dt values | E_A change / inj | W change / inj | max over m of the per-m residual change / ε_g | twin prediction, E_A and W |
|---|---|---|---|---|---|---|
| fixed 3.76e-4 | 2700 | 1 | −8.2e-5 | −1.2e-4 | 7.7e-5 | −8.4e-5, −1.6e-4 |
| adaptive 3.1e-4 to 4.7e-4 | 2700 | 108 | +6.9e-6 | −7.1e-5 | 4.4e-5 | +1.1e-5, −8.7e-5 |

In the same runs, float64 closes the ideal-step budgets to 7e-6 (E_A) and 6e-12 (W). The adaptive dt averages the E_A bias, whose sign varies with dt. The W bias is negative at all four dt of the scan and stays.

P1's 64³ runs take dt between 7e-4 and 2.5e-3, and its 128³ and 256³ runs between 2e-4 and 5e-4. In this scan the W bias reaches 5e-5 of the injection at the 64³ time steps and 1.7e-4 at the 128³ and 256³ time steps, the same size as the closure P1 reports.

## Trajectories and statistics

Relative difference from `f64` in the energy norms of z± and g:

| t − t₀ (τ_A) | `f32`: z, g | `f64-rounded`: z, g | `f64-seed1`: z, g |
|---|---|---|---|
| 0.15 | 1.1e-6, 1.9e-6 | 3.4e-8, 3.3e-8 | 0.40, 0.50 |
| 1.05 | 7.5e-6, 8.9e-6 | 5.3e-8, 5.7e-8 | 0.96, 1.2 |
| 3.01 | 2.6e-5, 2.7e-5 | 1.5e-7, 1.3e-7 | 1.4, 1.5 |
| 6.02 | 5.1e-5, 5.1e-5 | 4.0e-7, 4.0e-7 | 1.5, 1.5 |

The `f32` difference grows by per-step rounding. A one-time float32 perturbation grows twelvefold in 6 τ_A, so chaos in this run is weak. Over P1's 50 τ_A windows the trajectories would separate. The time averages then carry their usual statistical scatter, which a seed change shows.

Time averages over the second half of the window:

| variant | ⟨E_A⟩ | ⟨W⟩ | D_A | D_W | Π at m + 1/2 = 0.5, 4.5, 8.5 |
|---|---|---|---|---|---|
| `f64` | 0.5516 | 0.2469 | 0.3194 | 0.3369 | 0.2799, 0.2317, 0.1965 |
| `f32` | 0.5516 | 0.2469 | 0.3194 | 0.3369 | 0.2799, 0.2317, 0.1965 |
| `f64-seed1` | 0.5608 | 0.2504 | 0.3499 | 0.3243 | 0.2456, 0.2088, 0.1737 |

`f32` differs from `f64` by 7.5e-7 in ⟨E_A⟩, 2.2e-6 in ⟨W⟩, 4.7e-6 in D_A and 3.3e-6 in D_W. The seed change moves the same averages by 1.7%, 1.4%, 9.5% and 3.7%. The k⊥ spectra and W_m agree to the same level.

## Running a float32 state with float64 diagnostics

`DeviceLoop` steps in the precision of the state's z±. For a spectra or flux study, enable float64, cast the state to complex64, and upcast the fields in the accounting hook:

```python
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from krmhd.runloop import Accounting, DeviceLoop, FixedDt, RunConfig, StepParams

fields = ("z_plus", "z_minus", "g")
state32 = state.model_copy(update={k: getattr(state, k).astype(jnp.complex64) for k in fields})


def update(acc, before, after, increment, dt):
	g = after.g.astype(jnp.complex128)  # evaluate in float64
	return acc + jnp.sum(jnp.abs(g) ** 2)  # float64 accumulator


accounting = Accounting(init=jnp.zeros((), jnp.float64), update=update)
config = RunConfig(step=StepParams(eta=200.0, v_A=1.0, scheme="lawson_rk4"), dt=FixedDt(3e-3), accounting=accounting)
loop = DeviceLoop(state32, config)
```

The step stays in complex64 for all three schemes. The upcast in the hook makes complex128 temporaries the size of the fields it reads. On a memory-bound card, evaluate the diagnostics a few moments at a time.

The W bias might shrink if the step computed its factors in float64 and rounded them once. That change is inside the step kernels and is not tested here.

## Reproduce

The script is `scripts/precision_study.py`. Each command writes to the folder given by `--out`. The times are for one run on an 8-core Apple M1 Pro CPU shared with other jobs.

```bash
uv run python scripts/precision_study.py spinup --out DIR               # 6 τ_A float64 spin-up, 4 min
uv run python scripts/precision_study.py window --out DIR               # the four variants, 7 min
uv run python scripts/precision_study.py dtscan --out DIR               # state rounding against dt, 1 min
uv run python scripts/precision_study.py check --out DIR                # fixed dt = window dt / 8, 4 min
uv run python scripts/precision_study.py check --out DIR --cfl 0.15     # adaptive dt, 4 min
uv run python scripts/precision_study.py report --out DIR               # prints the tables above
```
