# Long forced runs on current code (Issue #136)

Issue #136 reported that every 128³ forced run of `alfvenic_cascade_benchmark.py` on v0.4.4 piled energy up near the grid cutoff and went to NaN, at t ≈ 120-570 τ_A at 128³ and ≈ 100-115 τ_A at 32³ and 64³. This page reruns those configurations on current code and finds where the old blowup came from.

## Result

- Current code is stable in every rerun. Six runs at 64³ and 128³, with both forcings, reached 500 to 1000 τ_A. That is 3 to 8 times the old blowup times. All six pass every criterion below. These runs test the Alfvénic sector only: g starts at zero and is not forced, so it stays zero, and every run uses `imex_rk222` at a fixed dt.
- Nothing piles up at the cutoff. Every time-averaged E(n⊥) falls monotonically from n_edge/2 to the 2/3 edge. Once the cascade reaches the hyper-resistive range, the energy fraction near the edge levels off. Over the last 100 τ_A it decays in four runs, and in the other two it e-folds in 190 τ_A or more. The stepper's energy error stays below 1.3% of the injection.
- dt stays at the Alfvén limit for the whole of every run. The largest Elsasser advection speed, max|∇⊥z±| = 0.36 at 128³, is below v_A.
- The old blowup came from the integrating-factor phase on the nonlinear increment, which #148 fixed. The v0.4.4 benchmark reproduces the blowup at 32³. The commit before #148 still blows up at the rescaled amplitude, and the commit with #148 does not. The bisect ran at 32³ only. That the 64³ and 128³ blowups had the same cause is inferred.
- Two features that the issue reported as symptoms come from the setup, not from the numerics. Shell forcing grows an n_z = 0 condensate that keeps E(t) rising. At 128³ the rescaled amplitudes inject 1/16 of the 64³ power, so their cascade needs 500 to 1000 τ_A to reach the cutoff.

## Setup

Each run is the benchmark's configuration, replayed by `scripts/long_run_study.py`:

- The benchmark's initial spectrum (seed 42), M = 10, β_i = 1, ν = 0, v_A = 1, unit box.
- dt = `compute_cfl_timestep(state, v_A=1, cfl_safety=0.3)`, evaluated once at t = 0 and then held fixed, as the benchmark does. It is 0.3/N at every N, the Alfvén limit.
- Force, then step, with the same PRNG key sequence. Shell forcing is `force_alfven_modes` on 1 ≤ |n| ≤ 2. Balanced forcing is `force_alfven_modes_balanced` on 1 ≤ n⊥ ≤ 2, |n_z| = 1, independent draws for z⁺ and z⁻.
- `scheme="imex_rk222"`, η = 2 with r = 2 (the 128³ and 64³ defaults), hyper_n = 2.

The driver runs the steps inside a jitted `lax.fori_loop` and calls the same forcing and step kernels as the benchmark. Over 60 steps at 32³ it matches the benchmark's Python loop to 2.8e-7 in z± (float32 rounding), with equal PRNG keys. g stays zero in this configuration (no pressure perturbation and no g forcing), so only z± evolve.

Every step the driver also sums three energy terms: the energy the forcing added, the energy the hyper-resistive factor removed, and the energy change of the nonlinear advance. The ideal bracket conserves energy, so the third term is the stepper's own energy error. A numerical instability would show as that term turning positive and growing. Over a full run the three terms close the energy budget to within 6e-7 of the final energy.

### Amplitudes

The pre-#144 nonlinearity was twice the correct one. If z obeys the old equations with forcing F, then 2z obeys the corrected equations with forcing 2F, and dt is the same 0.3/N in both. An old run at amplitude A is therefore the same dynamical system as a current run at amplitude 2A. The two strongest configurations of the issue become:

| issue configuration | blowup on v0.4.4 | current amplitude |
|---|---|---|
| shell, A = 0.01, η = 2, r = 2 | 165 τ_A | 0.02 |
| balanced, A = 0.05, η = 2, r = 2 | 120 τ_A | 0.10 |

The forcing adds amplitude/√dt per Fourier coefficient, and the code energy is N⁴ times the energy density. The injection rate in code units is the same at 64³ and 128³ (7.7 and 7.9 for shell 0.02, 198 and 199 for balanced 0.10), so the physical injection rate at a fixed amplitude falls as N⁻⁴. At 128³ the rescaled amplitudes inject 1/16 of the 64³ power, and their cascade reaches the dissipation range late. Two more 128³ runs inject the 64³ power, at amplitudes 0.08 (shell) and 0.40 (balanced), so that 128³ is also tested in saturated turbulence.

### Pass criteria

These criteria were written before the runs. The windows are the last 100 τ_A and the 100 τ_A before them.

- The run reaches t_end with finite z±.
- Plateau: the mean energy of the last window is within 25% of the window before.
- Saturation: the largest max|z±| of the last window is below 1.5 times that of the window before.
- Fall-off: the time-averaged E(n⊥) of the last window decreases at every shell from n_edge/2 to n_edge, where n_edge = (N-1)//3.
- Edge: the fraction of energy at n⊥ ≥ 0.8 n_edge does not grow faster than e-folding in 50 τ_A over the last window.

## Runs

The windows are the last 100 τ_A of each run. "E ratio" and "max z± ratio" compare the last window with the window before. "largest E(n+1)/E(n)" is the largest ratio of neighbouring shells from n_edge/2 to n_edge, and a value below 1 means monotone fall-off. "edge e-fold" is the e-folding time of the energy fraction at n⊥ ≥ 0.8 n_edge over the last window. A negative value means that the fraction decays. The rates are over the last window, and "inj" is the expected injection rate of the forcing.

| N | forcing | amplitude | t_end [τ_A] | E ratio | max z± ratio | max ∇⊥z± | largest E(n+1)/E(n) | E(n_edge)/peak | edge e-fold [τ_A] | diss/inj | stepper error/inj | criteria |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 64³ | shell | 0.02 | 500 | 1.16 | 1.02 | 0.061 | 0.98 | 2.2e-05 | +640 | 0.55 | 2.7e-03 | pass |
| 64³ | balanced | 0.1 | 500 | 0.96 | 1.12 | 0.296 | 0.82 | 2.3e-04 | -654 | 1.33 | 2.9e-03 | pass |
| 128³ | shell | 0.02 | 1000 | 1.18 | 1.14 | 0.025 | 0.79 | 3.3e-08 | -106 | 0.50 | 4.5e-03 | pass |
| 128³ | balanced | 0.1 | 1000 | 1.05 | 0.97 | 0.152 | 0.90 | 2.9e-06 | +191 | 0.98 | 1.3e-02 | pass |
| 128³ | shell | 0.08 | 500 | 1.05 | 1.04 | 0.065 | 0.90 | 6.1e-06 | -204 | 0.42 | 1.3e-03 | pass |
| 128³ | balanced | 0.4 | 500 | 1.13 | 0.92 | 0.359 | 0.97 | 1.5e-04 | -698 | 0.87 | 3.4e-03 | pass |

The two 128³ runs at the rescaled amplitudes were still spinning up at 500 τ_A, and there they failed the edge check (balanced also failed the plateau check, with an E ratio of 1.57). Their dissipation was 0.11 (shell) and 0.005 (balanced) of the injection, and their edge fraction was growing with an e-folding time of 18 and 8 τ_A as the cascade front advanced. Between 500 and 1000 τ_A the front reached the hyper-resistive range, and the dissipation rose to 0.50 and 0.98 of the injection. The edge fraction then decayed (shell) or slowed to an e-folding time of 191 τ_A (balanced). A numerical instability would have kept growing at its own rate.

The stepper's energy error is positive and proportional to E, at 6e-6 to 1.4e-5 of E per τ_A over the last window of every run. It is largest relative to the injection in the runs that hold the most energy per unit injection.

## The old blowup came from the integrating-factor phase (#148)

The v0.4.4 benchmark at its 32³ defaults (A = 0.01, η = 1, r = 2) reproduces the failure locally. The energy is smooth until t ≈ 43 τ_A, then grows 320-fold in 3.7 τ_A, with σ_c falling to −0.63 as z⁻ runs away, and is NaN at t = 47.8 τ_A.

Two commits bracket the fix. Both run the same benchmark at A = 0.02, the rescaled amplitude, as does current code:

| commit | change | result |
|---|---|---|
| 6cb6776 | #144 (nonlinearity factor 2) | NaN at t = 45.9 τ_A, the same z⁻ runaway |
| d4d89bc | #148 (integrating-factor phase) | finite to t = 200 τ_A |
| 44f7b5b (current) | `scripts/long_run_study.py`, same configuration | finite to t = 270 τ_A |

On current code the energy of that 32³ run was still rising at 270 τ_A (E ratio 1.40, the shell condensate below), while its edge fraction decayed and its stepper error was −1.7e-4 of the injection.

At steps 3000 and 4000 the energy at 6cb6776 is 233.63 and 350.05, four times the 58.415 and 87.515 of v0.4.4 to within 2e-4. #144 is therefore the exact rescaling, and the phase fix is the only change between a blowup and none. Before #148 the full step multiplied the nonlinear increment by e^(±2i k_z dt) where the interaction-picture midpoint uses e^(±i k_z dt/2). The extra rotation of 1.5 k_z dt is about 0.9 rad at the k_z dealiasing edge at every N, because dt is 0.3/N and k_z at the edge grows as N. A rotated increment no longer conserves energy. z⁺ and z⁻ rotate in opposite senses, so the error can feed one field and drain the other, which fits the one-signed σ_c of the runaway (inferred, not measured mode by mode). `TestConvergence.test_second_order_convergence_nonlinear_3d` in `tests/test_timestepping.py` guards the fix.

## Shell forcing grows a box-scale condensate

Under shell forcing the energy keeps rising long after the balanced runs level off. The rise is at the box scale, not at the cutoff. The shell 1 ≤ |n| ≤ 2 includes the n_z = 0 plane, and `force_alfven_modes` forces φ only, so the forced 2D modes cascade inversely and collect at n⊥ = 1. At 64³ the n_z = 0 plane holds 84% of the energy and n⊥ = 1 holds 86% by t = 400 τ_A. This is the "no true steady state plateau" of the issue. It is physical, and `band="perp_lowkz"` avoids it.

## Reproduce

The runs used JAX 0.9.1 on one A100 each, through `modal_app.py::run_script`. A 128³ run takes about 5.3 s per τ_A (12 ms per step). Outputs are on the `gandalf-results` volume under `issue136_*`.

```bash
modal run modal_app.py::run_script --project-dir . --include scripts/long_run_study.py \
	--script scripts/long_run_study.py --output-subdir issue136_128_shell_a0.02 \
	--args "--resolution 128 --forcing shell --amplitude 0.02 --t-end 500 --out {output_dir}"
modal volume get gandalf-results issue136_128_shell_a0.02 ./issue136_128_shell_a0.02
uv run python scripts/long_run_report.py ./issue136_128_shell_a0.02
```

A run resumes from the newest checkpoint in its output folder, so a larger `--t-end` extends it.
