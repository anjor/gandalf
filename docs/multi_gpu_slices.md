# Lawson-RK4 on several GPUs: sharding the Hermite slices

Issue #191, item 2. In the eigenbasis of the streaming matrix the M + 1 Hermite slices are independent between collision updates (#184), so `lawson_rk4` can advance them on different GPUs. This cuts wall time, not cost: the 64³, M = 768 run took 56 h on one B200. The GPU speed-up below is an estimate. `scripts/multidevice_benchmark.py` measures it.

## Usage

```python
import jax
from jax.sharding import Mesh
from krmhd import gandalf_step
from krmhd.runloop import DeviceLoop, RunConfig, StepParams

mesh = Mesh(jax.devices(), ("slices",))  # one axis, any name
state = gandalf_step(state, dt, eta, v_A, scheme="lawson_rk4", mesh=mesh)

config = RunConfig(step=StepParams(eta=eta, v_A=1.0, scheme="lawson_rk4", mesh=mesh), dt=...)
loop = DeviceLoop(state, config, key=key)
```

- The mesh needs `scheme="lawson_rk4"`, one axis, M ≥ 1 and Nz divisible by the number of devices. Otherwise `gandalf_step` and `DeviceLoop` raise `ValueError`.
- The state comes back with g split along k_z (Nz/D planes on each of D devices) and z± whole on every device. Diagnostics, checkpoints and accounting hooks use it as any `jax.Array`: XLA adds the reductions over the split axis, and `np.asarray` gathers it.
- Results match one device to round-off. Without a mesh the step traces exactly as before.
- `hermite_chunk` and the in-place rule (`hermite.advance_in_place`) apply to each device's share of g.

## Design

Between steps g stays in the Hermite basis, split along k_z. One step on device d:

1. Advance z± and compute the gradients of Φ and Ψ at the three stage times. Every device does this itself.
2. Project its k_z planes of g onto all slices, h = P⁻¹g. This is a matmul at each Fourier point and needs no communication.
3. All-to-all: device d keeps every k_z plane of its own ⌈(M+1)/D⌉ slices and sends the other slices away.
4. Advance its slices through the four RK4 stages, in place when its share is large.
5. All-to-all back, then g = P·C·h with the collision factors C. This is again local.

The two all-to-alls are the only collectives. The compiled step and the compiled forced, adaptive-dt `DeviceLoop` body each contain exactly two all-to-alls and no other collective (`tests/multidevice`). Zero slices pad M + 1 to a multiple of D: at most D − 1 slices, 0.9% at M = 768 on 8 GPUs.

**Why two exchanges.** The slice FFTs need every k_z plane of a slice. The projections and the collisions need every slice at a Fourier point. The data must therefore change hands twice per step. Keeping g in the slice basis between steps does not avoid this, because the collision matrix P⁻¹CP still couples all slices at every point at every step. One collective could apply it: each device multiplies its own slices into a whole-size partial result, and a reduce-scatter sums the partials. Each device then sends (D−1)/D·|g| per step instead of 2(D−1)/D²·|g|. That is the same at D = 2, twice as much at D = 4 and four times as much at D = 8, and it needs a buffer the size of g on every device. Through all-to-alls it needs the same two as this design. The slice basis would also hide the Hermite moments from the g_0 forcing and the per-m accounting.

**Why z± are replicated.** The compiled one-device step makes 36 FFTs of single 3D fields (the two Elsasser right-hand sides and the 12 gradient transforms) and 12 FFTs per slice. The z± work therefore costs about 3 slices: 0.4% of the step's FFTs at M = 768 and 1.2% at M = 256. Sharing it would need z± or the 12 gradient fields gathered on every device each step (25 MB at 64³), adding collectives and latency. Replicating it caps the speed-up at 1/(f + (1 − f)/D) with f ≈ 3/(M + 4): 7.3 at M = 256 and 7.8 at M = 768 on 8 GPUs.

## Bytes and memory per step

64³, complex128. |g| = 64·64·33·(M+1)·16 B. Each device sends 2(D−1)/D of its share per step (both all-to-alls).

| M | \|g\| | D | g per device | sent per device per step |
|---|---|---|---|---|
| 256 | 0.56 GB | 2 / 4 / 8 | 0.28 / 0.14 / 0.07 GB | 0.28 / 0.21 / 0.12 GB |
| 512 | 1.11 GB | 2 / 4 / 8 | 0.56 / 0.28 / 0.14 GB | 0.56 / 0.42 / 0.25 GB |
| 768 | 1.66 GB | 2 / 4 / 8 | 0.83 / 0.42 / 0.21 GB | 0.83 / 0.63 / 0.37 GB |

Between steps each device holds its share of g. The table below gives the memory of one step per device, in GB and in shares of g (|g|/D). These are XLA:CPU compiled estimates of one `gandalf_step` with nothing donated: arguments, outputs and temporaries, at 64³, complex128. The GPU compiler assigns buffers differently.

| M | D | in place | all at once |
|---|---|---|---|
| 256 | 1 / 2 / 4 / 8 | 1.17 / 1.26 / 0.67 / 0.34 GB (2.1 / 4.5 / 4.9 / 4.9) | 4.45 / 2.52 / 1.28 / 0.66 GB (8.0 / 9.1 / 9.2 / 9.4) |
| 768 | 1 / 2 / 4 / 8 | 3.41 / 3.77 / 2.00 / 1.04 GB (2.1 / 4.5 / 4.8 / 5.0) | 13.29 / 7.50 / 3.77 / 1.91 GB (8.0 / 9.0 / 9.1 / 9.2) |

One device in place holds about two copies of g, the input and the output. On a mesh the in-place path holds 4.5 to 5 shares per device. The exchanges add about 2.5 shares: the copy padded to D·n_own slices, the all-to-all's input chunks and its output. In place, sharding therefore saves memory only from D = 4 on. All at once, each device holds about 9 shares, against 8 on one device. A GPU runs its share in place only when 14 shares exceed its memory, since in place is about 4 times slower on a GPU (`hermite.advance_in_place`). At 64³, M = 768 every device count on a B200 with 180 GB therefore runs all at once. Sharding for memory matters at 256³ and above. Exchanging the slices in groups could bound the extra shares, if that becomes the limit.

## Expected GPU scaling (estimate, not measured)

Model: T(D) = T_rep + (T₁ − T_rep)/D + T_a2a(D), with T_a2a = bytes sent per device per step / link bandwidth, and no overlap of communication with compute.

- T₁, one B200 at 64³, M = 768, complex128, all at once: about 45 ms. The FFT and elementwise stages move about 170 GB per step, about 27 ms at 6.5 TB/s. The projections are 6.4e11 FP64 flops, about 17 ms at 37 TFLOPS. Neither number is measured, and either could be off by 2×.
- T_rep ≈ 0.4% of T₁, about 0.2 ms.
- NVLink 5 through NVSwitch gives 900 GB/s per direction per GPU. Assume 700 GB/s for the all-to-all.

| D | T_a2a | T(D) | speed-up | 56 h run |
|---|---|---|---|---|
| 2 | 1.2 ms | 23.4 ms | 1.9× | ~29 h |
| 4 | 0.9 ms | 12.1 ms | 3.7× | ~15 h |
| 8 | 0.5 ms | 6.2 ms | 7.2× | ~8 h |

GPU-hours rise by 5–12%. The estimate needs NVLink. Over PCIe at 25–50 GB/s, T_a2a at D = 2 is 17–33 ms, which cancels the gain. The benchmark times the all-to-all pair alone, so its "all-to-all pair ms" column shows which case applies. If T₁ is smaller than estimated, communication takes a larger share. At T₁ = 20 ms, for example, D = 8 gives about 6×.

## CPU validation

`tests/test_multidevice.py` runs `tests/multidevice` in a subprocess with four virtual CPU devices (21 tests, about 20 s, on JAX 0.6.2 and 0.7.2). In float64, sharded steps match one device to < 1e-13. On CPU most came out bitwise equal, and the largest difference seen was 8e-16. The tests cover:

- 1, 2 and 4 devices
- M + 1 divisible by the device count or not, and fewer slices than devices
- M = 1, both closures, and ν = 0 and ν > 0
- the all-at-once path, the in-place path in groups of one and two, and the in-place path by size
- complex64
- a forced `DeviceLoop` with a CFL dt and an accounting hook

A mutation that gives every device the eigenvalues of slice group 0 fails every comparison on two or more devices. Without pinning the forcing draws to every device, XLA split the draws and gathered them back, adding 8 collective-permutes and 2 all-gathers per step. The collective-count test catches that.

The overhead check uses 32³, M = 64 on 4 virtual CPU devices that share the machine's 8 cores, with the machine loaded by other work. These are not speed-ups. One device ran 130–137 ms per step and kept 1.5 cores busy. Four devices ran 41–46 ms per step and kept 5.4–6.0 cores busy. CPU time per step was 199–202 ms on one device and 243–250 ms on four. The extra 23% is the z± work done on four devices (9 slice-equivalents more than one device's 68) and the padding from 65 to 68 slices. That is 18% by count, and the all-to-all copies take the rest. The all-to-all pair alone took 4.5 ms.

## GPU measurement plan

Run from a checkout that contains this change:

```bash
MODAL_SCRIPT_GPU=B200:4 modal run modal_app.py::run_script --project-dir . \
    --include scripts/multidevice_benchmark.py --script scripts/multidevice_benchmark.py \
    --args "--out {output_dir}" --output-subdir multigpu_slices_b200x4 \
    --env "JAX_PLATFORMS=cuda"
modal volume get gandalf-results multigpu_slices_b200x4 ./multigpu_slices_b200x4
```

- `modal run` builds an ephemeral app from the local checkout, so `MODAL_SCRIPT_GPU=B200:4` gives this run four B200s and leaves the deployed app's GPU type unchanged. Modal's parser (modal 1.6.4) reads `B200:4` as four B200s, so `modal_app.py` needs no change. The call is tied to the local client and stops if the laptop sleeps.
- The defaults are 64³, M = 256, 512 and 768, devices 1, 2 and 4, and paths `auto`, `in_place` and `batched`. Each (M, path, device count) is 20 steps per sample over 5 interleaved rounds, after a compiling call. `--paths auto batched` cuts the run by a third. `MODAL_SCRIPT_GPU=B200:8` adds D = 8. `--memory` adds the GPU compiler's memory per device, at the cost of one more compile per loop.
- The run length is a guess: 30–60 min, mostly compiles. At Modal's listed B200 rate (check it), that is about $15–30 for four GPUs.
- Read the table in `modal_run.log` or `multidevice_benchmark.json`:
  - `speed-up` for path `auto` is what a run gets. For `in_place` it isolates the sharding, because every device count takes the same path. The baseline is the faster one-GPU path, `batched` or `in_place`.
  - Compare `all-to-all pair ms` with T_a2a above. It is the communication cost.
  - `g rel diff` must be round-off, about 1e-12 or less. Otherwise the timing is not valid.
  - `compiled GB/device` (with `--memory`) is XLA's memory estimate for one device.
