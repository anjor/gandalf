#!/usr/bin/env python
"""
Lawson-RK4 step time on one device against its slices sharded over N devices (Issue #191).

Setup: the P1 m_* physics (Λ = √5, β_i = 1, zero closure, η = 200 with r = 2,
ν = 30 with n = 6, white-noise forcing of Φ and g_0 on k⊥L/2π ∈ [1, 2], |n_z| = 1)
at --n³ and each --M, in DeviceLoop at a fixed dt. For each M and each Hermite
path, one loop per device count advances --steps steps per call, the device
counts interleaved (forward, then reversed) for --rounds rounds. The paths force
how each device advances its slices: "auto" is the default threshold
(hermite.advance_in_place on each device's share of g), "in_place" one
slice at a time, "batched" all at once.

Reported per (M, path, devices): median ms per step and its range over the
rounds, the speed-up against the fastest one-device path at that M, the time of
the step's two all-to-alls alone (a shard_map of the same block shape), the
bytes each device sends per step, and the largest relative difference of g and
z⁺ from one device after all rounds. That difference should be round-off. NaN
or a large value means the run is wrong and its time meaningless. --memory adds
XLA's compiled memory per device, at the cost of one more compile per loop.

On virtual CPU devices (XLA_FLAGS=--xla_force_host_platform_device_count=N) all
"devices" share the same cores, so the times check overhead only; they are not
a speed-up.

Usage:
    # Modal, four B200s; prints the table and writes multidevice_benchmark.json
    MODAL_SCRIPT_GPU=B200:4 modal run modal_app.py::run_script --project-dir . \\
        --include scripts/multidevice_benchmark.py --script scripts/multidevice_benchmark.py \\
        --args "--out {output_dir}" --output-subdir multigpu_slices_b200x4 \\
        --env "JAX_PLATFORMS=cuda"
    # CPU overhead check
    XLA_FLAGS=--xla_force_host_platform_device_count=4 JAX_PLATFORMS=cpu \\
        uv run python scripts/multidevice_benchmark.py --n 32 --M 32 64 --steps 3
"""

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.sharding import Mesh, NamedSharding, PartitionSpec  # noqa: E402

from krmhd import hermite  # noqa: E402
from krmhd.forcing import WhiteNoiseBand  # noqa: E402
from krmhd.physics import KRMHDState, initialize_random_spectrum  # noqa: E402
from krmhd.runloop import (  # noqa: E402
    DeviceLoop,
    FixedDt,
    Forcing,
    RunConfig,
    StepParams,
    _advance_impl,
)
from krmhd.spectral import SpectralGrid3D  # noqa: E402
from krmhd.timestepping import _shard_map, compute_cfl_timestep  # noqa: E402

LAMBDA, BETA = math.sqrt(5.0), 1.0
ETA, HYPER_R, NU, HYPER_N = 200.0, 2, 30.0, 6
BAND = dict(n_min=1, n_max=2, band="perp_lowkz", max_nz=1, include_nz0=False)
FORCING = Forcing(alfven=WhiteNoiseBand(amplitude=0.05, **BAND),
                  hermite=WhiteNoiseBand(amplitude=0.05, **BAND), moments=(0,))
# hermite.HERMITE_IN_PLACE_ENTRIES for each path
PATHS = {"auto": None, "in_place": 0, "batched": 2**62}


def initial_fields(grid: SpectralGrid3D, M: int, dtype) -> tuple:
    """Host copies of a random k⁻⁵ᐟ³ z± spectrum and a small in-band g."""
    s = initialize_random_spectrum(grid, M=M, alpha=5 / 3, amplitude=0.5, k_min=1.0, k_max=8.0,
                                   beta_i=BETA, nu=NU, Lambda=LAMBDA, seed=1)
    rng = np.random.default_rng(2)
    shape = s.g.shape
    g = 1e-2 * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))
    g = g * np.asarray(grid.dealias_mask)[..., None]
    return tuple(np.asarray(x, dtype) for x in (s.z_plus, s.z_minus, g))


def make_state(grid, M, fields, dtype) -> KRMHDState:
    zp, zm, g = (jnp.array(x, dtype) for x in fields)  # fresh buffers: the loop donates them
    return KRMHDState(z_plus=zp, z_minus=zm, g=g, M=M, beta_i=BETA, v_th=1.0, nu=NU,
                      Lambda=LAMBDA, time=0.0, grid=grid)


def mesh_of(n_devices: int) -> Optional[Mesh]:
    return None if n_devices == 1 else Mesh(np.array(jax.devices()[:n_devices]), ("slices",))


def padded_slices(M: int, n_devices: int) -> int:
    return -(-(M + 1) // n_devices) * n_devices


def bytes_sent_per_step(n: int, M: int, n_devices: int, itemsize: int) -> float:
    """Bytes one device sends per step: two all-to-alls of its share of g, each
    keeping 1/D of it at home."""
    share = padded_slices(M, n_devices) * (n // n_devices) * n * (n // 2 + 1) * itemsize
    return 2.0 * share * (n_devices - 1) / n_devices


def all_to_all_pair_ms(n: int, M: int, n_devices: int, dtype, reps: int) -> Optional[float]:
    """Median time of the step's two all-to-alls alone, on a g-sized array split along k_z."""
    if n_devices == 1:
        return None
    mesh = mesh_of(n_devices)
    axis = mesh.axis_names[0]
    spec = PartitionSpec(None, axis)

    def pair(x):
        y = jax.lax.all_to_all(x, axis, 0, 1, tiled=True) + 1.0
        return jax.lax.all_to_all(y, axis, 1, 0, tiled=True)

    fn = jax.jit(_shard_map(pair, mesh=mesh, in_specs=spec, out_specs=spec))
    shape = (padded_slices(M, n_devices), n, n, n // 2 + 1)
    x = jax.device_put(jnp.zeros(shape, dtype), NamedSharding(mesh, spec))
    jax.block_until_ready(fn(x))
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(x))
        times.append(time.perf_counter() - t0)
    return 1e3 * float(np.median(times))


def compiled_bytes(loop: DeviceLoop, steps: int) -> int:
    """XLA's memory estimate per device for loop.advance: arguments, outputs not
    aliased to them, and temporaries. The grid goes in as leaves, since lowering
    cannot rebuild a SpectralGrid3D from placeholder leaves."""
    leaves, treedef = jax.tree.flatten(loop._grid)
    fn = jax.jit(lambda carry, acc, n, grid_leaves, consts: _advance_impl(
        carry, acc, n, jax.tree.unflatten(treedef, grid_leaves), consts, loop._plan),
        donate_argnums=(0,))
    m = fn.lower(loop._carry, loop._acc0, jnp.asarray(steps, jnp.int32), leaves,
                 loop._consts).compile().memory_analysis()
    return (m.argument_size_in_bytes + m.output_size_in_bytes - m.alias_size_in_bytes
            + m.temp_size_in_bytes)


def rel_diff(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


def run_path(grid, M, path, device_counts, dt, dtype, steps, rounds, memory) -> List[Dict]:
    """Time every device count on one Hermite path, interleaved; return one row each."""
    hermite.HERMITE_IN_PLACE_ENTRIES = PATHS[path]
    jax.clear_caches()  # the in-place choice is made at trace time
    fields = initial_fields(grid, M, dtype)
    loops, first_call, mem = {}, {}, {}
    for D in device_counts:
        config = RunConfig(
            step=StepParams(eta=ETA, v_A=1.0, nu=NU, hyper_r=HYPER_R, hyper_n=HYPER_N,
                            scheme="lawson_rk4", mesh=mesh_of(D)),
            dt=FixedDt(dt), forcing=FORCING,
        )
        loops[D] = DeviceLoop(make_state(grid, M, fields, dtype), config, key=jax.random.PRNGKey(0))
        mem[D] = compiled_bytes(loops[D], steps) if memory else None
        t0 = time.perf_counter()
        loops[D].advance(steps)  # compiles
        jax.block_until_ready(loops[D].state.g)
        first_call[D] = time.perf_counter() - t0
    times = {D: [] for D in device_counts}
    for r in range(rounds):
        for D in (device_counts if r % 2 == 0 else device_counts[::-1]):
            t0 = time.perf_counter()
            loops[D].advance(steps)
            jax.block_until_ready(loops[D].state.g)
            times[D].append((time.perf_counter() - t0) / steps)
    ref = loops[device_counts[0]].state
    rows = []
    for D in device_counts:
        st = loops[D].state
        share = jax.ShapeDtypeStruct((grid.Nz // D, grid.Ny, grid.Nx // 2 + 1, M + 1), st.g.dtype)
        ms = 1e3 * np.asarray(times[D])
        rows.append(dict(
            M=M, path=path, devices=D,
            device_path="in_place" if hermite.advance_in_place(share) else "batched",
            ms_per_step_median=float(np.median(ms)), ms_min=float(ms.min()), ms_max=float(ms.max()),
            first_call_s=first_call[D],
            g_rel_diff_vs_first=rel_diff(st.g, ref.g),
            zp_rel_diff_vs_first=rel_diff(st.z_plus, ref.z_plus),
            finite=bool(np.isfinite(np.asarray(st.g)).all()),
            compiled_bytes_per_device=mem[D],
        ))
    hermite.HERMITE_IN_PLACE_ENTRIES = PATHS["auto"]
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--n", type=int, default=64, help="grid points per side (Nz divisible by the device counts)")
    parser.add_argument("--M", type=int, nargs="+", default=[256, 512, 768])
    parser.add_argument("--devices", type=int, nargs="+", default=None,
                        help="device counts, first is the baseline (default: 1, 2, 4, ... up to all)")
    parser.add_argument("--paths", nargs="+", default=["auto", "in_place", "batched"], choices=sorted(PATHS))
    parser.add_argument("--steps", type=int, default=20, help="steps per timed advance()")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--float32", action="store_true", help="complex64 state")
    parser.add_argument("--memory", action="store_true",
                        help="report XLA's compiled memory per device (one more compile per loop)")
    parser.add_argument("--out", type=Path, default=None, help="folder for multidevice_benchmark.json")
    args = parser.parse_args()

    n_all = jax.device_count()
    device_counts = args.devices or [d for d in (1, 2, 4, 8, 16) if d <= n_all]
    dtype = np.complex64 if args.float32 else np.complex128
    itemsize = np.dtype(dtype).itemsize
    grid = SpectralGrid3D.create(Nx=args.n, Ny=args.n, Nz=args.n, Lx=1.0, Ly=1.0, Lz=1.0)
    print(f"JAX {jax.__version__}, backend {jax.default_backend()}, {n_all} devices: "
          f"{[d.device_kind for d in jax.devices()][:1]}; {args.n}^3, {np.dtype(dtype).name}, "
          f"{args.steps} steps x {args.rounds} rounds", flush=True)

    rows = []
    for M in args.M:
        probe = make_state(grid, M, initial_fields(grid, M, dtype), dtype)
        dt = 0.5 * float(compute_cfl_timestep(probe, v_A=1.0, cfl_safety=0.3))
        comm = {D: all_to_all_pair_ms(args.n, M, D, dtype, reps=10) for D in device_counts}
        for path in args.paths:
            for row in run_path(grid, M, path, device_counts, dt, dtype, args.steps, args.rounds,
                                args.memory):
                row.update(dt=dt, all_to_all_pair_ms=comm[row["devices"]],
                           bytes_sent_per_step=bytes_sent_per_step(args.n, M, row["devices"], itemsize))
                rows.append(row)
                print(json.dumps(row), flush=True)
        base = min(r["ms_per_step_median"] for r in rows if r["M"] == M and r["devices"] == device_counts[0])
        for r in rows:
            if r["M"] == M:
                r["speedup_vs_best_baseline"] = base / r["ms_per_step_median"]

    print("\n| M | path | devices | per-device path | ms/step (median, range) | speed-up | all-to-all pair ms | "
          "GB sent/step/device | compiled GB/device | g rel diff |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        comm = "-" if r["all_to_all_pair_ms"] is None else f"{r['all_to_all_pair_ms']:.2f}"
        peak = "-" if r["compiled_bytes_per_device"] is None else f"{r['compiled_bytes_per_device'] / 1e9:.2f}"
        print(f"| {r['M']} | {r['path']} | {r['devices']} | {r['device_path']} | "
              f"{r['ms_per_step_median']:.1f} ({r['ms_min']:.1f}-{r['ms_max']:.1f}) | "
              f"{r['speedup_vs_best_baseline']:.2f} | {comm} | {r['bytes_sent_per_step'] / 1e9:.3f} | "
              f"{peak} | {r['g_rel_diff_vs_first']:.1e} |")
    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "multidevice_benchmark.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
