#!/usr/bin/env python3
"""
Long forced Alfvénic runs for Issue #136, with the stability diagnostics on the device.

The run is the alfvenic_cascade_benchmark.py configuration: the same initial
spectrum (seed 42), the timestep compute_cfl_timestep(state, v_A=1, 0.3)
evaluated once at t = 0 and then held fixed, the same PRNG key sequence, and
force-then-step with gandalf_step(scheme="imex_rk222"), nu = 0, M = 10. The
steps run inside one jitted lax.fori_loop per sample interval instead of the
benchmark's Python loop, calling the same forcing and step kernels.

Every step also folds three energy terms into the sample's totals:

    inj   energy the forcing added,
    diss  energy the post-step hyper-resistive factor removed (exact:
          sum_k e_k(after) * expm1(2 eta (k_perp^2/k_perp_max^2)^r dt)),
    num   E(after) - E(forced) + diss, the energy change of the nonlinear
          advance. The ideal bracket conserves E, so num is the time
          stepper's energy error; a numerical instability shows as num > 0.

Each sample (every --sample-every tau_A) records E, max|z±|, max|grad_perp z±|
(the Elsasser advection speeds), the current Courant dt against the fixed dt,
E(n_perp), E(|n_z|), the Hermite moment energies, and the energy fractions near
the grid cutoff. The run stops at --t-end, at a non-finite z±, or when E grows
past --blowup-factor times its running minimum after t = 20.

Outputs in --out: timeseries.csv, spectra.npz, spectra2d.npz, checkpoints/,
run_config.json, and a copy of this script. A run resumes from the latest
checkpoint in --out.

Exit status: 0 finished, 2 blew up.
"""

import argparse
import csv
import json
import math
import shutil
import sys
import time
from functools import partial
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from krmhd import SpectralGrid3D, compute_cfl_timestep, initialize_random_spectrum
from krmhd.forcing import _perp_lowkz_noise, _shell_noise
from krmhd.spectral import rfftn_inverse
from krmhd.timestepping import KRMHDFields, _gandalf_step_imex222_jit, cfl_timestep_estimate

FIELDS = [
    "t", "step", "E", "E_kin", "E_mag", "E_plus", "E_minus",
    "zp_max", "zm_max", "U_plus", "U_minus", "u_max", "b_max",
    "dt_cfl", "dt_over_cfl", "y_max",
    "f_half", "f_edge", "f_kz_half",
    "eps_inj", "eps_diss", "eps_num",
    "W_g", "g_finite", "wall_s",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--resolution", type=int, default=64)
    p.add_argument("--forcing", choices=["shell", "balanced"], default="shell")
    p.add_argument("--amplitude", type=float, required=True)
    p.add_argument("--eta", type=float, default=2.0)
    p.add_argument("--hyper-r", type=int, default=2)
    p.add_argument("--hyper-n", type=int, default=2)
    p.add_argument("--eta-z", type=float, default=0.0)
    p.add_argument("--M", type=int, default=10)
    p.add_argument("--t-end", type=float, required=True)
    p.add_argument("--sample-every", type=float, default=0.5, help="tau_A between samples")
    p.add_argument("--spectrum2d-every", type=float, default=10.0, help="tau_A between E(n_perp, n_z) records")
    p.add_argument("--checkpoint-every", type=float, default=10.0, help="tau_A between checkpoints")
    p.add_argument("--keep-checkpoints", type=int, default=6, help="rolling checkpoints kept (plus every 50 tau_A)")
    p.add_argument("--blowup-factor", type=float, default=1e3)
    p.add_argument("--max-wall-hours", type=float, default=23.0)
    p.add_argument("--out", type=str, required=True)
    return p.parse_args()


def benchmark_initial_state(N: int, M: int):
    """The initial state of alfvenic_cascade_benchmark.py at resolution N."""
    grid = SpectralGrid3D.create(Nx=N, Ny=N, Nz=N, Lx=1.0, Ly=1.0, Lz=1.0)
    state = initialize_random_spectrum(
        grid, M=M, alpha=5.0 / 3.0, amplitude=0.05, k_min=1.0, k_max=15.0,
        v_th=1.0, beta_i=1.0, seed=42,
    )
    return grid, state


class Cfg(NamedTuple):
    """Trace-time constants of a run (a static jit argument)."""

    forcing: str
    amplitude: float
    eta: float
    hyper_r: int
    hyper_n: int
    eta_z: float
    dt: float
    M: int
    beta_i: float
    Lambda: float


class Consts(NamedTuple):
    """Per-mode weights, passed to the jitted functions as data."""

    ew: jax.Array  # E = sum ew (|z+|^2 + |z-|^2): 0.25/(Nx Ny) w k_perp^2, as physics.energy
    damp_expm1: jax.Array  # expm1(2 eta (k_perp^2/k_perp_max^2)^r dt)
    wg: jax.Array  # 0.5/(Nx Ny) w, the |g_m|^2 weight
    n_perp: jax.Array  # integer shell of k_perp/2pi
    n_z: jax.Array  # integer |k_z|/2pi


def make_consts(grid: SpectralGrid3D, cfg: Cfg) -> tuple:
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    kx3 = grid.kx[None, None, :]
    ky3 = grid.ky[None, :, None]
    kz3 = grid.kz[:, None, None]
    kperp2 = kx3**2 + ky3**2
    ix = jnp.arange(Nx // 2 + 1)
    w = jnp.where((ix == 0) | ((Nx % 2 == 0) & (ix == Nx // 2)), 1.0, 2.0)[None, None, :]
    kpm2 = grid.kx[(Nx - 1) // 3] ** 2 + grid.ky[(Ny - 1) // 3] ** 2
    kappa = (kperp2 / kpm2) ** cfg.hyper_r
    two_pi = 2.0 * math.pi
    shape = (Nz, Ny, Nx // 2 + 1)
    n_perp = jnp.broadcast_to(jnp.floor(jnp.sqrt(kperp2) / two_pi + 0.5).astype(jnp.int32), shape)
    n_z = jnp.broadcast_to(jnp.floor(jnp.abs(kz3) / two_pi + 0.5).astype(jnp.int32), shape)
    consts = Consts(
        ew=jnp.broadcast_to((0.25 / (Nx * Ny)) * w * kperp2, shape).astype(jnp.float32),
        damp_expm1=jnp.broadcast_to(jnp.expm1(2.0 * cfg.eta * kappa * cfg.dt), shape).astype(jnp.float32),
        wg=jnp.broadcast_to((0.5 / (Nx * Ny)) * w, shape).astype(jnp.float32),
        n_perp=n_perp,
        n_z=n_z,
    )
    bins = (int(n_perp.max()) + 1, int(n_z.max()) + 1)
    return consts, float(jnp.sqrt(kpm2)), bins


def force(z_plus, z_minus, key, grid: SpectralGrid3D, cfg: Cfg):
    """The benchmark's forcing call, with its key handling."""
    key, subkey = jax.random.split(key)
    if cfg.forcing == "shell":
        # force_alfven_modes(band="shell", n 1-2): one draw added to z+ and z-
        F, key = _shell_noise(grid, cfg.amplitude, 1, 2, cfg.dt, subkey)
        return z_plus + F, z_minus + F, key
    # force_alfven_modes_balanced(n 1-2, max_nz=1, include_nz0=False, correlation=0)
    key, kA, kB, _ = jax.random.split(subkey, 4)
    Fp, _ = _perp_lowkz_noise(grid, cfg.amplitude, 1, 2, 1, False, cfg.dt, kA)
    Fm, _ = _perp_lowkz_noise(grid, cfg.amplitude, 1, 2, 1, False, cfg.dt, kB)
    return z_plus + Fp, z_minus + Fm, key


def step(fields: KRMHDFields, grid: SpectralGrid3D, cfg: Cfg) -> KRMHDFields:
    """gandalf_step(scheme="imex_rk222", nu=0, v_A=1): the kernel it calls."""
    return _gandalf_step_imex222_jit(
        fields, cfg.dt, grid.kx, grid.ky, grid.kz, grid.dealias_mask, cfg.eta, 1.0, cfg.beta_i,
        0.0, cfg.Lambda, cfg.M, grid.Nz, grid.Ny, grid.Nx, cfg.hyper_r, cfg.hyper_n,
        eta_z=cfg.eta_z, hyper_rz=1, closure="zero",
    )


def _chunk_impl(carry, grid: SpectralGrid3D, c: Consts, cfg: Cfg, n_steps: int):
    """n_steps of force-then-step, folding the energy budget of each step."""

    def body(_, state):
        fields, key, inj, diss, num = state
        zp, zm, key = force(fields.z_plus, fields.z_minus, key, grid, cfg)
        # |a|^2 - |b|^2 = Re((a - b) conj(a + b)), per mode before the reduction
        inj = inj + jnp.sum(c.ew * (
            jnp.real((zp - fields.z_plus) * jnp.conj(zp + fields.z_plus))
            + jnp.real((zm - fields.z_minus) * jnp.conj(zm + fields.z_minus))))
        after = step(fields._replace(z_plus=zp, z_minus=zm), grid, cfg)
        d = jnp.sum(c.ew * (jnp.abs(after.z_plus) ** 2 + jnp.abs(after.z_minus) ** 2) * c.damp_expm1)
        num = num + d + jnp.sum(c.ew * (
            jnp.real((after.z_plus - zp) * jnp.conj(after.z_plus + zp))
            + jnp.real((after.z_minus - zm) * jnp.conj(after.z_minus + zm))))
        return after, key, inj, diss + d, num

    fields, key = carry
    zero = jnp.zeros((), jnp.float32)
    fields, key, inj, diss, num = jax.lax.fori_loop(0, n_steps, body, (fields, key, zero, zero, zero))
    return (fields, key), (inj, diss, num)


_JIT = dict(static_argnames=("cfg", "n_steps"), donate_argnums=(0,))
try:
    # As krmhd.runloop: keeps the loop body's FFT count equal to the single step's.
    chunk = jax.jit(_chunk_impl, compiler_options={"xla_disable_hlo_passes": "while-loop-invariant-code-motion"}, **_JIT)
except TypeError:
    chunk = jax.jit(_chunk_impl, **_JIT)


@partial(jax.jit, static_argnames=("cfg", "bins"))
def diagnostics(fields: KRMHDFields, grid: SpectralGrid3D, c: Consts, cfg: Cfg, bins: tuple):
    Nz, Ny, Nx = grid.Nz, grid.Ny, grid.Nx
    zp, zm = fields.z_plus, fields.z_minus
    ep = c.ew * jnp.abs(zp) ** 2
    em = c.ew * jnp.abs(zm) ** 2
    e = ep + em
    phi = 0.5 * (zp + zm)
    A = 0.5 * (zp - zm)

    def real(f):
        return rfftn_inverse(f, Nz, Ny, Nx)

    def grad_max(f):
        fx = real(1j * grid.kx[None, None, :] * f)
        fy = real(1j * grid.ky[None, :, None] * f)
        return jnp.sqrt(fx**2 + fy**2).max()

    dt_cfl = cfl_timestep_estimate(zp, zm, grid, cfg.M, cfg.beta_i, cfg.Lambda, 1.0, 0.3, "zero")
    E_nperp = jnp.zeros(bins[0]).at[c.n_perp.ravel()].add(e.ravel())
    E_nz = jnp.zeros(bins[1]).at[c.n_z.ravel()].add(e.ravel())
    W_m = jnp.sum(c.wg[..., None] * jnp.abs(fields.g) ** 2, axis=(0, 1, 2))
    finite = jnp.stack([jnp.isfinite(zp).all() & jnp.isfinite(zm).all(), jnp.isfinite(fields.g).all()])
    scal = jnp.stack([
        jnp.sum(e), jnp.sum(2.0 * c.ew * jnp.abs(phi) ** 2), jnp.sum(2.0 * c.ew * jnp.abs(A) ** 2),
        jnp.sum(ep), jnp.sum(em), jnp.abs(real(zp)).max(), jnp.abs(real(zm)).max(),
        grad_max(zp), grad_max(zm), grad_max(phi), grad_max(A), dt_cfl,
    ])
    return scal, E_nperp, E_nz, W_m, finite


@partial(jax.jit, static_argnames=("bins",))
def spectrum2d(fields: KRMHDFields, c: Consts, bins: tuple):
    """E(n_perp, |n_z|)."""
    e = c.ew * (jnp.abs(fields.z_plus) ** 2 + jnp.abs(fields.z_minus) ** 2)
    flat = c.n_perp.ravel() * bins[1] + c.n_z.ravel()
    return jnp.zeros(bins[0] * bins[1]).at[flat].add(e.ravel()).reshape(bins)


def save_checkpoint(path: Path, fields: KRMHDFields, key, step: int, t: float, dt: float) -> None:
    tmp = path.with_suffix(".tmp.npz")
    np.savez(
        tmp, z_plus=np.asarray(fields.z_plus), z_minus=np.asarray(fields.z_minus),
        g=np.asarray(fields.g), key=np.asarray(key), step=step, t=t, dt=dt,
    )
    tmp.replace(path)


def main() -> int:
    args = parse_args()
    out = Path(args.out)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    shutil.copy(__file__, out / Path(__file__).name)
    wall0 = time.time()

    grid, state = benchmark_initial_state(args.resolution, args.M)
    dt = compute_cfl_timestep(state, v_A=1.0, cfl_safety=0.3)  # once, as the benchmark
    cfg = Cfg(forcing=args.forcing, amplitude=args.amplitude, eta=args.eta, hyper_r=args.hyper_r,
              hyper_n=args.hyper_n, eta_z=args.eta_z, dt=dt, M=state.M, beta_i=float(state.beta_i),
              Lambda=float(state.Lambda))
    consts, kperp_max, bins = make_consts(grid, cfg)
    n_edge = (args.resolution - 1) // 3  # 2/3-rule edge mode number per axis

    n_per_sample = max(1, int(round(args.sample_every / dt)))
    n_per_spec2d = max(1, int(round(args.spectrum2d_every / args.sample_every)))
    n_per_ckpt = max(1, int(round(args.checkpoint_every / args.sample_every)))

    config = dict(vars(args), dt=dt, steps_per_sample=n_per_sample, kperp_max=kperp_max,
                  n_edge=n_edge, jax_version=jax.__version__,
                  devices=[str(d) for d in jax.devices()])
    (out / "run_config.json").write_text(json.dumps(config, indent=2))

    ckpts = sorted((out / "checkpoints").glob("ck_step*.npz"))
    if ckpts:
        d = np.load(ckpts[-1])
        fields = KRMHDFields(z_plus=jnp.asarray(d["z_plus"]), z_minus=jnp.asarray(d["z_minus"]),
                             g=jnp.asarray(d["g"]), time=jnp.asarray(float(d["t"])))
        key = jnp.asarray(d["key"])
        step = int(d["step"])
        assert abs(float(d["dt"]) - dt) < 1e-12 * dt
        print(f"resumed from {ckpts[-1].name} at t = {float(d['t']):.3f}, step {step}", flush=True)
        spec = dict(np.load(out / "spectra.npz")) if (out / "spectra.npz").exists() else {}
        spec = {k: list(v) for k, v in spec.items()}
        spec2d = dict(np.load(out / "spectra2d.npz")) if (out / "spectra2d.npz").exists() else {}
        spec2d = {k: list(v) for k, v in spec2d.items()}
        # drop samples written after the checkpoint
        rows = []
        with open(out / "timeseries.csv") as fh:
            rows = [r for r in csv.DictReader(fh) if int(r["step"]) <= step]
        with open(out / "timeseries.csv", "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=FIELDS)
            wr.writeheader()
            wr.writerows(rows)
        keep = [i for i, s in enumerate(spec.get("step", [])) if s <= step]
        spec = {k: [v[i] for i in keep] for k, v in spec.items()}
        keep2 = [i for i, s in enumerate(spec2d.get("step", [])) if s <= step]
        spec2d = {k: [v[i] for i in keep2] for k, v in spec2d.items()}
        E_min = min([float(r["E"]) for r in rows if float(r["t"]) >= 20.0] or [math.inf])
    else:
        fields = KRMHDFields(z_plus=jnp.asarray(state.z_plus), z_minus=jnp.asarray(state.z_minus),
                             g=jnp.asarray(state.g), time=jnp.asarray(float(state.time)))
        key = jax.random.PRNGKey(42)
        step = 0
        spec = {}
        spec2d = {}
        E_min = math.inf
        with open(out / "timeseries.csv", "w", newline="") as fh:
            csv.DictWriter(fh, fieldnames=FIELDS).writeheader()

    print(f"N={args.resolution} forcing={args.forcing} amplitude={args.amplitude} eta={args.eta} "
          f"r={args.hyper_r} M={args.M} dt={dt:.6g} steps/sample={n_per_sample} "
          f"kperp_max={kperp_max:.2f} devices={jax.devices()}", flush=True)

    def record(fields, step, budget, t_wall):
        nonlocal E_min
        scal, E_nperp, E_nz, W_m, finite = jax.device_get(diagnostics(fields, grid, consts, cfg, bins))
        inj, diss, num = (float(x) for x in jax.device_get(budget)) if budget is not None else (0.0, 0.0, 0.0)
        span = n_per_sample * dt
        E = float(scal[0])
        tot = max(E, 1e-300)
        nperp = np.arange(len(E_nperp))
        f_half = float(E_nperp[nperp > n_edge / 2].sum() / tot)
        f_edge = float(E_nperp[nperp >= 0.8 * n_edge].sum() / tot)
        f_kz = float(E_nz[np.arange(len(E_nz)) > n_edge / 2].sum() / tot)
        U = max(float(scal[7]), float(scal[8]))
        t = step * dt
        row = dict(
            t=f"{t:.6f}", step=step, E=E, E_kin=float(scal[1]), E_mag=float(scal[2]),
            E_plus=float(scal[3]), E_minus=float(scal[4]), zp_max=float(scal[5]), zm_max=float(scal[6]),
            U_plus=float(scal[7]), U_minus=float(scal[8]), u_max=float(scal[9]), b_max=float(scal[10]),
            dt_cfl=float(scal[11]), dt_over_cfl=dt / float(scal[11]), y_max=dt * kperp_max * U,
            f_half=f_half, f_edge=f_edge, f_kz_half=f_kz,
            eps_inj=inj / span, eps_diss=diss / span, eps_num=num / span,
            W_g=float(W_m.sum()), g_finite=int(finite[1]), wall_s=round(t_wall, 1),
        )
        with open(out / "timeseries.csv", "a", newline="") as fh:
            csv.DictWriter(fh, fieldnames=FIELDS).writerow(row)
        for k, v in (("t", t), ("step", step), ("E_nperp", E_nperp), ("E_nz", E_nz), ("W_m", W_m)):
            spec.setdefault(k, []).append(v)
        np.savez(out / "spectra.npz", **{k: np.asarray(v) for k, v in spec.items()})
        if t >= 20.0 and np.isfinite(E):
            E_min = min(E_min, E)
        return row, bool(finite[0]) and np.isfinite(E)

    if step == 0:
        record(fields, 0, None, 0.0)

    n_samples = int(math.ceil(args.t_end / (n_per_sample * dt)))
    sample = step // n_per_sample
    status = 0
    while sample < n_samples:
        (fields, key), budget = chunk((fields, key), grid, consts, cfg, n_per_sample)
        step += n_per_sample
        sample += 1
        row, ok = record(fields, step, budget, time.time() - wall0)
        if sample % n_per_spec2d == 0 or not ok:
            s2 = np.asarray(jax.device_get(spectrum2d(fields, consts, bins)))
            spec2d.setdefault("t", []).append(step * dt)
            spec2d.setdefault("step", []).append(step)
            spec2d.setdefault("E2d", []).append(s2)
            np.savez(out / "spectra2d.npz", **{k: np.asarray(v) for k, v in spec2d.items()})
        if sample % 20 == 0 or not ok:
            print(f"t={step * dt:8.2f} E={row['E']:.4e} zmax={max(row['zp_max'], row['zm_max']):.3e} "
                  f"U={max(row['U_plus'], row['U_minus']):.3f} y={row['y_max']:.3f} dt/cfl={row['dt_over_cfl']:.3f} "
                  f"f_edge={row['f_edge']:.2e} inj={row['eps_inj']:.3e} diss={row['eps_diss']:.3e} "
                  f"num={row['eps_num']:+.2e} W_g={row['W_g']:.3e} g_ok={row['g_finite']} "
                  f"wall={row['wall_s']:.0f}s", flush=True)
        blew_up = (not ok) or (math.isfinite(E_min) and row["E"] > args.blowup_factor * E_min)
        if sample % n_per_ckpt == 0 and not blew_up:
            t = step * dt
            save_checkpoint(out / "checkpoints" / f"ck_step{step:09d}.npz", fields, key, step, t, dt)
            rolling = sorted((out / "checkpoints").glob("ck_step*.npz"))
            for old in rolling[:-args.keep_checkpoints]:
                t_old = float(np.load(old)["t"])
                if abs(t_old / 50.0 - round(t_old / 50.0)) * 50.0 > 0.6 * args.sample_every:
                    old.unlink()
        if blew_up:
            print(f"BLOWUP at t={step * dt:.3f} (step {step}): E={row['E']:.4e}, finite={ok}", flush=True)
            status = 2
            break
        if (time.time() - wall0) / 3600.0 > args.max_wall_hours:
            print(f"paused at t = {step * dt:.6f}", flush=True)
            save_checkpoint(out / "checkpoints" / f"ck_step{step:09d}.npz", fields, key, step, step * dt, dt)
            return 3

    print(f"done: t={step * dt:.3f}, status={status}, wall={time.time() - wall0:.0f}s", flush=True)
    return status


if __name__ == "__main__":
    sys.exit(main())
