#!/usr/bin/env python
"""
Multirate slice stepping against single-rate Lawson-RK4 (Issue #191, item 1).

scheme="lawson_rk4_multirate" advances z± with the macro step H and slice j of
the Hermite hierarchy with n_j = ceil(H·v_j / (C·Δ)) substeps (krmhd.multirate).
This script measures its stability, order, conservation and cost against
scheme="lawson_rk4" in P1-like turbulence. docs/multirate_slices_study.md
reports the numbers.

Physics: P1's forced_beta1 at P1's test parameters, as in precision_study.py.
Λ = √5, β_i = 1, zero closure, η = 200 (r = 2), ν = 30 (n = 6), white-noise
forcing of Φ and g_0 on k⊥L/2π ∈ [1, 2], |n_z| = 1, at ε_A = ε_g = 0.3, unit box.
Two step rules, each applied to both schemes:

  courant  CflDt(0.3, every=10): the Courant estimate with the Hermite term for
           lawson_rk4, without it (mode="multirate") for the multirate step,
           whose substeps use the same Courant number 0.3 slice by slice.
  exact    P1's rule: a fixed dt of 0.8 times the exact limit at safety 1
           (compute_cfl_timestep(mode="exact")), recomputed every --recal τ_A.
           The multirate step takes 0.8 times the Elsasser and v_A limit, and
           substep Courant numbers from exact_substep_cfl(state, 0.8): every
           slice at 0.8 of its own RK4 limit.

Commands (each writes into --out; run from the repository root):
    spinup    forced lawson_rk4 run from rest (or from a smaller state), saves a state
    frozen    per-slice stability in a frozen shear flow
    radii     slice speeds, exact radii, substeps and speed-up bounds of a state
    run       forced or decaying run of one scheme, history of energies and work
    accuracy  errors against a fine lawson_rk4 reference, unforced
    cost      wall time per τ_A of both schemes, alternating repetitions
    report    prints the tables of the doc
"""

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from krmhd.forcing import WhiteNoiseBand  # noqa: E402
from krmhd.hermite import compute_streaming_eigensystem, free_energy_weights  # noqa: E402
from krmhd.multirate import exact_substep_cfl, multirate_substeps  # noqa: E402
from krmhd.physics import KRMHDState, enforce_hermitian_state  # noqa: E402
from krmhd.runloop import (  # noqa: E402
    Accounting,
    CflDt,
    DeviceLoop,
    FixedDt,
    Forcing,
    RunConfig,
    StepParams,
)
from krmhd.spectral import SpectralGrid3D  # noqa: E402
from krmhd.timestepping import (  # noqa: E402
    RK4_IMAGINARY_AXIS_LIMIT,
    _perp_gradients,
    compute_cfl_timestep,
    gandalf_step,
    slice_advection_radius,
)

LAMBDA, BETA = math.sqrt(5.0), 1.0
ETA, HYPER_R, NU, HYPER_N = 200.0, 2, 30.0, 6
EPS_A = EPS_G = 0.3
DT_REF = 2e-3  # the forcing injects at ε for any dt (WhiteNoiseBand.dt_ref)
BAND = dict(n_min=1, n_max=2, band="perp_lowkz", max_nz=1, include_nz0=False)
CFL = 0.3
EVERY = 10  # steps between dt updates
P1_SAFETY = 0.8
SR, MR = "lawson_rk4", "lawson_rk4_multirate"


def grid_of(N: int) -> SpectralGrid3D:
    return SpectralGrid3D.create(Nx=N, Ny=N, Nz=N, Lx=1.0, Ly=1.0, Lz=1.0)


def params(scheme: str, ideal: bool = False, substep_cfl=CFL) -> StepParams:
    return StepParams(
        eta=0.0 if ideal else ETA, v_A=1.0, nu=0.0 if ideal else NU, hyper_r=HYPER_R,
        hyper_n=HYPER_N, scheme=scheme, substep_cfl=substep_cfl,
    )


def rfft_weights(N: int) -> np.ndarray:
    return np.where(np.isin(np.arange(N // 2 + 1), (0, N // 2)), 1.0, 2.0)[None, None, :]


def forcing(grid: SpectralGrid3D, M: int) -> Forcing:
    """Φ and g_0 forcing at ε_A, ε_g (precision_study.forcing)."""
    N = grid.Nx
    unit = WhiteNoiseBand(amplitude=1.0, **BAND)
    mask = np.abs(np.asarray(unit.draw(grid, 1.0, jax.random.PRNGKey(12345))[0])) > 0
    v = np.where(np.arange(N // 2 + 1) == 0, 1.0, 2.0)[None, None, :]
    kp2 = np.asarray(grid.kx)[None, None, :] ** 2 + np.asarray(grid.ky)[None, :, None] ** 2
    W = rfft_weights(N)
    c_A = float(np.sum(mask * 0.5 * W * v * kp2)) / float(N**3) ** 2
    c_g = float(free_energy_weights(M, LAMBDA)[0] * np.sum(mask * W * v)) / float(N**3) ** 2
    return Forcing(
        alfven=WhiteNoiseBand(amplitude=DT_REF * math.sqrt(EPS_A / c_A), dt_ref=DT_REF, **BAND),
        hermite=WhiteNoiseBand(amplitude=DT_REF * math.sqrt(EPS_G / c_g), dt_ref=DT_REF, **BAND),
        moments=(0,),
    )


def make_state(zp, zm, g, t: float = 0.0) -> KRMHDState:
    N, M = zp.shape[1], g.shape[-1] - 1
    return KRMHDState(
        z_plus=jnp.array(zp, jnp.complex128), z_minus=jnp.array(zm, jnp.complex128),
        g=jnp.array(g, jnp.complex128), M=M, beta_i=BETA, v_th=1.0, nu=NU, Lambda=LAMBDA,
        time=t, grid=grid_of(N),
    )


def load(path: Path) -> KRMHDState:
    d = np.load(path)
    return make_state(d["zp"], d["zm"], d["g"], float(d["t"]))


def host(state: KRMHDState) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return tuple(np.asarray(x, np.complex128) for x in (state.z_plus, state.z_minus, state.g))


def energies(state: KRMHDState) -> Tuple[float, float]:
    """E_A = (1/4)<k⊥²(|z⁺|² + |z⁻|²)> and W = Σ_m p_m <g_m²>, volume averages."""
    zp, zm, g = host(state)
    grid, N = state.grid, state.grid.Nx
    W = rfft_weights(N)
    kp2 = np.asarray(grid.kx)[None, None, :] ** 2 + np.asarray(grid.ky)[None, :, None] ** 2
    n6 = float(N**3) ** 2
    E = float(np.sum(0.25 * W * kp2 * (np.abs(zp) ** 2 + np.abs(zm) ** 2))) / n6
    p = free_energy_weights(state.M, LAMBDA)
    Wg = float(np.sum(p * np.sum(W[..., None] * np.abs(g) ** 2, axis=(0, 1, 2)))) / n6
    return E, Wg


def slice_speeds(M: int) -> np.ndarray:
    _, lam, _, _ = compute_streaming_eigensystem(M, LAMBDA, "zero")
    return math.sqrt(BETA) * np.asarray(lam)


def substeps(state: KRMHDState, dt: float, substep_cfl=CFL) -> np.ndarray:
    grid = state.grid
    return np.asarray(multirate_substeps(
        state.z_plus, state.z_minus, jnp.asarray(slice_speeds(state.M)), dt,
        jnp.asarray(substep_cfl) / grid.Nx, grid.kx, grid.ky, grid.Nz, grid.Ny, grid.Nx,
    ))


def slice_update_counter(state: KRMHDState, substep_cfl=CFL) -> Accounting:
    """Σ_j n_j of every multirate step, from the forced fields the kernel sees."""
    grid, speeds = state.grid, jnp.asarray(slice_speeds(state.M))
    courant = jnp.asarray(substep_cfl) / grid.Nx

    def update(acc, before, after, increment, dt):
        zp = before.z_plus + increment.alfven
        zm = before.z_minus + increment.alfven
        n = multirate_substeps(zp, zm, speeds, dt, courant, grid.kx, grid.ky,
                               grid.Nz, grid.Ny, grid.Nx)
        return acc + jnp.sum(n)

    return Accounting(init=jnp.zeros((), jnp.int64), update=update)


def policy(state: KRMHDState, scheme: str, rule: str, ideal: bool = False, cfl: float = CFL,
           substep_cfl: float = CFL):
    """StepParams and dt policy of a scheme under a step rule (see the module docstring)."""
    if rule == "courant":
        return params(scheme, ideal, substep_cfl), CflDt(cfl, every=EVERY)
    if scheme == SR:
        return params(SR, ideal), FixedDt(P1_SAFETY * compute_cfl_timestep(state, 1.0, 1.0, mode="exact"))
    H = P1_SAFETY * compute_cfl_timestep(state, 1.0, 1.0, mode="multirate")
    C = tuple(float(x) for x in exact_substep_cfl(state, P1_SAFETY))
    return params(MR, ideal, C), FixedDt(H)


def save_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=1))
    print(f"wrote {path}", flush=True)


# =============================================================================
# spinup
# =============================================================================


def pad(state: KRMHDState, N: int, M: int) -> KRMHDState:
    """Zero-pad z± and g to an N³ grid and M + 1 moments (spectral interpolation)."""
    zp, zm, g = host(state)
    n0 = zp.shape[1]
    scale = (N / n0) ** 3  # GANDALF's forward FFT is unnormalised
    h = n0 // 2

    def up(a):
        out = np.zeros((N, N, N // 2 + 1) + a.shape[3:], np.complex128)
        for zs, zd in ((slice(0, h), slice(0, h)), (slice(n0 - h, n0), slice(N - h, N))):
            for ys, yd in ((slice(0, h), slice(0, h)), (slice(n0 - h, n0), slice(N - h, N))):
                out[zd, yd, :h] = a[zs, ys, :h]
        return out * scale

    g = up(g)
    g = np.concatenate([g, np.zeros(g.shape[:3] + (M + 1 - g.shape[-1],))], axis=-1)
    return enforce_hermitian_state(make_state(up(zp), up(zm), g, state.time))


def cmd_spinup(a) -> None:
    out = Path(a.out)
    if a.start:
        state = pad(load(out / a.start), a.N, a.M)
    else:
        z = np.zeros((a.N, a.N, a.N // 2 + 1))
        state = make_state(z, z, np.zeros(z.shape + (a.M + 1,)))
    grid = state.grid
    config = RunConfig(step=params(SR), dt=CflDt(CFL, every=EVERY), forcing=forcing(grid, a.M))
    loop = DeviceLoop(state, config, key=jax.random.PRNGKey(2026))
    t_end, wall0 = state.time + a.t, time.time()
    while loop.state.time < t_end:
        loop.advance(50)
        E, Wg = energies(loop.state)
        if not np.isfinite(E + Wg):
            raise RuntimeError(f"non-finite energy at t = {loop.state.time}")
        print(f"t = {loop.state.time:6.3f}  E_A = {E:.4f}  W = {Wg:.4f}  dt = {float(loop.dt):.2e}"
              f"  wall {time.time() - wall0:.0f} s", flush=True)
    zp, zm, g = host(loop.state)
    np.savez(out / a.name, zp=zp, zm=zm, g=g, t=loop.state.time)
    print(f"saved {out / a.name}", flush=True)


# =============================================================================
# frozen: per-slice stability in a frozen flow
# =============================================================================


def cmd_frozen(a) -> None:
    """Shear flow z± = (1 ± 1/2)Φ(x): z± are steady and slice j is advected by
    (1 + c_j/2)Φ, so the slices are independent and ρ_j/v_j is one constant κ. The
    substep Courant number puts the largest h_j·ρ_j at 0.95 or 1.15 of 2√2."""
    M, N = 16, 16
    grid = SpectralGrid3D.create(Nx=N, Ny=N, Nz=8)
    x = np.arange(N) / N
    phi = jnp.fft.rfftn(np.broadcast_to(np.cos(2 * np.pi * x) + 0.5 * np.sin(4 * np.pi * x),
                                        (8, N, N)))
    rng = np.random.default_rng(0)
    shape = (8, N, N // 2 + 1, M + 1)
    g = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) * np.asarray(
        grid.dealias_mask)[..., None]
    state0 = enforce_hermitian_state(KRMHDState(
        z_plus=1.5 * phi, z_minus=0.5 * phi, g=jnp.asarray(g), M=M, beta_i=1.0, v_th=1.0,
        nu=0.0, Lambda=LAMBDA, time=0.0, grid=grid))
    _, lam, _, P_inv = compute_streaming_eigensystem(M, LAMBDA, "zero")
    c = np.asarray(lam)
    rho = np.asarray(slice_advection_radius(
        state0.z_plus, state0.z_minus, jnp.asarray(c), grid.kx, grid.ky, grid.dealias_mask,
        grid.Nz, grid.Ny, grid.Nx, tol=1e-10).rho)
    dphi = np.asarray(_perp_gradients(state0.z_plus, state0.z_minus, grid.kx, grid.ky,
                                      grid.Nz, grid.Ny, grid.Nx)[0])
    v = np.abs(1 + c / 2) * np.abs(dphi).max()
    kappa = float(rho[np.argmax(rho)] / v[np.argmax(rho)])
    spacing = grid.Lx / N
    H = 5.0 * RK4_IMAGINARY_AXIS_LIMIT / rho.max()

    def norms(s):
        h = np.einsum("jm,zyxm->jzyx", np.asarray(P_inv), np.asarray(s.g))
        r = np.fft.irfftn(h, s=(8, N, N), axes=(1, 2, 3))
        return np.sqrt(np.sum(r**2, axis=(1, 2, 3)))

    n0, rows = norms(state0), {}
    for factor in (0.95, 1.15):
        sc = factor * RK4_IMAGINARY_AXIS_LIMIT / (kappa * spacing)
        counts = np.asarray(multirate_substeps(
            state0.z_plus, state0.z_minus, jnp.asarray(c), H, sc * spacing, grid.kx, grid.ky,
            grid.Nz, grid.Ny, grid.Nx))
        s = state0
        for _ in range(12):
            s = gandalf_step(s, H, eta=0.0, v_A=1.0, nu=0.0, scheme=MR, substep_cfl=sc)
        y = H / counts * rho / RK4_IMAGINARY_AXIS_LIMIT
        rows[str(factor)] = dict(lam=c.tolist(), counts=counts.tolist(), y=y.tolist(),
                                 growth=(norms(s) / n0).tolist())
        print(f"C·κ = {factor}·2√2: slice, λ_j, n_j, h_jρ_j/2√2, norm ratio after 12 steps")
        for j in np.argsort(c):
            print(f"  {j:3d} {c[j]:+7.3f} {counts[j]:3d} {y[j]:6.3f} {rows[str(factor)]['growth'][j]:.3e}")
    save_json(Path(a.out) / "frozen.json", dict(kappa=kappa, H=H, rows=rows))


# =============================================================================
# radii: speeds, exact radii, substeps and speed-up bounds of a state
# =============================================================================


def cmd_radii(a) -> None:
    out = Path(a.out)
    result = {}
    for name in a.states:
        state = load(out / name)
        grid, M = state.grid, state.M
        c = slice_speeds(M)
        grads = _perp_gradients(state.z_plus, state.z_minus, grid.kx, grid.ky,
                                grid.Nz, grid.Ny, grid.Nx)
        dphi_dx, dphi_dy, dpsi_dx, dpsi_dy = (np.asarray(x) for x in grads)
        v = np.array([np.hypot(dphi_dx + cj * dpsi_dx, dphi_dy + cj * dpsi_dy).max() for cj in c])
        rad = slice_advection_radius(state.z_plus, state.z_minus, jnp.asarray(c), grid.kx, grid.ky,
                                     grid.dealias_mask, grid.Nz, grid.Ny, grid.Nx)
        rho = np.asarray(rad.rho)
        H = compute_cfl_timestep(state, 1.0, CFL, mode="multirate")
        dt_sr = compute_cfl_timestep(state, 1.0, CFL)
        dt_exact = compute_cfl_timestep(state, 1.0, CFL, mode="exact")
        n = substeps(state, H)
        grad_phi = float(np.hypot(dphi_dx, dphi_dy).max())
        grad_psi = float(np.hypot(dpsi_dx, dpsi_dy).max())
        r = dict(
            M=M, N=grid.Nx, lam=c.tolist(), v=v.tolist(), rho=rho.tolist(), n=n.tolist(),
            converged=bool(np.all(np.asarray(rad.converged))), H=H, dt_sr=dt_sr, dt_exact=dt_exact,
            grad_phi=grad_phi, grad_psi=grad_psi,
            mean_rho=float(rho.mean() / rho.max()), mean_v=float(v.mean() / v.max()),
            # Single-rate slice updates per multirate macro step over Σ n_j
            bound_rho=float(rho.max() / rho.mean()), bound_v=float(v.max() / v.mean()),
            speedup_updates=float((M + 1) * H / dt_sr / n.sum()),
        )
        result[name] = r
        print(f"{name}: N={grid.Nx} M={M} |∇Φ|={grad_phi:.2f} |∇Ψ|={grad_psi:.2f}  "
              f"mean ρ/ρmax={r['mean_rho']:.3f}  mean v/vmax={r['mean_v']:.3f}  "
              f"H/dt_sr={H / dt_sr:.2f}  dt_exact/dt_sr={dt_exact / dt_sr:.2f}  n={n.tolist()}  "
              f"updates ratio={r['speedup_updates']:.2f}", flush=True)
    save_json(out / f"radii_{a.tag}.json", result)


# =============================================================================
# run: forced or decaying run of one scheme
# =============================================================================


def cmd_run(a) -> None:
    out = Path(a.out)
    state = load(out / a.state)
    ideal = a.ideal
    forced = not a.decaying
    fx = forcing(state.grid, state.M) if forced else Forcing()

    def make_loop(s: KRMHDState, key) -> DeviceLoop:
        step, dt = policy(s, a.scheme, a.rule, ideal, a.cfl, a.substep_cfl)
        counter = slice_update_counter(s, step.substep_cfl) if a.scheme == MR else None
        return DeviceLoop(s, RunConfig(step=step, dt=dt, forcing=fx, accounting=counter),
                          key=key)

    loop = make_loop(state, jax.random.PRNGKey(a.seed))
    t0, wall0 = loop.state.time, time.time()
    next_recal = t0 + a.recal
    E0, W0 = energies(loop.state)
    hist = [dict(t=t0, E=E0, W=W0, steps=0, updates=0, dt=float(loop.dt), zmax=0.0, gmax=0.0, wall=0.0)]
    steps = updates = 0
    status = "ok"
    while loop.state.time < t0 + a.t:
        if a.rule == "exact" and loop.state.time >= next_recal:
            loop = make_loop(make_state(*host(loop.state), loop.state.time), loop.key)
            next_recal += a.recal
        acc = loop.advance(a.chunk)
        s = loop.state
        steps += a.chunk
        updates += int(acc) if a.scheme == MR else a.chunk * (s.M + 1)
        E, Wg = energies(s)
        zmax = float(max(jnp.abs(s.z_plus).max(), jnp.abs(s.z_minus).max()))
        gmax = float(jnp.abs(s.g).max())
        hist.append(dict(t=s.time, E=E, W=Wg, steps=steps, updates=updates, dt=float(loop.dt),
                         zmax=zmax, gmax=gmax, wall=time.time() - wall0))
        if not np.isfinite(E + Wg) or E > 1e6 * max(E0, 1e-30):
            status = f"blew up at t = {s.time:.3f}"
            break
        if len(hist) % 10 == 0:
            print(f"{a.tag}: t = {s.time - t0:6.3f}  E_A = {E:.4f}  W = {Wg:.4f}  dt = "
                  f"{float(loop.dt):.2e}  updates/step = {updates / steps:.1f}  wall "
                  f"{time.time() - wall0:.0f} s", flush=True)
    print(f"{a.tag}: {status}, {steps} steps, wall {time.time() - wall0:.0f} s", flush=True)
    save_json(out / f"run_{a.tag}.json", dict(
        state=a.state, scheme=a.scheme, rule=a.rule, ideal=ideal, forced=forced, cfl=a.cfl,
        substep_cfl=a.substep_cfl, status=status, M=state.M, N=state.grid.Nx, history=hist))


# =============================================================================
# accuracy: errors against a fine lawson_rk4 reference
# =============================================================================


def advance_fixed(state: KRMHDState, scheme: str, dt: float, n: int, ideal: bool,
                  substep_cfl: float = CFL) -> KRMHDState:
    config = RunConfig(step=params(scheme, ideal, substep_cfl), dt=FixedDt(dt))
    loop = DeviceLoop(make_state(*host(state), state.time), config)
    loop.advance(n)
    return loop.state


def rel_errors(s: KRMHDState, ref: KRMHDState) -> Dict[str, float]:
    a, b = host(s), host(ref)
    z = math.sqrt((np.sum(np.abs(a[0] - b[0]) ** 2) + np.sum(np.abs(a[1] - b[1]) ** 2))
                  / (np.sum(np.abs(b[0]) ** 2) + np.sum(np.abs(b[1]) ** 2)))
    g = float(np.linalg.norm(a[2] - b[2]) / np.linalg.norm(b[2]))
    E, Wg = energies(s)
    E_ref, W_ref = energies(ref)
    return dict(z=z if np.isfinite(z) else None, g=g if np.isfinite(g) else None,
                dE=(E - E_ref) / E_ref, dW=(Wg - W_ref) / W_ref)


def cmd_accuracy(a) -> None:
    out = Path(a.out)
    state = load(out / a.state)
    H = compute_cfl_timestep(state, 1.0, CFL, mode="multirate")
    dt_sr = compute_cfl_timestep(state, 1.0, CFL)
    n_macro = a.steps
    T = n_macro * H
    result = dict(state=a.state, H=H, dt_sr=dt_sr, T=T, counts=substeps(state, H).tolist())
    for ideal in (True, False):
        key = "ideal" if ideal else "dissipative"
        t0 = time.time()
        ref = advance_fixed(state, SR, H / a.ref_factor, n_macro * a.ref_factor, ideal)
        E0, W0 = energies(state)
        E1, W1 = energies(ref)
        r = dict(ref=dict(dt=H / a.ref_factor, dE=(E1 - E0) / E0, dW=(W1 - W0) / W0))
        # Convergence: dt and the substep Courant number refined together
        r["multirate_converge"] = [
            dict(dt=H / 2**k, substep_cfl=CFL / 2**k,
                 **rel_errors(advance_fixed(state, MR, H / 2**k, n_macro * 2**k, ideal, CFL / 2**k), ref))
            for k in range(4)
        ]
        # Equal macro dt, production substep Courant number
        r["multirate_equal_dt"] = [
            dict(dt=H / 2**k, **rel_errors(advance_fixed(state, MR, H / 2**k, n_macro * 2**k, ideal), ref))
            for k in range(4)
        ]
        r["lawson_equal_dt"] = [
            dict(dt=H / 2**k, **rel_errors(advance_fixed(state, SR, H / 2**k, n_macro * 2**k, ideal), ref))
            for k in range(4)
        ]
        # Each scheme at its own Courant dt (0.3): lawson at dt_sr, rounded to fit T
        n_sr = math.ceil(T / dt_sr)
        r["lawson_own_dt"] = dict(dt=T / n_sr, **rel_errors(advance_fixed(state, SR, T / n_sr, n_sr, ideal), ref))
        r["multirate_own_dt"] = dict(dt=H, **rel_errors(advance_fixed(state, MR, H, n_macro, ideal), ref))
        result[key] = r
        print(f"{a.state} {key}: done in {time.time() - t0:.0f} s", flush=True)
        for name in ("multirate_converge", "multirate_equal_dt", "lawson_equal_dt"):
            print(f"  {name}: " + "  ".join(
                f"dt={e['dt']:.2e} z={e['z']} g={e['g']}" for e in r[name]), flush=True)
        print(f"  own dt: lawson {r['lawson_own_dt']}  multirate {r['multirate_own_dt']}", flush=True)
    save_json(out / f"accuracy_{a.tag}.json", result)


# =============================================================================
# cost: wall time per τ_A, alternating repetitions
# =============================================================================


def cmd_cost(a) -> None:
    out = Path(a.out)
    state0 = load(out / a.state)
    M = state0.M
    fx = forcing(state0.grid, M)

    policies = {scheme: policy(state0, scheme, a.rule) for scheme in (SR, MR)}

    def loop_for(scheme: str, accounting=None, elsasser: bool = False) -> DeviceLoop:
        zp, zm, g = host(state0)
        step, dt = policies[scheme]
        if elsasser:  # the z± part of a lawson_rk4 step: the same step at M = 0
            g, step, f = g[..., :1], step._replace(nu=0.0), Forcing(alfven=fx.alfven)
            dt = FixedDt(dt_sr)
        else:
            f = fx
        cfg = RunConfig(step=step, dt=dt, forcing=f, accounting=accounting)
        return DeviceLoop(make_state(zp, zm, g, state0.time), cfg, key=jax.random.PRNGKey(7))

    def timed(scheme: str, n: int, elsasser: bool = False) -> Tuple[float, float, int]:
        loop = loop_for(scheme, elsasser=elsasser)
        loop.advance(2)  # compile, and leave the first two steps out
        loop.state.g.block_until_ready()
        t_start = loop.state.time
        w0 = time.perf_counter()
        loop.advance(n)
        loop.state.g.block_until_ready()
        wall = time.perf_counter() - w0
        s = loop.state
        if not all(bool(jnp.all(jnp.isfinite(x))) for x in (s.z_plus, s.g)):
            raise RuntimeError(f"{scheme}: non-finite state")
        return wall, s.time - t_start, int(loop.steps) - 2

    # Steps that cover the same simulated time, about a.window τ_A
    def first_dt(scheme: str) -> float:
        step, dt = policies[scheme]
        if isinstance(dt, FixedDt):
            return dt.dt
        mode = "multirate" if scheme == MR else "courant"
        return compute_cfl_timestep(state0, 1.0, dt.cfl_safety, mode=mode)

    dt_sr, H = first_dt(SR), first_dt(MR)
    n_sr = max(10, round(a.window / dt_sr))
    n_mr = max(10, round(a.window / H))

    # Slice updates of the same multirate trajectory (untimed, with the counting hook)
    loop = loop_for(MR, accounting=slice_update_counter(state0, policies[MR][0].substep_cfl))
    loop.advance(2)
    updates_mr = int(loop.advance(n_mr))

    runs = {SR: [], MR: [], "elsasser": []}
    for rep in range(a.reps):
        for scheme in (SR, MR):
            wall, dt_sim, steps = timed(scheme, n_sr if scheme == SR else n_mr)
            runs[scheme].append(dict(wall=wall, t=dt_sim, steps=steps))
            print(f"rep {rep} {scheme}: {steps} steps, {dt_sim:.4f} τ_A, {wall:.2f} s, "
                  f"{wall / dt_sim:.1f} s/τ_A", flush=True)
    for rep in range(3):
        wall, dt_sim, steps = timed(SR, n_sr, elsasser=True)
        runs["elsasser"].append(dict(wall=wall, t=dt_sim, steps=steps))
    per_tau = {k: [r["wall"] / r["t"] for r in v] for k, v in runs.items()}
    result = dict(
        state=a.state, rule=a.rule, N=state0.grid.Nx, M=M, H=H, dt_sr=dt_sr, n_sr=n_sr,
        n_mr=n_mr, runs=runs,
        updates_mr=updates_mr,
        per_tau={k: dict(median=float(np.median(v)), min=float(np.min(v)), max=float(np.max(v)))
                 for k, v in per_tau.items()},
    )
    sr = runs[SR][0]
    result["updates_per_tau"] = {
        SR: (M + 1) * sr["steps"] / sr["t"],
        MR: updates_mr / runs[MR][0]["t"],
    }
    result["speedup_wall"] = result["per_tau"][SR]["median"] / result["per_tau"][MR]["median"]
    result["speedup_updates"] = result["updates_per_tau"][SR] / result["updates_per_tau"][MR]
    print(f"{a.state}: wall per τ_A {result['per_tau']}  speed-up {result['speedup_wall']:.2f}  "
          f"slice updates per τ_A {result['updates_per_tau']}  ratio {result['speedup_updates']:.2f}",
          flush=True)
    save_json(out / f"cost_{a.tag}.json", result)


# =============================================================================
# report
# =============================================================================


def cmd_report(a) -> None:
    out = Path(a.out)

    def rows(pattern: str) -> List[Tuple[str, dict]]:
        return [(p.stem, json.loads(p.read_text())) for p in sorted(out.glob(pattern))]

    print("## Runs")
    print("| run | N | M | scheme | rule | forced | ideal | status | t (τ_A) | steps | updates/τ_A | "
          "W end/start | E_A end/start | mean E_A | mean W |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name, r in rows("run_*.json"):
        h = r["history"]
        t = h[-1]["t"] - h[0]["t"]
        late = [x for x in h if x["t"] - h[0]["t"] >= t / 2]
        print(f"| {name[4:]} | {r['N']} | {r['M']} | {r['scheme']} | {r.get('rule', 'courant')} | "
              f"{r['forced']} | {r['ideal']} | "
              f"{r['status']} | {t:.2f} | {h[-1]['steps']} | {h[-1]['updates'] / t:.0f} | "
              f"{h[-1]['W'] / h[0]['W']:.6f} | {h[-1]['E'] / h[0]['E']:.6f} | "
              f"{np.mean([x['E'] for x in late]):.4f} | {np.mean([x['W'] for x in late]):.4f} |")
    for name, r in rows("accuracy_*.json"):
        print(f"\n## {name}: H = {r['H']:.3e}, lawson Courant dt = {r['dt_sr']:.3e}, T = {r['T']:.3e}")
        print(f"counts {r['counts']}")
        for key in ("ideal", "dissipative"):
            x = r[key]
            print(f"\n{key}: reference dt {x['ref']['dt']:.2e}, dE/E {x['ref']['dE']:.2e}, "
                  f"dW/W {x['ref']['dW']:.2e}")
            for k in ("multirate_converge", "multirate_equal_dt", "lawson_equal_dt"):
                errs = [e["g"] for e in x[k]]
                rates = [math.log2(e0 / e1) if e0 and e1 else None for e0, e1 in zip(errs, errs[1:])]
                print(f"{k}: " + ", ".join(
                    f"dt {e['dt']:.2e}: z {e['z']:.2e} g {e['g']:.2e} dW {e['dW']:.1e}"
                    if e["g"] is not None else f"dt {e['dt']:.2e}: non-finite" for e in x[k]))
                print(f"   g rates {['%.2f' % q if q else None for q in rates]}")
            for k in ("lawson_own_dt", "multirate_own_dt"):
                e = x[k]
                print(f"{k}: dt {e['dt']:.2e} z {e['z']:.2e} g {e['g']:.2e} dE {e['dE']:.1e} dW {e['dW']:.1e}")
    for name, r in rows("cost_*.json"):
        p = r["per_tau"]
        el = r["runs"]["elsasser"]
        sr_ms = [x["wall"] / x["steps"] * 1e3 for x in r["runs"][SR]]
        mr_ms = [x["wall"] / x["steps"] * 1e3 for x in r["runs"][MR]]
        print(f"\n## {name}: rule {r.get('rule', 'courant')} N={r['N']} M={r['M']} dt_sr = "
              f"{r['dt_sr']:.2e} H = {r['H']:.2e} H/dt_sr = {r['H'] / r['dt_sr']:.2f}")
        print(f"wall s/τ_A: lawson {p[SR]['median']:.1f} [{p[SR]['min']:.1f}, {p[SR]['max']:.1f}], "
              f"multirate {p[MR]['median']:.1f} [{p[MR]['min']:.1f}, {p[MR]['max']:.1f}]; "
              f"speed-up {r['speedup_wall']:.2f}")
        print(f"ms per step: lawson {np.median(sr_ms):.1f}, multirate {np.median(mr_ms):.1f}, "
              f"lawson at M = 0 {np.median([x['wall'] / x['steps'] * 1e3 for x in el]):.1f}")
        u = r["updates_per_tau"]
        print(f"slice updates/τ_A: lawson {u[SR]:.0f}, multirate {u[MR]:.0f}, ratio {r['speedup_updates']:.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["spinup", "frozen", "radii", "run", "accuracy", "cost", "report"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--N", type=int, default=32)
    ap.add_argument("--M", type=int, default=16)
    ap.add_argument("--t", type=float, default=6.0, help="simulated time (τ_A)")
    ap.add_argument("--start", help="spinup: state file to pad to N, M")
    ap.add_argument("--name", help="spinup: output state file")
    ap.add_argument("--state", help="run, accuracy, cost: state file")
    ap.add_argument("--states", nargs="*", help="radii: state files")
    ap.add_argument("--scheme", default=MR, choices=[SR, MR])
    ap.add_argument("--decaying", action="store_true")
    ap.add_argument("--ideal", action="store_true", help="η = ν = 0")
    ap.add_argument("--rule", default="courant", choices=["courant", "exact"])
    ap.add_argument("--recal", type=float, default=0.25, help="run --rule exact: τ_A between dt updates")
    ap.add_argument("--cfl", type=float, default=CFL)
    ap.add_argument("--substep-cfl", type=float, default=CFL)
    ap.add_argument("--chunk", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=8, help="accuracy: macro steps")
    ap.add_argument("--ref-factor", type=int, default=64)
    ap.add_argument("--window", type=float, default=0.05, help="cost: simulated time per rep")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    dict(spinup=cmd_spinup, frozen=cmd_frozen, radii=cmd_radii, run=cmd_run,
         accuracy=cmd_accuracy, cost=cmd_cost, report=cmd_report)[a.command](a)


if __name__ == "__main__":
    main()
