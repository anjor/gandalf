#!/usr/bin/env python
"""
Float32 state against float64 in a forced, P1-like KRMHD run (Issue #191).

Question: can a budget-closure run (krmhd-research P1: Lawson-RK4, float64,
per-m free-energy budgets closed to ~1e-4 of the injection rate) keep its state
in float32 and evaluate and sum its budgets in float64?

Setup: 32³, M = 16, Λ = √5, β_i = 1, zero closure, lawson_rk4, η = 200 (r = 2),
ν = 30 (n = 6). White-noise forcing of Φ and of g_0 on k⊥L/2π ∈ [1, 2], |n_z| = 1,
at ε_A = ε_g = 0.3 (P1's forced_beta1 physics at P1's 32³ test parameters).

`spinup` runs float64 from rest and saves the state and the fixed dt of the window
(0.8 × the smallest exact Lawson-RK4 limit, compute_cfl_timestep(mode="exact"),
seen over the last τ_A). `window` then runs four variants in lockstep from that
state, all under jax_enable_x64, at the fixed dt:

  f64          complex128 state, forcing seed 0 (the reference)
  f32          complex64 state, seed 0: the same forcing draws, rounded to float32
  f64-rounded  complex128 state from the float32-rounded initial state, seed 0:
               chaos amplifying a one-time float32-sized perturbation
  f64-seed1    complex128 state, seed 1: the statistical yardstick

Every step the accounting hook (runloop.Accounting) records P1's exact budget
terms (krmhd-research h1lib.make_accounting), evaluated in float64 from the fields
(upcast for f32): the injection, the hyper-resistive and hyper-collisional losses
per m, the change of E_A and of each W_m = p_m<g_m²> in the ideal part of the step,
and the field-aligned Hermite flux Π_{m+1/2} = 2√β_i p_m T_{m,m+1}<g_m ∇∥g_{m+1}>,
∇∥ = ∂z + {Ψ, ·}, at the start and the end of the ideal step (trapezoid rule). For
f32 it also records the same terms evaluated in float32 arithmetic and, every
TWIN_EVERY steps, the float64 step of the same input. Sums are formed on the host,
so the three error sources of a float32 run separate:

  (a) state rounding: the float32 step against the float64 step of its input;
  (b) evaluation rounding: budget terms computed in float32 against float64;
  (c) accumulator rounding: the per-step terms summed in float32 (sequential,
      Kahan) against float64.

`dtscan` measures (a) at four smaller fixed dt, and `check` runs f64 and f32 in
lockstep at a smaller fixed or adaptive dt to measure the budget change directly.
`report` prints the tables of docs/precision_float32_study.md.

Usage (from the repository root; about 20 min on 8 CPU cores):
    uv run python scripts/precision_study.py spinup --out DIR
    uv run python scripts/precision_study.py window --out DIR
    uv run python scripts/precision_study.py dtscan --out DIR
    uv run python scripts/precision_study.py check --out DIR
    uv run python scripts/precision_study.py check --out DIR --cfl 0.15
    uv run python scripts/precision_study.py report --out DIR
"""

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from krmhd.forcing import WhiteNoiseBand  # noqa: E402
from krmhd.hermite import (  # noqa: E402
    compute_streaming_eigensystem,
    compute_streaming_matrix,
    free_energy_weights,
)
from krmhd.physics import KRMHDState  # noqa: E402
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
    KRMHDFields,
    _gandalf_step_lawson_rk4_jit,
    compute_cfl_timestep,
)

N, M = 32, 16
LAMBDA, BETA = math.sqrt(5.0), 1.0
ETA, HYPER_R, NU, HYPER_N = 200.0, 2, 30.0, 6
EPS_A = EPS_G = 0.3
DT_REF = 2e-3  # amplitude reference of the spin-up forcing (WhiteNoiseBand.dt_ref)
BAND = dict(n_min=1, n_max=2, band="perp_lowkz", max_nz=1, include_nz0=False)
CHUNK = 50  # steps per advance(): the per-step records of a chunk come back together
TWIN_EVERY = 10  # f32: one float64 step of the same input every TWIN_EVERY steps
STEP = StepParams(eta=ETA, v_A=1.0, nu=NU, hyper_r=HYPER_R, hyper_n=HYPER_N, scheme="lawson_rk4")
VARIANTS = {  # name: (state dtype, forcing seed, start from the float32-rounded state)
    "f64": (jnp.complex128, 0, False),
    "f32": (jnp.complex64, 0, False),
    "f64-rounded": (jnp.complex128, 0, True),
    "f64-seed1": (jnp.complex128, 1, False),
}

GRID = SpectralGrid3D.create(Nx=N, Ny=N, Nz=N, Lx=1.0, Ly=1.0, Lz=1.0)
K1 = 2.0 * math.pi
N6 = float(N**3) ** 2
_kx, _ky, _kz = (np.asarray(a, np.float64) for a in (GRID.kx, GRID.ky, GRID.kz))
KP2 = _kx[None, None, :] ** 2 + _ky[None, :, None] ** 2  # [1, Ny, Nkx]
W = np.where(np.isin(np.arange(N // 2 + 1), (0, N // 2)), 1.0, 2.0)[None, None, :]  # rfft weights
_i = (N - 1) // 3
RK = (KP2 / (_kx[_i] ** 2 + _ky[_i] ** 2)) ** HYPER_R  # GANDALF's normalised k⊥^(2r)
_m = np.arange(M + 1)
CRATES = np.where(_m >= 2, NU * (_m / M) ** HYPER_N, 0.0)  # hyper-collision rates per m
P_M = free_energy_weights(M, LAMBDA, "zero")  # W = Σ_m p_m <g_m²>
_T = np.asarray(compute_streaming_matrix(M, LAMBDA, "zero"))
COEF = 2.0 * math.sqrt(BETA) * P_M[:M] * np.array([_T[j, j + 1] for j in range(M)])
SHELL = np.rint(np.sqrt(KP2) / K1).astype(int)  # integer k⊥ shells
N_SHELL = int(SHELL[np.asarray(GRID.dealias_mask).any(axis=0)[None]].max()) + 1

# Per-step record layouts: (name, length). "64" rows are evaluated in float64.
ROW64 = [("E", 1), ("inj_A", 1), ("loss_A", 1), ("err_A", 1), ("inj_g", 1), ("Wm", M + 1),
         ("loss_eta", M + 1), ("loss_nu", M + 1), ("dW_ideal", M + 1), ("Pi", M), ("dt", 1)]
ROW32 = ROW64[:-2]  # the same terms in float32 arithmetic, without the flux
TWIN = [("dE_A", 1), ("dW", M + 1), ("rel_z", 1), ("rel_g", 1)]


def _size(layout: List[Tuple[str, int]]) -> int:
    return sum(n for _, n in layout)


def unpack(rows: np.ndarray, layout: List[Tuple[str, int]]) -> Dict[str, np.ndarray]:
    """Split [n_steps, width] records into named columns (scalars as [n_steps])."""
    out, j = {}, 0
    for name, n in layout:
        out[name] = rows[:, j] if n == 1 else rows[:, j:j + n]
        j += n
    return out


def _consts(real) -> Dict[str, jax.Array]:
    c = lambda a: jnp.asarray(a, real)  # noqa: E731
    return dict(W=c(W), KP2=c(KP2), RK=c(RK), CR=c(CRATES), P=c(P_M), COEF=c(COEF),
                KX=c(_kx), KY=c(_ky), KZ=c(_kz), MASK=GRID.dealias_mask)


def alfven_energy(zp, zm, C):
    """E_A = (1/2)<|∇⊥Φ|² + |∇⊥Ψ|²> = (1/4) Σ w k⊥² (|z⁺|² + |z⁻|²) / N⁶."""
    return jnp.sum(0.25 * C["W"] * C["KP2"] * (jnp.abs(zp) ** 2 + jnp.abs(zm) ** 2)) / N6


def free_energy_m(g, C):
    """W_m = p_m <g_m²>, [M+1]."""
    return C["P"] * jnp.sum(C["W"][..., None] * jnp.abs(g) ** 2, axis=(0, 1, 2)) / N6


def hermite_flux(zp, zm, g, C):
    """Π_{m+1/2} = 2√β_i p_m T_{m,m+1} <g_m ∇∥g_{m+1}>, ∇∥ = ∂z + {Ψ, ·}, [M] (P1's Pi_fa)."""
    psi = 0.5 * (zp - zm)
    gm = jnp.moveaxis(g, -1, 0)  # [M+1, Nz, Ny, Nkx]
    ikx = 1j * C["KX"][None, None, :]
    iky = 1j * C["KY"][:, None]

    def real(f):
        return jnp.fft.irfftn(f, s=(N, N, N), axes=(-3, -2, -1))

    bracket = real(ikx * psi) * real(iky * gm) - real(iky * psi) * real(ikx * gm)
    gpar = 1j * C["KZ"][:, None, None] * gm + jnp.fft.rfftn(bracket, axes=(-3, -2, -1)) * C["MASK"]
    corr = jnp.sum(C["W"] * jnp.real(jnp.conj(gm[:-1]) * gpar[1:]), axis=(-3, -2, -1)) / N6
    return C["COEF"] * corr


def budget_terms(before, after, inc, dt, C, flux: bool):
    """P1's exact per-step terms (h1lib.make_accounting) in the dtype of C.

    The loop forces first (before -> forced), then steps (forced -> after); the
    step's dissipation is a multiplicative factor applied after the ideal update,
    so the pre-dissipation energies are the post-step ones plus the losses.
    """
    zp_f = before.z_plus + inc.alfven
    zm_f = before.z_minus + inc.alfven
    g_f = before.g.at[..., 0].add(inc.hermite)
    E_b, E_f, E_a = (alfven_energy(a, b, C) for a, b in
                     ((before.z_plus, before.z_minus), (zp_f, zm_f), (after.z_plus, after.z_minus)))
    W_b, W_f, W_a = (free_energy_m(g, C) for g in (before.g, g_f, after.g))
    fa = jnp.expm1(2.0 * ETA * C["RK"] * dt)  # D_η⁻² - 1
    fn = jnp.expm1(2.0 * C["CR"] * dt)  # D_ν⁻² - 1
    eA = 0.25 * C["W"] * C["KP2"] * (jnp.abs(after.z_plus) ** 2 + jnp.abs(after.z_minus) ** 2) / N6
    g2k = C["W"][..., None] * jnp.abs(after.g) ** 2 / N6
    loss_A = jnp.sum(fa * eA)
    loss_eta = C["P"] * jnp.sum(fa[..., None] * g2k, axis=(0, 1, 2)) * (1.0 + fn)
    loss_nu = C["P"] * fn * jnp.sum(g2k, axis=(0, 1, 2))
    terms = dict(
        E=E_a, inj_A=E_f - E_b, loss_A=loss_A, err_A=E_a + loss_A - E_f, inj_g=W_f[0] - W_b[0],
        Wm=W_a, loss_eta=loss_eta, loss_nu=loss_nu, dW_ideal=W_a + loss_eta + loss_nu - W_f,
    )
    if flux:
        terms["Pi"] = 0.5 * dt * (hermite_flux(zp_f, zm_f, g_f, C)
                                  + hermite_flux(after.z_plus, after.z_minus, after.g, C))
        terms["dt"] = dt
    return terms


def _pack(terms, layout, real):
    return jnp.concatenate([jnp.ravel(jnp.asarray(terms[name], real)) for name, _ in layout])


def _upcast(f: KRMHDFields) -> KRMHDFields:
    return KRMHDFields(*(x.astype(jnp.complex128) for x in (f.z_plus, f.z_minus, f.g)), time=f.time)


def make_accounting(float32_state: bool, twin_every: int = TWIN_EVERY) -> Accounting:
    """Per-step records of one chunk: rows of ROW64 (and ROW32, TWIN for a float32 state)."""
    C64 = _consts(jnp.float64)
    C32 = _consts(jnp.float32)
    _, lam, P, P_inv = compute_streaming_eigensystem(M, LAMBDA, "zero")  # float64 under x64
    init = dict(i=jnp.int32(0), r64=jnp.zeros((CHUNK, _size(ROW64)), jnp.float64))
    if float32_state:
        init.update(r32=jnp.zeros((CHUNK, _size(ROW32)), jnp.float32),
                    twin=jnp.full((CHUNK, _size(TWIN)), jnp.nan, jnp.float64))

    def twin(forced, after, dt):
        """The float64 step of the float32 step's input: the state's own rounding."""
        exact = _gandalf_step_lawson_rk4_jit(
            _upcast(forced), dt.astype(jnp.float64), GRID.kx, GRID.ky, GRID.kz, GRID.dealias_mask,
            ETA, 1.0, BETA, NU, LAMBDA, M, N, N, N, HYPER_R, HYPER_N,
            streaming_eigenvalues=lam, streaming_P_T=P.T, streaming_P_inv_T=P_inv.T, closure="zero",
        )
        a = _upcast(after)

        def rel(x, y, weight):
            return jnp.sqrt(jnp.sum(weight * jnp.abs(x - y) ** 2) / jnp.sum(weight * jnp.abs(y) ** 2))

        wz = C64["W"] * C64["KP2"]
        wg = C64["W"][..., None] * C64["P"]
        return jnp.concatenate([
            jnp.atleast_1d(alfven_energy(a.z_plus, a.z_minus, C64)
                           - alfven_energy(exact.z_plus, exact.z_minus, C64)),
            free_energy_m(a.g, C64) - free_energy_m(exact.g, C64),
            jnp.atleast_1d(rel(jnp.stack([a.z_plus, a.z_minus]), jnp.stack([exact.z_plus, exact.z_minus]), wz)),
            jnp.atleast_1d(rel(a.g, exact.g, wg)),
        ])

    def update(acc, before, after, inc, dt):
        i = acc["i"]
        acc = dict(acc, i=i + 1)
        acc["r64"] = acc["r64"].at[i].set(_pack(
            budget_terms(_upcast(before), _upcast(after),
                         inc._replace(alfven=inc.alfven.astype(jnp.complex128),
                                      hermite=inc.hermite.astype(jnp.complex128)),
                         dt.astype(jnp.float64), C64, flux=True),
            ROW64, jnp.float64))
        if float32_state:
            acc["r32"] = acc["r32"].at[i].set(_pack(
                budget_terms(before, after, inc, dt, C32, flux=False), ROW32, jnp.float32))
            forced = KRMHDFields(before.z_plus + inc.alfven, before.z_minus + inc.alfven,
                                 before.g.at[..., 0].add(inc.hermite), before.time)
            row = jax.lax.cond(i % twin_every == 0, lambda: twin(forced, after, dt),
                               lambda: jnp.full(_size(TWIN), jnp.nan, jnp.float64))
            acc["twin"] = acc["twin"].at[i].set(row)
        return acc

    return Accounting(init=init, update=update)


def forcing(dt_ref) -> Forcing:
    """Φ and g_0 forcing at ε_A, ε_g: amplitude dt_ref √(ε/c), c the expected energy per A²/dt."""
    key = jax.random.PRNGKey(12345)
    unit = WhiteNoiseBand(amplitude=1.0, **BAND)
    mask = np.abs(np.asarray(unit.draw(GRID, 1.0, key)[0])) > 0
    v = np.where(np.arange(N // 2 + 1) == 0, 1.0, 2.0)[None, None, :]  # E|ξ|² after the kx=0 projection
    c_A = float(np.sum(mask * 0.5 * W * v * KP2)) / N6
    c_g = float(P_M[0] * np.sum(mask * W * v)) / N6
    return Forcing(
        alfven=WhiteNoiseBand(amplitude=dt_ref * math.sqrt(EPS_A / c_A), dt_ref=dt_ref, **BAND),
        hermite=WhiteNoiseBand(amplitude=dt_ref * math.sqrt(EPS_G / c_g), dt_ref=dt_ref, **BAND),
        moments=(0,),
    )


def make_state(zp, zm, g, dtype) -> KRMHDState:
    """A fresh state (DeviceLoop donates the arrays it is given)."""
    return KRMHDState(z_plus=jnp.array(zp, dtype), z_minus=jnp.array(zm, dtype), g=jnp.array(g, dtype),
                      M=M, beta_i=BETA, v_th=1.0, nu=NU, Lambda=LAMBDA, time=0.0, grid=GRID)


def host_fields(state: KRMHDState) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return tuple(np.asarray(x, np.complex128) for x in (state.z_plus, state.z_minus, state.g))


def spectra(zp, zm, g) -> Tuple[np.ndarray, np.ndarray]:
    """E_A(k⊥) and W(k⊥) = Σ_m p_m <g_m²> in integer k⊥ shells."""
    eA = 0.25 * W * KP2 * (np.abs(zp) ** 2 + np.abs(zm) ** 2) / N6
    wg = W * np.einsum("zyxm,m->zyx", np.abs(g) ** 2, P_M) / N6
    shells = np.broadcast_to(SHELL, eA.shape).ravel()
    return (np.bincount(shells, eA.ravel(), N_SHELL + 1)[:N_SHELL],
            np.bincount(shells, wg.ravel(), N_SHELL + 1)[:N_SHELL])


def spinup(out: Path, t_end: float) -> None:
    """Float64 from rest to t_end; save the state and the fixed dt of the window."""
    zeros = np.zeros((N, N, N // 2 + 1), np.complex128)
    state = make_state(zeros, zeros, np.zeros(zeros.shape + (M + 1,)), jnp.complex128)
    config = RunConfig(step=STEP, dt=CflDt(cfl_safety=0.3, every=10), forcing=forcing(DT_REF))
    loop = DeviceLoop(state, config, key=jax.random.PRNGKey(2026))
    history, limits, wall0 = [], [], time.time()
    while True:
        loop.advance(CHUNK)
        st = loop.state
        zp, zm, g = host_fields(st)
        E = float(alfven_energy(zp, zm, _consts(jnp.float64)))
        Wt = float(np.sum(free_energy_m(g, _consts(jnp.float64))))
        if not np.isfinite(E + Wt):
            raise RuntimeError(f"non-finite energy at t = {st.time}")
        history.append((st.time, E, Wt, float(loop.dt)))
        if st.time > t_end - 1.0:
            limits.append(compute_cfl_timestep(st, 1.0, cfl_safety=1.0, mode="exact"))
        if len(history) % 20 == 0:
            print(f"t = {st.time:6.3f}  E_A = {E:.4f}  W = {Wt:.4f}  dt = {float(loop.dt):.2e}  "
                  f"wall {time.time() - wall0:.0f} s", flush=True)
        if st.time >= t_end:
            break
    dt_window = min(5e-3, 0.8 * min(limits))
    np.savez(out / "spinup.npz", zp=zp, zm=zm, g=g, history=np.array(history),
             limits=np.array(limits), dt_window=dt_window)
    print(f"saved {out / 'spinup.npz'}: t = {st.time:.3f}, dt_window = {dt_window:.3e}", flush=True)


def window(out: Path, t_window: float) -> None:
    """The four variants in lockstep from the spun-up state at the fixed dt."""
    sp = np.load(out / "spinup.npz")
    zp0, zm0, g0 = sp["zp"], sp["zm"], sp["g"]
    dt = float(sp["dt_window"])
    n_chunks = int(math.ceil(t_window / (dt * CHUNK)))
    rounded = tuple(np.asarray(a, np.complex64).astype(np.complex128) for a in (zp0, zm0, g0))
    loops = {}
    for name, (dtype, seed, from_rounded) in VARIANTS.items():
        start = rounded if from_rounded else (zp0, zm0, g0)
        config = RunConfig(step=STEP, dt=FixedDt(dt), forcing=forcing(dt),
                           accounting=make_accounting(dtype == jnp.complex64))
        loops[name] = DeviceLoop(make_state(*start, dtype), config, key=jax.random.PRNGKey(seed))
    rec: Dict[str, Dict[str, list]] = {name: {"r64": [], "r32": [], "twin": [], "EAk": [], "Wk": [],
                                              "div_z": [], "div_g": [], "wall": []} for name in loops}
    times = []
    wz = W * KP2
    wg = W[..., None] * P_M
    print(f"window: dt = {dt:.3e}, {n_chunks} chunks of {CHUNK} steps, t = {n_chunks * CHUNK * dt:.2f}",
          flush=True)
    for c in range(n_chunks):
        fields = {}
        for name, loop in loops.items():
            t0 = time.time()
            acc = jax.device_get(loop.advance(CHUNK))
            rec[name]["wall"].append(time.time() - t0)
            for k in ("r64", "r32", "twin"):
                if k in acc:
                    rec[name][k].append(np.asarray(acc[k]))
            fields[name] = host_fields(loop.state)
            if not np.all(np.isfinite(acc["r64"])):
                raise RuntimeError(f"{name}: non-finite budget at chunk {c}")
        times.append(loops["f64"].state.time)
        zp_r, zm_r, g_r = fields["f64"]
        for name, (zp, zm, g) in fields.items():
            EAk, Wk = spectra(zp, zm, g)
            rec[name]["EAk"].append(EAk)
            rec[name]["Wk"].append(Wk)
            dz = np.sum(wz * (np.abs(zp - zp_r) ** 2 + np.abs(zm - zm_r) ** 2))
            rec[name]["div_z"].append(math.sqrt(dz / np.sum(wz * (np.abs(zp_r) ** 2 + np.abs(zm_r) ** 2))))
            rec[name]["div_g"].append(math.sqrt(np.sum(wg * np.abs(g - g_r) ** 2) / np.sum(wg * np.abs(g_r) ** 2)))
        if (c + 1) % 10 == 0 or c == n_chunks - 1:
            r = rec["f32"]
            walls = ", ".join(f"{n} {np.mean(rec[n]['wall'][-10:]):.1f}" for n in loops)
            print(f"t = {times[-1]:6.3f}  E_A = {r['r64'][-1][-1, 0]:.4f}  divergence f32/rounded "
                  f"{r['div_z'][-1]:.1e}/{rec['f64-rounded']['div_z'][-1]:.1e}  s/chunk {walls}", flush=True)
    C64 = _consts(jnp.float64)
    arrays = dict(t=np.array(times), dt=dt, t_spinup=float(sp["history"][-1, 0]),
                  W0=np.asarray(free_energy_m(g0, C64)), E0=float(alfven_energy(zp0, zm0, C64)))
    for name, r in rec.items():
        for k, v in r.items():
            if v:
                arrays[f"{name}/{k}"] = np.concatenate(v) if k in ("r64", "r32", "twin") else np.array(v)
    np.savez(out / "window.npz", **arrays)
    print(f"saved {out / 'window.npz'}", flush=True)


def dtscan(out: Path, factors=(1.0, 0.5, 0.25, 0.125), n_steps: int = 200) -> None:
    """State rounding per step against dt: the float32 run from the spun-up state, twin every step.

    The budget terms of a step scale with dt; the rounding of the step need not.
    """
    sp = np.load(out / "spinup.npz")
    dt_w = float(sp["dt_window"])
    result = {}
    for f in factors:
        dt = dt_w * f
        config = RunConfig(step=STEP, dt=FixedDt(dt), forcing=forcing(dt),
                           accounting=make_accounting(True, twin_every=1))
        loop = DeviceLoop(make_state(sp["zp"], sp["zm"], sp["g"], jnp.complex64), config,
                          key=jax.random.PRNGKey(0))
        tw = unpack(np.concatenate([np.asarray(jax.device_get(loop.advance(CHUNK))["twin"])
                                    for _ in range(n_steps // CHUNK)]), TWIN)
        dW = tw["dW"].sum(1)
        result[f] = dict(dt=dt, n=len(dW),
                         E_A=(float(np.mean(tw["dE_A"])) / (EPS_A * dt), float(np.std(tw["dE_A"])) / (EPS_A * dt)),
                         W=(float(np.mean(dW)) / (EPS_G * dt), float(np.std(dW)) / (EPS_G * dt)),
                         Wm_max=float(np.max(np.abs(np.mean(tw["dW"], 0)))) / (EPS_G * dt),
                         dE_A_abs=float(np.mean(tw["dE_A"])), dW_abs=float(np.mean(dW)))
        print(f"dt = {dt:.3e}: per step / (eps dt): E_A bias {result[f]['E_A'][0]:+.2e} (std {result[f]['E_A'][1]:.2e}),"
              f" W bias {result[f]['W'][0]:+.2e} (std {result[f]['W'][1]:.2e}), max_m |W_m bias| "
              f"{result[f]['Wm_max']:.2e}; absolute bias per step E_A {result[f]['dE_A_abs']:+.2e}, "
              f"W {result[f]['dW_abs']:+.2e}", flush=True)
    (out / "dtscan.json").write_text(json.dumps(result, indent=1))


def check(out: Path, t_end: float, dt_factor: float, cfl: Optional[float]) -> None:
    """f64 and f32 in lockstep from the spun-up state: the budget change at a smaller dt, measured.

    dt is fixed at dt_factor × the window's dt, or with `cfl` set, adaptive
    (CflDt(cfl, every=25), as P1 recomputes its dt every 25 steps) with the
    forcing amplitude referred to dt_factor × the window's dt.
    """
    sp = np.load(out / "spinup.npz")
    dt0 = float(sp["dt_window"]) * dt_factor
    policy = FixedDt(dt0) if cfl is None else CflDt(cfl_safety=cfl, every=25)
    label = f"fixed{dt_factor:g}" if cfl is None else f"cfl{cfl:g}"
    C64 = _consts(jnp.float64)
    loops, W0 = {}, {}
    for name, dtype in (("f64", jnp.complex128), ("f32", jnp.complex64)):
        config = RunConfig(step=STEP, dt=policy, forcing=forcing(dt0),
                           accounting=make_accounting(dtype == jnp.complex64))
        state = make_state(sp["zp"], sp["zm"], sp["g"], dtype)
        W0[name] = np.asarray(free_energy_m(state.g.astype(jnp.complex128), C64))
        loops[name] = DeviceLoop(state, config, key=jax.random.PRNGKey(0))
    rows: Dict[str, list] = {"f64": [], "f32": [], "twin": []}
    while loops["f64"].state.time < t_end:
        for name, loop in loops.items():
            acc = jax.device_get(loop.advance(CHUNK))
            rows[name].append(np.asarray(acc["r64"]))
            if "twin" in acc:
                rows["twin"].append(np.asarray(acc["twin"]))
    r = {n: unpack(np.concatenate(rows[n]), ROW64) for n in ("f64", "f32")}
    b = {n: budgets(r[n], W0[n]) for n in r}
    tw = unpack(np.concatenate(rows["twin"]), TWIN)
    ok = np.isfinite(tw["dE_A"])
    dts = r["f32"]["dt"]
    result = dict(
        label=label, n_steps=len(dts), dt_min=float(dts.min()), dt_max=float(dts.max()), dt_mean=float(dts.mean()),
        n_dt_values=len(np.unique(dts)),
        delta_err_A=float(b["f32"]["err_A"] - b["f64"]["err_A"]),
        delta_err_W=float(b["f32"]["err_W"] - b["f64"]["err_W"]),
        delta_res_m=float(np.max(np.abs(b["f32"]["res_m"] - b["f64"]["res_m"]))),
        twin_E_A=float(np.sum(tw["dE_A"][ok]) / (EPS_A * np.sum(dts[ok]))),
        twin_W=float(np.sum(tw["dW"][ok]) / (EPS_G * np.sum(dts[ok]))),
        err_A_f64=float(b["f64"]["err_A"]), err_W_f64=float(b["f64"]["err_W"]),
    )
    print(json.dumps(result, indent=1), flush=True)
    (out / f"check_{label}.json").write_text(json.dumps(result, indent=1))


# ------------------------------------------------------------------------------------------------
# report
# ------------------------------------------------------------------------------------------------
def kahan(x: np.ndarray) -> np.ndarray:
    """Compensated float32 sum over axis 0, one step at a time."""
    s = np.zeros(x.shape[1:], np.float32)
    c = np.zeros_like(s)
    for v in x.astype(np.float32):
        y = v - c
        t = s + y
        c = (t - s) - y
        s = t
    return s


def sequential32(x: np.ndarray) -> np.ndarray:
    """Plain float32 running sum over axis 0, as a float32 accumulator in the loop carry."""
    return np.cumsum(x.astype(np.float32), axis=0, dtype=np.float32)[-1]


def per_m_residual(Pi_sum: np.ndarray, inj_g: float, loss_m: np.ndarray, dWm: np.ndarray) -> np.ndarray:
    """P1's per-m budget (Π_{m-1/2} - Π_{m+1/2}) + inj_m - D_m - dW_m/dt over ε_g; all terms times T."""
    Pe = np.concatenate([[0.0], Pi_sum, [0.0]])
    inj_m = np.zeros(M + 1)
    inj_m[0] = inj_g
    return ((Pe[:-1] - Pe[1:]) + inj_m - loss_m - dWm) / inj_g


def chunked32(x: np.ndarray) -> np.ndarray:
    """float32 running sum within each advance() chunk, float64 across chunks (the DeviceLoop pattern)."""
    n = len(x) // CHUNK * CHUNK
    parts = np.cumsum(x[:n].astype(np.float32).reshape((-1, CHUNK) + x.shape[1:]), axis=1, dtype=np.float32)
    return np.sum(parts[:, -1].astype(np.float64), axis=0) + np.sum(x[n:], axis=0)


def budgets(r: Dict[str, np.ndarray], W0: np.ndarray) -> Dict[str, object]:
    """Window budgets of one variant from its float64 per-step records, summed in float64."""
    S = {k: np.sum(v, axis=0) for k, v in r.items()}
    loss_m = S["loss_eta"] + S["loss_nu"]
    dWm = r["Wm"][-1] - W0
    return dict(
        err_A=S["err_A"] / S["inj_A"],
        err_W=np.sum(S["dW_ideal"]) / S["inj_g"],
        inj_A=S["inj_A"], inj_g=S["inj_g"], loss_A=S["loss_A"], loss_W=np.sum(loss_m),
        res_m=per_m_residual(S["Pi"], S["inj_g"], loss_m, dWm),
        # the accounting identity: telescopes to zero up to the summation's rounding
        identity_W=(np.sum(dWm) - (S["inj_g"] - np.sum(loss_m) + np.sum(S["dW_ideal"]))) / S["inj_g"],
    )


def p1_cadence(r: Dict[str, np.ndarray], W0: np.ndarray, every: int = 20) -> float:
    """max_m |per-m residual| with Π sampled every `every` steps, as P1 samples it (forced_beta1: 20).

    Each sample (the step's trapezoid flux) stands for the interval that ends at it,
    weighted by its length (h1lib.time_average).
    """
    n = len(r["dt"]) // every * every
    flux = r["Pi"][:n] / r["dt"][:n, None]
    idx = np.arange(every - 1, n, every)
    Pi_sum = np.sum(flux[idx] * np.add.reduceat(r["dt"][:n], idx - (every - 1))[:, None], 0)
    loss_m = np.sum(r["loss_eta"][:n] + r["loss_nu"][:n], 0)
    res = per_m_residual(Pi_sum, float(np.sum(r["inj_g"][:n])), loss_m, r["Wm"][n - 1] - W0)
    return float(np.max(np.abs(res)))


def report(out: Path) -> None:
    z = np.load(out / "window.npz")
    W0 = z["W0"]
    names = list(VARIANTS)
    r64 = {n: unpack(z[f"{n}/r64"], ROW64) for n in names}
    n_steps = len(r64["f64"]["dt"])
    T = float(np.sum(r64["f64"]["dt"]))
    eps_A = np.sum(r64["f64"]["inj_A"]) / T
    eps_g = np.sum(r64["f64"]["inj_g"]) / T
    summary: Dict[str, object] = dict(n_steps=n_steps, T=T, dt=float(z["dt"]), t_spinup=float(z["t_spinup"]),
                                      eps_A=eps_A, eps_g=eps_g)
    print(f"window: {n_steps} steps of dt = {float(z['dt']):.3e}, T = {T:.2f} tau_A after a {float(z['t_spinup']):.1f}"
          f" tau_A spin-up; measured eps_A = {eps_A:.3f}, eps_g = {eps_g:.3f}\n")

    # 1. P1's closure metrics, float64 evaluation and float64 sums
    print("| variant | ideal-step error E_A / inj | ideal-step error W / inj | max_m per-m residual / eps_g "
          "| per-m residual at P1's cadence (every 20 steps) / eps_g | inj = diss (A, W) |")
    print("|---|---|---|---|---|---|")
    for n in names:
        b = budgets(r64[n], W0)
        p1 = p1_cadence(r64[n], W0)
        summary[n] = dict(err_A=b["err_A"], err_W=b["err_W"], res_max=float(np.max(np.abs(b["res_m"]))),
                          res_m=b["res_m"].tolist(), res_p1=p1, identity_W=b["identity_W"])
        print(f"| {n} | {b['err_A']:+.2e} | {b['err_W']:+.2e} | {np.max(np.abs(b['res_m'])):.2e} | {p1:.2e} | "
              f"{(b['inj_A'] - b['loss_A']) / b['inj_A']:+.3f}, {(b['inj_g'] - b['loss_W']) / b['inj_g']:+.3f} |")
    print()
    print("change against f64 (the trajectories stay within 5e-5 of each other, so this is the precision effect):")
    print("| variant | ideal-step error E_A / inj | ideal-step error W / inj | max_m per-m residual / eps_g |")
    print("|---|---|---|---|")
    for n in ("f32", "f64-rounded"):
        d = {k: np.asarray(summary[n][k]) - np.asarray(summary["f64"][k]) for k in ("err_A", "err_W", "res_m")}
        summary[n]["delta"] = dict(err_A=float(d["err_A"]), err_W=float(d["err_W"]),
                                   res_m=float(np.max(np.abs(d["res_m"]))))
        print(f"| {n} | {float(d['err_A']):+.1e} | {float(d['err_W']):+.1e} | {np.max(np.abs(d['res_m'])):.1e} |")
    print()
    print("per-m residual / eps_g, f64 and f32 (m = 0..M):")
    for n in ("f64", "f32"):
        print(f"  {n}: " + " ".join(f"{x:+.1e}" for x in summary[n]["res_m"]))
    print()

    # 2. error sources of the float32 run
    f32_64 = r64["f32"]
    f32_32 = unpack(z["f32/r32"].astype(np.float64), ROW32)
    inj = dict(A=np.sum(f32_64["inj_A"]), g=np.sum(f32_64["inj_g"]))
    tw = unpack(z["f32/twin"], TWIN)
    ok = np.isfinite(tw["dE_A"])
    per_step_inj_A = eps_A * float(z["dt"])
    per_step_inj_g = eps_g * float(z["dt"])
    dW_tot = tw["dW"][ok].sum(1)
    src = {}
    src["a_state_A"] = (np.mean(tw["dE_A"][ok]) / per_step_inj_A, np.std(tw["dE_A"][ok]) / per_step_inj_A)
    src["a_state_W"] = (np.mean(dW_tot) / per_step_inj_g, np.std(dW_tot) / per_step_inj_g)
    m_worst = int(np.argmax(np.abs(np.mean(tw["dW"][ok], 0))))
    src["a_state_Wm"] = (np.mean(tw["dW"][ok, m_worst]) / per_step_inj_g,
                         np.std(tw["dW"][ok, m_worst]) / per_step_inj_g, m_worst)
    src["a_rel"] = (float(np.median(tw["rel_z"][ok])), float(np.median(tw["rel_g"][ok])))
    print(f"(a) state rounding, {int(ok.sum())} twin steps (float64 step of the float32 step's input), "
          f"per step, in units of the per-step injection eps*dt:")
    print("| quantity | mean (bias) | std | window, bias + noise/sqrt(n) |")
    print("|---|---|---|---|")
    for label, (mu, sd, *rest) in (("E_A", src["a_state_A"]), ("W", src["a_state_W"]),
                                   (f"W_m, worst m = {m_worst}", src["a_state_Wm"])):
        print(f"| {label} | {mu:+.2e} | {sd:.2e} | {abs(mu) + sd / math.sqrt(n_steps):.2e} |")
    print(f"median relative state error per step (energy norms): z {src['a_rel'][0]:.1e}, g {src['a_rel'][1]:.1e}\n")

    print("(b) evaluation and (c) accumulation, float32 run, window totals relative to the window injection:")
    print("| term | (b) float32 evaluation, float64 sum | (c) float32 sequential sum over the window "
          f"| (c) float32 Kahan sum over the window | (c) float32 within chunks of {CHUNK} steps, float64 across "
          "| (b)+(c) float32 evaluation, float32 sequential sum |")
    print("|---|---|---|---|---|---|")
    rows = {}
    for label, key, ref in (("ideal-step error E_A", "err_A", "A"), ("ideal-step error W", "dW_ideal", "g"),
                            ("injection E_A", "inj_A", "A"), ("injection W", "inj_g", "g"),
                            ("loss E_A", "loss_A", "A"), ("loss W (eta + nu)", None, "g")):
        if key is None:
            x64 = f32_64["loss_eta"] + f32_64["loss_nu"]
            x32 = f32_32["loss_eta"] + f32_32["loss_nu"]
        else:
            x64, x32 = f32_64[key], f32_32[key]
        exact = np.sum(x64, axis=0)
        vals = (np.sum(x32, axis=0) - exact, sequential32(x64) - exact, kahan(x64) - exact,
                chunked32(x64) - exact, sequential32(x32) - exact)
        vals = [float(np.sum(v)) / inj[ref] for v in vals]
        rows[label] = vals
        print(f"| {label} | " + " | ".join(f"{v:+.1e}" for v in vals) + " |")
    src["bc"] = rows
    # per-m worst case of (b) and (c) for the ideal-step change
    x64, x32 = f32_64["dW_ideal"], f32_32["dW_ideal"]
    exact = np.sum(x64, axis=0)
    per_m = dict(b=np.max(np.abs(np.sum(x32, 0) - exact)) / inj["g"],
                 c=np.max(np.abs(sequential32(x64) - exact)) / inj["g"],
                 bc=np.max(np.abs(sequential32(x32) - exact)) / inj["g"])
    src["per_m"] = per_m
    print(f"per-m ideal-step change, max over m / window injection: (b) {per_m['b']:.1e}, (c) {per_m['c']:.1e},"
          f" (b)+(c) {per_m['bc']:.1e}\n")
    summary["sources"] = src

    # 3. trajectories
    t = z["t"]
    print("| t - t0 (tau_A) | f32 vs f64: z, g | f64-rounded vs f64: z, g | f64-seed1 vs f64: z, g |")
    print("|---|---|---|---|")
    for target in (0.1, 0.25, 0.5, 1, 2, 3, 4, 6, 8):
        if target > t[-1] + 1e-9:
            break
        j = int(np.argmin(np.abs(t - target)))
        cells = [f"{z[f'{n}/div_z'][j]:.1e}, {z[f'{n}/div_g'][j]:.1e}" for n in ("f32", "f64-rounded", "f64-seed1")]
        print(f"| {t[j]:.2f} | " + " | ".join(cells) + " |")
    print()

    # 4. statistics over the second half of the window (trajectories decorrelated)
    half = slice(n_steps // 2, None)
    print("time averages over the second half of the window:")
    print("| variant | <E_A> | <W> | D_A | D_W | W_m at m = 0, 4, 8, 12 | Pi at m+1/2 = 0.5, 4.5, 8.5 | "
          "E_A(k⊥) slope n⊥ = 2..6 | W(k⊥) slope |")
    print("|---|---|---|---|---|---|---|---|---|")
    stats = {}
    for n in names:
        r = r64[n]
        dtw = r["dt"][half]
        Th = float(np.sum(dtw))
        Wm = np.sum(r["Wm"][half] * dtw[:, None], 0) / Th
        Pi = np.sum(r["Pi"][half], 0) / Th
        jh = len(t) // 2
        EAk = np.mean(z[f"{n}/EAk"][jh:], 0)
        Wk = np.mean(z[f"{n}/Wk"][jh:], 0)
        sl = slice(2, 7)
        nn = np.log(np.arange(N_SHELL)[sl])
        s = dict(E=float(np.sum(r["E"][half] * dtw) / Th), W=float(np.sum(Wm)),
                 D_A=float(np.sum(r["loss_A"][half]) / Th),
                 D_W=float((np.sum(r["loss_eta"][half]) + np.sum(r["loss_nu"][half])) / Th),
                 Wm=Wm[[0, 4, 8, 12]].tolist(), Pi=Pi[[0, 4, 8]].tolist(),
                 slope_A=float(np.polyfit(nn, np.log(EAk[sl]), 1)[0]),
                 slope_W=float(np.polyfit(nn, np.log(Wk[sl]), 1)[0]))
        stats[n] = s
        print(f"| {n} | {s['E']:.4f} | {s['W']:.4f} | {s['D_A']:.4f} | {s['D_W']:.4f} | "
              + ", ".join(f"{x:.4f}" for x in s["Wm"]) + " | " + ", ".join(f"{x:.4f}" for x in s["Pi"])
              + f" | {s['slope_A']:.2f} | {s['slope_W']:.2f} |")
    summary["stats"] = stats
    rel = lambda n, k: abs(stats[n][k] - stats["f64"][k]) / abs(stats["f64"][k])  # noqa: E731
    print("\nrelative difference from f64: " + "; ".join(
        f"{k}: f32 {rel('f32', k):.1e}, rounded {rel('f64-rounded', k):.1e}, seed1 {rel('f64-seed1', k):.1e}"
        for k in ("E", "W", "D_A", "D_W")))
    scan = out / "dtscan.json"
    if scan.exists():
        print("\nstate rounding against dt (dtscan): bias per step in units of eps*dt, +- standard error of the mean")
        print("| dt | E_A | W | max over m of W_m | E_A bias per step / E_A |")
        print("|---|---|---|---|---|")
        for f, d in sorted(json.loads(scan.read_text()).items(), key=lambda kv: -float(kv[0])):
            se = 1.0 / math.sqrt(d["n"])
            print(f"| {d['dt']:.2e} | {d['E_A'][0]:+.1e} ± {d['E_A'][1] * se:.0e} | {d['W'][0]:+.1e} ± "
                  f"{d['W'][1] * se:.0e} | {d['Wm_max']:.1e} | {d['dE_A_abs'] / stats['f64']['E']:+.1e} |")
    checks = sorted(out.glob("check_*.json"))
    if checks:
        print("\nf32 against f64 at a smaller dt (check), changes of the budgets / injection:")
        print("| run | steps | dt (mean, range) | distinct dt values | E_A | W | max_m per-m residual "
              "| twin prediction E_A, W |")
        print("|---|---|---|---|---|---|---|---|")
        for f in checks:
            c = json.loads(f.read_text())
            print(f"| {c['label']} | {c['n_steps']} | {c['dt_mean']:.2e} ({c['dt_min']:.2e}..{c['dt_max']:.2e}) | "
                  f"{c['n_dt_values']} | {c['delta_err_A']:+.1e} | {c['delta_err_W']:+.1e} | {c['delta_res_m']:.1e} | "
                  f"{c['twin_E_A']:+.1e}, {c['twin_W']:+.1e} |")
    walls = {n: float(np.median(z[f"{n}/wall"])) / CHUNK for n in names}
    summary["s_per_step_with_hooks"] = walls
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(f"\nwrote {out / 'summary.json'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=("spinup", "window", "dtscan", "check", "report"))
    ap.add_argument("--out", type=Path, required=True, help="data folder")
    ap.add_argument("--t-spinup", type=float, default=6.0, help="spin-up length in tau_A")
    ap.add_argument("--t-window", type=float, default=6.0, help="window length in tau_A")
    ap.add_argument("--t-check", type=float, default=1.0, help="check: run length in tau_A")
    ap.add_argument("--dt-factor", type=float, default=0.125, help="check: dt (or dt_ref) / the window's dt")
    ap.add_argument("--cfl", type=float, default=None, help="check: adaptive CflDt(cfl, every=25) instead")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.command == "spinup":
        spinup(args.out, args.t_spinup)
    elif args.command == "window":
        window(args.out, args.t_window)
    elif args.command == "dtscan":
        dtscan(args.out)
    elif args.command == "check":
        check(args.out, args.t_check, args.dt_factor, args.cfl)
    else:
        report(args.out)


if __name__ == "__main__":
    main()
