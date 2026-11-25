#!/usr/bin/env python
"""
Stability of scheme="imex_ars343" with parallel streaming (Issue #200).

The frozen shear flow of Issue #200: z± = (1 ± 1/2)Φ(x) with
Φ = cos(2πx/L) + ½ sin(4πx/L), so every Elsasser bracket vanishes and z± stay
fixed. Grid 16×16×8, L = 2π, M = 8, Λ = √5, β_i = 1, zero closure, float64.
ρ is the exact slice advection radius (timestepping.slice_advection_radius),
and y = dt·ρ/2√2 is the step as a fraction of the RK4 limit.

With z± fixed, one gandalf_step is a real-linear map of g. The flow depends
on x only, so the map keeps each pair of (k_y, k_z) planes {(k_y, k_z),
(-k_y, -k_z)}. The script builds the map on each pair by stepping basis
vectors (all pairs at once, 198 steps in all) and reports its spectral
radius: the amplification per step of the most amplified mode. A value above
1 means g grows without bound. The (0, 0) plane is left out: it is neither
advected nor streamed.

`table` prints the radius for each scheme, step and dissipation setting.
`threshold` bisects, for imex_ars343 and imex_rk222, the smallest ν = η
(r = n = 2) at which the radius is at most 1. --amplitude scales Φ, and
with it ρ, at fixed k_z.

`scalar` gives the worst case over the streaming rate, for one mode with
advection rate ω (explicit) and streaming rate σ and damping rate κ
(implicit): Y = ω·dt, s = σ·dt, d = κ·dt. It prints max over s of the IMEX
amplification |R(iY, is)|, the smallest d with max_s |R(iY, is - d)| ≤ 1
(hyper-collisions, which sit in the implicit operator), and ln max_s |R|,
the smallest e of a post-step factor exp(-e) (hyper-resistivity).

Run from the repository root:
    uv run python scripts/ars343_stability_study.py table
    uv run python scripts/ars343_stability_study.py threshold [--amplitude 4]
    uv run python scripts/ars343_stability_study.py scalar
docs/numerical_methods.md reports the numbers.
"""

import argparse
import math

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from krmhd import timestepping as ts  # noqa: E402
from krmhd.physics import KRMHDState  # noqa: E402
from krmhd.spectral import SpectralGrid3D  # noqa: E402
from krmhd.timestepping import (  # noqa: E402
    RK4_IMAGINARY_AXIS_LIMIT,
    _streaming_eigenvalue_extremes,
    compute_cfl_timestep,
    gandalf_step,
    slice_advection_radius,
)

N, NZ, M = 16, 8, 8
K, KZ = (N - 1) // 3, (NZ - 1) // 3  # largest dealiased mode numbers
SCHEMES = ("imex_ars343", "imex_rk222", "lawson_rk4")
# (label, eta, nu, hyper_r, hyper_n); the docs recommend η ≈ ν ≈ 0.5-1 with r = n = 2
SETTINGS = (
    ("ideal", 0.0, 0.0, 1, 1),
    ("ν = η = 0.5, r = n = 2", 0.5, 0.5, 2, 2),
    ("ν = η = 1, r = n = 2", 1.0, 1.0, 2, 2),
)

# Plane pairs and, per pair, the complex coordinates (sign, kx, m): sign +1 is the
# (k_y, k_z) plane with kx = 0..K, sign -1 the (-k_y, -k_z) plane with kx = 1..K
# (its kx = 0 entries are the conjugates of the + plane's).
PAIRS = [(ky, kz) for ky in range(1, K + 1) for kz in range(-KZ, KZ + 1)] + [
    (0, kz) for kz in range(1, KZ + 1)
]
COORDS = [(1, kx, m) for kx in range(K + 1) for m in range(M + 1)] + [
    (-1, kx, m) for kx in range(1, K + 1) for m in range(M + 1)
]


def _index(sign, kx, m):
    """Index arrays into g of coordinate (sign, kx, m) for every pair."""
    iz = np.array([(sign * kz) % NZ for _, kz in PAIRS])
    iy = np.array([(sign * ky) % N for ky, _ in PAIRS])
    return iz, iy, np.full(len(PAIRS), kx), np.full(len(PAIRS), m)


IDX = [_index(*c) for c in COORDS]
IDX_ALL = tuple(np.stack([ix[a] for ix in IDX], axis=1) for a in range(4))  # [pair, coord]


def frozen_shear_state(amplitude: float = 1.0) -> KRMHDState:
    grid = SpectralGrid3D.create(Nx=N, Ny=N, Nz=NZ)
    x = np.arange(N) / N
    phi = amplitude * jnp.fft.rfftn(
        np.broadcast_to(np.cos(2 * np.pi * x) + 0.5 * np.sin(4 * np.pi * x), (NZ, N, N))
    )
    return KRMHDState(
        z_plus=1.5 * phi, z_minus=0.5 * phi, g=jnp.zeros(phi.shape + (M + 1,), jnp.complex128),
        M=M, beta_i=1.0, v_th=1.0, nu=0.0, Lambda=math.sqrt(5.0), time=0.0, grid=grid,
    )


def radius(state: KRMHDState) -> float:
    lo, hi, _ = _streaming_eigenvalue_extremes(state.M, state.Lambda, "zero")
    grid = state.grid
    rad = slice_advection_radius(
        state.z_plus, state.z_minus, math.sqrt(state.beta_i) * jnp.array([lo, hi]),
        grid.kx, grid.ky, grid.dealias_mask, grid.Nz, grid.Ny, grid.Nx, tol=1e-10,
    )
    return float(rad.rho.max())


def step_maps(state0: KRMHDState, scheme: str, dt: float, eta: float, nu: float,
              hyper_r: int, hyper_n: int) -> np.ndarray:
    """The step map g -> g' with z± fixed, one real matrix per plane pair.

    Row and column 2c + p are the real (p = 0) and imaginary (p = 1) parts of
    coordinate COORDS[c].
    """
    n = 2 * len(COORDS)
    maps = np.zeros((len(PAIRS), n, n))
    for b, ((sign, kx, m), idx) in enumerate(zip(COORDS, IDX)):
        for part, val in enumerate((1.0, 1j)):
            g = np.zeros(state0.g.shape, complex)
            g[idx] = val
            if kx == 0:
                g[(-idx[0]) % NZ, (-idx[1]) % N, 0, m] = np.conj(val)
            out = gandalf_step(
                state0.model_copy(update=dict(g=jnp.asarray(g))), dt, eta=eta, v_A=1.0,
                nu=nu, hyper_r=hyper_r, hyper_n=hyper_n, scheme=scheme,
            )
            column = np.asarray(out.g)[IDX_ALL]  # [pair, coord]
            maps[:, 0::2, 2 * b + part] = column.real
            maps[:, 1::2, 2 * b + part] = column.imag
    return maps


def step_radius(*args) -> float:
    """Spectral radius of the step map: the amplification per step of the most amplified mode."""
    return float(max(np.abs(np.linalg.eigvals(a)).max() for a in step_maps(*args)))


def cmd_table(a) -> None:
    state0 = frozen_shear_state(a.amplitude)
    rho = radius(state0)
    y_courant = compute_cfl_timestep(state0, v_A=1.0, cfl_safety=0.3) * rho / RK4_IMAGINARY_AXIS_LIMIT
    print(f"ρ = {rho:.4f}. compute_cfl_timestep(cfl_safety=0.3) gives y = {y_courant:.3f} "
          f"(courant) and 0.3 (exact).")
    ys = (round(y_courant, 2), 0.3, 0.5, 0.95, 1.05)
    print("| Setting | Scheme | " + " | ".join(f"y = {y:g}" for y in ys) + " |")
    print("|---|---|" + "---|" * len(ys))
    for label, eta, nu, r, n in SETTINGS:
        for scheme in SCHEMES:
            amps = [step_radius(state0, scheme, y * RK4_IMAGINARY_AXIS_LIMIT / rho, eta, nu, r, n)
                    for y in ys]
            print(f"| {label} | `{scheme}` | " + " | ".join(f"{v:.6f}" for v in amps) + " |",
                  flush=True)


def cmd_threshold(a) -> None:
    state0 = frozen_shear_state(a.amplitude)
    rho = radius(state0)
    print(f"ρ = {rho:.4f}")
    print("| Scheme | y | dt·ρ | ν = η (r = n = 2) | ν/ρ | ν·dt |")
    print("|---|---|---|---|---|---|")
    for scheme in ("imex_ars343", "imex_rk222"):
        for y in (0.3, 0.5, 0.95):
            dt = y * RK4_IMAGINARY_AXIS_LIMIT / rho
            lo, hi = 1e-3, 1e3
            for _ in range(a.bisections):
                mid = math.sqrt(lo * hi)
                if step_radius(state0, scheme, dt, mid, mid, 2, 2) <= 1.0 + 1e-12:
                    hi = mid
                else:
                    lo = mid
            print(f"| `{scheme}` | {y} | {dt * rho:.3f} | {hi:.3g} | {hi / rho:.3g} | "
                  f"{hi * dt:.3g} |", flush=True)


def _tableaus():
    """(implicit A, explicit Â, b, b̂) of both IMEX schemes, from timestepping's constants."""
    g3, g2, d2 = ts._ARS343_GAMMA, ts._IMEX_GAMMA, ts._IMEX_DELTA
    b343 = np.array([0, ts._ARS343_B1, ts._ARS343_B2, g3])
    return {
        "imex_ars343": (
            np.array([[0, 0, 0, 0], [0, g3, 0, 0], [0, (1 - g3) / 2, g3, 0], b343]),
            np.array([[0, 0, 0, 0], [g3, 0, 0, 0], [ts._ARS343_A31, ts._ARS343_A32, 0, 0],
                      [ts._ARS343_A41, ts._ARS343_A42, ts._ARS343_A43, 0]]),
            b343, b343,
        ),
        "imex_rk222": (
            np.array([[0, 0, 0], [0, g2, 0], [0, 1 - g2, g2]]),
            np.array([[0, 0, 0], [g2, 0, 0], [d2, 1 - d2, 0]]),
            np.array([0, 1 - g2, g2]), np.array([d2, 1 - d2, 0]),
        ),
    }


def cmd_scalar(a) -> None:
    tableaus = _tableaus()
    s_grid = np.concatenate([np.linspace(0.0, 20.0, 4001), np.logspace(1.3, 5, 100)])

    def max_amp(scheme, Y, d):
        A, Ah, b, bh = tableaus[scheme]
        one = np.ones(len(b))
        return max(
            abs(1 + (1j * Y * bh + (1j * s - d) * b)
                @ np.linalg.solve(np.eye(len(b)) - 1j * Y * Ah - (1j * s - d) * A, one))
            for s in s_grid
        )

    print("| y | Y = ω·dt | Scheme | max_s |R| | d (κ/ω = d/Y) | e (e/Y) |")
    print("|---|---|---|---|---|---|")
    for y in (0.19, 0.3, 0.5, 0.95):
        Y = y * RK4_IMAGINARY_AXIS_LIMIT
        for scheme in tableaus:
            r0 = max_amp(scheme, Y, 0.0)
            lo, hi = 0.0, 10.0
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                lo, hi = (lo, mid) if max_amp(scheme, Y, mid) <= 1.0 + 1e-12 else (mid, hi)
            e = math.log(r0)
            print(f"| {y} | {Y:.3f} | `{scheme}` | {r0:.4f} | {hi:.3g} ({hi / Y:.3g}) | "
                  f"{e:.3g} ({e / Y:.3g}) |", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=["table", "threshold", "scalar"])
    ap.add_argument("--bisections", type=int, default=16)
    ap.add_argument("--amplitude", type=float, default=1.0, help="scales Φ, hence ρ")
    a = ap.parse_args()
    dict(table=cmd_table, threshold=cmd_threshold, scalar=cmd_scalar)[a.command](a)


if __name__ == "__main__":
    main()
