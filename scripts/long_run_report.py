#!/usr/bin/env python3
"""
Verdict and figures for the Issue #136 runs of scripts/long_run_study.py.

For each run folder (timeseries.csv, spectra.npz, run_config.json) it prints
the pre-registered checks and writes <run name>.png in --fig-dir:

  reached  the run reached t_end with finite z±
  plateau  mean E over the last 100 tau_A within 25% of the 100 tau_A before
  zmax     max|z±| over the last 100 tau_A below 1.5x the 100 tau_A before
  falloff  time-averaged E(n_perp) over the last 100 tau_A decreases
           monotonically over n_perp in [n_edge/2, n_edge] (no bump)
  edge     e-folding time of the near-cutoff energy fraction over the last
           100 tau_A is not a growth faster than 1/(50 tau_A)
  num      the stepper's numerical energy source, mean over the last 100 tau_A,
           as a fraction of the expected injection rate

The expected injection rate is the mean of sum_k ew_k (|F+_k|^2 + |F-_k|^2)/dt
over 200 draws of the run's own forcing increments: the cross term z*F
averages to zero, so this is <eps_inj> without the sampling noise of the
per-sample injection.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def load(run: Path):
    with open(run / "timeseries.csv") as fh:
        rows = list(csv.DictReader(fh))
    ts = {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}
    spec = dict(np.load(run / "spectra.npz"))
    cfg = json.loads((run / "run_config.json").read_text())
    return ts, spec, cfg


def expected_injection(cfg, draws: int = 200) -> float:
    """<eps_inj> of the run's forcing, from its own noise generators."""
    import jax

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import long_run_study as drv

    grid, state = drv.benchmark_initial_state(cfg["resolution"], cfg["M"])
    c = drv.Cfg(cfg["forcing"], cfg["amplitude"], cfg["eta"], cfg["hyper_r"], cfg["hyper_n"],
                cfg["eta_z"], cfg["dt"], state.M, float(state.beta_i), float(state.Lambda))
    consts, _, _ = drv.make_consts(grid, c)
    zero = jax.numpy.zeros_like(state.z_plus)

    @jax.jit
    def one(key):
        zp, zm, key = drv.force(zero, zero, key, grid, c)
        return jax.numpy.sum(consts.ew * (jax.numpy.abs(zp) ** 2 + jax.numpy.abs(zm) ** 2)), key

    key = jax.random.PRNGKey(7)
    total = 0.0
    for _ in range(draws):
        e, key = one(key)
        total += float(e)
    return total / draws / cfg["dt"]


def running_mean(x, n):
    n = max(1, int(n))
    return np.convolve(x, np.ones(n) / n, mode="valid")


def checks(ts, spec, cfg, window: float = 100.0):
    t = ts["t"]
    t_last = t[-1]
    reached = t_last >= cfg["t_end"] - 1e-6 and np.isfinite(ts["E"][-1])
    w2 = t > t_last - window
    w1 = (t > t_last - 2 * window) & ~w2
    zmax = np.maximum(ts["zp_max"], ts["zm_max"])
    out = dict(t_last=t_last, reached=bool(reached))
    out["E_ratio"] = float(ts["E"][w2].mean() / ts["E"][w1].mean())
    out["E_cv_last"] = float(ts["E"][w2].std() / ts["E"][w2].mean())
    out["plateau"] = bool(0.75 <= out["E_ratio"] <= 1.25)
    out["zmax_ratio"] = float(zmax[w2].max() / zmax[w1].max())
    out["zmax_last"] = float(zmax[w2].max())
    out["zmax_ok"] = bool(out["zmax_ratio"] < 1.5)
    n_edge = cfg["n_edge"]
    st = spec["t"]
    sw = st > t_last - window
    Eavg = spec["E_nperp"][sw].mean(axis=0)
    n = np.arange(len(Eavg))
    band = (n >= n_edge // 2) & (n <= n_edge)
    ratios = Eavg[band][1:] / Eavg[band][:-1]
    out["falloff_max_ratio"] = float(ratios.max())
    out["falloff"] = bool(ratios.max() < 1.0)
    out["E_edge_over_peak"] = float(Eavg[n_edge] / Eavg[1:].max())
    f = ts["f_edge"][w2]
    slope = np.polyfit(t[w2], np.log(np.maximum(f, 1e-300)), 1)[0]
    out["f_edge_mean"] = float(f.mean())
    out["f_edge_efold"] = float(1.0 / slope) if slope != 0 else float("inf")
    out["edge_ok"] = bool(slope < 1.0 / 50.0)
    out["eps_inj_expected"] = cfg["eps_inj_expected"]
    out["eps_inj_sampled"] = float(ts["eps_inj"][w2].mean())
    out["eps_diss"] = float(ts["eps_diss"][w2].mean())
    out["eps_num"] = float(ts["eps_num"][w2].mean())
    out["num_over_inj"] = out["eps_num"] / cfg["eps_inj_expected"]
    out["diss_over_inj"] = out["eps_diss"] / cfg["eps_inj_expected"]
    out["dEdt_over_inj"] = float(np.polyfit(t[w2], ts["E"][w2], 1)[0]) / cfg["eps_inj_expected"]
    out["U_max"] = float(np.maximum(ts["U_plus"], ts["U_minus"])[w2].max())
    out["y_max"] = float(ts["y_max"][w2].max())
    out["dt_over_cfl_max"] = float(ts["dt_over_cfl"].max())
    span = np.diff(t)
    flux = (ts["eps_inj"][1:] - ts["eps_diss"][1:] + ts["eps_num"][1:]) * span
    out["budget_closure"] = float((ts["E"][-1] - ts["E"][0] - flux.sum()) / ts["E"][-1])
    out["stable"] = all(out[k] for k in ("reached", "plateau", "zmax_ok", "falloff", "edge_ok"))
    return out, Eavg


def figure(name: str, ts, spec, cfg, Eavg, path: Path) -> None:
    t = ts["t"]
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    a = ax[0, 0]
    a.semilogy(t, ts["E"], "k-", lw=1)
    a.set(xlabel="t [τ_A]", ylabel="E (code units)", title="E(t)")
    a = ax[0, 1]
    a.semilogy(t, ts["zp_max"], label="max|z+|")
    a.semilogy(t, ts["zm_max"], label="max|z−|")
    a.semilogy(t, np.maximum(ts["U_plus"], ts["U_minus"]), label="max|∇⊥z±|")
    a.set(xlabel="t [τ_A]", title="field maxima")
    a.legend()
    a = ax[0, 2]
    a.semilogy(t, ts["f_edge"], label="E(n⊥ ≥ 0.8 n_edge)/E")
    a.semilogy(t, ts["f_half"], label="E(n⊥ > n_edge/2)/E")
    a.semilogy(t, np.maximum(ts["f_kz_half"], 1e-30), label="E(|n_z| > n_edge/2)/E")
    a.set(xlabel="t [τ_A]", title="energy near the cutoff")
    a.legend()
    a = ax[1, 0]
    n = int(round(20.0 / (t[2] - t[1])))
    tm = running_mean(t[1:], n)
    a.axhline(cfg["eps_inj_expected"], color="C0", ls="--", lw=1, label="injection, expected")
    a.plot(tm, running_mean(ts["eps_inj"][1:], n), color="C0", lw=0.8, label="injection, sampled")
    a.plot(tm, running_mean(ts["eps_diss"][1:], n), color="C1", label="hyper-resistive dissipation")
    a.plot(tm, running_mean(ts["eps_num"][1:], n), color="C2", label="stepper energy error")
    a.set(xlabel="t [τ_A]", ylabel="dE/dt (code units)", title="energy budget, 20 τ_A running means")
    a.legend(fontsize=8)
    a = ax[1, 1]
    st = spec["t"]
    n = np.arange(spec["E_nperp"].shape[1])
    for tt in np.linspace(st[len(st) // 10], st[-1], 6):
        i = int(np.argmin(np.abs(st - tt)))
        a.loglog(n[1:], spec["E_nperp"][i][1:], lw=0.8, label=f"t={st[i]:.0f}")
    a.loglog(n[1:], Eavg[1:], "k-", lw=2, label="avg last 100 τ_A")
    a.axvline(cfg["n_edge"], color="gray", ls="--", lw=0.8)
    nr = np.array([3.0, 20.0])
    a.loglog(nr, Eavg[3] * (nr / 3.0) ** (-5.0 / 3.0), "k:", label="n^(-5/3)")
    a.set(xlabel="n⊥", ylabel="E(n⊥)", title="perpendicular spectrum")
    a.set_ylim(max(Eavg[1:].max() * 1e-14, Eavg[Eavg > 0].min() / 10), Eavg[1:].max() * 10)
    a.legend(fontsize=7)
    a = ax[1, 2]
    sw = st > st[-1] - 100.0
    Ez = spec["E_nz"][sw].mean(axis=0)
    nz = np.arange(len(Ez))
    a.loglog(nz[1:], Ez[1:], "k-")
    a.axvline(cfg["n_edge"], color="gray", ls="--", lw=0.8)
    a.set(xlabel="|n_z|", ylabel="E(|n_z|)", title="parallel spectrum, avg last 100 τ_A")
    fig.suptitle(f"{name}: N={cfg['resolution']}³, {cfg['forcing']} forcing, amplitude {cfg['amplitude']}, "
                 f"η={cfg['eta']}, r={cfg['hyper_r']}, dt={cfg['dt']:.6g}")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--fig-dir", default=".")
    p.add_argument("--window", type=float, default=100.0)
    p.add_argument("--markdown", action="store_true", help="also print one table row per run")
    a = p.parse_args()
    rows = []
    for r in a.runs:
        run = Path(r)
        ts, spec, cfg = load(run)
        cfg["eps_inj_expected"] = expected_injection(cfg)
        out, Eavg = checks(ts, spec, cfg, a.window)
        print(f"== {run.name}")
        for k, v in out.items():
            print(f"   {k:18s} {v:.4g}" if isinstance(v, float) else f"   {k:18s} {v}")
        figure(run.name, ts, spec, cfg, Eavg, Path(a.fig_dir) / f"{run.name}.png")
        rows.append(
            f"| {cfg['resolution']}³ | {cfg['forcing']} | {cfg['amplitude']} | {out['t_last']:.0f} "
            f"| {out['E_ratio']:.2f} | {out['zmax_ratio']:.2f} | {out['U_max']:.3f} "
            f"| {out['falloff_max_ratio']:.2f} | {out['E_edge_over_peak']:.1e} "
            f"| {out['f_edge_efold']:+.0f} | {out['diss_over_inj']:.2f} | {out['num_over_inj']:.1e} "
            f"| {'pass' if out['stable'] else 'fail: ' + ', '.join(k for k in ('reached', 'plateau', 'zmax_ok', 'falloff', 'edge_ok') if not out[k])} |"
        )
    if a.markdown:
        print("| N | forcing | amplitude | t_end [τ_A] | E ratio | max z± ratio | max ∇⊥z± "
              "| largest E(n+1)/E(n) | E(n_edge)/peak | edge e-fold [τ_A] | diss/inj | stepper error/inj | criteria |")
        print("|" + "---|" * 13)
        print("\n".join(rows))


if __name__ == "__main__":
    main()
