"""
Tests for the device-resident run loop (krmhd.runloop.DeviceLoop, Issue #188).

The reference for every test is the Python loop it replaces: gandalf_step called
once per step, with force_alfven_modes / force_hermite_moments before each step
and compute_cfl_timestep for adaptive dt. The loop must reproduce that sequence
to round-off, carry the PRNG key exactly, hand the accounting hook the actual
step inputs, and own (donate) its buffers without warnings.
"""

import dataclasses
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import enable_x64

from krmhd.forcing import WhiteNoiseBand, force_alfven_modes, force_hermite_moments
from krmhd.physics import KRMHDState, initialize_random_spectrum
from krmhd.spectral import SpectralGrid3D
from krmhd.timestepping import compute_cfl_timestep, gandalf_step
from krmhd.runloop import (
    Accounting,
    CflDt,
    DeviceLoop,
    FixedDt,
    Forcing,
    RunConfig,
    StepParams,
)

GRID = SpectralGrid3D.create(Nx=16, Ny=16, Nz=16, Lx=1.0, Ly=1.0, Lz=1.0)
M = 4
# Relative round-off bound for a few float32 steps: the loop runs the same
# kernels as gandalf_step, compiled in a different graph.
RTOL = 5e-6


def _state(grid: SpectralGrid3D = GRID) -> KRMHDState:
    """Random z± spectrum and small in-band random g."""
    s = initialize_random_spectrum(grid, M=M, alpha=5 / 3, amplitude=0.5, k_min=1.0, k_max=8.0, seed=1)
    g = jax.random.normal(jax.random.PRNGKey(3), s.g.shape) * 1e-2
    g = (g + 0j).astype(s.z_plus.dtype) * grid.dealias_mask[..., None]
    return s.model_copy(update=dict(g=g))


def _zero_state(grid: SpectralGrid3D = GRID, dtype=jnp.complex64) -> KRMHDState:
    """A state at rest, with z⁺ and z⁻ sharing one buffer (as the 'zero' initial condition)."""
    zeros = jnp.zeros((grid.Nz, grid.Ny, grid.Nx // 2 + 1), dtype=dtype)
    g = jnp.zeros(zeros.shape + (M + 1,), dtype=dtype)
    return KRMHDState(
        z_plus=zeros, z_minus=zeros, g=g, M=M, beta_i=1.0, v_th=1.0, nu=0.5,
        Lambda=1.0, time=0.0, grid=grid,
    )


def _assert_close(ref: KRMHDState, out: KRMHDState, rtol: float = RTOL) -> None:
    for name in ("z_plus", "z_minus", "g"):
        a, b = np.asarray(getattr(ref, name)), np.asarray(getattr(out, name))
        scale = np.abs(a).max()
        assert np.abs(a - b).max() <= rtol * scale, (name, np.abs(a - b).max() / scale)


PARAMS = StepParams(eta=0.5, v_A=1.0, nu=0.5, hyper_r=2, hyper_n=2)
# Strong enough that a state at rest outgrows v_A within a few steps, so the
# CFL dt changes during the spin-up test.
FORCING = Forcing(
    alfven=WhiteNoiseBand(amplitude=5.0, n_min=1, n_max=2),
    hermite=WhiteNoiseBand(amplitude=0.5, n_min=1, n_max=2, band="perp_lowkz"),
    moments=(0, 2),
)


def _force_python(state, dt, key):
    """FORCING applied with the public functions, as a Python loop would."""
    a, h = FORCING.alfven, FORCING.hermite
    state, key = force_alfven_modes(state, a.amplitude, a.n_min, a.n_max, dt, key)
    return force_hermite_moments(
        state, h.amplitude, h.n_min, h.n_max, dt, key,
        forced_moments=FORCING.moments, band=h.band,
    )


def _python_loop(state, params, n_steps, dt=None, key=None, cfl_safety=0.3):
    """The reference: force, step, and (dt=None) re-evaluate the CFL dt each step."""
    dts = []
    for _ in range(n_steps):
        mode = "multirate" if params.scheme == "lawson_rk4_multirate" else "courant"
        step_dt = dt if dt is not None else compute_cfl_timestep(
            state, params.v_A, cfl_safety, params.closure, mode=mode
        )
        dts.append(step_dt)
        if key is not None:
            state, key = _force_python(state, step_dt, key)
        state = gandalf_step(state, step_dt, **params._asdict())
    return state, key, np.asarray(dts)


def _record_dt(n_steps: int) -> Accounting:
    """Accounting that records each step's dt."""
    def update(acc, before, after, increment, dt):
        dts, i = acc
        return dts.at[i].set(dt), i + 1

    return Accounting(init=(jnp.zeros(n_steps), jnp.zeros((), jnp.int32)), update=update)


@pytest.mark.parametrize(
    "scheme", ["lawson_rk4", "imex_rk222", "imex_ars343", "lawson_rk4_multirate"]
)
def test_fixed_dt_matches_gandalf_step(scheme):
    params = PARAMS._replace(scheme=scheme)
    ref, _, _ = _python_loop(_state(), params, 5, dt=1e-3)

    loop = DeviceLoop(_state(), RunConfig(step=params, dt=FixedDt(1e-3)))
    loop.advance(5)

    _assert_close(ref, loop.state)
    assert loop.state.time == ref.time
    assert int(loop.steps) == 5


def test_forcing_and_adaptive_dt_match_python_loop():
    """A forced spin-up from rest: key splits, increments and CFL dt as in the Python loop."""
    key = jax.random.PRNGKey(7)
    n = 6
    ref, ref_key, ref_dts = _python_loop(_zero_state(), PARAMS, n, key=key)

    config = RunConfig(step=PARAMS, dt=CflDt(0.3), forcing=FORCING, accounting=_record_dt(n))
    loop = DeviceLoop(_zero_state(), config, key=key)
    dts, count = loop.advance(n)

    assert int(count) == n
    np.testing.assert_allclose(np.asarray(dts), ref_dts, rtol=1e-6)
    assert ref_dts[-1] < ref_dts[0]  # the forcing spun the state up: dt fell
    _assert_close(ref, loop.state)
    np.testing.assert_array_equal(np.asarray(loop.key), np.asarray(ref_key))


@pytest.mark.parametrize("forced", [False, True])
@pytest.mark.parametrize(
    "scheme", ["lawson_rk4", "imex_rk222", "imex_ars343", "lawson_rk4_multirate"]
)
def test_adaptive_dt_matches_python_loop_float64(scheme, forced):
    """In float64 (the P1 runs' precision) the steps and the CFL dt agree to float64 round-off.

    The forcing increments are drawn in float32 by design (gaussian_white_noise_fourier).
    Compiled in a different graph they can round differently by one float32 ulp
    (seen on x86 with JAX 0.7.2), so the forced case is held to float32 precision.
    """
    with enable_x64():
        grid = SpectralGrid3D.create(Nx=16, Ny=16, Nz=16, Lx=1.0, Ly=1.0, Lz=1.0)
        params = PARAMS._replace(scheme=scheme)
        key = jax.random.PRNGKey(7) if forced else None
        start = (lambda: _zero_state(grid, jnp.complex128)) if forced else (lambda: _state(grid))
        ref, _, ref_dts = _python_loop(start(), params, 4, key=key)

        forcing = FORCING if forced else Forcing()
        config = RunConfig(step=params, dt=CflDt(0.3), forcing=forcing, accounting=_record_dt(4))
        loop = DeviceLoop(start(), config, key=key)
        dts, _ = loop.advance(4)

        rtol = 1e-6 if forced else 1e-12
        assert loop.state.g.dtype == jnp.complex128
        np.testing.assert_allclose(np.asarray(dts), ref_dts, rtol=rtol)
        _assert_close(ref, loop.state, rtol=rtol)


def test_mixed_precision_state_runs_in_the_promoted_dtype():
    """Under float64 a complex64 g comes out of gandalf_step as complex128; the loop starts there."""
    with enable_x64():
        grid = SpectralGrid3D.create(Nx=16, Ny=16, Nz=16, Lx=1.0, Ly=1.0, Lz=1.0)
        state = _state(grid)
        state = state.model_copy(update=dict(g=state.g.astype(jnp.complex64)))
        ref = gandalf_step(state, 1e-3, **PARAMS._replace(scheme="lawson_rk4")._asdict())

        loop = DeviceLoop(state, RunConfig(step=PARAMS._replace(scheme="lawson_rk4"), dt=FixedDt(1e-3)))
        loop.advance(2)

        assert loop.state.g.dtype == ref.g.dtype == jnp.complex128


@pytest.mark.parametrize(
    "scheme", ["lawson_rk4", "imex_rk222", "imex_ars343", "lawson_rk4_multirate"]
)
def test_complex64_state_with_float64_accounting(scheme):
    """Under float64 a complex64 state steps in float32 and the hook sums in float64 (Issue #191)."""
    with enable_x64():
        grid = SpectralGrid3D.create(Nx=16, Ny=16, Nz=16, Lx=1.0, Ly=1.0, Lz=1.0)
        fields = ("z_plus", "z_minus", "g")
        state = _state(grid)
        state32 = state.model_copy(update={k: getattr(state, k).astype(jnp.complex64) for k in fields})

        def free_energy(acc, before, after, increment, dt):
            return acc + jnp.sum(jnp.abs(after.g.astype(jnp.complex128)) ** 2)

        accounting = Accounting(init=jnp.zeros((), jnp.float64), update=free_energy)
        config = RunConfig(step=PARAMS._replace(scheme=scheme), dt=CflDt(0.3), forcing=FORCING,
                           accounting=accounting)
        key = jax.random.PRNGKey(7)
        ref = DeviceLoop(state, config, key=key)
        ref_acc = ref.advance(4)
        loop = DeviceLoop(state32, config, key=key)
        acc = loop.advance(4)

        assert all(getattr(loop.state, k).dtype == jnp.complex64 for k in fields)
        assert loop.dt.dtype == jnp.float32
        assert acc.dtype == jnp.float64
        np.testing.assert_allclose(float(acc), float(ref_acc), rtol=1e-5)
        _assert_close(ref.state, loop.state, rtol=1e-5)


def test_chunks_compose():
    """advance(2) then advance(3) is advance(5): the carry holds dt, key and step count."""
    key = jax.random.PRNGKey(11)
    config = RunConfig(step=PARAMS, dt=CflDt(0.3, every=2), forcing=FORCING)
    whole = DeviceLoop(_state(), config, key=key)
    whole.advance(5)
    split = DeviceLoop(_state(), config, key=key)
    split.advance(2)
    split.advance(3)

    for name in ("z_plus", "z_minus", "g"):
        np.testing.assert_array_equal(
            np.asarray(getattr(whole.state, name)), np.asarray(getattr(split.state, name))
        )
    assert float(whole.dt) == float(split.dt)
    assert int(split.steps) == 5


def test_dt_updates_every_k_steps():
    n, every = 6, 3
    config = RunConfig(
        step=PARAMS, dt=CflDt(0.3, every=every), forcing=FORCING, accounting=_record_dt(n)
    )
    loop = DeviceLoop(_zero_state(), config, key=jax.random.PRNGKey(5))
    dts, _ = loop.advance(n)
    dts = np.asarray(dts)

    assert np.all(dts[:every] == dts[0]) and np.all(dts[every:] == dts[every])
    assert dts[every] != dts[0]


def test_accounting_sees_step_inputs():
    """The hook gets the pre-forcing fields, the increment, the post-step fields and dt."""
    def update(acc, before, after, increment, dt):
        injected, n_dt, t_after, g0_before = acc
        forced_zp = before.z_plus + increment.alfven
        return (
            injected + jnp.sum(jnp.abs(forced_zp) ** 2 - jnp.abs(before.z_plus) ** 2),
            n_dt + dt,
            after.time,
            g0_before + jnp.sum(jnp.abs(before.g[..., 0] + increment.hermite) ** 2),
        )

    zero = jnp.zeros(())
    config = RunConfig(
        step=PARAMS, dt=FixedDt(1e-3), forcing=FORCING,
        accounting=Accounting(init=(zero, zero, zero, zero), update=update),
    )
    key = jax.random.PRNGKey(2)
    injected, sum_dt, t_after, g0_forced = DeviceLoop(_state(), config, key=key).advance(3)

    state, ref_injected, ref_g0 = _state(), 0.0, 0.0
    for _ in range(3):
        before = state
        state, key = _force_python(state, 1e-3, key)
        ref_injected += float(jnp.sum(jnp.abs(state.z_plus) ** 2 - jnp.abs(before.z_plus) ** 2))
        ref_g0 += float(jnp.sum(jnp.abs(state.g[..., 0]) ** 2))
        state = gandalf_step(state, 1e-3, **PARAMS._asdict())

    np.testing.assert_allclose(float(injected), ref_injected, rtol=1e-4)
    np.testing.assert_allclose(float(g0_forced), ref_g0, rtol=1e-5)
    np.testing.assert_allclose(float(sum_dt), 3e-3, rtol=1e-6)
    assert float(t_after) == state.time


def test_donates_input_buffers_without_warnings():
    state = _state()
    inputs = (state.z_plus, state.z_minus, state.g)
    loop = DeviceLoop(state, RunConfig(step=PARAMS, dt=FixedDt(1e-3)))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        loop.advance(2)
        loop.advance(1)
        out = loop.state

    assert all(a.is_deleted() for a in inputs)
    assert np.isfinite(np.asarray(out.g)).all() and out.time > 0
    first = out.z_plus
    loop.advance(1)
    assert first.is_deleted()  # a state read from the loop lives until the next advance


def test_rejects_forcing_without_key_and_bad_moments():
    with pytest.raises(ValueError, match="key"):
        DeviceLoop(_state(), RunConfig(step=PARAMS, dt=FixedDt(1e-3), forcing=FORCING))
    bad = Forcing(hermite=WhiteNoiseBand(0.1, 1, 2), moments=(M + 1,))
    with pytest.raises(ValueError, match="moment"):
        DeviceLoop(_state(), RunConfig(step=PARAMS, dt=FixedDt(1e-3), forcing=bad), key=jax.random.PRNGKey(0))
    with pytest.raises(ValueError, match="hyper_r"):
        DeviceLoop(_state(), RunConfig(step=PARAMS._replace(hyper_r=3), dt=FixedDt(1e-3)))
    with pytest.raises(ValueError, match="perp_lowkz"):
        WhiteNoiseBand(0.1, 1, 2, max_nz=1)
    with pytest.raises(ValueError, match="dt_ref"):
        WhiteNoiseBand(0.1, 1, 2, dt_ref=0.0)


def _with_dt_ref(forcing: Forcing, dt_ref: float) -> Forcing:
    return dataclasses.replace(
        forcing,
        alfven=dataclasses.replace(forcing.alfven, dt_ref=dt_ref),
        hermite=dataclasses.replace(forcing.hermite, dt_ref=dt_ref),
    )


def _record_injection(n_steps: int) -> Accounting:
    """Accounting that records each step's dt and Σ|F|²/dt of the Alfvénic and Hermite increments."""
    def update(acc, before, after, increment, dt):
        dts, rates, i = acc
        power = jnp.stack([jnp.sum(jnp.abs(increment.alfven) ** 2), jnp.sum(jnp.abs(increment.hermite) ** 2)])
        return dts.at[i].set(dt), rates.at[i].set(power / dt), i + 1

    return Accounting(init=(jnp.zeros(n_steps), jnp.zeros((n_steps, 2)), jnp.zeros((), jnp.int32)), update=update)


def _forced_run(dt_policy, forcing: Forcing, n_steps: int):
    key = jax.random.PRNGKey(7)
    config = RunConfig(step=PARAMS, dt=dt_policy, forcing=forcing, accounting=_record_injection(n_steps))
    dts, rates, _ = DeviceLoop(_zero_state(), config, key=key).advance(n_steps)
    return np.asarray(dts), np.asarray(rates)


def test_dt_ref_keeps_the_injection_rate_as_dt_adapts():
    """With dt_ref, each step injects Σ|F|²/dt of a fixed-dt run at dt_ref, whatever its dt.

    The noise ξ depends only on the key, so with one key the steps of every run
    draw the same ξ, and Σ|F|²/dt compares exactly, step by step.
    """
    n, dt_ref = 6, 0.01
    fixed_dts, fixed_rates = _forced_run(FixedDt(dt_ref), FORCING, n)
    dts, rates = _forced_run(CflDt(0.3), _with_dt_ref(FORCING, dt_ref), n)

    assert dts.max() > 1.5 * dts.min()  # the spin-up changes dt
    np.testing.assert_allclose(rates, fixed_rates, rtol=1e-5)

    # Without dt_ref the rate goes as 1/dt²
    free_dts, free_rates = _forced_run(CflDt(0.3), FORCING, n)
    np.testing.assert_allclose(free_rates, fixed_rates * (dt_ref / free_dts[:, None]) ** 2, rtol=1e-5)


def test_dt_ref_equal_to_dt_changes_nothing():
    key = jax.random.PRNGKey(7)
    loops = [
        DeviceLoop(_state(), RunConfig(step=PARAMS, dt=FixedDt(1e-3), forcing=forcing), key=key)
        for forcing in (FORCING, _with_dt_ref(FORCING, 1e-3))
    ]
    for loop in loops:
        loop.advance(3)
    ref, out = (loop.state for loop in loops)
    for name in ("z_plus", "z_minus", "g"):
        np.testing.assert_array_equal(np.asarray(getattr(out, name)), np.asarray(getattr(ref, name)))
    np.testing.assert_array_equal(np.asarray(loops[1].key), np.asarray(loops[0].key))
