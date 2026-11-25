"""
Lawson-RK4 with its Hermite slices sharded over several devices (Issue #191).

These tests need four devices. tests/test_multidevice.py runs them in a
subprocess with four virtual CPU devices
(XLA_FLAGS=--xla_force_host_platform_device_count=4); in a one-device session
they skip. To run them directly:

    XLA_FLAGS=--xla_force_host_platform_device_count=4 JAX_PLATFORMS=cpu \\
        uv run pytest tests/multidevice -q

The reference for every test is the same step or loop on one device: a sharded
step must match it to round-off, return g split along k_z and z± whole, and
communicate only through two all-to-alls per step.
"""

import math
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import enable_x64
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from krmhd.forcing import WhiteNoiseBand
from krmhd.hermite import compute_streaming_eigensystem
from krmhd.physics import KRMHDState, enforce_hermitian_state
from krmhd.runloop import (
    Accounting,
    CflDt,
    DeviceLoop,
    Forcing,
    RunConfig,
    StepParams,
    _advance_impl,
)
from krmhd.spectral import SpectralGrid3D
from krmhd.timestepping import (
    _fields_from_state,
    _gandalf_step_lawson_rk4_jit,
    gandalf_step,
    place_on_mesh,
)

pytestmark = pytest.mark.skipif(
    jax.device_count() < 4, reason="needs 4 devices; tests/test_multidevice.py runs these"
)

COLLECTIVE = re.compile(
    r" (all-to-all|all-reduce|all-gather|reduce-scatter|collective-permute|collective-broadcast)"
    r"(?:-start)?\("
)


def _mesh(n_devices: int) -> Mesh:
    return Mesh(np.array(jax.devices()[:n_devices]), ("slices",))


def _random_state(grid: SpectralGrid3D, M: int, nu: float, seed: int,
                  dtype=jnp.complex128) -> KRMHDState:
    """Random in-band z± and g, of order one in real space, with Hermitian pairs."""
    rng = np.random.default_rng(seed)
    shape = (grid.Nz, grid.Ny, grid.Nx // 2 + 1)
    mask = np.asarray(grid.dealias_mask)

    def draw(*s: int) -> jnp.ndarray:
        a = (rng.standard_normal(s) + 1j * rng.standard_normal(s)) * 50.0
        return jnp.asarray(a * mask.reshape(shape + (1,) * (len(s) - 3)), dtype=dtype)

    state = KRMHDState(
        z_plus=draw(*shape), z_minus=draw(*shape), g=draw(*shape, M + 1), M=M,
        beta_i=0.7, v_th=1.0, nu=nu, Lambda=math.sqrt(5.0), time=0.0, grid=grid,
    )
    return enforce_hermitian_state(state)


def _rel_diff(a: jnp.ndarray, b: jnp.ndarray) -> float:
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


def _collectives(hlo: str) -> list:
    return COLLECTIVE.findall(hlo)


# (devices, M, closure, nu, hyper_n, eta_z, hermite_chunk). M + 1 divisible by the
# device count or not (zero slices pad it), fewer slices than devices (M = 2 on 4),
# M = 1, both closures, collisions on and off, the all-at-once path (chunk None
# on this small grid) and the in-place path in groups of 1 and 2 with a remainder.
CASES = [
    (1, 4, "zero", 2.0, 6, 0.5, None),
    (2, 1, "zero", 0.0, 1, 0.3, None),
    (2, 4, "zero", 2.0, 6, 0.5, None),
    (4, 3, "zero", 0.0, 1, 0.0, None),
    (4, 4, "symmetric", 1.0, 2, 0.0, None),
    (4, 9, "symmetric", 5.0, 4, 0.5, None),
    (4, 2, "zero", 3.0, 6, 0.0, None),
    (2, 9, "zero", 2.0, 6, 0.0, 2),
    (4, 6, "symmetric", 2.0, 3, 0.5, 1),
]
STEP = dict(dt=3e-3, eta=2.0, v_A=1.0, hyper_r=2, hyper_rz=2, scheme="lawson_rk4")


@pytest.mark.parametrize("n_devices,M,closure,nu,hyper_n,eta_z,hermite_chunk", CASES)
def test_sharded_step_matches_one_device(n_devices, M, closure, nu, hyper_n, eta_z, hermite_chunk):
    """Three float64 steps on a mesh equal three steps on one device to round-off."""
    with enable_x64():
        grid = SpectralGrid3D.create(Nx=16, Ny=12, Nz=8)
        kw = dict(STEP, nu=nu, hyper_n=hyper_n, eta_z=eta_z, closure=closure,
                  hermite_chunk=hermite_chunk)
        one = many = _random_state(grid, M, nu, seed=10 * n_devices + M)
        for _ in range(3):
            one = gandalf_step(one, **kw)
            many = gandalf_step(many, mesh=_mesh(n_devices), **kw)
        assert many.g.dtype == jnp.complex128
        assert many.time == one.time
        for name in ("z_plus", "z_minus", "g"):
            err = _rel_diff(getattr(many, name), getattr(one, name))
            assert err < 1e-13, f"{name} differs by {err:.2e}"


def test_sharded_step_in_place_by_size(hermite_in_place):
    """In place, each device advances its slices one at a time, as one device
    does for large g."""
    with enable_x64():
        grid = SpectralGrid3D.create(Nx=12, Ny=10, Nz=4)
        kw = dict(STEP, nu=1.0, hyper_n=3, eta_z=0.0, closure="symmetric")
        state = _random_state(grid, 5, 1.0, seed=3)
        one = gandalf_step(state, **kw)
        many = gandalf_step(state, mesh=_mesh(4), **kw)
        assert _rel_diff(many.g, one.g) < 1e-13
        assert _rel_diff(many.z_plus, one.z_plus) < 1e-13


def test_sharded_step_float32():
    """The default precision (complex64) on a mesh: same dtype, round-off agreement."""
    grid = SpectralGrid3D.create(Nx=16, Ny=16, Nz=8)
    kw = dict(STEP, nu=2.0, hyper_n=6, eta_z=0.0, closure="zero")
    state = _random_state(grid, 6, 2.0, seed=4, dtype=jnp.complex64)
    one = gandalf_step(state, **kw)
    many = gandalf_step(state, mesh=_mesh(4), **kw)
    assert many.g.dtype == jnp.complex64
    assert _rel_diff(many.g, one.g) < 1e-6
    assert _rel_diff(many.z_plus, one.z_plus) < 1e-6


def test_sharded_state_layout():
    """The state a sharded step returns has g split along k_z, Nz/D planes per
    device, and z± whole on every device."""
    with enable_x64():
        grid = SpectralGrid3D.create(Nx=16, Ny=12, Nz=8)
        mesh = _mesh(4)
        new = gandalf_step(_random_state(grid, 4, 1.0, seed=5), mesh=mesh,
                           **dict(STEP, nu=1.0, hyper_n=2))
        assert new.g.sharding.is_equivalent_to(NamedSharding(mesh, PartitionSpec("slices")), 4)
        assert {s.data.shape for s in new.g.addressable_shards} == {(2, 12, 9, 5)}
        for z in (new.z_plus, new.z_minus):
            assert z.sharding.is_equivalent_to(NamedSharding(mesh, PartitionSpec()), 3)


@pytest.mark.parametrize("n_devices,M,hermite_chunk", [(2, 6, None), (4, 6, None), (4, 9, 2)])
def test_sharded_step_makes_two_all_to_alls(n_devices, M, hermite_chunk):
    """The compiled step communicates through exactly two all-to-alls (one to the
    slices, one back) and no other collective."""
    with enable_x64():
        grid = SpectralGrid3D.create(Nx=16, Ny=12, Nz=8)
        mesh = _mesh(n_devices)
        state = _random_state(grid, M, 1.0, seed=6)
        _, lam, P, P_inv = compute_streaming_eigensystem(M, state.Lambda, "zero")
        hlo = _gandalf_step_lawson_rk4_jit.lower(
            place_on_mesh(_fields_from_state(state), mesh), 3e-3, grid.kx, grid.ky, grid.kz,
            grid.dealias_mask, 2.0, 1.0, state.beta_i, 1.0, state.Lambda, M, grid.Nz, grid.Ny,
            grid.Nx, 2, 2, streaming_eigenvalues=lam, streaming_P_T=P.T,
            streaming_P_inv_T=P_inv.T, hermite_chunk=hermite_chunk, mesh=mesh,
        ).compile().as_text()
        assert _collectives(hlo) == ["all-to-all", "all-to-all"]


def _loop_config(mesh, accounting=None) -> RunConfig:
    return RunConfig(
        step=StepParams(eta=2.0, v_A=1.0, hyper_r=2, hyper_n=6, scheme="lawson_rk4", mesh=mesh),
        dt=CflDt(cfl_safety=0.3),
        forcing=Forcing(alfven=WhiteNoiseBand(0.5, 1, 2), hermite=WhiteNoiseBand(0.5, 1, 2),
                        moments=(0, 2)),
        accounting=accounting,
    )


def test_device_loop_on_mesh_matches_one_device():
    """A forced DeviceLoop with an adaptive dt and an accounting hook, on a mesh:
    fields, time, dt and the accumulator match one device to round-off, and g
    stays split along k_z across advance() calls."""
    with enable_x64():
        grid = SpectralGrid3D.create(Nx=16, Ny=16, Nz=8)
        M = 6
        accounting = Accounting(
            init=jnp.zeros(M + 1),
            update=lambda acc, before, after, inc, dt: acc + jnp.sum(jnp.abs(after.g) ** 2, axis=(0, 1, 2)),
        )
        results = []
        for mesh in (None, _mesh(4)):
            loop = DeviceLoop(_random_state(grid, M, 2.0, seed=7), _loop_config(mesh, accounting),
                              key=jax.random.PRNGKey(1))
            loop.advance(4)
            acc = loop.advance(3)
            results.append((loop.state, float(loop.dt), int(loop.steps), np.asarray(acc)))
        (one, dt_one, n_one, acc_one), (many, dt_many, n_many, acc_many) = results
        assert n_many == n_one == 7
        assert dt_many == pytest.approx(dt_one, rel=1e-13)
        assert many.time == pytest.approx(one.time, rel=1e-13)
        for name in ("z_plus", "z_minus", "g"):
            assert _rel_diff(getattr(many, name), getattr(one, name)) < 1e-13
        assert _rel_diff(acc_many, acc_one) < 1e-13
        assert {s.data.shape for s in many.g.addressable_shards} == {(2, 16, 9, M + 1)}


def test_device_loop_on_mesh_makes_two_all_to_alls_per_step():
    """The forced, adaptive-dt loop body adds no collective to the step's two: the
    forcing is drawn on every device rather than split and gathered."""
    with enable_x64():
        grid = SpectralGrid3D.create(Nx=16, Ny=16, Nz=8)
        loop = DeviceLoop(_random_state(grid, 4, 2.0, seed=8), _loop_config(_mesh(4)),
                          key=jax.random.PRNGKey(2))
        # The grid goes in as leaves: lowering cannot rebuild a SpectralGrid3D from
        # placeholder leaves.
        leaves, treedef = jax.tree.flatten(loop._grid)
        hlo = jax.jit(
            lambda carry, acc, n, grid_leaves, consts: _advance_impl(
                carry, acc, n, jax.tree.unflatten(treedef, grid_leaves), consts, loop._plan
            )
        ).lower(loop._carry, loop._acc0, jnp.asarray(5, jnp.int32), leaves, loop._consts)
        assert _collectives(hlo.compile().as_text()) == ["all-to-all", "all-to-all"]


@pytest.mark.parametrize("kw,match", [
    (dict(scheme="imex_rk222"), "lawson_rk4"),
    (dict(M=0), "M >= 1"),
    (dict(Nz=6), "divisible"),
    (dict(two_axes=True), "one axis"),
])
def test_mesh_rejects_what_it_cannot_shard(kw, match):
    grid = SpectralGrid3D.create(Nx=8, Ny=8, Nz=kw.get("Nz", 8))
    state = _random_state(grid, kw.get("M", 4), 0.0, seed=9, dtype=jnp.complex64)
    mesh = (Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("a", "b"))
            if kw.get("two_axes") else _mesh(4))
    step = dict(STEP, scheme=kw.get("scheme", "lawson_rk4"))
    with pytest.raises(ValueError, match=match):
        gandalf_step(state, mesh=mesh, **step)
    with pytest.raises(ValueError, match=match):
        DeviceLoop(state, RunConfig(step=StepParams(eta=2.0, v_A=1.0, scheme=step["scheme"],
                                                    mesh=mesh), dt=CflDt()))
