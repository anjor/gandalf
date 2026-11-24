"""
Device-resident multi-step run loop (Issue #188).

gandalf_step advances one step per call, and every call returns to the host: it
reads the new time as a Python float, rebuilds a Pydantic KRMHDState, and (IMEX
schemes) rebuilds the implicit operator in numpy. Forcing runs eagerly outside
jit. A driver loop therefore waits for each step before it dispatches the next.

DeviceLoop runs many steps inside one jitted lax.fori_loop. Each step applies,
in order:

1. forcing: white-noise increments drawn on the device from the carried PRNG
   key (RunConfig.forcing);
2. the scheme's step kernel, the same jitted kernel gandalf_step calls;
3. the accounting hook, a pure function that folds the step into a pytree
   accumulator (Accounting);
4. the dt update (RunConfig.dt): a fixed dt, or a jittable estimate evaluated
   on the post-step fields every `every` steps.

The host syncs only when the caller reads the state, typically at sample and
checkpoint times between advance() calls.

Example:
    >>> config = RunConfig(
    ...     step=StepParams(eta=1.0, v_A=1.0, hyper_r=2, hyper_n=2, scheme="lawson_rk4"),
    ...     dt=CflDt(cfl_safety=0.3),
    ...     forcing=Forcing(alfven=WhiteNoiseBand(amplitude=0.1, n_min=1, n_max=2)),
    ... )
    >>> loop = DeviceLoop(state, config, key=jax.random.PRNGKey(0))
    >>> for _ in range(10):
    ...     loop.advance(100)        # 100 steps, no host sync
    ...     snapshot = loop.state    # one sync; valid until the next advance()

Buffer ownership: advance() donates the loop's field buffers to the jitted
kernel, so XLA updates z±, g in place and the step needs no second copy of the
state (Issue #186). The arrays of the KRMHDState passed to DeviceLoop become
the loop's first buffers: they are invalid after the first advance(). Each
state read from loop.state is valid until the next advance().

Precision: the step runs in the precision of the state's z±. Under
jax_enable_x64 a complex64 state therefore steps in float32 while the
accounting hook upcasts the fields and sums in float64: the float32 state with
float64 budgets of Issue #191 (docs/precision_float32_study.md).
"""

import warnings
from dataclasses import dataclass
from typing import Any, Callable, NamedTuple, Optional, Protocol, Tuple, Union

import jax
import jax.numpy as jnp
from jax import Array

from krmhd.forcing import WhiteNoiseBand, _add_to_moments
from krmhd.hermite import compute_streaming_eigensystem
from krmhd.physics import KRMHDState
from krmhd.spectral import SpectralGrid3D
from krmhd.timestepping import (
    HermiteClosure,
    KRMHDFields,
    Scheme,
    _gandalf_step_imex222_jit,
    _gandalf_step_imex343_jit,
    _gandalf_step_lawson_rk4_jit,
    _validate_step_args,
    cfl_timestep_estimate,
)


class StepParams(NamedTuple):
    """gandalf_step's keyword arguments, fixed for a run (see gandalf_step).

    nu=None uses the state's nu, as in gandalf_step.
    """

    eta: float
    v_A: float
    nu: Optional[float] = None
    hyper_r: int = 1
    hyper_n: int = 1
    scheme: Scheme = "imex_rk222"
    eta_z: float = 0.0
    hyper_rz: int = 1
    closure: HermiteClosure = "zero"


@dataclass(frozen=True)
class FixedDt:
    """Every step uses dt."""

    dt: float


class AdaptiveDtPolicy(Protocol):
    """A dt rule evaluated on the device.

    estimate() returns the step size for the post-step fields. It must be
    traceable in z_plus and z_minus; the other arguments are trace-time
    constants (grid may be a traced pytree). The loop evaluates it after every
    `every`-th step and keeps dt in between.
    """

    every: int

    def estimate(
        self,
        z_plus: Array,
        z_minus: Array,
        grid: SpectralGrid3D,
        M: int,
        beta_i: float,
        Lambda: float,
        step: StepParams,
    ) -> Array: ...


@dataclass(frozen=True)
class CflDt:
    """dt = compute_cfl_timestep(state, v_A, cfl_safety, closure), on the device."""

    cfl_safety: float = 0.3
    every: int = 1

    def __post_init__(self) -> None:
        if self.every < 1:
            raise ValueError(f"every must be >= 1, got {self.every}")

    def estimate(
        self,
        z_plus: Array,
        z_minus: Array,
        grid: SpectralGrid3D,
        M: int,
        beta_i: float,
        Lambda: float,
        step: StepParams,
    ) -> Array:
        return cfl_timestep_estimate(
            z_plus, z_minus, grid, M, beta_i, Lambda, step.v_A, self.cfl_safety, step.closure
        )


DtPolicy = Union[FixedDt, AdaptiveDtPolicy]


@dataclass(frozen=True)
class Forcing:
    """White-noise forcing applied at the start of every step.

    alfven: increment F added to z⁺ and z⁻ alike, as force_alfven_modes does.
    hermite: increment added to each g_m, m in moments, as force_hermite_moments does.
    Both draw from one carried key, Alfvénic first, so the sequence matches a
    Python loop that calls force_alfven_modes and then force_hermite_moments
    with the same key.
    """

    alfven: Optional[WhiteNoiseBand] = None
    hermite: Optional[WhiteNoiseBand] = None
    moments: Tuple[int, ...] = (0,)

    @property
    def enabled(self) -> bool:
        return self.alfven is not None or self.hermite is not None


class ForcingIncrement(NamedTuple):
    """The forcing added in one step (zeros for an absent term).

    alfven: [Nz, Ny, Nx//2+1], added to z⁺ and z⁻.
    hermite: [Nz, Ny, Nx//2+1], added to each forced moment g_m.
    """

    alfven: Array
    hermite: Array


class Accounting(NamedTuple):
    """Per-step accounting folded into a pytree on the device.

    update(acc, before, after, increment, dt) -> acc is called once per step,
    with `before` the fields at the start of the step (before forcing),
    `after` the fields after the step, `increment` the forcing added in
    between, and dt the step size. It must be pure and traceable, and return
    a pytree with the structure and dtypes of `init`. Every advance() starts
    from `init` and returns the accumulator of its steps.

    Reading before.g keeps the pre-forcing g alive next to the forced one, an
    extra g-sized buffer; reading only some moments costs only those.
    """

    init: Any
    update: Callable[[Any, KRMHDFields, KRMHDFields, ForcingIncrement, Array], Any]


@dataclass(frozen=True)
class RunConfig:
    """Everything that stays fixed for a run of DeviceLoop."""

    step: StepParams
    dt: DtPolicy
    forcing: Forcing = Forcing()
    accounting: Optional[Accounting] = None


class RunState(NamedTuple):
    """The loop carry: everything that changes from step to step."""

    fields: KRMHDFields  # time is a device scalar here
    key: Optional[Array]  # PRNG key; None without forcing
    dt: Array  # step size of the next step
    steps: Array  # steps taken in the run (int32)


@dataclass(frozen=True)
class _Plan:
    """Trace-time constants of the loop: a static jit argument."""

    step: StepParams
    dt: DtPolicy
    forcing: Forcing
    update: Optional[Callable]
    M: int
    beta_i: float
    Lambda: float
    nu: float


def _cast_floats(tree: Any, dtype: Any) -> Any:
    """tree with its real floating-point arrays cast to dtype (masks unchanged)."""
    return jax.tree.map(
        lambda a: a.astype(dtype) if jnp.issubdtype(a.dtype, jnp.floating) else a, tree
    )


def _scheme_constants(state: KRMHDState, step: StepParams) -> Tuple[Array, ...]:
    """Per-run constants of the scheme's step kernel.

    lawson_rk4: the eigensystem of the streaming matrix. IMEX: none; the
    kernels factor the tridiagonal (I - γ dt L) themselves, so dt may be traced.
    """
    if step.scheme == "lawson_rk4":
        _, eigenvalues, P, P_inv = compute_streaming_eigensystem(
            state.M, state.Lambda, step.closure
        )
        return eigenvalues, P.T, P_inv.T
    return ()


def _scheme_step(
    fields: KRMHDFields,
    dt: Array,
    grid: SpectralGrid3D,
    consts: Tuple[Array, ...],
    plan: _Plan,
) -> KRMHDFields:
    """One step of the configured scheme: the kernels gandalf_step calls."""
    p = plan.step
    args = (
        fields, dt, grid.kx, grid.ky, grid.kz, grid.dealias_mask, p.eta, p.v_A, plan.beta_i,
        plan.nu, plan.Lambda, plan.M, grid.Nz, grid.Ny, grid.Nx, p.hyper_r, p.hyper_n,
    )
    if p.scheme == "lawson_rk4":
        eigenvalues, P_T, P_inv_T = consts
        return _gandalf_step_lawson_rk4_jit(
            *args, eta_z=p.eta_z, hyper_rz=p.hyper_rz,
            streaming_eigenvalues=eigenvalues, streaming_P_T=P_T,
            streaming_P_inv_T=P_inv_T, closure=p.closure,
        )
    kernel = _gandalf_step_imex343_jit if p.scheme == "imex_ars343" else _gandalf_step_imex222_jit
    return kernel(*args, eta_z=p.eta_z, hyper_rz=p.hyper_rz, closure=p.closure)


def _force(
    fields: KRMHDFields, dt: Array, key: Optional[Array], grid: SpectralGrid3D, forcing: Forcing
) -> Tuple[KRMHDFields, ForcingIncrement, Optional[Array]]:
    """Draw this step's forcing and add it to the fields."""
    zeros = jnp.zeros_like(fields.z_plus)
    z_plus, z_minus, g = fields.z_plus, fields.z_minus, fields.g
    alfven = hermite = zeros
    if forcing.alfven is not None:
        alfven, key = forcing.alfven.draw(grid, dt, key)
        alfven = alfven.astype(z_plus.dtype)
        z_plus = z_plus + alfven
        z_minus = z_minus + alfven
    if forcing.hermite is not None:
        hermite, key = forcing.hermite.draw(grid, dt, key)
        hermite = hermite.astype(g.dtype)
        g = _add_to_moments(g, hermite, forcing.moments)
    forced = KRMHDFields(z_plus=z_plus, z_minus=z_minus, g=g, time=fields.time)
    return forced, ForcingIncrement(alfven=alfven, hermite=hermite), key


def _next_dt(after: KRMHDFields, dt: Array, steps: Array, grid: SpectralGrid3D, plan: _Plan) -> Array:
    """dt for the next step under the plan's dt policy."""
    policy = plan.dt
    if isinstance(policy, FixedDt):
        return dt

    def estimate() -> Array:
        new_dt = policy.estimate(
            after.z_plus, after.z_minus, grid, plan.M, plan.beta_i, plan.Lambda, plan.step
        )
        return new_dt.astype(dt.dtype)

    if policy.every == 1:
        return estimate()
    return jax.lax.cond(steps % policy.every == 0, estimate, lambda: dt)


def _one_step(
    state: RunState, acc: Any, grid: SpectralGrid3D, consts: Tuple[Array, ...], plan: _Plan
) -> Tuple[RunState, Any]:
    """One loop step: forcing, the scheme's step, the accounting hook, the dt update."""
    before, dt = state.fields, state.dt
    forced, increment, key = _force(before, dt, state.key, grid, plan.forcing)
    after = _scheme_step(forced, dt, grid, consts, plan)
    if plan.update is not None:
        acc = plan.update(acc, before, after, increment, dt)
    steps = state.steps + 1
    dt = _next_dt(after, dt, steps, grid, plan)
    return RunState(fields=after, key=key, dt=dt, steps=steps), acc


def _advance_impl(
    carry: RunState,
    acc: Any,
    n_steps: Array,
    grid: SpectralGrid3D,
    consts: Tuple[Array, ...],
    plan: _Plan,
) -> Tuple[RunState, Any]:
    return jax.lax.fori_loop(
        0, n_steps, lambda _, loop: _one_step(*loop, grid, consts, plan), (carry, acc)
    )


# The carry (argument 0) is donated: XLA writes the new fields into its buffers.
#
# XLA's while-loop invariant code motion hoists each copy of a repeated
# loop-invariant subexpression (the 1j·kx, 1j·ky derivative factors of the
# separately inlined z_plus_rhs and z_minus_rhs, for example) into its own loop
# operand before CSE merges the copies. CSE then cannot merge the FFTs that
# consume them: at 16³, M = 2 the Lawson-RK4 loop body ran 72 inverse FFTs to
# the standalone step's 36. With the pass off, the body compiles to the
# standalone step's FFT count and recomputes the small invariants each step,
# as gandalf_step does.
_JIT = dict(static_argnames=("plan",), donate_argnums=(0,))
try:
    _advance = jax.jit(
        _advance_impl,
        compiler_options={"xla_disable_hlo_passes": "while-loop-invariant-code-motion"},
        **_JIT,
    )
except TypeError:  # jax.jit without compiler_options (JAX < 0.4.36): correct, slower loop
    _advance = jax.jit(_advance_impl, **_JIT)

# JAX warns when a backend cannot donate (e.g. jax-metal). The loop is still
# correct there; it only keeps the extra copy that donation avoids.
_NO_DONATION_WARNING = r"(?s)Some donated buffers were not usable.*Donation is not implemented"


class DeviceLoop:
    """
    Advances a KRMHD state many steps per call, on the device (Issue #188).

    Args:
        state: Initial state. Its arrays become the loop's buffers and are
            invalid after the first advance() (see the module docstring).
        config: Scheme, physics, dt policy, forcing and accounting of the run.
        key: PRNG key of the forcing; required when config.forcing is enabled.
            The loop copies it, so the caller's key stays valid.
        dt: Step size of the first step. Default: config.dt's (the fixed dt, or
            the estimate on `state`). A resumed run passes the dt it saved.
        steps: Steps already taken, for a resumed run: an adaptive policy then
            re-evaluates dt on the same steps as an uninterrupted run.

    Raises:
        ValueError: As gandalf_step for the step parameters (checked once, with
            the initial dt), and for forcing without a key or with a moment
            outside 0..M.
    """

    def __init__(
        self,
        state: KRMHDState,
        config: RunConfig,
        key: Optional[Array] = None,
        dt: Optional[float] = None,
        steps: int = 0,
    ):
        step = config.step
        nu = state.nu if step.nu is None else step.nu
        forcing = config.forcing
        if forcing.enabled and key is None:
            raise ValueError("forcing needs a PRNG key")
        if forcing.hermite is not None:
            for m in forcing.moments:
                if not isinstance(m, int) or not 0 <= m <= state.M:
                    raise ValueError(f"forced moment {m!r} must be an int in 0..M={state.M}")

        self._plan = _Plan(
            step=step,
            dt=config.dt,
            forcing=forcing,
            update=None if config.accounting is None else config.accounting.update,
            M=state.M,
            beta_i=state.beta_i,
            Lambda=state.Lambda,
            nu=nu,
        )
        self._acc0 = () if config.accounting is None else config.accounting.init
        # The step runs in the precision of z±: wavenumbers, eigensystem and dt in
        # its real dtype. A complex64 state then stays complex64 under float64,
        # where the accounting hook can sum in float64 (Issue #191).
        real = jnp.finfo(state.z_plus.dtype).dtype
        self._grid = _cast_floats(state.grid, real)
        self._meta = dict(
            M=state.M, beta_i=state.beta_i, v_th=state.v_th, nu=state.nu,
            Lambda=state.Lambda, grid=state.grid,
        )

        if dt is not None:
            dt0 = dt
        elif isinstance(config.dt, FixedDt):
            dt0 = config.dt.dt
        else:
            dt0 = config.dt.estimate(
                state.z_plus, state.z_minus, state.grid, state.M, state.beta_i, state.Lambda, step
            )
        _validate_step_args(
            state, float(dt0), step.eta, nu, step.hyper_r, step.hyper_n, step.scheme,
            step.eta_z, step.hyper_rz, step.closure,
        )
        self._consts = _cast_floats(_scheme_constants(state, step), real)

        # A buffer may be donated only once: give z⁻ its own if it shares z⁺'s
        # (zero initial conditions do).
        z_plus = jnp.asarray(state.z_plus)
        z_minus = jnp.asarray(state.z_minus)
        if z_minus is z_plus:
            z_minus = jnp.copy(z_minus)
        carry = RunState(
            fields=KRMHDFields(
                z_plus=z_plus,
                z_minus=z_minus,
                g=jnp.asarray(state.g),
                # the dtype jit gives gandalf_step's float time
                time=jnp.asarray(state.time, dtype=jnp.result_type(float)),
            ),
            key=None if key is None else jnp.copy(key),  # the caller keeps its key
            dt=jnp.asarray(dt0, dtype=real),
            steps=jnp.asarray(steps, dtype=jnp.int32),
        )
        # The step can still promote a field: a complex64 g next to complex128 z±
        # comes out complex128. The loop carry keeps one dtype, so start from the
        # dtypes one step produces.
        stepped, _ = jax.eval_shape(
            lambda c, a: _one_step(c, a, self._grid, self._consts, self._plan), carry, self._acc0
        )
        self._carry = jax.tree.map(
            lambda x, out: x if x.dtype == out.dtype else x.astype(out.dtype), carry, stepped
        )

    def advance(self, n_steps: int) -> Any:
        """Take n_steps steps on the device; return their accounting accumulator.

        Dispatches asynchronously and does not sync with the host. The previous
        loop.state snapshot is invalid afterwards.
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=_NO_DONATION_WARNING)
            self._carry, acc = _advance(
                self._carry,
                self._acc0,
                jnp.asarray(n_steps, dtype=jnp.int32),
                self._grid,
                self._consts,
                plan=self._plan,
            )
        return acc

    @property
    def state(self) -> KRMHDState:
        """The current state (one host sync to read the time).

        It shares the loop's buffers and is valid until the next advance().
        """
        f = self._carry.fields
        return KRMHDState(z_plus=f.z_plus, z_minus=f.z_minus, g=f.g, time=float(f.time), **self._meta)

    @property
    def key(self) -> Optional[Array]:
        """The forcing PRNG key after the steps taken (save it to resume)."""
        return self._carry.key

    @property
    def dt(self) -> Array:
        """Step size of the next step (device scalar)."""
        return self._carry.dt

    @property
    def steps(self) -> Array:
        """Steps taken so far (device int32 scalar)."""
        return self._carry.steps
