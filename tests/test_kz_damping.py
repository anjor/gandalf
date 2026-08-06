"""
Tests for parallel (kz) hyper-dissipation in gandalf_step (audit finding F4).

The original CUDA GANDALF applies parallel hyper-dissipation every step to the
Elsasser fields and to every Hermite moment (timestep.cu:94-108, `dampz`), with
kernel (damping_kernel.cu:4-33):

    znew *= exp(-nu_kz * dt * |kz / kz(idmax)|^(2*alpha_z)),  idmax = (Nz-1)/3

i.e. normalized parallel hyper-dissipation exp(-eta_z * (kz^2/kz_max^2)^rz * dt)
with kz_max at the 2/3 dealias boundary — the exact parallel analogue of the
perpendicular normalized hyper-resistivity already in timestepping.py.

This suite validates:
- Default no-op guarantee: eta_z=0.0, hyper_rz=1 changes nothing
- Nz=2 (2D-style) grids run without NaN under defaults (static idz_max guard)
- Single-Alfven-mode amplitude decay matches exp(-eta_z*(kz^2/kz_max^2)^rz * t)
  for hyper_rz in {1, 2} and both integrator schemes
- Hermite moments g decay at the same kz-dependent rate, uniformly across moments
- Input validation: hyper_rz allow-list, eta_z*dt overflow, Nz too small in z
"""

import pytest
import jax.numpy as jnp

from krmhd.spectral import SpectralGrid3D
from krmhd.physics import (
    KRMHDState,
    initialize_alfven_wave,
    initialize_hermite_moments,
    initialize_orszag_tang,
    initialize_random_spectrum,
)
from krmhd.timestepping import gandalf_step


def _kz_max_squared(grid: SpectralGrid3D) -> float:
    """kz_max^2 at the 2/3 dealias boundary, computed the same way the kernel does.

    idz_max = (Nz - 1) // 3 indexes into grid.kz which uses fftfreq ordering
    [0, 1, ..., Nz/2-1, -Nz/2, ..., -1]; idz_max < Nz/2 lands in the
    positive-kz range (matches original GANDALF damping_kernel.cu idmax).
    """
    idz_max = (grid.Nz - 1) // 3
    return float(grid.kz[idz_max]) ** 2


class TestKzDampingNoOp:
    """Defaults must give exactly zero behavior change."""

    def test_defaults_identical_to_explicit_zero(self):
        """gandalf_step with and without explicit eta_z=0.0, hyper_rz=1 must agree."""
        grid = SpectralGrid3D.create(Nx=16, Ny=16, Nz=16)
        state = initialize_random_spectrum(
            grid, M=4, amplitude=0.1, g_perturbation_amplitude=1e-3
        )
        dt = 0.01

        s_default = gandalf_step(state, dt, eta=0.01, v_A=1.0)
        s_explicit = gandalf_step(state, dt, eta=0.01, v_A=1.0, eta_z=0.0, hyper_rz=1)

        # Only the dynamically evolved fields are compared; B_parallel is
        # frozen pass-through state and is being removed entirely in
        # refactor/remove-b-parallel (PR #149).
        assert jnp.allclose(s_default.z_plus, s_explicit.z_plus, rtol=0.0, atol=0.0)
        assert jnp.allclose(s_default.z_minus, s_explicit.z_minus, rtol=0.0, atol=0.0)
        assert jnp.allclose(s_default.g, s_explicit.g, rtol=0.0, atol=0.0)

    def test_nz2_grid_runs_without_nan_with_defaults(self):
        """2D-style Nz=2 grid must not NaN: static idz_max guard skips kz damping.

        Without the static `if idz_max >= 1` guard, kz[0]=0 would give a
        divide-by-zero inf, and 0 * inf = NaN inside exp even for eta_z=0.
        """
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=2, Lx=1.0, Ly=1.0, Lz=1.0)
        state = initialize_orszag_tang(grid, M=0)

        current = state
        for _ in range(3):
            current = gandalf_step(current, dt=0.001, eta=0.0, v_A=1.0, nu=0.0)

        assert jnp.all(jnp.isfinite(current.z_plus))
        assert jnp.all(jnp.isfinite(current.z_minus))
        assert jnp.all(jnp.isfinite(current.g))


class TestKzDampingDecayRate:
    """Amplitude decay must match exp(-eta_z*(kz^2/kz_max^2)^rz * t) within ~1%."""

    @pytest.mark.parametrize("scheme", ["imex_rk222", "lawson_rk4"])
    @pytest.mark.parametrize("hyper_rz", [1, 2])
    def test_alfven_mode_decay_rate(self, hyper_rz, scheme):
        """Single Alfven mode at kz!=0: linear propagation is unitary, so |z|
        isolates the kz damping factor exactly."""
        grid = SpectralGrid3D.create(Nx=16, Ny=16, Nz=16)
        kz_mode = 2.0
        # M=0 keeps it fluid; small amplitude so nonlinearity is negligible
        # (a single perpendicular mode has vanishing Poisson bracket anyway).
        state = initialize_alfven_wave(grid, M=0, kz_mode=kz_mode, amplitude=1e-3)

        eta_z = 1.0
        dt = 0.01
        n_steps = 50

        current = state
        for _ in range(n_steps):
            current = gandalf_step(
                current,
                dt,
                eta=0.0,
                v_A=1.0,
                nu=0.0,
                eta_z=eta_z,
                hyper_rz=hyper_rz,
                scheme=scheme,
            )

        t = n_steps * dt
        kz_max_sq = _kz_max_squared(grid)
        expected = float(jnp.exp(-eta_z * (kz_mode**2 / kz_max_sq) ** hyper_rz * t))

        for f0, f1 in [(state.z_plus, current.z_plus), (state.z_minus, current.z_minus)]:
            amp0 = float(jnp.linalg.norm(f0))
            amp1 = float(jnp.linalg.norm(f1))
            assert amp0 > 0.0
            assert amp1 / amp0 == pytest.approx(expected, rel=1e-2), (
                f"kz damping decay mismatch (hyper_rz={hyper_rz}, scheme={scheme}): "
                f"measured {amp1 / amp0:.6f}, expected {expected:.6f}"
            )

    def test_kz_zero_mode_unaffected(self):
        """A kz=0 mode must not be damped by parallel dissipation."""
        grid = SpectralGrid3D.create(Nx=16, Ny=16, Nz=16)
        state = initialize_alfven_wave(grid, M=0, kz_mode=0.0, amplitude=1e-3)

        current = state
        for _ in range(20):
            current = gandalf_step(
                current, 0.01, eta=0.0, v_A=1.0, nu=0.0, eta_z=1.0, hyper_rz=2
            )

        amp0 = float(jnp.linalg.norm(state.z_plus))
        amp1 = float(jnp.linalg.norm(current.z_plus))
        assert amp1 / amp0 == pytest.approx(1.0, rel=1e-3)


class TestKzDampingHermiteMoments:
    """Hermite moments must be damped at the same kz-dependent rate, all m equally."""

    @pytest.mark.parametrize("scheme", ["imex_rk222", "lawson_rk4"])
    def test_g_damped_uniformly_across_moments(self, scheme):
        """With z+-=0 the g evolution is linear and block-diagonal per kz, so the
        damped run must equal the undamped run times the analytic kz factor,
        identically for every Hermite moment m."""
        grid = SpectralGrid3D.create(Nx=8, Ny=8, Nz=8)
        M = 4
        g = initialize_hermite_moments(grid, M=M, perturbation_amplitude=0.01)
        zeros = jnp.zeros((grid.Nz, grid.Ny, grid.Nx // 2 + 1), dtype=jnp.complex64)
        state = KRMHDState(
            z_plus=zeros,
            z_minus=zeros,
            B_parallel=zeros,  # required pre-#149; ignored extra kwarg after it merges
            g=g,
            M=M,
            beta_i=1.0,
            v_th=1.0,
            nu=0.0,
            Lambda=1.0,
            time=0.0,
            grid=grid,
        )
        assert float(jnp.linalg.norm(state.g)) > 0.0

        eta_z = 1.0
        hyper_rz = 2
        dt = 0.01
        n_steps = 20

        s_ref = state
        s_damped = state
        for _ in range(n_steps):
            s_ref = gandalf_step(s_ref, dt, eta=0.0, v_A=1.0, nu=0.0, scheme=scheme)
            s_damped = gandalf_step(
                s_damped,
                dt,
                eta=0.0,
                v_A=1.0,
                nu=0.0,
                eta_z=eta_z,
                hyper_rz=hyper_rz,
                scheme=scheme,
            )

        # z+- must remain zero (no spurious excitation)
        assert jnp.allclose(s_damped.z_plus, 0.0, atol=1e-12)
        assert jnp.allclose(s_damped.z_minus, 0.0, atol=1e-12)

        t = n_steps * dt
        kz_max_sq = _kz_max_squared(grid)
        kz_3d = grid.kz[:, jnp.newaxis, jnp.newaxis]
        expected_factor = jnp.exp(-eta_z * (kz_3d**2 / kz_max_sq) ** hyper_rz * t)

        # Same factor for every moment m (broadcast across the trailing m-axis)
        assert jnp.allclose(
            s_damped.g,
            s_ref.g * expected_factor[:, :, :, jnp.newaxis],
            rtol=1e-3,
            atol=1e-8,
        ), f"g not damped by the analytic kz factor (scheme={scheme})"

        # kz=0 plane untouched
        assert jnp.allclose(s_damped.g[0], s_ref.g[0], rtol=1e-5, atol=1e-10)

        # Streaming spread energy into moments beyond the seeded g_1, so the
        # uniform-in-m check above is non-trivial: verify several moments are live.
        moment_norms = jnp.sqrt(jnp.sum(jnp.abs(s_ref.g) ** 2, axis=(0, 1, 2)))
        assert int(jnp.sum(moment_norms > 0)) >= 3


class TestKzDampingValidation:
    """Validation mirrors the existing perpendicular eta checks."""

    @staticmethod
    def _small_state():
        grid = SpectralGrid3D.create(Nx=16, Ny=16, Nz=16)
        return initialize_alfven_wave(grid, M=0, kz_mode=1.0, amplitude=1e-3)

    def test_invalid_hyper_rz_raises(self):
        state = self._small_state()
        with pytest.raises(ValueError, match="hyper_rz"):
            gandalf_step(state, 0.01, eta=0.0, v_A=1.0, nu=0.0, eta_z=0.1, hyper_rz=3)

    def test_eta_z_overflow_raises(self):
        state = self._small_state()
        # eta_z * dt = 50 >= MAX_DAMPING_RATE_THRESHOLD
        with pytest.raises(ValueError, match="overflow"):
            gandalf_step(state, 0.01, eta=0.0, v_A=1.0, nu=0.0, eta_z=5000.0)

    def test_eta_z_moderate_rate_warns(self):
        state = self._small_state()
        # eta_z * dt = 25 in [20, 50): RuntimeWarning, same thresholds as eta
        with pytest.warns(RuntimeWarning, match="eta_z"):
            gandalf_step(state, 0.01, eta=0.0, v_A=1.0, nu=0.0, eta_z=2500.0)

    def test_eta_z_with_tiny_nz_raises(self):
        """eta_z > 0 requires (Nz-1)//3 >= 1: normalized kz damping is undefined
        when no nonzero kz mode exists inside the dealias boundary (original
        GANDALF guarded dampz with `if (Nz > 1)`)."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=2, Lx=1.0, Ly=1.0, Lz=1.0)
        state = initialize_orszag_tang(grid, M=0)
        with pytest.raises(ValueError, match="Nz"):
            gandalf_step(state, 0.001, eta=0.0, v_A=1.0, nu=0.0, eta_z=0.1)
