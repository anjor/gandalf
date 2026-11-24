"""
Tests for the exact slice-advection radius (slice_advection_radius and
compute_cfl_timestep(mode="exact"), Issue #189).

The radius ρ_c is the spectral radius of a ↦ P{Φ + cΨ, a} on the 2/3 band.
The tests diagonalise that operator densely on small grids, compare with a
scipy eigsh reference (the krmhd-research method) on turbulent-like fields,
and check that the 2√2/ρ step limit is sharp for RK4 on the frozen operator
and for the real lawson_rk4 and imex_ars343 steps.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import enable_x64
from scipy.sparse.linalg import LinearOperator, eigsh

from krmhd.physics import KRMHDState, poisson_bracket_3d
from krmhd.spectral import SpectralGrid3D
from krmhd.timestepping import (
    RK4_IMAGINARY_AXIS_LIMIT,
    _slice_advection_rhs,
    _streaming_eigenvalue_extremes,
    compute_cfl_timestep,
    gandalf_step,
    slice_advection_radius,
)


def _turbulent_fields(grid: SpectralGrid3D, seed: int, dtype=np.complex128):
    """z± with a k⊥^(-5/3)-like spectrum, k∥ ≲ k⊥^(2/3), random phases.

    Filtered real white noise, so the Hermitian pairs on the kx = 0 plane are
    exact. Content outside the 2/3 band is kept on purpose: the radius must
    project it away.
    """
    rng = np.random.default_rng(seed)
    nx = np.fft.rfftfreq(grid.Nx, 1.0 / grid.Nx)[None, None, :]
    ny = np.fft.fftfreq(grid.Ny, 1.0 / grid.Ny)[None, :, None]
    nz = np.fft.fftfreq(grid.Nz, 1.0 / grid.Nz)[:, None, None]
    n_perp = np.sqrt(nx**2 + ny**2)
    shape = np.maximum(n_perp, 1.0) ** (-11.0 / 6.0) * np.exp(
        -np.abs(nz) / (1.0 + np.maximum(n_perp, 1.0) ** (2.0 / 3.0))
    )
    fields = []
    for _ in range(2):
        noise = rng.standard_normal((grid.Nz, grid.Ny, grid.Nx))
        fields.append(jnp.asarray((np.fft.rfftn(noise) * shape).astype(dtype)))
    return fields


def _slice_speeds(M: int, Lambda: float, closure: str, beta_i: float) -> np.ndarray:
    lam_lo, lam_hi, _ = _streaming_eigenvalue_extremes(M, Lambda, closure)
    return math.sqrt(beta_i) * np.array([lam_lo, lam_hi])


def _radius(grid, z_plus, z_minus, speeds, **kw):
    return slice_advection_radius(
        z_plus, z_minus, jnp.asarray(speeds), grid.kx, grid.ky, grid.dealias_mask,
        grid.Nz, grid.Ny, grid.Nx, **kw,
    )


def _dense_bracket_operator(grid, z_plus, z_minus, c):
    """Real-space matrix of a ↦ P{Φ + cΨ, P a}, built from poisson_bracket_3d.

    Φ and Ψ are projected onto the band. Column i is the image of the i-th
    grid delta function, so the matrix acts on the Euclidean real space.
    """
    Nz, Ny, Nx = grid.Nz, grid.Ny, grid.Nx
    n = Nz * Ny * Nx
    band = grid.dealias_mask.at[0, 0, 0].set(False)
    f = ((z_plus + z_minus) / 2 + c * (z_plus - z_minus) / 2) * grid.dealias_mask
    deltas = jnp.eye(n, dtype=jnp.real(z_plus).dtype).reshape(n, Nz, Ny, Nx)
    a = jnp.fft.rfftn(deltas, axes=(1, 2, 3)) * band
    bracket = jax.vmap(
        lambda ai: poisson_bracket_3d(f, ai, grid.kx, grid.ky, Nz, Ny, Nx, grid.dealias_mask)
    )(a) * band
    return np.asarray(jnp.fft.irfftn(bracket, s=(Nz, Ny, Nx), axes=(1, 2, 3)).reshape(n, n)).T


class TestSliceAdvectionRadius:
    """slice_advection_radius against dense and eigsh references."""

    @pytest.mark.parametrize("closure", ["zero", "symmetric"])
    @pytest.mark.parametrize("dims", [(8, 8, 4), (12, 10, 6)])
    def test_matches_dense_operator(self, dims, closure):
        """ρ equals the largest |eigenvalue| of the dense operator, which is skew."""
        Nx, Ny, Nz = dims
        with enable_x64():
            grid = SpectralGrid3D.create(Nx=Nx, Ny=Ny, Nz=Nz)
            z_plus, z_minus = _turbulent_fields(grid, seed=Nx + len(closure))
            speeds = _slice_speeds(8, math.sqrt(5.0), closure, beta_i=0.7)
            rad = _radius(grid, z_plus, z_minus, speeds, tol=1e-12)
            assert bool(jnp.all(rad.converged))
            for j, c in enumerate(speeds):
                D = _dense_bracket_operator(grid, z_plus, z_minus, c)
                eigs = np.linalg.eigvals(D)
                rho_dense = np.abs(eigs).max()
                # Skew-adjoint on the band: ρ = ‖A‖ and the spectrum is imaginary
                assert np.abs(D + D.T).max() <= 1e-12 * np.abs(D).max()
                assert np.abs(eigs.real).max() <= 1e-12 * rho_dense
                assert float(rad.rho[j]) == pytest.approx(rho_dense, rel=1e-10)

    def test_float32_matches_float64(self):
        grid = SpectralGrid3D.create(Nx=12, Ny=10, Nz=6)
        speeds = _slice_speeds(8, math.sqrt(5.0), "zero", beta_i=1.0)
        with enable_x64():
            grid64 = SpectralGrid3D.create(Nx=12, Ny=10, Nz=6)
            zp64, zm64 = _turbulent_fields(grid64, seed=3)
            rho64 = np.asarray(_radius(grid64, zp64, zm64, speeds, tol=1e-12).rho)
        zp, zm = _turbulent_fields(grid, seed=3, dtype=np.complex64)
        rad = _radius(grid, zp, zm, speeds)
        assert rad.rho.dtype == jnp.float32
        assert bool(jnp.all(rad.converged))
        np.testing.assert_allclose(np.asarray(rad.rho), rho64, rtol=1e-5)

    def test_matches_eigsh_reference(self):
        """Within tol/2 of scipy eigsh on -A², the krmhd-research reference."""
        tol = 1e-3
        with enable_x64():
            grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=32)
            Nz, Ny, Nx = grid.Nz, grid.Ny, grid.Nx
            z_plus, z_minus = _turbulent_fields(grid, seed=7)
            speeds = _slice_speeds(32, math.sqrt(5.0), "zero", beta_i=1.0)
            rad = _radius(grid, z_plus, z_minus, speeds, tol=tol)
            assert bool(jnp.all(rad.converged))
            mask = grid.dealias_mask
            for j, c in enumerate(speeds):
                f = ((z_plus + z_minus) / 2 + c * (z_plus - z_minus) / 2) * mask

                @jax.jit
                def apply(v, f=f):
                    a = jnp.fft.rfftn(v.reshape(Nz, Ny, Nx)) * mask
                    b = poisson_bracket_3d(f, a, grid.kx, grid.ky, Nz, Ny, Nx, mask)
                    return jnp.fft.irfftn(b, s=(Nz, Ny, Nx)).ravel()

                op = LinearOperator(
                    (Nz * Ny * Nx,) * 2,
                    matvec=lambda v: -np.asarray(apply(apply(jnp.asarray(v.ravel())))),
                    dtype=float,
                )
                v0 = np.random.default_rng(0).standard_normal(Nz * Ny * Nx)
                lam = eigsh(op, k=1, which="LA", tol=1e-10, v0=v0, return_eigenvectors=False)
                assert float(rad.rho[j]) == pytest.approx(math.sqrt(lam[0]), rel=tol / 2)

    def test_runs_under_jit_with_traced_fields(self):
        grid = SpectralGrid3D.create(Nx=16, Ny=16, Nz=8)
        zp, zm = _turbulent_fields(grid, seed=1, dtype=np.complex64)
        speeds = jnp.asarray(_slice_speeds(16, math.sqrt(5.0), "zero", 1.0), jnp.float32)

        @jax.jit
        def rho(z_plus, z_minus, c, tol):
            return _radius(grid, z_plus, z_minus, c, tol=tol).rho

        np.testing.assert_allclose(
            np.asarray(rho(zp, zm, speeds, 1e-3)),
            np.asarray(_radius(grid, zp, zm, speeds).rho),
            rtol=1e-6,
        )

    def test_zero_fields(self):
        grid = SpectralGrid3D.create(Nx=16, Ny=16, Nz=8)
        zeros = jnp.zeros((8, 16, 9), jnp.complex64)
        rad = _radius(grid, zeros, zeros, np.array([-3.0, 3.0]))
        assert np.all(np.asarray(rad.rho) == 0.0)
        assert bool(jnp.all(rad.converged))


class TestSharpStabilityLimit:
    """dt = 2√2/ρ is the exact RK4 limit for the frozen slice operator."""

    @staticmethod
    def _rk4_growth(advect, a0, dt, n_steps):
        """‖a_n‖/‖a_0‖ for classical RK4 on da/dt = advect(a)."""

        @jax.jit
        def run(a):
            def step(_, a):
                k1 = advect(a)
                k2 = advect(a + dt / 2 * k1)
                k3 = advect(a + dt / 2 * k2)
                k4 = advect(a + dt * k3)
                return a + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

            return jax.lax.fori_loop(0, n_steps, step, a)

        norm = lambda a: float(jnp.linalg.norm(jnp.fft.irfftn(a, axes=(-3, -2, -1))))
        return norm(run(a0)) / norm(a0)

    def test_rk4_on_frozen_slice_operator(self):
        with enable_x64():
            grid = SpectralGrid3D.create(Nx=16, Ny=16, Nz=8)
            Nz, Ny, Nx = grid.Nz, grid.Ny, grid.Nx
            z_plus, z_minus = _turbulent_fields(grid, seed=11)
            z_plus, z_minus = z_plus * grid.dealias_mask, z_minus * grid.dealias_mask
            speeds = _slice_speeds(16, math.sqrt(5.0), "zero", beta_i=1.0)
            rad = _radius(grid, z_plus, z_minus, speeds, tol=1e-10)
            noise = np.random.default_rng(0).standard_normal((Nz, Ny, Nx))
            a0 = (jnp.fft.rfftn(noise) * grid.dealias_mask)[jnp.newaxis]
            for j, c in enumerate(speeds):
                c_j = jnp.array([c])

                def advect(a, c_j=c_j):
                    return _slice_advection_rhs(
                        z_plus, z_minus, a, c_j, grid.kx, grid.ky, grid.dealias_mask, Nz, Ny, Nx
                    )

                dt_limit = RK4_IMAGINARY_AXIS_LIMIT / float(rad.rho[j])
                assert self._rk4_growth(advect, a0, 0.95 * dt_limit, 200) <= 1.0 + 1e-10
                # |R(iy)| ≈ 1.41 per step at y = 1.05·2√2 for the top mode
                assert self._rk4_growth(advect, a0, 1.05 * dt_limit, 200) > 1e6

    @staticmethod
    def _frozen_shear_state(g_on_kz0_plane: bool) -> KRMHDState:
        """z± = (1 ± 1/2)Φ with Φ = Φ(x) and random g, η = ν = 0.

        Every Elsasser bracket has a ∂y z± = 0 factor, so z± stay exactly
        fixed, and slice advection commutes with streaming. The slice-basis
        norm of g is then conserved by the continuous dynamics.
        """
        grid = SpectralGrid3D.create(Nx=16, Ny=16, Nz=8)
        x = np.arange(16) / 16
        phi_real = np.broadcast_to(
            np.cos(2 * np.pi * x) + 0.5 * np.sin(4 * np.pi * x), (8, 16, 16)
        )
        phi = jnp.fft.rfftn(phi_real).astype(jnp.complex64)
        rng = np.random.default_rng(0)
        shape = (8, 16, 9, 9)
        g = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) * np.asarray(
            grid.dealias_mask
        )[..., None]
        if g_on_kz0_plane:
            g[1:] = 0.0
        from krmhd.physics import enforce_hermitian_state

        return enforce_hermitian_state(KRMHDState(
            z_plus=1.5 * phi, z_minus=0.5 * phi, g=jnp.asarray(g, jnp.complex64), M=8,
            beta_i=1.0, v_th=1.0, nu=0.0, Lambda=math.sqrt(5.0), time=0.0, grid=grid,
        ))

    @staticmethod
    def _slice_norm(state: KRMHDState) -> float:
        from krmhd.hermite import compute_streaming_eigensystem

        _, _, _, P_inv = compute_streaming_eigensystem(state.M, state.Lambda, "zero")
        h = jnp.einsum("jm,zyxm->jzyx", jnp.real(P_inv), state.g)
        return float(jnp.linalg.norm(jnp.fft.irfftn(h, axes=(-3, -2, -1))))

    @pytest.mark.parametrize(
        "scheme, g_on_kz0_plane",
        [("lawson_rk4", False), ("imex_ars343", True)],
        ids=["lawson_rk4", "imex_ars343_no_streaming"],
    )
    def test_gandalf_step_threshold(self, scheme, g_on_kz0_plane):
        """The real step is bounded at 0.95·2√2/ρ and grows at 1.05·2√2/ρ.

        imex_ars343 meets the bound only without streaming (g on the kz = 0
        plane); see compute_cfl_timestep's Physics notes.
        """
        state0 = self._frozen_shear_state(g_on_kz0_plane)
        grid = state0.grid
        speeds = _slice_speeds(state0.M, state0.Lambda, "zero", state0.beta_i)
        rho = float(jnp.max(_radius(grid, state0.z_plus, state0.z_minus, speeds, tol=1e-6).rho))
        n0 = self._slice_norm(state0)
        ratios = {}
        for factor in (0.95, 1.05):
            state = state0
            for _ in range(60):
                state = gandalf_step(
                    state, factor * RK4_IMAGINARY_AXIS_LIMIT / rho, eta=0.0, v_A=1.0,
                    nu=0.0, scheme=scheme,
                )
            assert float(jnp.max(jnp.abs(state.z_plus - state0.z_plus))) == 0.0
            ratios[factor] = self._slice_norm(state) / n0
        assert ratios[0.95] <= 1.0 + 1e-4
        assert ratios[1.05] > 1e3


class TestCFLExactMode:
    """compute_cfl_timestep(mode="exact")."""

    @staticmethod
    def _state(M=16, beta_i=1.0, Lambda=math.sqrt(5.0), amplitude=1000.0, seed=5):
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)
        z_plus, z_minus = _turbulent_fields(grid, seed=seed, dtype=np.complex64)
        z_plus, z_minus = z_plus * grid.dealias_mask, z_minus * grid.dealias_mask
        return KRMHDState(
            z_plus=amplitude * z_plus, z_minus=amplitude * z_minus,
            g=jnp.zeros(z_plus.shape + (M + 1,), jnp.complex64), M=M, beta_i=beta_i,
            v_th=1.0, nu=0.0, Lambda=Lambda, time=0.0, grid=grid,
        )

    def test_exact_dt_is_safety_times_rk4_limit(self):
        state = self._state()
        safety = 0.5
        dt_exact = compute_cfl_timestep(state, v_A=1.0, cfl_safety=safety, mode="exact")
        dt_courant = compute_cfl_timestep(state, v_A=1.0, cfl_safety=safety)
        speeds = _slice_speeds(state.M, state.Lambda, "zero", state.beta_i)
        rho = float(jnp.max(_radius(state.grid, state.z_plus, state.z_minus, speeds).rho))
        assert dt_exact == pytest.approx(safety * RK4_IMAGINARY_AXIS_LIMIT / rho, rel=1e-6)
        # The Courant estimate overestimates ρ for the Hermite slices
        assert dt_exact > 1.2 * dt_courant

    def test_elsasser_and_alfven_terms_still_bind(self):
        """Small β_i: the slices move slower than z±, so both modes agree."""
        state = self._state(beta_i=1e-4)
        dt_exact = compute_cfl_timestep(state, v_A=1.0, mode="exact")
        assert dt_exact == pytest.approx(compute_cfl_timestep(state, v_A=1.0), rel=1e-6)

    def test_M0_and_zero_fields_match_courant(self):
        for state in (self._state(M=0), self._state(amplitude=0.0)):
            assert compute_cfl_timestep(state, v_A=1.0, mode="exact") == pytest.approx(
                compute_cfl_timestep(state, v_A=1.0), rel=1e-6
            )

    def test_closure_changes_the_radius(self):
        state = self._state(M=8, beta_i=10.0)
        dt_zero = compute_cfl_timestep(state, v_A=1.0, closure="zero", mode="exact")
        dt_sym = compute_cfl_timestep(state, v_A=1.0, closure="symmetric", mode="exact")
        # The symmetric closure has larger |λ| extremes, hence faster slices
        assert dt_sym < dt_zero

    def test_rejects_complex_eigenvalues_and_unknown_mode(self):
        with pytest.raises(ValueError, match="complex"):
            compute_cfl_timestep(self._state(Lambda=0.5), v_A=1.0, mode="exact")
        with pytest.raises(ValueError, match="mode"):
            compute_cfl_timestep(self._state(), v_A=1.0, mode="lanczos")

    def test_falls_back_to_courant_without_convergence(self, monkeypatch):
        import krmhd.timestepping as ts

        state = self._state()
        real = ts.slice_advection_radius

        def unconverged(*args, **kwargs):
            out = real(*args, **kwargs)
            return out._replace(converged=jnp.zeros_like(out.converged))

        monkeypatch.setattr(ts, "slice_advection_radius", unconverged)
        with pytest.warns(RuntimeWarning, match="did not converge"):
            dt = compute_cfl_timestep(state, v_A=1.0, mode="exact")
        assert dt == pytest.approx(compute_cfl_timestep(state, v_A=1.0), rel=1e-6)
