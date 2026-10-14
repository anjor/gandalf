"""
Tests for Hermitian pairs on the self-paired rfft planes (Issues #156, #165).

In rfftn storage [Nz, Ny, Nx//2+1] the kx=0 plane (and the kx=Nyquist plane
for even Nx) stores both partners (kx, ky, kz) and (kx, -ky, -kz). Reality
requires f(kx, -ky, -kz) = f*(kx, ky, kz). Any violation is an anti-Hermitian
"ghost": irfftn drops it, but it stays in the Fourier state and every
Fourier-space diagnostic counts it.

This module covers:
- spectral.enforce_hermitian / anti_hermitian_fraction on 3D and 4D arrays,
  for even and odd Nx
- every built-in initializer returning a ghost-free state
- io.load_checkpoint removing (or, on request, keeping) a ghost

SpectralGrid3D only admits even grid sizes, so odd Nx is exercised on raw
arrays through the helpers.
"""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from krmhd import (
    SpectralGrid3D,
    anti_hermitian_fraction,
    enforce_hermitian,
    enforce_hermitian_state,
    energy,
    initialize_alfven_wave,
    initialize_hermite_moments,
    initialize_orszag_tang,
    initialize_random_spectrum,
)
from krmhd.config import GridConfig, InitialConditionConfig, SimulationConfig
from krmhd.io import HERMITIAN_GHOST_TOLERANCE, load_checkpoint, save_checkpoint
from krmhd.physics import physical_to_elsasser
from krmhd.spectral import rfftn_inverse

# float32 states: an exactly paired plane gives 0; allow round-off headroom
ROUND_OFF = 1e-12


def _random_rfft(shape: tuple, seed: int = 0) -> jnp.ndarray:
    """Complex64 noise in rfftn layout, with independent draws at every entry."""
    rng = np.random.default_rng(seed)
    f = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    return jnp.asarray(f.astype(np.complex64))


def _irfftn(field, Nx: int) -> np.ndarray:
    """irfftn over the three spatial axes, in float64, for any trailing axes."""
    f = np.asarray(field, dtype=np.complex128)
    return np.fft.irfftn(f, s=(f.shape[0], f.shape[1], Nx), axes=(0, 1, 2))


def _round_trip(field, Nx: int) -> np.ndarray:
    """rfftn(irfftn(f)): the part of f that irfftn keeps."""
    return np.fft.rfftn(_irfftn(field, Nx), axes=(0, 1, 2))


def _assert_state_hermitian(state) -> None:
    Nx = state.grid.Nx
    for name, arr in (("z_plus", state.z_plus), ("z_minus", state.z_minus), ("g", state.g)):
        share = float(anti_hermitian_fraction(arr, Nx))
        assert share < ROUND_OFF, f"{name}: anti-Hermitian share {share:.3e} on kx=0/Nyquist planes"


# =============================================================================
# enforce_hermitian / anti_hermitian_fraction
# =============================================================================


@pytest.mark.parametrize("Nx", [8, 9])
@pytest.mark.parametrize("trailing", [(), (4,)])
class TestEnforceHermitian:
    """Projection onto Hermitian pairs, for 3D fields and 4D Hermite arrays."""

    def _field(self, Nx, trailing):
        return _random_rfft((6, 8, Nx // 2 + 1) + trailing)

    def test_matches_irfftn_round_trip(self, Nx, trailing):
        """The projection is exactly what irfftn keeps."""
        f = self._field(Nx, trailing)
        np.testing.assert_allclose(enforce_hermitian(f, Nx), _round_trip(f, Nx), atol=1e-5)

    def test_leaves_irfftn_unchanged(self, Nx, trailing):
        f = self._field(Nx, trailing)
        np.testing.assert_allclose(
            _irfftn(enforce_hermitian(f, Nx), Nx), _irfftn(f, Nx), atol=1e-6
        )

    def test_idempotent(self, Nx, trailing):
        h = enforce_hermitian(self._field(Nx, trailing), Nx)
        np.testing.assert_array_equal(enforce_hermitian(h, Nx), h)

    def test_removes_ghost(self, Nx, trailing):
        f = self._field(Nx, trailing)
        assert float(anti_hermitian_fraction(f, Nx)) > 0.3
        assert float(anti_hermitian_fraction(enforce_hermitian(f, Nx), Nx)) == 0.0

    def test_only_paired_planes_change(self, Nx, trailing):
        """kx > 0 planes are untouched; for odd Nx that includes the last index."""
        f = self._field(Nx, trailing)
        h = enforce_hermitian(f, Nx)
        last = -1 if Nx % 2 == 0 else None
        np.testing.assert_array_equal(h[:, :, 1:last], f[:, :, 1:last])

    def test_self_conjugate_entries_real(self, Nx, trailing):
        """Entries with ky, kz each 0 or Nyquist are their own partner."""
        f = self._field(Nx, trailing)
        h = np.asarray(enforce_hermitian(f, Nx))
        Nz, Ny = f.shape[:2]
        planes = [0, Nx // 2] if Nx % 2 == 0 else [0]
        for ix in planes:
            for iz in (0, Nz // 2):
                for iy in (0, Ny // 2):
                    np.testing.assert_array_equal(h[iz, iy, ix].imag, 0.0)
                    np.testing.assert_array_equal(h[iz, iy, ix].real, np.asarray(f)[iz, iy, ix].real)

    def test_fraction_matches_round_trip_loss(self, Nx, trailing):
        """anti_hermitian_fraction(total) equals the power irfftn discards."""
        f = self._field(Nx, trailing)
        lost = np.sum(np.abs(np.asarray(f, dtype=np.complex128) - _round_trip(f, Nx)) ** 2)
        total = np.sum(np.abs(np.asarray(f, dtype=np.complex128)) ** 2)
        share = float(anti_hermitian_fraction(f, Nx, relative_to="total"))
        assert share == pytest.approx(lost / total, rel=1e-4)


def test_enforce_hermitian_under_jit_and_vmap():
    """Nx is static, so the helper composes with jit and vmap."""
    Nx = 8
    f = _random_rfft((3, 4, 6, Nx // 2 + 1))  # leading batch axis

    @jax.jit
    def project(x):
        return enforce_hermitian(x, Nx)

    batched = jax.vmap(project)(f)
    for b in range(f.shape[0]):
        np.testing.assert_array_equal(batched[b], enforce_hermitian(f[b], Nx))


def test_projection_halves_expected_power_of_independent_draws():
    """E|f'|² = E|f|²/2 for independent equal-variance partners (same as Re f)."""
    N = 64
    f = _random_rfft((N, N, N // 2 + 1), seed=1)
    p_raw = float(jnp.sum(jnp.abs(f[:, :, 0]) ** 2))
    p_proj = float(jnp.sum(jnp.abs(enforce_hermitian(f, N)[:, :, 0]) ** 2))
    assert p_proj / p_raw == pytest.approx(0.5, abs=0.03)


def test_anti_hermitian_fraction_edge_cases():
    Nx = 8
    zero = jnp.zeros((4, 4, Nx // 2 + 1), dtype=jnp.complex64)
    assert float(anti_hermitian_fraction(zero, Nx)) == 0.0
    assert float(anti_hermitian_fraction(zero, Nx, relative_to="total")) == 0.0

    f = _random_rfft((4, 4, Nx // 2 + 1))
    planes = float(anti_hermitian_fraction(f, Nx))
    total = float(anti_hermitian_fraction(f, Nx, relative_to="total"))
    assert 0.0 < total < planes < 1.0

    with pytest.raises(ValueError, match="relative_to"):
        anti_hermitian_fraction(f, Nx, relative_to="kx0")
    with pytest.raises(ValueError, match="rfftn storage"):
        enforce_hermitian(f, 16)


def test_enforce_hermitian_state_keeps_real_space_and_scalars():
    grid = SpectralGrid3D.create(Nx=8, Ny=8, Nz=4)
    base = initialize_random_spectrum(grid, M=3, seed=3)
    shape = base.z_plus.shape
    dirty = base.model_copy(update={
        "z_plus": _random_rfft(shape, 1),
        "z_minus": _random_rfft(shape, 2),
        "g": _random_rfft(base.g.shape, 3),
        "time": 1.25,
    })
    clean = enforce_hermitian_state(dirty)

    _assert_state_hermitian(clean)
    for name in ("z_plus", "z_minus", "g"):
        np.testing.assert_allclose(
            _irfftn(getattr(clean, name), grid.Nx), _irfftn(getattr(dirty, name), grid.Nx), atol=1e-5
        )
    for name in ("M", "beta_i", "v_th", "nu", "Lambda", "time"):
        assert getattr(clean, name) == getattr(dirty, name)


def test_forcing_alias_is_spectral_helper():
    """forcing keeps its private name, bound to the shared implementation."""
    from krmhd import forcing, spectral

    assert forcing._enforce_rfft_hermitian is spectral.enforce_hermitian
    assert forcing._rfft_partner is spectral._rfft_partner


# =============================================================================
# Built-in initializers
# =============================================================================


GRIDS = [(16, 16, 16), (32, 16, 8), (8, 8, 2)]


@pytest.mark.parametrize("Nx,Ny,Nz", GRIDS)
@pytest.mark.parametrize("k_max", [None, 1e3])
def test_random_spectrum_hermitian(Nx, Ny, Nz, k_max):
    """k_max=1e3 fills the Nyquist plane too."""
    grid = SpectralGrid3D.create(Nx=Nx, Ny=Ny, Nz=Nz)
    state = initialize_random_spectrum(
        grid, M=4, k_max=k_max, seed=5, g_perturbation_amplitude=1e-3
    )
    if k_max is not None:
        assert float(jnp.sum(jnp.abs(state.z_plus[:, :, -1]) ** 2)) > 0
    _assert_state_hermitian(state)


def test_random_spectrum_keeps_kx_positive_draws():
    """The default seed reproduces the pre-#165 draws on every kx > 0 entry."""
    grid = SpectralGrid3D.create(Nx=16, Ny=16, Nz=8)
    seed, alpha, amplitude, k_min = 42, 5.0 / 3.0, 1.0, 1.0
    state = initialize_random_spectrum(grid, M=2, alpha=alpha, amplitude=amplitude, seed=seed)

    # Same draws as the initializer, before any Hermitian projection
    key = jax.random.PRNGKey(seed)
    key, k1, k2 = jax.random.split(key, 3)
    shape = (grid.Nz, grid.Ny, grid.Nx // 2 + 1)
    phase_phi = 2.0 * jnp.pi * jax.random.uniform(k1, shape=shape)
    phase_A = 2.0 * jnp.pi * jax.random.uniform(k2, shape=shape)
    k_mag = jnp.sqrt(
        grid.kx[None, None, :] ** 2 + grid.ky[None, :, None] ** 2 + grid.kz[:, None, None] ** 2
    )
    k_max = 2.0 / 3.0 * jnp.max(grid.kx)
    spectrum = jnp.where(
        (k_mag >= k_min) & (k_mag <= k_max), amplitude * k_mag ** (-alpha / 2.0), 0.0
    )
    phi = spectrum * jnp.exp(1.0j * phase_phi).astype(jnp.complex64)
    A = spectrum * jnp.exp(1.0j * phase_A).astype(jnp.complex64)
    z_plus, z_minus = physical_to_elsasser(phi, A)

    np.testing.assert_array_equal(state.z_plus[:, :, 1:-1], z_plus[:, :, 1:-1])
    np.testing.assert_array_equal(state.z_minus[:, :, 1:-1], z_minus[:, :, 1:-1])


def test_random_spectrum_paired_plane_modulus_and_phase():
    """On kx=0, |φ| equals the target spectrum exactly and the phase is uniform."""
    grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=32)
    k_mag = np.sqrt(
        np.asarray(grid.kx)[None, None, :] ** 2
        + np.asarray(grid.ky)[None, :, None] ** 2
        + np.asarray(grid.kz)[:, None, None] ** 2
    )
    band = ((k_mag >= 1.0) & (k_mag <= 10.0))[:, :, 0]

    phases = []
    for seed in range(8):
        # alpha=0: flat target |f| = 1 in band, 0 outside
        state = initialize_random_spectrum(grid, M=0, alpha=0.0, k_max=10.0, seed=seed)
        phi0 = np.asarray(state.phi[:, :, 0])
        np.testing.assert_allclose(np.abs(phi0[band]), 1.0, rtol=1e-6)
        assert np.all(phi0[~band] == 0)
        phases.append(np.angle(phi0[band]))
    phases = np.concatenate(phases)
    # Uniform phase: low circular moments vanish (Hermitian pairs make the
    # sine moments cancel exactly, so only the cosines are informative)
    assert abs(np.mean(np.cos(phases))) < 0.05
    assert abs(np.mean(np.cos(2 * phases))) < 0.05


@pytest.mark.parametrize("Nx,Ny,Nz", GRIDS)
@pytest.mark.parametrize(
    "kx_mode,ky_mode,kz_mode",
    [
        (1.0, 0.0, 1.0),  # default-like kx > 0 mode
        (0.0, 2.0, 1.0),  # kx=0 plane
        (0.0, -1.0, -1.0),  # kx=0 plane, negative ky/kz indices
        (0.0, 0.0, 1.0),  # kx=0, k_perp=0 (configs/alfven_wave.yaml)
        ("nyquist", 1.0, 1.0),  # kx=Nyquist plane
    ],
)
def test_alfven_wave_hermitian(Nx, Ny, Nz, kx_mode, ky_mode, kz_mode):
    grid = SpectralGrid3D.create(Nx=Nx, Ny=Ny, Nz=Nz)
    if kx_mode == "nyquist":
        kx_mode = float(grid.kx[-1])
    state = initialize_alfven_wave(grid, M=4, kx_mode=kx_mode, ky_mode=ky_mode, kz_mode=kz_mode)
    _assert_state_hermitian(state)


def test_alfven_wave_kx0_sets_conjugate_partner():
    grid = SpectralGrid3D.create(Nx=16, Ny=16, Nz=16)
    a = 0.1
    state = initialize_alfven_wave(grid, M=2, kx_mode=0.0, ky_mode=2.0, kz_mode=1.0, amplitude=a)
    phi = np.asarray(state.phi)
    A = np.asarray(state.A_parallel)
    # Mode (kx, ky, kz) = (0, 2, 1) and its stored partner (0, -2, -1)
    assert phi[1, 2, 0] == pytest.approx(1j * a)
    assert A[1, 2, 0] == pytest.approx(a)
    assert phi[-1, -2, 0] == pytest.approx(np.conj(phi[1, 2, 0]))
    assert A[-1, -2, 0] == pytest.approx(np.conj(A[1, 2, 0]))
    assert np.count_nonzero(phi) == 2 and np.count_nonzero(A) == 2


def test_alfven_wave_kx0_matches_kx_positive_wave():
    """Same |k⊥|, kz and amplitude: same energy and real-space amplitude."""
    grid = SpectralGrid3D.create(Nx=16, Ny=16, Nz=16)
    wave_x = initialize_alfven_wave(grid, M=2, kx_mode=2.0, ky_mode=0.0, kz_mode=1.0)
    wave_y = initialize_alfven_wave(grid, M=2, kx_mode=0.0, ky_mode=2.0, kz_mode=1.0)

    E_x, E_y = energy(wave_x), energy(wave_y)
    for key in ("magnetic", "kinetic", "total"):
        assert E_y[key] == pytest.approx(E_x[key], rel=1e-6)

    shape = (grid.Nz, grid.Ny, grid.Nx)
    for name in ("phi", "A_parallel"):
        amp_x = float(jnp.max(jnp.abs(rfftn_inverse(getattr(wave_x, name), *shape))))
        amp_y = float(jnp.max(jnp.abs(rfftn_inverse(getattr(wave_y, name), *shape))))
        assert amp_y == pytest.approx(amp_x, rel=1e-5)


def test_alfven_wave_self_conjugate_mode_is_real():
    """(kx, ky, kz) = (0, Ny/2, 0) is its own partner: only the cosine survives."""
    grid = SpectralGrid3D.create(Nx=16, Ny=16, Nz=16)
    a = 0.1
    state = initialize_alfven_wave(
        grid, M=2, kx_mode=0.0, ky_mode=float(grid.ky[grid.Ny // 2]), kz_mode=0.0, amplitude=a
    )
    A = np.asarray(state.A_parallel)
    assert A[0, grid.Ny // 2, 0] == pytest.approx(2.0 * a)
    assert np.count_nonzero(A) == 1
    assert np.count_nonzero(np.asarray(state.phi)) == 0  # Re(i·a) = 0
    _assert_state_hermitian(state)


def test_alfven_wave_default_mode_unchanged():
    """A kx > 0 wave is one entry, with phi = i·a and A∥ = a exactly."""
    grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)
    a = 0.1
    state = initialize_alfven_wave(grid, M=2, amplitude=a)
    z_plus = np.asarray(state.z_plus)
    z_minus = np.asarray(state.z_minus)
    (iz, iy, ix), = np.argwhere(z_plus != 0)
    assert (iz, iy, ix) == (1, 0, 1)
    assert z_plus[1, 0, 1] == np.complex64(a + 1j * a)
    assert z_minus[1, 0, 1] == np.complex64(-a + 1j * a)


def test_orszag_tang_and_hermite_moments_hermitian():
    grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=2)
    _assert_state_hermitian(initialize_orszag_tang(grid, M=2))

    grid = SpectralGrid3D.create(Nx=16, Ny=16, Nz=8)
    g = initialize_hermite_moments(grid, M=4, perturbation_amplitude=0.1, seed=7)
    assert float(jnp.sum(jnp.abs(g[:, :, 0, 1]) ** 2)) > 0
    # rfftn of real data: Hermitian up to float32 FFT round-off
    assert float(anti_hermitian_fraction(g, grid.Nx)) < 1e-10


@pytest.mark.parametrize(
    "ic",
    [
        InitialConditionConfig(type="random_spectrum", M=4, k_max=1e3, g_perturbation_amplitude=1e-3),
        InitialConditionConfig(type="alfven_wave", M=4, k_wave=[0.0, 2.0, 1.0]),
        InitialConditionConfig(type="alfven_wave", M=4, k_wave=[0.0, 0.0, 1.0]),
        InitialConditionConfig(type="orszag_tang", M=2),
        InitialConditionConfig(type="zero", M=2),
    ],
    ids=lambda ic: f"{ic.type}-{ic.k_wave}",
)
def test_config_initial_states_hermitian(ic):
    config = SimulationConfig(grid=GridConfig(Nx=16, Ny=16, Nz=8), initial_condition=ic)
    grid = config.create_grid()
    _assert_state_hermitian(config.create_initial_state(grid))


# =============================================================================
# load_checkpoint
# =============================================================================


@pytest.fixture
def clean_state():
    grid = SpectralGrid3D.create(Nx=16, Ny=16, Nz=8)
    return initialize_random_spectrum(grid, M=3, seed=11, g_perturbation_amplitude=1e-2)


def _with_ghost(state, fields=("z_plus", "z_minus", "g")):
    """Add a purely anti-Hermitian part on the kx=0/Nyquist planes."""
    Nx = state.grid.Nx
    update = {}
    for i, name in enumerate(fields):
        arr = getattr(state, name)
        noise = 0.1 * _random_rfft(arr.shape, seed=20 + i)
        update[name] = arr + (noise - enforce_hermitian(noise, Nx))
    return state.model_copy(update=update)


def test_load_checkpoint_removes_ghost(clean_state, tmp_path):
    ghosted = _with_ghost(clean_state)
    filename = tmp_path / "ghost.h5"
    save_checkpoint(ghosted, str(filename))

    with pytest.warns(UserWarning, match=r"#156/#165") as record:
        loaded, _, _ = load_checkpoint(str(filename))
    (ghost_warning,) = [w for w in record if "#156/#165" in str(w.message)]
    assert "enforce_hermitian=False" in str(ghost_warning.message)
    assert ghost_warning.filename == __file__  # stacklevel points at the caller

    _assert_state_hermitian(loaded)
    for name in ("z_plus", "z_minus", "g"):
        np.testing.assert_allclose(getattr(loaded, name), getattr(clean_state, name), atol=1e-6)


def test_load_checkpoint_ghost_in_g_only(clean_state, tmp_path):
    filename = tmp_path / "ghost_g.h5"
    save_checkpoint(_with_ghost(clean_state, fields=("g",)), str(filename))

    with pytest.warns(UserWarning, match="anti-Hermitian"):
        loaded, _, _ = load_checkpoint(str(filename))
    _assert_state_hermitian(loaded)
    np.testing.assert_array_equal(loaded.z_plus, clean_state.z_plus)


def test_load_checkpoint_opt_out_keeps_ghost(clean_state, tmp_path):
    ghosted = _with_ghost(clean_state)
    filename = tmp_path / "ghost_raw.h5"
    save_checkpoint(ghosted, str(filename))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        loaded, _, _ = load_checkpoint(str(filename), enforce_hermitian=False)
    for name in ("z_plus", "z_minus", "g"):
        np.testing.assert_array_equal(getattr(loaded, name), getattr(ghosted, name))
    share = float(anti_hermitian_fraction(loaded.z_plus, loaded.grid.Nx, relative_to="total"))
    assert share > HERMITIAN_GHOST_TOLERANCE


def test_load_checkpoint_clean_is_silent_and_exact(clean_state, tmp_path):
    filename = tmp_path / "clean.h5"
    save_checkpoint(clean_state, str(filename))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        loaded, _, _ = load_checkpoint(str(filename))
    for name in ("z_plus", "z_minus", "g"):
        np.testing.assert_array_equal(getattr(loaded, name), getattr(clean_state, name))
