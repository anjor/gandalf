"""
Unit tests for balanced Elsasser forcing.

Tests the force_alfven_modes_balanced() function implementation,
focusing on:
- Hermitian symmetry enforcement (rfft reality condition)
- max_nz restriction
- nz=0 plane inclusion/exclusion
- correlation parameter behavior
- Reality condition preservation
"""

import jax
import jax.numpy as jnp
import pytest

from krmhd.forcing import force_alfven_modes_balanced
from krmhd.physics import KRMHDState
from krmhd.spectral import SpectralGrid3D


def create_zero_state(grid: SpectralGrid3D, M: int = 10) -> KRMHDState:
    """Helper to create a zero-initialized KRMHDState for testing."""
    z_plus = jnp.zeros((grid.Nz, grid.Ny, grid.Nx // 2 + 1), dtype=complex)
    z_minus = jnp.zeros((grid.Nz, grid.Ny, grid.Nx // 2 + 1), dtype=complex)
    B_parallel = jnp.zeros((grid.Nz, grid.Ny, grid.Nx // 2 + 1), dtype=complex)
    g = jnp.zeros((M + 1, grid.Nz, grid.Ny, grid.Nx // 2 + 1), dtype=complex)

    return KRMHDState(
        z_plus=z_plus,
        z_minus=z_minus,
        B_parallel=B_parallel,
        g=g,
        M=M,
        beta_i=1.0,
        v_th=1.0,
        nu=0.0,
        Lambda=1.0,
        time=0.0,
        grid=grid,
    )


def test_hermitian_symmetry_kx_zero_plane():
    """Test that kx=0 plane is real-valued (Hermitian symmetry)."""
    grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=32)
    state = create_zero_state(grid, M=10)

    key = jax.random.PRNGKey(42)
    amplitude = 0.1
    dt = 0.01

    # Force with balanced Elsasser
    state_forced, _ = force_alfven_modes_balanced(
        state,
        amplitude=amplitude,
        n_min=1,
        n_max=2,
        dt=dt,
        key=key,
        max_nz=1,
    )

    # Check kx=0 plane is real-valued in z+ and z-
    # For rfft, kx=0 is at index 0
    z_plus_kx0 = state_forced.z_plus[:, :, 0]
    z_minus_kx0 = state_forced.z_minus[:, :, 0]

    # Imaginary parts should be zero (within numerical precision)
    assert jnp.allclose(jnp.imag(z_plus_kx0), 0.0, atol=1e-10), \
        "kx=0 plane in z+ has non-zero imaginary part (violates Hermitian symmetry)"
    assert jnp.allclose(jnp.imag(z_minus_kx0), 0.0, atol=1e-10), \
        "kx=0 plane in z- has non-zero imaginary part (violates Hermitian symmetry)"


def test_hermitian_symmetry_kx_nyquist_plane():
    """Test that kx=Nyquist plane is real-valued (if Nx is even)."""
    grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=32)  # Nx=32 is even
    state = create_zero_state(grid, M=10)

    key = jax.random.PRNGKey(43)
    amplitude = 0.1
    dt = 0.01

    state_forced, _ = force_alfven_modes_balanced(
        state,
        amplitude=amplitude,
        n_min=1,
        n_max=2,
        dt=dt,
        key=key,
        max_nz=1,
    )

    # For rfft with Nx=32, Nyquist is at index Nx//2 = 16
    kx_nyquist_idx = grid.Nx // 2
    z_plus_nyquist = state_forced.z_plus[:, :, kx_nyquist_idx]
    z_minus_nyquist = state_forced.z_minus[:, :, kx_nyquist_idx]

    assert jnp.allclose(jnp.imag(z_plus_nyquist), 0.0, atol=1e-10), \
        "kx=Nyquist plane in z+ has non-zero imaginary part"
    assert jnp.allclose(jnp.imag(z_minus_nyquist), 0.0, atol=1e-10), \
        "kx=Nyquist plane in z- has non-zero imaginary part"


def test_max_nz_restriction():
    """Test that only |n_z| <= max_nz modes are forced."""
    grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=32)
    state = create_zero_state(grid, M=10)

    key = jax.random.PRNGKey(44)
    amplitude = 0.1
    dt = 0.01
    max_nz = 2  # Allow nz = -2, -1, 0, 1, 2

    state_forced, _ = force_alfven_modes_balanced(
        state,
        amplitude=amplitude,
        n_min=1,
        n_max=2,
        dt=dt,
        key=key,
        max_nz=max_nz,
        include_nz0=True,
    )

    # Check that modes with |n_z| > max_nz are zero
    # For Nz=32, nz ranges from -16 to 15 (FFTSHIFT convention)
    # Modes with |nz| > 2 should be zero

    # Get mode numbers
    nz = jnp.fft.fftfreq(grid.Nz, d=1.0/grid.Nz)  # Mode numbers

    # Find indices where |nz| > max_nz
    high_nz_mask = jnp.abs(nz) > max_nz

    # Check z+ and z- at these high-nz modes
    z_plus_high_nz = state_forced.z_plus[high_nz_mask, :, :]
    z_minus_high_nz = state_forced.z_minus[high_nz_mask, :, :]

    assert jnp.allclose(z_plus_high_nz, 0.0, atol=1e-10), \
        f"z+ has non-zero modes at |n_z| > {max_nz}"
    assert jnp.allclose(z_minus_high_nz, 0.0, atol=1e-10), \
        f"z- has non-zero modes at |n_z| > {max_nz}"


def test_nz0_exclusion():
    """Test that nz=0 plane is NOT forced when include_nz0=False."""
    grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=32)
    state = create_zero_state(grid, M=10)

    key = jax.random.PRNGKey(45)
    amplitude = 0.1
    dt = 0.01

    state_forced, _ = force_alfven_modes_balanced(
        state,
        amplitude=amplitude,
        n_min=1,
        n_max=2,
        dt=dt,
        key=key,
        max_nz=1,
        include_nz0=False,  # Exclude nz=0
    )

    # nz=0 is at index 0 in Fourier space
    z_plus_nz0 = state_forced.z_plus[0, :, :]
    z_minus_nz0 = state_forced.z_minus[0, :, :]

    assert jnp.allclose(z_plus_nz0, 0.0, atol=1e-10), \
        "z+ has non-zero nz=0 plane when include_nz0=False"
    assert jnp.allclose(z_minus_nz0, 0.0, atol=1e-10), \
        "z- has non-zero nz=0 plane when include_nz0=False"


def test_nz0_inclusion():
    """Test that nz=0 plane IS forced when include_nz0=True."""
    grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=32)
    state = create_zero_state(grid, M=10)

    key = jax.random.PRNGKey(46)
    amplitude = 0.1
    dt = 0.01

    state_forced, _ = force_alfven_modes_balanced(
        state,
        amplitude=amplitude,
        n_min=1,
        n_max=2,
        dt=dt,
        key=key,
        max_nz=1,
        include_nz0=True,  # Include nz=0
    )

    # nz=0 is at index 0
    z_plus_nz0 = state_forced.z_plus[0, :, :]
    z_minus_nz0 = state_forced.z_minus[0, :, :]

    # Should have some non-zero entries (within forcing band)
    # Check that at least some modes are non-zero
    assert not jnp.allclose(z_plus_nz0, 0.0, atol=1e-10), \
        "z+ has zero nz=0 plane when include_nz0=True (expected forcing)"
    assert not jnp.allclose(z_minus_nz0, 0.0, atol=1e-10), \
        "z- has zero nz=0 plane when include_nz0=True (expected forcing)"


def test_correlation_zero_independence():
    """Test that z+ and z- are independent when correlation=0.0."""
    grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=32)
    state = create_zero_state(grid, M=10)

    key = jax.random.PRNGKey(47)
    amplitude = 0.1
    dt = 0.01

    state_forced, _ = force_alfven_modes_balanced(
        state,
        amplitude=amplitude,
        n_min=1,
        n_max=2,
        dt=dt,
        key=key,
        max_nz=1,
        correlation=0.0,  # No correlation
    )

    # With correlation=0, z+ and z- should be forced independently
    # They should NOT be identical
    assert not jnp.allclose(state_forced.z_plus, state_forced.z_minus, atol=1e-6), \
        "z+ and z- are identical with correlation=0.0 (expected independence)"


def test_correlation_parameter_range():
    """Test that correlation parameter must be in [0, 1)."""
    grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=32)
    state = create_zero_state(grid, M=10)

    key = jax.random.PRNGKey(48)
    amplitude = 0.1
    dt = 0.01

    # Test invalid correlation values
    with pytest.raises(ValueError, match="correlation must be in"):
        force_alfven_modes_balanced(
            state,
            amplitude=amplitude,
            n_min=1,
            n_max=2,
            dt=dt,
            key=key,
            max_nz=1,
            correlation=-0.1,  # Invalid: negative
        )

    with pytest.raises(ValueError, match="correlation must be in"):
        force_alfven_modes_balanced(
            state,
            amplitude=amplitude,
            n_min=1,
            n_max=2,
            dt=dt,
            key=key,
            max_nz=1,
            correlation=1.0,  # Invalid: must be < 1.0
        )

    # Valid values should not raise
    force_alfven_modes_balanced(
        state,
        amplitude=amplitude,
        n_min=1,
        n_max=2,
        dt=dt,
        key=key,
        max_nz=1,
        correlation=0.0,  # Valid
    )

    force_alfven_modes_balanced(
        state,
        amplitude=amplitude,
        n_min=1,
        n_max=2,
        dt=dt,
        key=key,
        max_nz=1,
        correlation=0.9,  # Valid
    )


def test_forcing_preserves_other_fields():
    """Test that forcing only affects z+ and z-, not B_parallel or g."""
    grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=32)
    state = create_zero_state(grid, M=10)

    # Set non-zero values in B_parallel and g
    state = state.model_copy(
        update=dict(
            B_parallel=jnp.ones_like(state.B_parallel) * 0.5,
            g=jnp.ones_like(state.g) * 0.3,
        )
    )

    key = jax.random.PRNGKey(49)
    amplitude = 0.1
    dt = 0.01

    state_forced, _ = force_alfven_modes_balanced(
        state,
        amplitude=amplitude,
        n_min=1,
        n_max=2,
        dt=dt,
        key=key,
        max_nz=1,
    )

    # B_parallel and g should be unchanged
    assert jnp.allclose(state_forced.B_parallel, state.B_parallel), \
        "Forcing modified B_parallel (should only affect z± fields)"
    assert jnp.allclose(state_forced.g, state.g), \
        "Forcing modified Hermite moments (should only affect z± fields)"


def test_white_noise_scaling():
    """Test that forcing scales correctly with dt for white noise."""
    grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=32)
    state = create_zero_state(grid, M=10)

    key = jax.random.PRNGKey(50)
    amplitude = 0.1

    # Force with dt1
    dt1 = 0.01
    state1, _ = force_alfven_modes_balanced(
        state, amplitude=amplitude, n_min=1, n_max=2, dt=dt1, key=key, max_nz=1
    )

    # Force with dt2 = 4 * dt1
    dt2 = 0.04
    state2, _ = force_alfven_modes_balanced(
        state, amplitude=amplitude, n_min=1, n_max=2, dt=dt2, key=key, max_nz=1
    )

    # White noise forcing should scale as amplitude/sqrt(dt)
    # So energy injection rate ~ amplitude^2 / dt
    # Field amplitude ~ amplitude / sqrt(dt)
    # Therefore, state2 / state1 ~ sqrt(dt1/dt2) = sqrt(1/4) = 0.5

    # Compute RMS amplitudes
    rms1 = jnp.sqrt(jnp.mean(jnp.abs(state1.z_plus)**2))
    rms2 = jnp.sqrt(jnp.mean(jnp.abs(state2.z_plus)**2))

    ratio = rms2 / rms1
    expected_ratio = jnp.sqrt(dt1 / dt2)  # = sqrt(0.01/0.04) = 0.5

    # Allow 20% tolerance due to stochastic nature
    assert jnp.abs(ratio - expected_ratio) / expected_ratio < 0.2, \
        f"White noise scaling incorrect: got {ratio:.3f}, expected {expected_ratio:.3f}"


def test_forcing_band_restriction():
    """Test that forcing is restricted to n_min <= n <= n_max."""
    grid = SpectralGrid3D.create(Nx=64, Ny=64, Nz=64)
    state = create_zero_state(grid, M=10)

    key = jax.random.PRNGKey(51)
    amplitude = 0.1
    dt = 0.01
    n_min = 2
    n_max = 4

    state_forced, _ = force_alfven_modes_balanced(
        state,
        amplitude=amplitude,
        n_min=n_min,
        n_max=n_max,
        dt=dt,
        key=key,
        max_nz=1,
    )

    # Compute k_perp for each mode
    Lx, Ly = grid.Lx, grid.Ly
    L_min = min(Lx, Ly)

    # Mode numbers in x and y
    kx_grid, ky_grid = jnp.meshgrid(grid.kx, grid.ky, indexing='ij')
    k_perp = jnp.sqrt(kx_grid**2 + ky_grid**2)
    n_perp = k_perp * L_min / (2 * jnp.pi)

    # Modes outside [n_min, n_max] should be zero
    outside_band = (n_perp < n_min) | (n_perp > n_max)

    # Check z+ at nz=0 (simplest case)
    z_plus_nz0 = state_forced.z_plus[0, :, :]

    # Expand outside_band to match z_plus_nz0 shape (transpose for broadcasting)
    outside_band_2d = outside_band.T  # (Ny, Nx_rfft)

    z_plus_outside = z_plus_nz0[outside_band_2d]

    assert jnp.allclose(z_plus_outside, 0.0, atol=1e-10), \
        f"Forcing applied outside band [{n_min}, {n_max}]"


def test_deterministic_with_same_key():
    """Test that same key produces same forcing."""
    grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=32)
    state = create_zero_state(grid, M=10)

    key = jax.random.PRNGKey(52)
    amplitude = 0.1
    dt = 0.01

    # Force twice with same key
    state1, _ = force_alfven_modes_balanced(
        state, amplitude=amplitude, n_min=1, n_max=2, dt=dt, key=key, max_nz=1
    )
    state2, _ = force_alfven_modes_balanced(
        state, amplitude=amplitude, n_min=1, n_max=2, dt=dt, key=key, max_nz=1
    )

    # Results should be identical
    assert jnp.allclose(state1.z_plus, state2.z_plus), \
        "Same key produced different z+ (expected deterministic)"
    assert jnp.allclose(state1.z_minus, state2.z_minus), \
        "Same key produced different z- (expected deterministic)"


def test_different_with_different_key():
    """Test that different keys produce different forcing."""
    grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=32)
    state = create_zero_state(grid, M=10)

    key1 = jax.random.PRNGKey(53)
    key2 = jax.random.PRNGKey(54)
    amplitude = 0.1
    dt = 0.01

    state1, _ = force_alfven_modes_balanced(
        state, amplitude=amplitude, n_min=1, n_max=2, dt=dt, key=key1, max_nz=1
    )
    state2, _ = force_alfven_modes_balanced(
        state, amplitude=amplitude, n_min=1, n_max=2, dt=dt, key=key2, max_nz=1
    )

    # Results should be different
    assert not jnp.allclose(state1.z_plus, state2.z_plus, atol=1e-6), \
        "Different keys produced identical z+ (expected different)"
    assert not jnp.allclose(state1.z_minus, state2.z_minus, atol=1e-6), \
        "Different keys produced identical z- (expected different)"
