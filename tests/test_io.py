"""
Tests for HDF5 I/O functionality.

This module tests checkpoint and timeseries save/load operations, including:
- Roundtrip accuracy (save then load recovers original data)
- Error handling (file conflicts, missing files, invalid data)
- Metadata preservation
- Shape and dtype validation
- Compression and file size

Test organization:
- test_checkpoint_*: Checkpoint save/load tests
- test_timeseries_*: Timeseries save/load tests
- test_io_errors_*: Error handling tests
"""

import tempfile
from pathlib import Path
import pytest
import numpy as np
import jax.numpy as jnp
import h5py

from krmhd import KRMHDState, SpectralGrid3D, initialize_random_spectrum
from krmhd.diagnostics import EnergyHistory
from krmhd.io import (
    save_checkpoint,
    load_checkpoint,
    save_timeseries,
    load_timeseries,
    IO_FORMAT_VERSION,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def grid():
    """Create test grid."""
    return SpectralGrid3D.create(Nx=32, Ny=32, Nz=16, Lx=2*np.pi, Ly=2*np.pi, Lz=2*np.pi)


@pytest.fixture
def state(grid):
    """Create test state with random spectrum."""
    return initialize_random_spectrum(
        grid, M=10, beta_i=1.0, v_th=1.0, nu=0.01, Lambda=1.0, alpha=5/3, k_min=1, k_max=5
    )


@pytest.fixture
def tmpdir():
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield Path(tmpdirname)


@pytest.fixture
def energy_history(state):
    """Create test energy history with multiple timesteps."""
    history = EnergyHistory()
    # Add 10 timesteps
    for i in range(10):
        state.time = i * 0.1
        history.append(state)
    return history


# =============================================================================
# Checkpoint Tests
# =============================================================================


def test_checkpoint_roundtrip(state, tmpdir):
    """Test that save/load checkpoint recovers original state."""
    filename = tmpdir / "test_checkpoint.h5"

    # Save checkpoint
    save_checkpoint(state, str(filename))

    # Load checkpoint
    loaded_state, loaded_grid, metadata = load_checkpoint(str(filename))

    # Check grid parameters
    assert loaded_grid.Nx == state.grid.Nx
    assert loaded_grid.Ny == state.grid.Ny
    assert loaded_grid.Nz == state.grid.Nz
    assert np.allclose(loaded_grid.Lx, state.grid.Lx)
    assert np.allclose(loaded_grid.Ly, state.grid.Ly)
    assert np.allclose(loaded_grid.Lz, state.grid.Lz)

    # Check state fields (allow small numerical error from float32 storage)
    assert np.allclose(loaded_state.z_plus, state.z_plus, rtol=1e-6, atol=1e-7)
    assert np.allclose(loaded_state.z_minus, state.z_minus, rtol=1e-6, atol=1e-7)
    assert np.allclose(loaded_state.B_parallel, state.B_parallel, rtol=1e-6, atol=1e-7)
    assert np.allclose(loaded_state.g, state.g, rtol=1e-6, atol=1e-7)

    # Check state parameters
    assert loaded_state.M == state.M
    assert np.allclose(loaded_state.beta_i, state.beta_i)
    assert np.allclose(loaded_state.v_th, state.v_th)
    assert np.allclose(loaded_state.nu, state.nu)
    assert np.allclose(loaded_state.Lambda, state.Lambda)
    assert np.allclose(loaded_state.time, state.time)

    # Check metadata
    assert metadata['version'] == IO_FORMAT_VERSION
    assert 'timestamp' in metadata


def test_checkpoint_with_metadata(state, tmpdir):
    """Test checkpoint save/load with user metadata."""
    filename = tmpdir / "test_checkpoint_meta.h5"

    metadata_in = {
        'run_id': 'test_001',
        'description': 'Test simulation',
        'eta': 0.01,
        'parameters': {'key': 'value'}  # Will be converted to string
    }

    # Save with metadata
    save_checkpoint(state, str(filename), metadata=metadata_in)

    # Load and check metadata
    _, _, metadata_out = load_checkpoint(str(filename))

    assert metadata_out['run_id'] == 'test_001'
    assert metadata_out['description'] == 'Test simulation'
    assert metadata_out['eta'] == 0.01
    assert 'parameters' in metadata_out  # Dict converted to string


def test_checkpoint_overwrite(state, tmpdir):
    """Test checkpoint overwrite behavior."""
    filename = tmpdir / "test_checkpoint_overwrite.h5"

    # Save initial checkpoint
    save_checkpoint(state, str(filename))

    # Try to save again without overwrite - should fail
    with pytest.raises(FileExistsError):
        save_checkpoint(state, str(filename), overwrite=False)

    # Save again with overwrite=True - should succeed
    state.time = 10.0
    save_checkpoint(state, str(filename), overwrite=True)

    # Load and verify it's the updated state
    loaded_state, _, _ = load_checkpoint(str(filename))
    assert np.allclose(loaded_state.time, 10.0)


def test_checkpoint_creates_directory(state, tmpdir):
    """Test that save_checkpoint creates parent directories."""
    filename = tmpdir / "subdir1" / "subdir2" / "checkpoint.h5"

    # Should create nested directories
    save_checkpoint(state, str(filename))

    assert filename.exists()
    assert filename.parent.exists()


def test_checkpoint_file_not_found(tmpdir):
    """Test load_checkpoint with non-existent file."""
    filename = tmpdir / "nonexistent.h5"

    with pytest.raises(FileNotFoundError):
        load_checkpoint(str(filename))


def test_checkpoint_shape_validation(state, tmpdir):
    """Test that load_checkpoint validates array shapes."""
    filename = tmpdir / "test_checkpoint_shapes.h5"

    # Save valid checkpoint
    save_checkpoint(state, str(filename))

    # Manually corrupt the file by changing z_plus shape
    with h5py.File(filename, 'r+') as f:
        # Delete original dataset
        del f['state']['z_plus_real']
        # Create dataset with wrong shape
        wrong_shape = (10, 10, 10)  # Doesn't match grid
        f['state'].create_dataset('z_plus_real', data=np.zeros(wrong_shape))

    # Loading should fail with validation error
    with pytest.raises(ValueError, match="z_plus shape"):
        load_checkpoint(str(filename), validate_grid=True)


def test_checkpoint_complex_dtype(state, tmpdir):
    """Test that complex fields are correctly stored and loaded."""
    filename = tmpdir / "test_checkpoint_complex.h5"

    # Save checkpoint
    save_checkpoint(state, str(filename))

    # Manually check HDF5 file structure
    with h5py.File(filename, 'r') as f:
        # Should have real and imag parts
        assert 'z_plus_real' in f['state']
        assert 'z_plus_imag' in f['state']
        assert 'z_minus_real' in f['state']
        assert 'z_minus_imag' in f['state']
        assert 'B_parallel_real' in f['state']
        assert 'B_parallel_imag' in f['state']
        assert 'g_real' in f['state']
        assert 'g_imag' in f['state']

        # Check dtypes (should be float32)
        assert f['state']['z_plus_real'].dtype == np.float32
        assert f['state']['z_plus_imag'].dtype == np.float32

    # Load and verify complex reconstruction
    loaded_state, _, _ = load_checkpoint(str(filename))
    assert jnp.iscomplexobj(loaded_state.z_plus)
    assert jnp.iscomplexobj(loaded_state.z_minus)
    assert jnp.iscomplexobj(loaded_state.B_parallel)
    assert jnp.iscomplexobj(loaded_state.g)


def test_checkpoint_compression(state, tmpdir):
    """Test that checkpoint files use compression."""
    filename = tmpdir / "test_checkpoint_compression.h5"

    save_checkpoint(state, str(filename))

    # Check that compression is enabled
    with h5py.File(filename, 'r') as f:
        z_plus_real = f['state']['z_plus_real']
        assert z_plus_real.compression == 'gzip'
        assert z_plus_real.compression_opts == 4


def test_checkpoint_grid_reconstruction(state, tmpdir):
    """Test that grid is correctly reconstructed from attributes."""
    filename = tmpdir / "test_checkpoint_grid.h5"

    save_checkpoint(state, str(filename))
    _, loaded_grid, _ = load_checkpoint(str(filename))

    # Grid should have all computed arrays (kx, ky, kz, dealias_mask)
    assert loaded_grid.kx is not None
    assert loaded_grid.ky is not None
    assert loaded_grid.kz is not None
    assert loaded_grid.dealias_mask is not None

    # Check shapes
    assert len(loaded_grid.kx) == state.grid.Nx // 2 + 1
    assert len(loaded_grid.ky) == state.grid.Ny
    assert len(loaded_grid.kz) == state.grid.Nz
    assert loaded_grid.dealias_mask.shape == (state.grid.Nz, state.grid.Ny, state.grid.Nx // 2 + 1)


# =============================================================================
# Timeseries Tests
# =============================================================================


def test_timeseries_roundtrip(energy_history, tmpdir):
    """Test that save/load timeseries recovers original data."""
    filename = tmpdir / "test_timeseries.h5"

    # Save timeseries
    save_timeseries(energy_history, str(filename))

    # Load timeseries
    loaded_history, metadata = load_timeseries(str(filename))

    # Check data
    assert len(loaded_history.times) == len(energy_history.times)
    assert np.allclose(loaded_history.times, energy_history.times)
    assert np.allclose(loaded_history.E_magnetic, energy_history.E_magnetic)
    assert np.allclose(loaded_history.E_kinetic, energy_history.E_kinetic)
    assert np.allclose(loaded_history.E_compressive, energy_history.E_compressive)
    assert np.allclose(loaded_history.E_total, energy_history.E_total)

    # Check metadata
    assert metadata['version'] == IO_FORMAT_VERSION
    assert 'timestamp' in metadata
    assert metadata['n_timesteps'] == len(energy_history.times)
    assert np.allclose(metadata['t_start'], energy_history.times[0])
    assert np.allclose(metadata['t_end'], energy_history.times[-1])


def test_timeseries_with_metadata(energy_history, tmpdir):
    """Test timeseries save/load with user metadata."""
    filename = tmpdir / "test_timeseries_meta.h5"

    metadata_in = {
        'run_id': 'test_001',
        'eta': 0.01,
        'nu': 0.01,
    }

    # Save with metadata
    save_timeseries(energy_history, str(filename), metadata=metadata_in)

    # Load and check metadata
    _, metadata_out = load_timeseries(str(filename))

    assert metadata_out['run_id'] == 'test_001'
    assert metadata_out['eta'] == 0.01
    assert metadata_out['nu'] == 0.01


def test_timeseries_empty_history(tmpdir):
    """Test that saving empty history raises error."""
    filename = tmpdir / "test_timeseries_empty.h5"

    empty_history = EnergyHistory()

    with pytest.raises(ValueError, match="empty"):
        save_timeseries(empty_history, str(filename))


def test_timeseries_overwrite(energy_history, tmpdir):
    """Test timeseries overwrite behavior."""
    filename = tmpdir / "test_timeseries_overwrite.h5"

    # Save initial timeseries
    save_timeseries(energy_history, str(filename))

    # Try to save again without overwrite - should fail
    with pytest.raises(FileExistsError):
        save_timeseries(energy_history, str(filename), overwrite=False)

    # Save again with overwrite=True - should succeed
    save_timeseries(energy_history, str(filename), overwrite=True)


def test_timeseries_creates_directory(energy_history, tmpdir):
    """Test that save_timeseries creates parent directories."""
    filename = tmpdir / "subdir1" / "subdir2" / "timeseries.h5"

    # Should create nested directories
    save_timeseries(energy_history, str(filename))

    assert filename.exists()
    assert filename.parent.exists()


def test_timeseries_file_not_found(tmpdir):
    """Test load_timeseries with non-existent file."""
    filename = tmpdir / "nonexistent.h5"

    with pytest.raises(FileNotFoundError):
        load_timeseries(str(filename))


def test_timeseries_compression(energy_history, tmpdir):
    """Test that timeseries files use compression."""
    filename = tmpdir / "test_timeseries_compression.h5"

    save_timeseries(energy_history, str(filename))

    # Check that compression is enabled
    with h5py.File(filename, 'r') as f:
        times = f['times']
        assert times.compression == 'gzip'
        assert times.compression_opts == 4


def test_timeseries_dtype(energy_history, tmpdir):
    """Test that timeseries data is stored as float64."""
    filename = tmpdir / "test_timeseries_dtype.h5"

    save_timeseries(energy_history, str(filename))

    # Check dtypes
    with h5py.File(filename, 'r') as f:
        assert f['times'].dtype == np.float64
        assert f['E_magnetic'].dtype == np.float64
        assert f['E_kinetic'].dtype == np.float64
        assert f['E_compressive'].dtype == np.float64
        assert f['E_total'].dtype == np.float64


# =============================================================================
# Integration Tests
# =============================================================================


def test_checkpoint_restart_simulation(state, tmpdir):
    """Test checkpoint/restart workflow for simulation continuation."""
    from krmhd import gandalf_step

    checkpoint_file = tmpdir / "checkpoint.h5"

    # Run simulation for a few steps
    state_original = state
    for i in range(5):
        state_original = gandalf_step(state_original, dt=0.01, eta=0.01, nu=0.01, v_A=1.0)

    # Save checkpoint
    save_checkpoint(state_original, str(checkpoint_file))

    # Load checkpoint and continue simulation
    state_loaded, _, _ = load_checkpoint(str(checkpoint_file))

    # Continue from loaded state
    state_continued = state_loaded
    for i in range(5):
        state_continued = gandalf_step(state_continued, dt=0.01, eta=0.01, nu=0.01, v_A=1.0)

    # The continued simulation should produce different results than the loaded state
    # (but we can't verify exact values without re-running from scratch)
    assert state_continued.time > state_loaded.time
    assert not np.allclose(state_continued.z_plus, state_loaded.z_plus)


def test_multiple_checkpoints(state, tmpdir):
    """Test saving multiple checkpoints at different times."""
    from krmhd import gandalf_step

    # Run and save checkpoints at t=0, 0.5, 1.0
    times = [0.0, 0.5, 1.0]
    checkpoints = []

    for t in times:
        # Evolve to target time
        while state.time < t:
            state = gandalf_step(state, dt=0.01, eta=0.01, nu=0.01, v_A=1.0)

        # Save checkpoint
        filename = tmpdir / f"checkpoint_t{t:.1f}.h5"
        save_checkpoint(state, str(filename))
        checkpoints.append(filename)

    # Load all checkpoints and verify times are monotonic
    loaded_times = []
    for checkpoint_file in checkpoints:
        loaded_state, _, _ = load_checkpoint(str(checkpoint_file))
        loaded_times.append(loaded_state.time)

    # Times should be approximately equal to target times
    assert np.allclose(loaded_times, times, atol=0.01)


def test_timeseries_analysis(energy_history, tmpdir):
    """Test that loaded timeseries can be analyzed."""
    filename = tmpdir / "test_timeseries_analysis.h5"

    # Save timeseries
    save_timeseries(energy_history, str(filename))

    # Load timeseries
    loaded_history, _ = load_timeseries(str(filename))

    # Test analysis methods
    mag_frac = loaded_history.magnetic_fraction()
    assert len(mag_frac) == len(loaded_history.times)
    assert np.all(mag_frac >= 0) and np.all(mag_frac <= 1)

    dissipation = loaded_history.dissipation_rate()
    assert len(dissipation) == len(loaded_history.times) - 1  # Finite difference


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


def test_checkpoint_with_zero_fields(grid, tmpdir):
    """Test checkpoint save/load with zero fields."""
    # Create state with all zeros
    state_zero = KRMHDState(
        z_plus=jnp.zeros((grid.Nz, grid.Ny, grid.Nx // 2 + 1), dtype=complex),
        z_minus=jnp.zeros((grid.Nz, grid.Ny, grid.Nx // 2 + 1), dtype=complex),
        B_parallel=jnp.zeros((grid.Nz, grid.Ny, grid.Nx // 2 + 1), dtype=complex),
        g=jnp.zeros((grid.Nz, grid.Ny, grid.Nx // 2 + 1, 11), dtype=complex),
        M=10,
        beta_i=1.0,
        v_th=1.0,
        nu=0.01,
        Lambda=1.0,
        time=0.0,
        grid=grid,
    )

    filename = tmpdir / "test_checkpoint_zeros.h5"

    save_checkpoint(state_zero, str(filename))
    loaded_state, _, _ = load_checkpoint(str(filename))

    assert np.allclose(loaded_state.z_plus, 0.0)
    assert np.allclose(loaded_state.z_minus, 0.0)
    assert np.allclose(loaded_state.B_parallel, 0.0)
    assert np.allclose(loaded_state.g, 0.0)


def test_checkpoint_disable_validation(state, tmpdir):
    """Test loading checkpoint with validation disabled."""
    filename = tmpdir / "test_checkpoint_no_validation.h5"

    save_checkpoint(state, str(filename))

    # Load without validation (should still work)
    loaded_state, _, _ = load_checkpoint(str(filename), validate_grid=False)

    assert loaded_state.grid.Nx == state.grid.Nx


def test_version_warning(state, tmpdir):
    """Test that version mismatch produces warning."""
    filename = tmpdir / "test_checkpoint_version.h5"

    save_checkpoint(state, str(filename))

    # Manually change version in file
    with h5py.File(filename, 'r+') as f:
        f['metadata'].attrs['version'] = '0.0.0'

    # Loading should produce warning
    with pytest.warns(UserWarning, match="version"):
        load_checkpoint(str(filename))


def test_timeseries_version_warning(energy_history, tmpdir):
    """Test that version mismatch produces warning for timeseries."""
    filename = tmpdir / "test_timeseries_version.h5"

    save_timeseries(energy_history, str(filename))

    # Manually change version in file
    with h5py.File(filename, 'r+') as f:
        f.attrs['version'] = '0.0.0'

    # Loading should produce warning
    with pytest.warns(UserWarning, match="version"):
        load_timeseries(str(filename))
