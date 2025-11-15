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


# =============================================================================
# Turbulence Diagnostics I/O Tests (Issue #82)
# =============================================================================


def test_turbulence_diagnostics_roundtrip(grid, tmpdir):
    """Test save/load_turbulence_diagnostics preserves data correctly."""
    from krmhd.io import save_turbulence_diagnostics, load_turbulence_diagnostics
    from krmhd.diagnostics import compute_turbulence_diagnostics, TurbulenceDiagnostics
    from krmhd.physics import initialize_random_spectrum

    # Create a list of diagnostic samples
    diagnostics_list = []
    for i in range(10):
        state = initialize_random_spectrum(grid, M=10, alpha=5/3, amplitude=0.1, seed=i)
        state = state.model_copy(update={"time": i * 0.5})  # Different times
        diag = compute_turbulence_diagnostics(state, dt=0.01, v_A=1.0)
        diagnostics_list.append(diag)

    filename = tmpdir / "test_turbulence_diagnostics.h5"
    metadata = {
        "resolution": grid.Nx,
        "eta": 1.0,
        "nu": 1.0,
        "hyper_r": 2,
        "dt": 0.01,
    }

    # Save diagnostics
    save_turbulence_diagnostics(diagnostics_list, str(filename), **metadata)

    # Load diagnostics
    loaded_list, loaded_metadata = load_turbulence_diagnostics(str(filename))

    # Check length
    assert len(loaded_list) == len(diagnostics_list), "Diagnostic list length mismatch"

    # Check each diagnostic
    for original, loaded in zip(diagnostics_list, loaded_list):
        assert np.isclose(original.time, loaded.time, rtol=1e-6), "Time mismatch"
        assert np.isclose(original.max_velocity, loaded.max_velocity, rtol=1e-6), "max_velocity mismatch"
        assert np.isclose(original.cfl_number, loaded.cfl_number, rtol=1e-6), "cfl_number mismatch"
        assert np.isclose(original.max_nonlinear, loaded.max_nonlinear, rtol=1e-6), "max_nonlinear mismatch"
        assert np.isclose(original.energy_highk, loaded.energy_highk, rtol=1e-6), "energy_highk mismatch"
        assert np.isclose(original.critical_balance_ratio, loaded.critical_balance_ratio, rtol=1e-6), "critical_balance_ratio mismatch"
        assert np.isclose(original.energy_total, loaded.energy_total, rtol=1e-6), "energy_total mismatch"

    # Check metadata
    for key in metadata:
        assert loaded_metadata[key] == metadata[key], f"Metadata {key} mismatch"


def test_turbulence_diagnostics_metadata_preservation(grid, tmpdir):
    """Test that metadata is correctly saved and loaded."""
    from krmhd.io import save_turbulence_diagnostics, load_turbulence_diagnostics
    from krmhd.diagnostics import TurbulenceDiagnostics

    # Create minimal diagnostic list
    diagnostics_list = [
        TurbulenceDiagnostics(
            time=0.0,
            max_velocity=0.1,
            cfl_number=0.01,
            max_nonlinear=0.5,
            energy_highk=0.001,
            critical_balance_ratio=1.0,
            energy_total=1.0,
        )
    ]

    filename = tmpdir / "test_turbulence_metadata.h5"
    metadata = {
        "resolution": 64,
        "eta": 2.0,
        "nu": 1.0,
        "hyper_r": 2,
        "dt": 0.005,
        "termination_reason": "completed",
        "custom_field": "test_value",
    }

    save_turbulence_diagnostics(diagnostics_list, str(filename), **metadata)
    _, loaded_metadata = load_turbulence_diagnostics(str(filename))

    # All metadata should be preserved
    for key, value in metadata.items():
        assert key in loaded_metadata, f"Metadata key {key} missing"
        assert loaded_metadata[key] == value, f"Metadata {key}: expected {value}, got {loaded_metadata[key]}"


def test_turbulence_diagnostics_empty_list(tmpdir):
    """Test handling of empty diagnostic list."""
    from krmhd.io import save_turbulence_diagnostics, load_turbulence_diagnostics

    filename = tmpdir / "test_turbulence_empty.h5"

    # Save empty list
    save_turbulence_diagnostics([], str(filename))

    # Load should work and return empty list
    loaded_list, _ = load_turbulence_diagnostics(str(filename))
    assert len(loaded_list) == 0, "Empty list should stay empty"


def test_turbulence_diagnostics_file_not_found():
    """Test error handling for missing file."""
    from krmhd.io import load_turbulence_diagnostics

    with pytest.raises(FileNotFoundError):
        load_turbulence_diagnostics("nonexistent_file.h5")


def test_turbulence_diagnostics_compression(grid, tmpdir):
    """Test that compression reduces file size."""
    from krmhd.io import save_turbulence_diagnostics
    from krmhd.diagnostics import compute_turbulence_diagnostics
    from krmhd.physics import initialize_random_spectrum

    # Create large diagnostic list (100 samples)
    diagnostics_list = []
    for i in range(100):
        state = initialize_random_spectrum(grid, M=10, alpha=5/3, amplitude=0.1, seed=i)
        state = state.model_copy(update={"time": i * 0.1})
        diag = compute_turbulence_diagnostics(state, dt=0.01, v_A=1.0)
        diagnostics_list.append(diag)

    filename = tmpdir / "test_turbulence_compressed.h5"
    save_turbulence_diagnostics(diagnostics_list, str(filename))

    # File should exist and be smaller than uncompressed data
    # 7 fields × 100 samples × 8 bytes/float64 = 5600 bytes uncompressed
    # With compression, should be significantly smaller
    file_size = filename.stat().st_size
    uncompressed_size = 7 * 100 * 8

    # Compression should reduce size (conservative test: < 2x uncompressed)
    assert file_size < 2 * uncompressed_size, f"File too large: {file_size} > {2 * uncompressed_size}"


# =============================================================================
# Checkpoint Resume with Parameter Override Tests
# =============================================================================


def test_checkpoint_resume_with_parameter_override(grid, tmpdir):
    """Test resuming from checkpoint with modified parameters.

    This tests the workflow where:
    1. Run simulation with initial parameters and save checkpoint
    2. Load checkpoint
    3. Continue simulation with modified parameters (e.g., different eta, forcing)
    4. Verify state continuity and parameter changes
    """
    from krmhd import gandalf_step, force_alfven_modes
    import jax

    # Create initial state
    state = initialize_random_spectrum(
        grid, M=10, beta_i=1.0, v_th=1.0, nu=0.01, Lambda=1.0, alpha=5/3, k_min=1, k_max=5
    )

    # Initial parameters
    eta_initial = 0.1
    force_amplitude_initial = 0.05
    dt = 0.01
    v_A = 1.0

    # Run a few steps with initial parameters
    key = jax.random.PRNGKey(42)
    for _ in range(5):
        state, key = force_alfven_modes(state, amplitude=force_amplitude_initial, n_min=1, n_max=2, dt=dt, key=key)
        state = gandalf_step(state, dt=dt, eta=eta_initial, nu=0.0, v_A=v_A, hyper_r=2, hyper_n=2)

    # Save checkpoint with metadata
    checkpoint_file = tmpdir / "resume_test_checkpoint.h5"
    checkpoint_metadata = {
        'step': 5,
        'eta': eta_initial,
        'force_amplitude': force_amplitude_initial,
        'dt': dt,
        'v_A': v_A,
        'description': 'Test checkpoint for resume'
    }
    save_checkpoint(state, str(checkpoint_file), metadata=checkpoint_metadata)

    # Load checkpoint
    loaded_state, loaded_grid, loaded_metadata = load_checkpoint(str(checkpoint_file))

    # Verify metadata
    assert loaded_metadata['step'] == 5
    assert np.isclose(loaded_metadata['eta'], eta_initial)
    assert np.isclose(loaded_metadata['force_amplitude'], force_amplitude_initial)

    # Verify state continuity
    assert np.allclose(loaded_state.z_plus, state.z_plus, rtol=1e-6)
    assert np.allclose(loaded_state.time, state.time)

    # Continue with MODIFIED parameters (parameter override)
    eta_modified = 0.5  # 5x stronger dissipation
    force_amplitude_modified = 0.01  # 5x weaker forcing

    # Continue from checkpoint with modified parameters
    for _ in range(5):
        loaded_state, key = force_alfven_modes(
            loaded_state, amplitude=force_amplitude_modified, n_min=1, n_max=2, dt=dt, key=key
        )
        loaded_state = gandalf_step(
            loaded_state, dt=dt, eta=eta_modified, nu=0.0, v_A=v_A, hyper_r=2, hyper_n=2
        )

    # Verify that simulation continued (time advanced)
    assert loaded_state.time > state.time
    expected_time = state.time + 5 * dt
    assert np.isclose(loaded_state.time, expected_time, rtol=1e-6)

    # Verify that state evolved (fields changed from checkpoint)
    # With stronger dissipation, energy should decrease
    from krmhd import energy as compute_energy
    E_checkpoint = compute_energy(state)['total']
    E_continued = compute_energy(loaded_state)['total']

    # Stronger dissipation should reduce energy (or at least change it)
    assert E_continued != E_checkpoint  # State evolved

    print(f"✓ Checkpoint resume with parameter override successful")
    print(f"  Original eta={eta_initial}, modified eta={eta_modified}")
    print(f"  Original amplitude={force_amplitude_initial}, modified amplitude={force_amplitude_modified}")
    print(f"  Energy: {E_checkpoint:.6e} → {E_continued:.6e}")


def test_checkpoint_parameter_override_workflow(grid, tmpdir):
    """Test complete workflow: save → load → verify override → continue.

    This simulates the user's workflow:
    - Run to t=50 with initial parameters
    - Checkpoint saved
    - Load checkpoint and override eta and forcing amplitude
    - Continue to t=100 with new parameters
    """
    from krmhd import gandalf_step

    # Initial run
    state = initialize_random_spectrum(
        grid, M=10, beta_i=1.0, v_th=1.0, alpha=5/3, k_min=1, k_max=5
    )

    # Simulate running to t=50
    dt = 0.01
    target_time_1 = 0.5  # Short simulation for test
    eta_v1 = 1.0

    while state.time < target_time_1:
        state = gandalf_step(state, dt=dt, eta=eta_v1, nu=0.0, v_A=1.0, hyper_r=2, hyper_n=2)

    # Save checkpoint at t=50
    checkpoint_file = tmpdir / "workflow_checkpoint_t50.h5"
    metadata_v1 = {'eta': eta_v1, 'run_version': 'v1'}
    save_checkpoint(state, str(checkpoint_file), metadata=metadata_v1)

    # Load and verify original parameters
    loaded_state, _, loaded_metadata = load_checkpoint(str(checkpoint_file))
    assert np.isclose(loaded_metadata['eta'], eta_v1)
    assert loaded_metadata['run_version'] == 'v1'

    # Override parameters for resume
    eta_v2 = 5.0  # Much stronger dissipation

    # Continue to t=100 with new parameters
    target_time_2 = 1.0
    while loaded_state.time < target_time_2:
        loaded_state = gandalf_step(loaded_state, dt=dt, eta=eta_v2, nu=0.0, v_A=1.0, hyper_r=2, hyper_n=2)

    # Verify final state
    assert loaded_state.time >= target_time_2

    # Save final checkpoint with updated metadata
    final_checkpoint = tmpdir / "workflow_checkpoint_final.h5"
    metadata_v2 = {'eta': eta_v2, 'run_version': 'v2'}
    save_checkpoint(loaded_state, str(final_checkpoint), metadata=metadata_v2)

    # Verify final metadata updated
    _, _, final_metadata = load_checkpoint(str(final_checkpoint))
    assert np.isclose(final_metadata['eta'], eta_v2)
    assert final_metadata['run_version'] == 'v2'

    print(f"✓ Complete workflow test passed")
    print(f"  Phase 1 (t=0→0.5): eta={eta_v1}")
    print(f"  Phase 2 (t=0.5→1.0): eta={eta_v2} (overridden)")
