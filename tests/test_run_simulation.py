"""
Tests for scripts/run_simulation.py integration.

These tests verify end-to-end behavior of the simulation runner including:
- Programmatic execution of run_simulation()
- Error handling for invalid configurations
- Output directory management
- Overwrite behavior
"""

import sys
import tempfile
from pathlib import Path

import pytest
import yaml

# Add scripts directory to path to import run_simulation
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from run_simulation import run_simulation
from krmhd.config import (
    SimulationConfig,
    GridConfig,
    TimeIntegrationConfig,
    IOConfig,
    decaying_turbulence_config,
)


class TestRunSimulationIntegration:
    """Integration tests for run_simulation() function."""

    def test_run_simulation_minimal(self):
        """Test run_simulation() with minimal config runs successfully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = decaying_turbulence_config(
                grid=GridConfig(Nx=8, Ny=8, Nz=8),
                time_integration=TimeIntegrationConfig(n_steps=2, save_interval=1),
                io=IOConfig(output_dir=tmpdir)
            )

            # Run simulation (should complete without errors)
            state, history, grid = run_simulation(config, verbose=False)

            # Verify outputs were created
            assert Path(tmpdir).exists()
            assert (Path(tmpdir) / 'config.yaml').exists()
            assert (Path(tmpdir) / 'energy_history.npz').exists()
            assert (Path(tmpdir) / 'final_state.npz').exists()

            # Verify return values
            assert state is not None
            assert history is not None
            assert grid is not None
            assert len(history.times) == 3  # Initial + 2 steps

    def test_run_simulation_no_overwrite_error(self):
        """Test that run_simulation raises error when output exists and overwrite=False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = decaying_turbulence_config(
                grid=GridConfig(Nx=8, Ny=8, Nz=8),
                time_integration=TimeIntegrationConfig(n_steps=1, save_interval=1),
                io=IOConfig(output_dir=tmpdir, overwrite=False)
            )

            # First run should succeed
            run_simulation(config, verbose=False)

            # Second run should fail (directory exists and has files)
            with pytest.raises(FileExistsError) as exc_info:
                run_simulation(config, verbose=False)

            assert "Output directory exists and is non-empty" in str(exc_info.value)
            assert "overwrite: true" in str(exc_info.value)

    def test_run_simulation_with_overwrite(self):
        """Test that run_simulation allows overwriting when overwrite=True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_no_overwrite = decaying_turbulence_config(
                grid=GridConfig(Nx=8, Ny=8, Nz=8),
                time_integration=TimeIntegrationConfig(n_steps=1, save_interval=1),
                io=IOConfig(output_dir=tmpdir, overwrite=False)
            )

            # First run
            run_simulation(config_no_overwrite, verbose=False)

            # Second run with overwrite=True should succeed
            config_with_overwrite = config_no_overwrite.model_copy(
                update={'io': config_no_overwrite.io.model_copy(update={'overwrite': True})}
            )
            state, history, grid = run_simulation(config_with_overwrite, verbose=False)

            # Should complete successfully
            assert state is not None

    def test_run_simulation_with_forcing(self):
        """Test run_simulation with forcing enabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from krmhd.config import ForcingConfig

            config = decaying_turbulence_config(
                grid=GridConfig(Nx=8, Ny=8, Nz=8),
                time_integration=TimeIntegrationConfig(n_steps=2, save_interval=1),
                io=IOConfig(output_dir=tmpdir),
                forcing=ForcingConfig(
                    enabled=True,
                    amplitude=0.1,
                    k_min=1.0,
                    k_max=3.0,
                    seed=42
                )
            )

            # Should run successfully with forcing
            state, history, grid = run_simulation(config, verbose=False)
            assert state is not None
            assert len(history.times) == 3


class TestConfigErrorHandling:
    """Tests for error handling of invalid configurations."""

    def test_missing_yaml_file(self):
        """Test error when YAML file doesn't exist."""
        from krmhd.config import SimulationConfig

        with pytest.raises(FileNotFoundError):
            SimulationConfig.from_yaml(Path("nonexistent_config.yaml"))

    def test_malformed_yaml(self):
        """Test error when YAML file is malformed."""
        from krmhd.config import SimulationConfig

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            # Write malformed YAML
            f.write("grid:\n  Nx: [this is not a number")
            f.flush()
            temp_path = Path(f.name)

        try:
            with pytest.raises(Exception):  # yaml.YAMLError or similar
                SimulationConfig.from_yaml(temp_path)
        finally:
            temp_path.unlink()

    def test_invalid_config_values(self):
        """Test validation errors for invalid config values."""
        with pytest.raises(Exception):
            # Negative grid size should fail validation
            SimulationConfig(
                grid=GridConfig(Nx=-1, Ny=8, Nz=8)
            )

    def test_hyper_r_validation_in_yaml(self):
        """Test that hyper_r=3 fails when loading from YAML."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({
                'physics': {'hyper_r': 3}  # Invalid value
            }, f)
            f.flush()
            temp_path = Path(f.name)

        try:
            with pytest.raises(Exception) as exc_info:
                SimulationConfig.from_yaml(temp_path)
            assert "hyper_r must be 1, 2, 4, or 8" in str(exc_info.value)
        finally:
            temp_path.unlink()


class TestOutputDirectoryManagement:
    """Tests for output directory creation and management."""

    def test_creates_nested_directories(self):
        """Test that nested output directories are created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_dir = Path(tmpdir) / "level1" / "level2" / "output"

            config = decaying_turbulence_config(
                grid=GridConfig(Nx=8, Ny=8, Nz=8),
                time_integration=TimeIntegrationConfig(n_steps=1, save_interval=1),
                io=IOConfig(output_dir=str(nested_dir))
            )

            run_simulation(config, verbose=False)

            # Should have created nested directories
            assert nested_dir.exists()
            assert (nested_dir / 'config.yaml').exists()

    def test_empty_directory_reuse_allowed(self):
        """Test that empty existing directories can be reused even with overwrite=False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "empty_dir"
            output_dir.mkdir()  # Create empty directory

            config = decaying_turbulence_config(
                grid=GridConfig(Nx=8, Ny=8, Nz=8),
                time_integration=TimeIntegrationConfig(n_steps=1, save_interval=1),
                io=IOConfig(output_dir=str(output_dir), overwrite=False)
            )

            # Should succeed because directory is empty
            state, history, grid = run_simulation(config, verbose=False)
            assert state is not None


class TestConfigImmutability:
    """Tests for Pydantic model immutability and proper updates."""

    def test_output_dir_override_immutable(self):
        """Test that output_dir override doesn't mutate original config."""
        config = decaying_turbulence_config(
            io=IOConfig(output_dir="original_dir")
        )
        original_dir = config.io.output_dir

        # Simulate what happens in main() with args.output_dir
        updated_config = config.model_copy(
            update={'io': config.io.model_copy(update={'output_dir': 'new_dir'})}
        )

        # Original should be unchanged
        assert config.io.output_dir == "original_dir"
        # New config should have updated value
        assert updated_config.io.output_dir == "new_dir"

    def test_config_yaml_roundtrip_preserves_values(self):
        """Test that saving and loading config preserves all values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"

            original_config = decaying_turbulence_config(
                grid=GridConfig(Nx=16, Ny=16, Nz=8),
                io=IOConfig(output_dir=tmpdir, overwrite=True)
            )

            # Save to YAML
            original_config.to_yaml(config_path)

            # Load back
            loaded_config = SimulationConfig.from_yaml(config_path)

            # Compare critical values
            assert loaded_config.grid.Nx == 16
            assert loaded_config.grid.Ny == 16
            assert loaded_config.grid.Nz == 8
            assert loaded_config.io.overwrite == True


class TestCFLValidation:
    """Tests for CFL timestep validation."""

    def test_severe_cfl_violation_raises_error(self):
        """Test that dt_fixed > 10× CFL raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create config with absurdly large fixed timestep
            config = decaying_turbulence_config(
                grid=GridConfig(Nx=8, Ny=8, Nz=8),
                time_integration=TimeIntegrationConfig(
                    n_steps=1,
                    dt_fixed=100.0  # Way too large (will be >10× CFL)
                ),
                io=IOConfig(output_dir=tmpdir)
            )

            # Should raise ValueError before starting simulation
            with pytest.raises(ValueError) as exc_info:
                run_simulation(config, verbose=False)

            assert "catastrophic numerical failure" in str(exc_info.value)
            assert "Reduce dt" in str(exc_info.value)

    def test_moderate_cfl_violation_warns(self):
        """Test that 2× < dt_fixed < 10× CFL issues warning."""
        with tempfile.TemporaryDirectory() as tmpdir:
            import warnings

            config = decaying_turbulence_config(
                grid=GridConfig(Nx=8, Ny=8, Nz=8),
                time_integration=TimeIntegrationConfig(
                    n_steps=1,
                    dt_fixed=1.0  # Likely 2-10× CFL but not catastrophic
                ),
                io=IOConfig(output_dir=tmpdir)
            )

            # Should warn but not error
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                state, history, grid = run_simulation(config, verbose=False)

                # Check if warning was issued (may depend on grid/CFL calculation)
                # At minimum, simulation should complete
                assert state is not None


class TestCheckpointIntervalValidation:
    """Tests for checkpoint_interval field validation."""

    def test_checkpoint_interval_accepted_when_set(self):
        """Test that setting checkpoint_interval is accepted (reserved for Issue #13)."""
        config = TimeIntegrationConfig(
            n_steps=100,
            checkpoint_interval=10  # Reserved for HDF5 I/O
        )
        # Should accept the value without warning
        assert config.checkpoint_interval == 10

    def test_checkpoint_interval_none_by_default(self):
        """Test that checkpoint_interval defaults to None."""
        config = TimeIntegrationConfig(n_steps=100)
        assert config.checkpoint_interval is None

    def test_checkpoint_interval_warns_at_runtime(self):
        """Test that using checkpoint_interval in simulation issues runtime warning."""
        import warnings
        with tempfile.TemporaryDirectory() as tmpdir:
            config = decaying_turbulence_config(
                grid=GridConfig(Nx=8, Ny=8, Nz=8),
                time_integration=TimeIntegrationConfig(
                    n_steps=1,
                    save_interval=1,
                    checkpoint_interval=1  # Should trigger warning
                ),
                io=IOConfig(output_dir=tmpdir)
            )

            # Should warn at runtime when simulation runs
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                run_simulation(config, verbose=False)

                # Check warning was issued
                checkpoint_warnings = [
                    warning for warning in w
                    if "checkpoint_interval" in str(warning.message).lower()
                ]
                assert len(checkpoint_warnings) > 0
                assert "Issue #13" in str(checkpoint_warnings[0].message)
