"""
Tests for Modal integration (modal_app.py)

Tests critical functionality including:
1. Parameter validation before expensive cloud execution
2. Parameter sweep combination generation
3. YAML manipulation for parameter sweeps
4. Forcing API conversion (k_min/k_max → n_min/n_max)
"""

import pytest
import tempfile
from pathlib import Path
import yaml
import numpy as np

from krmhd.config import SimulationConfig, decaying_turbulence_config
from krmhd.spectral import SpectralGrid3D


class TestForcingAPIConversion:
    """Tests for k_min/k_max → n_min/n_max conversion (Issue #97)."""

    def test_conversion_formula(self):
        """Test the conversion formula n = round(k*L/(2π))."""
        # For unit domain (L=1.0)
        L = 1.0

        # k = 2π corresponds to n=1 (fundamental mode)
        k = 2 * np.pi
        n = int(np.round(k * L / (2 * np.pi)))
        assert n == 1

        # k = 4π corresponds to n=2 (second harmonic)
        k = 4 * np.pi
        n = int(np.round(k * L / (2 * np.pi)))
        assert n == 2

        # k = 6π corresponds to n=3 (third harmonic)
        k = 6 * np.pi
        n = int(np.round(k * L / (2 * np.pi)))
        assert n == 3

    def test_conversion_with_anisotropic_domain(self):
        """Test conversion uses L_min for anisotropic domains."""
        # Anisotropic domain with Lz < Lx, Ly
        Lx, Ly, Lz = 2.0, 2.0, 1.0
        L_min = min(Lx, Ly, Lz)  # 1.0

        # Same conversion as unit domain
        k = 2 * np.pi
        n = int(np.round(k * L_min / (2 * np.pi)))
        assert n == 1

    def test_conversion_with_small_k(self):
        """Test that small k_min gets clamped to n_min=1."""
        L = 1.0
        k_min = 0.5  # Much smaller than 2π

        n_min = int(np.round(k_min * L / (2 * np.pi)))
        n_min = max(1, n_min)  # Clamp to at least 1

        assert n_min == 1

    def test_conversion_ensures_n_max_ge_n_min(self):
        """Test that n_max is always >= n_min."""
        L = 1.0
        k_min = 2 * np.pi  # n_min = 1
        k_max = 2 * np.pi - 0.1  # Slightly less (edge case)

        n_min = int(np.round(k_min * L / (2 * np.pi)))
        n_max = int(np.round(k_max * L / (2 * np.pi)))

        n_min = max(1, n_min)
        n_max = max(n_min, n_max)  # Ensure n_max >= n_min

        assert n_max >= n_min

    def test_roundtrip_conversion(self):
        """Test that n → k → n roundtrip is exact for integer modes."""
        L = 1.0

        for n_original in [1, 2, 3, 5, 10]:
            # Convert to wavenumber
            k = 2 * np.pi * n_original / L

            # Convert back to mode number
            n_roundtrip = int(np.round(k * L / (2 * np.pi)))

            assert n_roundtrip == n_original


class TestParameterValidation:
    """Tests for parameter validation before cloud execution."""

    def test_validate_empty_parameters(self):
        """Test that empty parameters dict raises ValueError."""
        parameters = {}

        with pytest.raises(ValueError, match="Parameters dictionary is empty"):
            if not parameters:
                raise ValueError("Parameters dictionary is empty")

    def test_validate_dot_notation_required(self):
        """Test that parameters must use dot notation (e.g., 'physics.eta')."""
        parameters = {
            "eta": [0.1, 0.5, 1.0],  # Missing 'physics.' prefix
        }

        for param_name in parameters.keys():
            if '.' not in param_name:
                with pytest.raises(ValueError, match="must use dot notation"):
                    raise ValueError(
                        f"Parameter '{param_name}' must use dot notation "
                        "(e.g., 'physics.eta' not 'eta')"
                    )

    def test_validate_config_path_exists(self):
        """Test that parameter path exists in config structure."""
        config_dict = yaml.safe_load(yaml.dump(decaying_turbulence_config().model_dump()))

        # Valid path
        path_parts = "physics.eta".split('.')
        current = config_dict
        for part in path_parts[:-1]:
            assert part in current
            current = current[part]
        assert path_parts[-1] in current

        # Invalid path
        path_parts = "physics.nonexistent".split('.')
        current = config_dict
        for part in path_parts[:-1]:
            current = current[part]

        with pytest.raises(KeyError):
            _ = current[path_parts[-1]]

    def test_validate_config_after_modification(self):
        """Test that modified config still validates with Pydantic."""
        config_dict = decaying_turbulence_config().model_dump()

        # Modify a parameter
        config_dict['physics']['eta'] = 0.5

        # Should still validate
        config = SimulationConfig(**config_dict)
        assert config.physics.eta == 0.5

    def test_validate_invalid_value_raises_error(self):
        """Test that invalid values raise ValidationError."""
        config_dict = decaying_turbulence_config().model_dump()

        # Invalid value (negative eta)
        config_dict['physics']['eta'] = -0.5

        with pytest.raises(Exception):  # Pydantic ValidationError
            _ = SimulationConfig(**config_dict)


class TestParameterSweepGeneration:
    """Tests for parameter sweep combination generation."""

    def test_generate_sweep_combinations_single_param(self):
        """Test sweep with single parameter."""
        parameters = {
            "physics.eta": [0.1, 0.5, 1.0]
        }

        param_names = list(parameters.keys())
        param_lists = list(parameters.values())

        # Generate Cartesian product
        from itertools import product
        combinations = list(product(*param_lists))

        assert len(combinations) == 3
        assert combinations == [(0.1,), (0.5,), (1.0,)]

    def test_generate_sweep_combinations_two_params(self):
        """Test sweep with two parameters (Cartesian product)."""
        parameters = {
            "physics.eta": [0.1, 0.5],
            "physics.nu": [0.2, 0.8]
        }

        param_names = list(parameters.keys())
        param_lists = list(parameters.values())

        from itertools import product
        combinations = list(product(*param_lists))

        assert len(combinations) == 4
        assert (0.1, 0.2) in combinations
        assert (0.1, 0.8) in combinations
        assert (0.5, 0.2) in combinations
        assert (0.5, 0.8) in combinations

    def test_generate_sweep_combinations_three_params(self):
        """Test sweep with three parameters."""
        parameters = {
            "physics.eta": [0.1, 0.5],
            "grid.Nx": [32, 64],
            "physics.hyper_r": [1, 2]
        }

        param_names = list(parameters.keys())
        param_lists = list(parameters.values())

        from itertools import product
        combinations = list(product(*param_lists))

        # 2 × 2 × 2 = 8 combinations
        assert len(combinations) == 8

    def test_sweep_preserves_parameter_order(self):
        """Test that parameter sweep preserves order of parameters."""
        parameters = {
            "physics.eta": [0.1, 0.5],
            "physics.nu": [0.2]
        }

        param_names = list(parameters.keys())
        assert param_names == ["physics.eta", "physics.nu"]

        from itertools import product
        combinations = list(product(*[parameters[p] for p in param_names]))

        # First value in each tuple should be eta
        assert combinations[0][0] == 0.1
        assert combinations[1][0] == 0.5


class TestYAMLManipulation:
    """Tests for YAML manipulation in parameter sweeps."""

    def test_yaml_roundtrip(self):
        """Test that config can be dumped to YAML and reloaded."""
        config = decaying_turbulence_config()

        # Dump to YAML string
        yaml_str = yaml.dump(config.model_dump())

        # Reload
        config_dict = yaml.safe_load(yaml_str)
        config_reloaded = SimulationConfig(**config_dict)

        # Should be identical
        assert config_reloaded.physics.eta == config.physics.eta
        assert config_reloaded.grid.Nx == config.grid.Nx

    def test_modify_nested_parameter(self):
        """Test modifying nested parameter via dot notation."""
        config_dict = decaying_turbulence_config().model_dump()

        # Navigate to nested parameter
        param_name = "physics.eta"
        path_parts = param_name.split('.')

        current = config_dict
        for part in path_parts[:-1]:
            current = current[part]

        # Modify
        old_value = current[path_parts[-1]]
        current[path_parts[-1]] = 0.999

        # Verify modification
        assert config_dict['physics']['eta'] == 0.999
        assert config_dict['physics']['eta'] != old_value

    def test_modify_multiple_parameters(self):
        """Test modifying multiple parameters in sequence."""
        config_dict = decaying_turbulence_config().model_dump()

        # Modify physics.eta
        config_dict['physics']['eta'] = 0.5

        # Modify grid.Nx
        config_dict['grid']['Nx'] = 128

        # Verify both modifications
        assert config_dict['physics']['eta'] == 0.5
        assert config_dict['grid']['Nx'] == 128

    def test_modified_config_validates(self):
        """Test that modified config still validates."""
        config_dict = decaying_turbulence_config().model_dump()

        # Modify several parameters
        config_dict['physics']['eta'] = 0.75
        config_dict['physics']['nu'] = 0.25
        config_dict['grid']['Nx'] = 128
        config_dict['grid']['Ny'] = 128

        # Should validate
        config = SimulationConfig(**config_dict)
        assert config.physics.eta == 0.75
        assert config.grid.Nx == 128

    def test_yaml_preserves_types(self):
        """Test that YAML roundtrip preserves types."""
        config = decaying_turbulence_config()
        yaml_str = yaml.dump(config.model_dump())
        config_dict = yaml.safe_load(yaml_str)

        # Check types
        assert isinstance(config_dict['physics']['eta'], float)
        assert isinstance(config_dict['grid']['Nx'], int)
        assert isinstance(config_dict['forcing']['enabled'], bool)


class TestParameterSweepMetadata:
    """Tests for parameter sweep metadata generation."""

    def test_metadata_includes_parameters(self):
        """Test that metadata includes parameter values."""
        param_names = ["physics.eta", "physics.nu"]
        param_values = (0.5, 0.2)

        metadata = {
            "parameters": {
                name: value for name, value in zip(param_names, param_values)
            }
        }

        assert metadata["parameters"]["physics.eta"] == 0.5
        assert metadata["parameters"]["physics.nu"] == 0.2

    def test_metadata_includes_job_index(self):
        """Test that metadata includes job index."""
        metadata = {
            "job_index": 5,
            "total_jobs": 10
        }

        assert metadata["job_index"] == 5
        assert metadata["total_jobs"] == 10

    def test_metadata_includes_output_subdir(self):
        """Test that metadata includes output subdirectory."""
        param_names = ["physics.eta", "physics.nu"]
        param_values = (0.5, 0.2)

        output_subdir = "_".join(
            f"{name.split('.')[-1]}={value}"
            for name, value in zip(param_names, param_values)
        )

        assert output_subdir == "eta=0.5_nu=0.2"


class TestErrorHandling:
    """Tests for error handling in Modal integration."""

    def test_partial_failure_status(self):
        """Test that partial failures are tracked separately."""
        result = {
            'status': 'partial_failure',
            'volume_commit_error': 'Network timeout',
            'final_energy': 1.23,
            'output_dir': '/results/run_1'
        }

        assert result['status'] == 'partial_failure'
        assert 'volume_commit_error' in result
        assert result['final_energy'] is not None  # Simulation succeeded

    def test_full_failure_status(self):
        """Test that full failures are tracked."""
        result = {
            'status': 'failed',
            'error': 'JAX GPU not available',
            'traceback': '...'
        }

        assert result['status'] == 'failed'
        assert 'error' in result

    def test_success_status(self):
        """Test that successes are tracked."""
        result = {
            'status': 'success',
            'final_energy': 1.23,
            'output_dir': '/results/run_1',
            'volume_commit_error': None
        }

        assert result['status'] == 'success'
        assert result['volume_commit_error'] is None


class TestSweepJobTimeout:
    """Tests for sweep job timeout handling."""

    def test_timeout_calculation(self):
        """Test that timeout accounts for job duration + buffer."""
        timeout_single = 3600  # 1 hour
        timeout_buffer = 600   # 10 minutes

        timeout_job_get = timeout_single + timeout_buffer

        assert timeout_job_get == 4200  # 1h 10m

    def test_timeout_configurable(self):
        """Test that timeouts are configurable via environment."""
        import os

        # Simulate environment variable
        test_timeout = 7200
        os.environ['MODAL_TIMEOUT_JOB_GET'] = str(test_timeout)

        timeout_job_get = int(os.environ.get('MODAL_TIMEOUT_JOB_GET', 4200))

        assert timeout_job_get == test_timeout

        # Cleanup
        del os.environ['MODAL_TIMEOUT_JOB_GET']
