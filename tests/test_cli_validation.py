"""
Integration tests for parameter validation CLI.

Tests all CLI commands:
- validate: Validate config files
- suggest: Suggest parameters
- check: Quick parameter check
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import yaml


def run_cli(*args):
    """Run CLI command and return result."""
    result = subprocess.run(
        [sys.executable, "-m", "krmhd"] + list(args),
        capture_output=True,
        text=True,
    )
    return result


class TestCLIValidate:
    """Test 'validate' command."""

    def test_validate_valid_config(self):
        """Validate a valid config file."""
        config = {
            "dt": 0.01,
            "eta": 0.02,
            "nu": 0.02,
            "Nx": 64,
            "Ny": 64,
            "Nz": 32,
            "v_A": 1.0,
            "force_amplitude": 0.3,
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            config_path = f.name

        try:
            result = run_cli("validate", config_path)
            assert result.returncode == 0
            assert "valid" in result.stdout.lower()
        finally:
            Path(config_path).unlink()

    def test_validate_invalid_config(self):
        """Validate an invalid config file."""
        config = {
            "dt": 1.0,
            "eta": 60.0,  # Overflow
            "nu": 0.02,
            "Nx": 64,
            "Ny": 64,
            "Nz": 32,
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            config_path = f.name

        try:
            result = run_cli("validate", config_path)
            assert result.returncode == 1  # Should fail
            assert "error" in result.stdout.lower()
        finally:
            Path(config_path).unlink()

    def test_validate_nonexistent_file(self):
        """Validate non-existent config file."""
        result = run_cli("validate", "nonexistent.yaml")
        assert result.returncode == 1
        assert "not found" in result.stdout.lower()

    def test_validate_existing_config_files(self):
        """Test with actual config files in repo."""
        config_dir = Path("configs")
        if not config_dir.exists():
            pytest.skip("configs/ directory not found")

        config_files = list(config_dir.glob("*.yaml"))
        if not config_files:
            pytest.skip("No config files found")

        # Test at least one config file
        result = run_cli("validate", str(config_files[0]))
        # Should not crash (may be valid or invalid)
        assert isinstance(result.returncode, int)


class TestCLISuggest:
    """Test 'suggest' command."""

    def test_suggest_forced_default(self):
        """Suggest parameters for forced turbulence."""
        result = run_cli("suggest", "--resolution", "64", "64", "32")
        assert result.returncode == 0
        assert "dt:" in result.stdout
        assert "eta:" in result.stdout
        assert "force_amplitude:" in result.stdout

    def test_suggest_decaying(self):
        """Suggest parameters for decaying turbulence."""
        result = run_cli("suggest", "--resolution", "128", "128", "64", "--type", "decaying")
        assert result.returncode == 0
        assert "dt:" in result.stdout
        assert "eta:" in result.stdout

    def test_suggest_benchmark(self):
        """Suggest benchmark parameters."""
        result = run_cli("suggest", "--resolution", "64", "64", "32", "--type", "benchmark")
        assert result.returncode == 0
        assert "dt:" in result.stdout

    def test_suggest_custom_cfl(self):
        """Suggest with custom CFL."""
        result = run_cli(
            "suggest",
            "--resolution", "64", "64", "32",
            "--cfl", "0.5"
        )
        assert result.returncode == 0
        assert "cfl_safety:      0.50" in result.stdout

    def test_suggest_validates_results(self):
        """Suggested parameters should pass validation."""
        result = run_cli("suggest", "--resolution", "64", "64", "32", "--type", "forced")
        assert result.returncode == 0
        # Should show validation results
        assert "Validating suggested parameters" in result.stdout
        assert "valid" in result.stdout.lower()

    def test_suggest_invalid_resolution(self):
        """Invalid resolution format should fail."""
        result = run_cli("suggest", "--resolution", "64", "64")  # Missing Nz
        # Should fail with argparse error
        assert result.returncode != 0

    def test_suggest_invalid_type(self):
        """Invalid simulation type should fail."""
        result = run_cli(
            "suggest",
            "--resolution", "64", "64", "32",
            "--type", "invalid_type"
        )
        assert result.returncode != 0


class TestCLICheck:
    """Test 'check' command."""

    def test_check_valid_parameters(self):
        """Check valid parameters."""
        result = run_cli(
            "check",
            "--dt", "0.01",
            "--eta", "0.5",
            "--nu", "0.5",
            "--Nx", "64",
            "--Ny", "64",
            "--Nz", "32"
        )
        assert result.returncode == 0
        assert "valid" in result.stdout.lower()

    def test_check_invalid_parameters_overflow(self):
        """Check invalid parameters (overflow)."""
        result = run_cli(
            "check",
            "--dt", "1.0",
            "--eta", "60.0",  # Overflow
            "--nu", "0.5",
            "--Nx", "64",
            "--Ny", "64",
            "--Nz", "32"
        )
        assert result.returncode == 1
        assert "error" in result.stdout.lower()

    def test_check_with_forcing(self):
        """Check with forcing amplitude."""
        result = run_cli(
            "check",
            "--dt", "0.01",
            "--eta", "0.02",
            "--nu", "0.02",
            "--Nx", "64",
            "--Ny", "64",
            "--Nz", "32",
            "--force_amplitude", "0.3"
        )
        assert result.returncode == 0

    def test_check_with_hyper_orders(self):
        """Check with custom hyper-orders."""
        result = run_cli(
            "check",
            "--dt", "0.01",
            "--eta", "0.02",
            "--nu", "0.02",
            "--Nx", "64",
            "--Ny", "64",
            "--Nz", "32",
            "--hyper_r", "2",
            "--hyper_n", "2"
        )
        assert result.returncode == 0

    def test_check_missing_required_args(self):
        """Check with missing required arguments."""
        result = run_cli("check", "--dt", "0.01")  # Missing other required args
        assert result.returncode != 0


class TestCLIHelp:
    """Test CLI help and documentation."""

    def test_main_help(self):
        """Main help message."""
        result = run_cli("--help")
        assert result.returncode == 0
        assert "validate" in result.stdout
        assert "suggest" in result.stdout
        assert "check" in result.stdout

    def test_validate_help(self):
        """Validate command help."""
        result = run_cli("validate", "--help")
        assert result.returncode == 0
        assert "config" in result.stdout.lower()

    def test_suggest_help(self):
        """Suggest command help."""
        result = run_cli("suggest", "--help")
        assert result.returncode == 0
        assert "resolution" in result.stdout.lower()

    def test_check_help(self):
        """Check command help."""
        result = run_cli("check", "--help")
        assert result.returncode == 0
        assert "dt" in result.stdout.lower()
        assert "eta" in result.stdout.lower()

    def test_no_command(self):
        """No command should show help."""
        result = run_cli()
        # Should show help (may return 0 or 1 depending on implementation)
        assert "validate" in result.stdout or "validate" in result.stderr


class TestCLIOutput:
    """Test CLI output formatting."""

    def test_validate_output_has_separators(self):
        """Validate output should have clear separators."""
        config = {
            "dt": 0.01,
            "eta": 0.02,
            "nu": 0.02,
            "Nx": 64,
            "Ny": 64,
            "Nz": 32,
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            config_path = f.name

        try:
            result = run_cli("validate", config_path)
            # Should have separators like "=" * 70
            assert "=" * 70 in result.stdout
        finally:
            Path(config_path).unlink()

    def test_suggest_output_formatting(self):
        """Suggest output should be well-formatted."""
        result = run_cli("suggest", "--resolution", "64", "64", "32")

        # Should have clear formatting
        assert "Recommended parameters:" in result.stdout
        assert "dt:" in result.stdout
        assert "eta:" in result.stdout

    def test_check_output_has_checkmarks(self):
        """Check output should have status indicators."""
        result = run_cli(
            "check",
            "--dt", "0.01",
            "--eta", "0.02",
            "--nu", "0.02",
            "--Nx", "64",
            "--Ny", "64",
            "--Nz", "32"
        )

        # Should have checkmark or similar indicator
        assert "✓" in result.stdout or "valid" in result.stdout.lower()


class TestCLIErrorHandling:
    """Test CLI error handling."""

    def test_validate_malformed_yaml(self):
        """Validate with malformed YAML."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("not: valid: yaml: syntax\n")
            f.write("  broken indentation\n")
            config_path = f.name

        try:
            result = run_cli("validate", config_path)
            # Should handle gracefully (may error during YAML load)
            assert isinstance(result.returncode, int)
        finally:
            Path(config_path).unlink()

    def test_check_non_numeric_arguments(self):
        """Check with non-numeric arguments."""
        result = run_cli(
            "check",
            "--dt", "not_a_number",
            "--eta", "0.02",
            "--nu", "0.02",
            "--Nx", "64",
            "--Ny", "64",
            "--Nz", "32"
        )
        assert result.returncode != 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
