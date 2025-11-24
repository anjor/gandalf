"""
Unit tests for parameter validation module.

Tests all validation functions including:
- Overflow safety checks
- CFL condition validation
- Forcing stability checks
- Parameter suggestion
- Config dict validation
"""

import pytest
import jax.numpy as jnp

from krmhd.validation import (
    ValidationResult,
    validate_overflow_safety,
    validate_cfl_condition,
    validate_forcing_stability,
    validate_parameters,
    validate_config_dict,
    suggest_parameters,
)


class TestValidationResult:
    """Test ValidationResult dataclass."""

    def test_valid_result_no_warnings(self):
        """Valid result with no warnings."""
        result = ValidationResult(valid=True, warnings=[], errors=[], suggestions=[])
        assert result.valid
        assert len(result.warnings) == 0
        assert len(result.errors) == 0

    def test_invalid_result_with_errors(self):
        """Invalid result with errors."""
        result = ValidationResult(
            valid=False,
            warnings=["Warning 1"],
            errors=["Error 1", "Error 2"],
            suggestions=["Fix this"]
        )
        assert not result.valid
        assert len(result.errors) == 2
        assert len(result.warnings) == 1

    def test_print_report_no_crash(self, capsys):
        """Ensure print_report doesn't crash."""
        result = ValidationResult(
            valid=False,
            warnings=["Test warning"],
            errors=["Test error"],
            suggestions=["Test suggestion"]
        )
        result.print_report()
        captured = capsys.readouterr()
        assert "ERRORS" in captured.out
        assert "WARNINGS" in captured.out
        assert "SUGGESTIONS" in captured.out


class TestOverflowSafety:
    """Test overflow safety validation."""

    def test_safe_parameters(self):
        """Safe parameters should pass."""
        result = validate_overflow_safety(eta=0.5, dt=0.01, nu=0.5)
        assert result.valid
        assert len(result.errors) == 0

    def test_resistivity_overflow(self):
        """Resistivity overflow should be caught."""
        result = validate_overflow_safety(eta=60.0, dt=1.0, nu=0.5)
        assert not result.valid
        assert len(result.errors) >= 1
        assert any("Resistivity overflow" in err for err in result.errors)

    def test_collision_overflow(self):
        """Collision overflow should be caught."""
        result = validate_overflow_safety(eta=0.5, dt=1.0, nu=60.0)
        assert not result.valid
        assert len(result.errors) >= 1
        assert any("Collision overflow" in err for err in result.errors)

    def test_both_overflow(self):
        """Both resistivity and collision overflow."""
        result = validate_overflow_safety(eta=60.0, dt=1.0, nu=60.0)
        assert not result.valid
        assert len(result.errors) >= 2

    def test_near_threshold_warning(self):
        """Near threshold should produce warning."""
        # 0.8 * 50 = 40, so eta*dt = 45 should warn
        result = validate_overflow_safety(eta=45.0, dt=1.0, nu=0.5)
        assert result.valid  # Still valid
        assert len(result.warnings) >= 1

    def test_high_hyper_order_suggestion(self):
        """High hyper-order should produce suggestion."""
        result = validate_overflow_safety(
            eta=0.5, dt=0.01, nu=0.5, hyper_r=4, hyper_n=3
        )
        assert result.valid
        assert len(result.suggestions) >= 1
        assert any("hyper-order" in sug.lower() for sug in result.suggestions)

    def test_no_nu_provided(self):
        """Should handle nu=None gracefully."""
        result = validate_overflow_safety(eta=0.5, dt=0.01, nu=None)
        assert result.valid
        # Should not check collision overflow

    def test_custom_threshold(self):
        """Custom threshold should be respected."""
        result = validate_overflow_safety(
            eta=30.0, dt=1.0, nu=0.5, threshold=25.0
        )
        assert not result.valid  # 30 > 25
        assert len(result.errors) >= 1

    def test_zero_dt_edge_case(self):
        """Zero timestep should not crash (though physically invalid)."""
        result = validate_overflow_safety(eta=0.5, dt=0.0, nu=0.5)
        # Should handle gracefully (may warn or error depending on implementation)
        assert isinstance(result, ValidationResult)

    def test_negative_parameters(self):
        """Negative parameters (physically invalid but should not crash)."""
        result = validate_overflow_safety(eta=-0.5, dt=0.01, nu=-0.5)
        assert isinstance(result, ValidationResult)


class TestCFLCondition:
    """Test CFL condition validation."""

    def test_safe_cfl(self):
        """Safe CFL should pass."""
        result = validate_cfl_condition(dt=0.01, dx=0.1, v_A=1.0, cfl_limit=1.0)
        assert result.valid
        assert len(result.errors) == 0

    def test_violated_cfl(self):
        """Violated CFL should error."""
        result = validate_cfl_condition(dt=0.5, dx=0.1, v_A=1.0, cfl_limit=1.0)
        assert not result.valid
        assert len(result.errors) >= 1
        assert any("CFL condition violated" in err for err in result.errors)

    def test_near_cfl_limit_warning(self):
        """Near CFL limit should warn."""
        # CFL = 0.9 > 0.8 * 1.0 should warn
        result = validate_cfl_condition(dt=0.09, dx=0.1, v_A=1.0, cfl_limit=1.0)
        assert result.valid
        assert len(result.warnings) >= 1

    def test_custom_cfl_limit(self):
        """Custom CFL limit should be respected."""
        result = validate_cfl_condition(dt=0.01, dx=0.1, v_A=1.0, cfl_limit=0.5)
        # CFL = 0.1, which is < 0.5, so valid
        assert result.valid

    def test_high_velocity(self):
        """High Alfvén velocity increases CFL number."""
        result = validate_cfl_condition(dt=0.5, dx=0.1, v_A=10.0, cfl_limit=1.0)
        # CFL = 0.5 * 10.0 / 0.1 = 50, way over limit
        assert not result.valid

    def test_zero_dx_edge_case(self):
        """Zero grid spacing should not crash."""
        result = validate_cfl_condition(dt=0.01, dx=0.0, v_A=1.0)
        # Should handle gracefully with error
        assert isinstance(result, ValidationResult)
        assert not result.valid  # Should be invalid
        assert len(result.errors) >= 1
        assert any("grid spacing" in err.lower() for err in result.errors)

    def test_zero_v_A_edge_case(self):
        """Zero Alfvén velocity should not crash."""
        result = validate_cfl_condition(dt=0.01, dx=0.1, v_A=0.0)
        # Should handle gracefully with error
        assert isinstance(result, ValidationResult)
        assert not result.valid
        assert len(result.errors) >= 1
        assert any("alfvén velocity" in err.lower() for err in result.errors)

    def test_negative_v_A_edge_case(self):
        """Negative Alfvén velocity should error."""
        result = validate_cfl_condition(dt=0.01, dx=0.1, v_A=-1.0)
        assert not result.valid
        assert len(result.errors) >= 1
        assert any("alfvén velocity" in err.lower() for err in result.errors)


class TestForcingStability:
    """Test forcing stability validation."""

    def test_low_resolution_safe(self):
        """Low resolution with moderate forcing."""
        result = validate_forcing_stability(
            force_amplitude=0.3, eta=0.02, resolution=64, hyper_r=1
        )
        # Always returns valid (warnings only)
        assert result.valid

    def test_high_resolution_high_amplitude_warning(self):
        """High resolution + high amplitude should warn."""
        result = validate_forcing_stability(
            force_amplitude=0.5, eta=0.02, resolution=128, hyper_r=1
        )
        assert result.valid  # Still valid
        assert len(result.warnings) >= 1

    def test_very_high_resolution(self):
        """256³ resolution with high amplitude."""
        result = validate_forcing_stability(
            force_amplitude=0.4, eta=0.02, resolution=256, hyper_r=1
        )
        assert result.valid
        # Should warn about high amplitude
        assert len(result.warnings) >= 1

    def test_injection_dissipation_imbalance(self):
        """Large injection vs small dissipation."""
        # Need injection²/(eta*k²) > 10
        # With amplitude=2.0, eta=0.001, n_max=2:
        # injection=4.0, dissipation≈0.158, ratio≈25 > 10
        result = validate_forcing_stability(
            force_amplitude=2.0,  # Very large
            eta=0.001,  # Very small
            resolution=64,
            hyper_r=1,
            n_force_max=2
        )
        assert result.valid
        assert len(result.warnings) >= 1
        assert any("injection" in warn.lower() for warn in result.warnings)

    def test_high_hyper_order_warning(self):
        """High hyper-order should warn."""
        result = validate_forcing_stability(
            force_amplitude=0.3, eta=0.02, resolution=64, hyper_r=4
        )
        assert result.valid
        assert len(result.warnings) >= 1
        assert any("hyper-order" in warn.lower() for warn in result.warnings)

    def test_low_amplitude_no_warning(self):
        """Low forcing amplitude should not warn."""
        result = validate_forcing_stability(
            force_amplitude=0.05, eta=0.02, resolution=128, hyper_r=2
        )
        assert result.valid
        # May still have some suggestions but no critical warnings


class TestValidateParameters:
    """Test comprehensive parameter validation."""

    def test_all_valid_parameters(self):
        """All parameters valid."""
        result = validate_parameters(
            dt=0.01, eta=0.02, nu=0.02,
            Nx=64, Ny=64, Nz=32,
            v_A=1.0, force_amplitude=0.3,
            hyper_r=1, hyper_n=1
        )
        assert result.valid

    def test_overflow_caught(self):
        """Overflow should be caught."""
        result = validate_parameters(
            dt=1.0, eta=60.0, nu=0.02,
            Nx=64, Ny=64, Nz=32
        )
        assert not result.valid
        assert any("overflow" in err.lower() for err in result.errors)

    def test_cfl_violation_caught(self):
        """CFL violation should be caught."""
        result = validate_parameters(
            dt=1.0, eta=0.02, nu=0.02,
            Nx=64, Ny=64, Nz=32,
            v_A=10.0
        )
        assert not result.valid
        assert any("CFL" in err for err in result.errors)

    def test_forcing_warnings_included(self):
        """Forcing stability warnings should be included."""
        # Use smaller dt to avoid CFL violation with 256³
        result = validate_parameters(
            dt=0.001, eta=0.001, nu=0.02,  # Smaller dt
            Nx=256, Ny=256, Nz=128,
            force_amplitude=0.5,
            hyper_r=4
        )
        # Should have warnings (but still valid now that CFL is OK)
        assert result.valid
        assert len(result.warnings) > 0

    def test_no_forcing(self):
        """No forcing should skip forcing validation."""
        result = validate_parameters(
            dt=0.01, eta=0.02, nu=0.02,
            Nx=64, Ny=64, Nz=32,
            force_amplitude=None  # No forcing
        )
        assert result.valid

    def test_zero_forcing(self):
        """Zero forcing amplitude should skip forcing checks."""
        result = validate_parameters(
            dt=0.01, eta=0.02, nu=0.02,
            Nx=64, Ny=64, Nz=32,
            force_amplitude=0.0
        )
        assert result.valid


class TestSuggestParameters:
    """Test parameter suggestion function."""

    def test_suggest_forced_64cubed(self):
        """Suggest parameters for 64³ forced turbulence."""
        params = suggest_parameters(64, 64, 32, simulation_type="forced")

        assert "dt" in params
        assert "eta" in params
        assert "nu" in params
        assert "force_amplitude" in params

        # Check reasonable values
        assert params["dt"] > 0
        assert params["eta"] > 0
        assert params["force_amplitude"] > 0

    def test_suggest_decaying_128cubed(self):
        """Suggest parameters for 128³ decaying turbulence."""
        params = suggest_parameters(128, 128, 64, simulation_type="decaying")

        assert params["force_amplitude"] == 0.0  # No forcing
        assert params["eta"] > 0
        assert params["dt"] > 0

    def test_suggest_benchmark(self):
        """Suggest benchmark parameters."""
        params = suggest_parameters(64, 64, 32, simulation_type="benchmark")

        assert params["eta"] == 0.01  # Standard benchmark value
        assert params["nu"] == 0.01

    def test_higher_resolution_smaller_eta(self):
        """Higher resolution should suggest smaller eta."""
        params_64 = suggest_parameters(64, 64, 32, simulation_type="forced")
        params_256 = suggest_parameters(256, 256, 128, simulation_type="forced")

        # Higher res should have smaller dissipation
        assert params_256["eta"] < params_64["eta"]

    def test_cfl_safety_respected(self):
        """CFL safety factor should be respected."""
        params = suggest_parameters(64, 64, 32, target_cfl=0.5)

        # dt should be larger with higher CFL safety
        assert params["cfl_safety"] == 0.5

        # Verify suggested dt satisfies CFL
        dx_min = 1.0 / 64
        cfl_actual = params["dt"] * params["v_A"] / dx_min
        assert cfl_actual <= 0.5

    def test_suggested_params_pass_validation(self):
        """Suggested parameters should pass validation."""
        params = suggest_parameters(64, 64, 32, simulation_type="forced")

        result = validate_parameters(
            params["dt"], params["eta"], params["nu"],
            64, 64, 32,
            params["v_A"],
            params["force_amplitude"],
            hyper_r=1, hyper_n=1
        )

        assert result.valid


class TestValidateConfigDict:
    """Test config dictionary validation."""

    def test_valid_flat_config(self):
        """Flat config dict should work."""
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
        result = validate_config_dict(config)
        assert result.valid

    def test_nested_config(self):
        """Nested config structure should work."""
        config = {
            "grid": {"Nx": 64, "Ny": 64, "Nz": 32},
            "physics": {"hyper_r": 2, "hyper_n": 2},
            "forcing": {"amplitude": 0.3, "n_max": 2},
            "dt": 0.01,
            "eta": 0.02,
            "nu": 0.02,
        }
        result = validate_config_dict(config)
        assert result.valid

    def test_missing_parameters_use_defaults(self):
        """Missing parameters should use defaults."""
        config = {"Nx": 64, "Ny": 64, "Nz": 32}
        result = validate_config_dict(config)
        # Should not crash, uses defaults
        assert isinstance(result, ValidationResult)

    def test_invalid_nested_config(self):
        """Invalid nested config should be caught."""
        config = {
            "grid": {"Nx": 64, "Ny": 64, "Nz": 32},
            "dt": 1.0,
            "eta": 60.0,  # Overflow
            "nu": 0.02,
        }
        result = validate_config_dict(config)
        assert not result.valid


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_timestep(self):
        """Zero timestep should not crash."""
        result = validate_parameters(
            dt=0.0, eta=0.02, nu=0.02,
            Nx=64, Ny=64, Nz=32
        )
        # Should handle gracefully
        assert isinstance(result, ValidationResult)

    def test_negative_parameters(self):
        """Negative (unphysical) parameters should not crash."""
        result = validate_parameters(
            dt=-0.01, eta=-0.02, nu=-0.02,
            Nx=64, Ny=64, Nz=32
        )
        assert isinstance(result, ValidationResult)

    def test_very_small_resolution(self):
        """Very small resolution."""
        params = suggest_parameters(8, 8, 4, simulation_type="forced")
        assert params["dt"] > 0
        # Should not crash

    def test_very_large_resolution(self):
        """Very large resolution."""
        params = suggest_parameters(512, 512, 256, simulation_type="forced")
        assert params["dt"] > 0
        assert params["eta"] > 0

    def test_extreme_cfl_target(self):
        """Extreme CFL safety factors."""
        params_low = suggest_parameters(64, 64, 32, target_cfl=0.01)
        params_high = suggest_parameters(64, 64, 32, target_cfl=0.9)

        # Higher CFL should give larger dt
        assert params_high["dt"] > params_low["dt"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
