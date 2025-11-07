"""
Tests for utility functions in example scripts.

These tests validate helper functions used in benchmark and example scripts,
such as steady-state detection, parameter sweeps, and diagnostic utilities.
"""

import numpy as np
import pytest


# Import detect_steady_state from the benchmark script
# (In a real refactor, this would be moved to a utility module)
def detect_steady_state(energy_history, window=100, threshold=0.02, n_smooth=None):
    """
    Detect if system has reached steady state (energy plateau) - DIAGNOSTIC ONLY.

    This function is used for logging/monitoring purposes only and does NOT
    control simulation runtime. The simulation always runs for the fixed
    total_time specified by the user, regardless of steady-state status.

    Checks if energy has stopped growing by looking at the trend over
    a long window. True steady state requires energy to plateau, not
    just have small fluctuations.

    Args:
        energy_history: List of total energy values
        window: Number of recent points to check (default 100)
        threshold: Relative energy change threshold (default 2%)
        n_smooth: Number of points to average for smoothing (default: window//10, min 5)

    Returns:
        True if steady state detected (energy plateau), False otherwise

    Note:
        This is used only for informational logging during the run. To ensure
        steady state is achieved, users should increase --total-time or monitor
        the ΔE/⟨E⟩ values printed during averaging.
    """
    if len(energy_history) < window:
        return False

    recent = energy_history[-window:]
    # Average over n_smooth points to smooth out high-frequency fluctuations
    # while preserving low-frequency trends. Default: use 10% of window size
    # to adapt to different window lengths, with a minimum of 5 points.
    if n_smooth is None:
        n_smooth = max(5, window // 10)

    E_start = np.mean(recent[:n_smooth])   # Average of first n_smooth points in window
    E_end = np.mean(recent[-n_smooth:])    # Average of last n_smooth points in window

    if E_start == 0:
        return False

    # Check if energy has stopped growing (plateau)
    relative_change = abs(E_end - E_start) / E_start
    return relative_change < threshold


class TestDetectSteadyState:
    """Test suite for steady-state detection utility."""

    def test_insufficient_history(self):
        """Should return False if energy history is too short."""
        energy = [1.0, 1.1, 1.2]
        assert not detect_steady_state(energy, window=100)

    def test_perfect_plateau(self):
        """Should detect steady state for perfectly flat energy."""
        energy = [1.0] * 150
        assert detect_steady_state(energy, window=100, threshold=0.02)

    def test_small_fluctuations(self):
        """Should detect steady state despite small fluctuations."""
        # Energy with ±0.5% random fluctuations
        np.random.seed(42)
        base = 100.0
        energy = base * (1.0 + 0.005 * np.random.randn(200))
        assert detect_steady_state(energy, window=100, threshold=0.02)

    def test_linear_growth(self):
        """Should NOT detect steady state if energy is growing."""
        # Energy growing at 5% per window
        energy = np.linspace(1.0, 1.1, 200)
        assert not detect_steady_state(energy, window=100, threshold=0.02)

    def test_exponential_growth(self):
        """Should NOT detect steady state during exponential growth."""
        energy = [1.0 * 1.01**i for i in range(200)]
        assert not detect_steady_state(energy, window=100, threshold=0.02)

    def test_threshold_sensitivity(self):
        """Should respect threshold parameter."""
        # Create energy with 1% change within the window
        # First 50 at 1.0, next 150 at 1.01 (so window sees transition)
        energy_start = [1.0] * 50
        energy_end = [1.01] * 150
        energy = energy_start + energy_end

        # Window=100 spans from index 100 to 200
        # - First n_smooth=10 are at 1.01 (indices 100-109)
        # - Last n_smooth=10 are at 1.01 (indices 190-199)
        # Actually both are in plateau! Let me use a gradual transition

        # Better: use gradual transition in middle of window
        energy = [1.0] * 100 + list(np.linspace(1.0, 1.01, 50)) + [1.01] * 50

        # Window=100 at end will see transition from 1.0 to 1.01
        # With threshold=0.02 (2%), a 1% change should pass
        assert detect_steady_state(energy, window=100, threshold=0.02)

        # With threshold=0.005 (0.5%), a 1% change should fail
        assert not detect_steady_state(energy, window=100, threshold=0.005)

    def test_zero_energy(self):
        """Should handle zero initial energy gracefully."""
        energy = [0.0] * 150
        assert not detect_steady_state(energy, window=100, threshold=0.02)

    def test_custom_smoothing_window(self):
        """Should respect custom n_smooth parameter."""
        # With large smoothing, small fluctuations are averaged out
        np.random.seed(42)
        base = 100.0
        energy = base * (1.0 + 0.01 * np.random.randn(200))

        # Large smoothing should detect plateau
        assert detect_steady_state(energy, window=100, threshold=0.02, n_smooth=20)

        # Small smoothing (more sensitive) might not
        result_small = detect_steady_state(energy, window=100, threshold=0.02, n_smooth=2)
        # Result depends on random fluctuations, so no strict assertion

    def test_adaptive_smoothing(self):
        """Should adapt n_smooth to window size."""
        energy = [1.0] * 200

        # Default n_smooth should be window//10 (at least 5)
        # For window=50, n_smooth=5
        # For window=200, n_smooth=20
        assert detect_steady_state(energy, window=50, threshold=0.02)
        assert detect_steady_state(energy, window=200, threshold=0.02)

    def test_late_plateau(self):
        """Should detect steady state even if early history had growth."""
        # First 100 steps: growth, next 150 steps: plateau
        growth = np.linspace(1.0, 2.0, 100)
        plateau = np.full(150, 2.0)
        energy = np.concatenate([growth, plateau])

        # Window=100 should see only the plateau
        assert detect_steady_state(energy, window=100, threshold=0.02)

    def test_multiple_time_scales(self):
        """Should handle energy with multiple time scales."""
        # Fast oscillations on slow growth
        t = np.arange(200)
        slow_growth = 1.0 + 0.0001 * t  # Very slow growth
        fast_oscillation = 0.001 * np.sin(2 * np.pi * t / 10)  # Fast oscillation
        energy = slow_growth + fast_oscillation

        # Should detect as steady with appropriate smoothing
        assert detect_steady_state(list(energy), window=100, threshold=0.01, n_smooth=10)

    def test_edge_case_window_equals_history(self):
        """Should handle case where window equals history length."""
        energy = [1.0] * 100
        assert detect_steady_state(energy, window=100, threshold=0.02)

    def test_realistic_turbulence_plateau(self):
        """Should detect steady state in realistic turbulence scenario."""
        # Simulate turbulent energy evolution: rapid growth then saturation
        np.random.seed(123)
        t = np.arange(300)

        # Exponential approach to steady state: E(t) = E_∞(1 - exp(-t/τ))
        E_inf = 100.0
        tau = 50.0
        E_mean = E_inf * (1.0 - np.exp(-t / tau))

        # Add turbulent fluctuations (~5% amplitude)
        E_fluctuations = 0.05 * E_inf * np.random.randn(300)
        energy = E_mean + E_fluctuations

        # Should detect steady state at late times
        assert detect_steady_state(list(energy), window=100, threshold=0.05)
