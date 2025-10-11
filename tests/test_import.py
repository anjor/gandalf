"""
Bootstrap tests: Verify all core dependencies can be imported.

These tests validate the basic installation and ensure that:
1. Core package (krmhd) imports successfully
2. All required dependencies are available
3. JAX can detect available compute devices (Metal/CPU/CUDA)
"""

import sys


def test_import_krmhd():
    """Test that krmhd package imports and has version."""
    import krmhd
    
    assert hasattr(krmhd, "__version__")
    assert krmhd.__version__ == "0.1.0"


def test_import_core_dependencies():
    """Test that all core dependencies import successfully."""
    import h5py
    import jax
    import matplotlib.pyplot as plt
    import numpy

    # Basic smoke tests
    assert hasattr(jax, "numpy")
    assert hasattr(numpy, "array")
    assert hasattr(h5py, "File")
    assert plt is not None


def test_jax_device_detection():
    """Test that JAX can detect compute devices.

    This test verifies that JAX is properly installed and can see
    available hardware accelerators (Metal on macOS, CUDA on Linux,
    or CPU fallback).
    """
    import jax

    devices = jax.devices()
    assert len(devices) > 0, "JAX should detect at least one device (CPU fallback)"

    # Print device info for diagnostics
    device = devices[0]
    print(f"\nJAX detected device: {device.platform} - {device.device_kind}")

    # Verify device is one of the expected platforms
    # Note: JAX Metal backend uses uppercase "METAL"
    assert device.platform.upper() in ["CPU", "GPU", "TPU", "METAL"], \
        f"Unknown JAX platform: {device.platform}"


def test_jax_basic_operation():
    """Test that JAX can perform basic array operations.

    Note: This test may be skipped on Metal backend due to experimental
    limitations. Use CPU backend for testing if needed:
    JAX_PLATFORMS=cpu pytest tests/test_import.py
    """
    import jax
    import jax.numpy as jnp
    import pytest

    # Check if we're on Metal backend
    device = jax.devices()[0]
    is_metal = device.platform.upper() == "METAL"

    try:
        # Simple operation to verify JAX is functional
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.sum(x)

        assert float(y) == 6.0, "JAX array operations should work"
    except Exception as e:
        if is_metal and "default_memory_space" in str(e):
            pytest.skip(f"Known Metal backend limitation: {e}")
        else:
            raise


def test_numpy_version():
    """Test that numpy version is compatible."""
    import numpy as np
    
    major, minor, *_ = map(int, np.__version__.split('.'))
    assert (major, minor) >= (1, 24), \
        f"numpy version {np.__version__} is too old (need >=1.24.0)"


def test_matplotlib_backend():
    """Test that matplotlib has a usable backend."""
    import matplotlib
    import matplotlib.pyplot as plt
    
    # Ensure a non-interactive backend is set for headless testing
    backend = matplotlib.get_backend()
    assert backend is not None, "matplotlib should have a backend configured"


if __name__ == "__main__":
    # Allow running tests directly with: uv run python tests/test_import.py
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
