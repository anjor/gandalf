"""
Runs tests/multidevice in a subprocess with four virtual CPU devices (Issue #191).

XLA reads --xla_force_host_platform_device_count when JAX creates its CPU
backend, which this session has already done with one device, so the sharded
tests need a fresh process.
"""

import os
import subprocess
import sys
from pathlib import Path

MULTIDEVICE_TESTS = Path(__file__).parent / "multidevice"


def test_sharded_slices_on_four_cpu_devices():
    env = dict(
        os.environ,
        JAX_PLATFORMS="cpu",
        XLA_FLAGS=(os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=4"),
    )
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(MULTIDEVICE_TESTS)],
        env=env, capture_output=True, text=True, timeout=900,
    )
    report = result.stdout[-6000:] + result.stderr[-3000:]
    assert result.returncode == 0, report
    assert " passed" in result.stdout and " skipped" not in result.stdout, report
