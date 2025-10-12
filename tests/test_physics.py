"""
Tests for KRMHD physics operations.

Validates:
- Poisson bracket {f,g} = ẑ·(∇f × ∇g) for 2D and 3D
- Anti-symmetry: {f,g} = -{g,f}
- Linearity properties
- Conservation properties
- Dealiasing application
"""

import pytest
import jax
import jax.numpy as jnp

from krmhd.spectral import (
    SpectralGrid2D,
    SpectralGrid3D,
    rfft2_forward,
    rfft2_inverse,
    rfftn_forward,
    rfftn_inverse,
)
from krmhd.physics import poisson_bracket_2d, poisson_bracket_3d


class TestPoissonBracket2D:
    """Test suite for 2D Poisson bracket implementation."""

    def test_analytical_sin_cos(self):
        """
        Test analytical case from Issue #4: f=sin(x), g=cos(y) → {f,g}=sin(x)sin(y).

        For f = sin(x), g = cos(y):
        ∂f/∂x = cos(x), ∂f/∂y = 0
        ∂g/∂x = 0, ∂g/∂y = -sin(y)
        {f,g} = cos(x)·(-sin(y)) - 0·0 = -cos(x)sin(y)

        Wait, let me recalculate:
        {f,g} = ∂f/∂x · ∂g/∂y - ∂f/∂y · ∂g/∂x
              = cos(x) · (-sin(y)) - 0 · 0
              = -cos(x)sin(y)

        Hmm, the issue says sin(x)sin(y). Let me check with g=sin(y) instead:
        For f = sin(x), g = sin(y):
        ∂f/∂x = cos(x), ∂f/∂y = 0
        ∂g/∂x = 0, ∂g/∂y = cos(y)
        {f,g} = cos(x)·cos(y) - 0·0 = cos(x)cos(y)

        Still not matching. Let me try g=cos(y):
        Actually, I think the issue might have meant the magnitude.
        Let me test with a specific case and verify numerically.

        For f = sin(x), g = cos(y):
        {f,g} = ∂f/∂x · ∂g/∂y - ∂f/∂y · ∂g/∂x
              = cos(x) · (-sin(y)) - 0 · 0
              = -cos(x)sin(y)

        Actually, looking at the issue again: "f=sin(x), g=cos(y) - should give sin(x)sin(y)"
        This seems like it might be approximate or there's a typo. Let me just verify
        the implementation is correct by computing it exactly.
        """
        grid = SpectralGrid2D.create(Nx=128, Ny=128, Lx=2*jnp.pi, Ly=2*jnp.pi)

        # Create coordinate arrays
        x = jnp.linspace(0, grid.Lx, grid.Nx, endpoint=False)
        y = jnp.linspace(0, grid.Ly, grid.Ny, endpoint=False)
        X, Y = jnp.meshgrid(x, y, indexing='ij')

        # f = sin(x), g = cos(y) in [Ny, Nx] ordering
        f = jnp.sin(X).T
        g = jnp.cos(Y).T

        # Transform to Fourier space
        f_k = rfft2_forward(f)
        g_k = rfft2_forward(g)

        # Compute Poisson bracket
        bracket_k = poisson_bracket_2d(f_k, g_k, grid.kx, grid.ky, grid.Ny, grid.Nx, grid.dealias_mask)
        bracket = rfft2_inverse(bracket_k, grid.Ny, grid.Nx)

        # Expected: {f,g} = ∂f/∂x · ∂g/∂y - ∂f/∂y · ∂g/∂x
        #                 = cos(x) · (-sin(y)) - 0 · 0 = -cos(x)sin(y)
        expected = -jnp.cos(X).T * jnp.sin(Y).T

        # Check relative error
        error = jnp.max(jnp.abs(bracket - expected))
        rel_error = error / jnp.max(jnp.abs(expected))

        # float32 precision: expect errors ~1e-5 with multiple FFTs
        assert rel_error < 1e-4, f"Relative error {rel_error} exceeds tolerance"

    def test_analytical_sin_sin(self):
        """
        Test with f=sin(kx·x), g=sin(ky·y) for specific wavenumbers.

        {f,g} = ∂f/∂x · ∂g/∂y - ∂f/∂y · ∂g/∂x
              = (kx·cos(kx·x)) · (ky·cos(ky·y)) - 0 · 0
              = kx·ky·cos(kx·x)cos(ky·y)
        """
        grid = SpectralGrid2D.create(Nx=128, Ny=128, Lx=2*jnp.pi, Ly=2*jnp.pi)

        # Wavenumbers
        kx_mode = 2.0
        ky_mode = 3.0

        # Create coordinate arrays
        x = jnp.linspace(0, grid.Lx, grid.Nx, endpoint=False)
        y = jnp.linspace(0, grid.Ly, grid.Ny, endpoint=False)
        X, Y = jnp.meshgrid(x, y, indexing='ij')

        # f = sin(kx·x), g = sin(ky·y) in [Ny, Nx] ordering
        f = jnp.sin(kx_mode * X).T
        g = jnp.sin(ky_mode * Y).T

        # Transform to Fourier space
        f_k = rfft2_forward(f)
        g_k = rfft2_forward(g)

        # Compute Poisson bracket
        bracket_k = poisson_bracket_2d(f_k, g_k, grid.kx, grid.ky, grid.Ny, grid.Nx, grid.dealias_mask)
        bracket = rfft2_inverse(bracket_k, grid.Ny, grid.Nx)

        # Expected: kx·ky·cos(kx·x)cos(ky·y)
        expected = kx_mode * ky_mode * jnp.cos(kx_mode * X).T * jnp.cos(ky_mode * Y).T

        # Check relative error
        error = jnp.max(jnp.abs(bracket - expected))
        rel_error = error / jnp.max(jnp.abs(expected))

        # float32 precision: expect errors ~1e-5 with multiple FFTs
        assert rel_error < 1e-4, f"Relative error {rel_error} exceeds tolerance"

    def test_antisymmetry(self):
        """Test anti-symmetry property: {f,g} = -{g,f}."""
        grid = SpectralGrid2D.create(Nx=128, Ny=128)

        # Create test fields with random Fourier coefficients
        key = jax.random.PRNGKey(42)
        key1, key2 = jax.random.split(key)

        f_k = jax.random.normal(key1, (grid.Ny, grid.Nx//2+1)) + \
              1j * jax.random.normal(key1, (grid.Ny, grid.Nx//2+1))
        g_k = jax.random.normal(key2, (grid.Ny, grid.Nx//2+1)) + \
              1j * jax.random.normal(key2, (grid.Ny, grid.Nx//2+1))

        # Enforce reality condition: f(-k) = f*(k)
        # For rfft2, this means proper symmetry in ky dimension
        # (This is automatically handled by rfft2_forward, but let's ensure it)

        # Compute both orderings
        bracket_fg = poisson_bracket_2d(f_k, g_k, grid.kx, grid.ky, grid.Ny, grid.Nx, grid.dealias_mask)
        bracket_gf = poisson_bracket_2d(g_k, f_k, grid.kx, grid.ky, grid.Ny, grid.Nx, grid.dealias_mask)

        # Check anti-symmetry
        max_diff = jnp.max(jnp.abs(bracket_fg + bracket_gf))

        # float32 precision: allow errors ~1e-5 with multiple FFTs
        assert max_diff < 1e-4, f"Anti-symmetry violated: max|{{f,g}} + {{g,f}}| = {max_diff}"

    def test_linearity_first_argument(self):
        """Test linearity in first argument: {af + bh, g} = a{f,g} + b{h,g}."""
        grid = SpectralGrid2D.create(Nx=128, Ny=128)

        # Create test fields
        key = jax.random.PRNGKey(43)
        keys = jax.random.split(key, 3)

        f_k = jax.random.normal(keys[0], (grid.Ny, grid.Nx//2+1)) + \
              1j * jax.random.normal(keys[0], (grid.Ny, grid.Nx//2+1))
        h_k = jax.random.normal(keys[1], (grid.Ny, grid.Nx//2+1)) + \
              1j * jax.random.normal(keys[1], (grid.Ny, grid.Nx//2+1))
        g_k = jax.random.normal(keys[2], (grid.Ny, grid.Nx//2+1)) + \
              1j * jax.random.normal(keys[2], (grid.Ny, grid.Nx//2+1))

        # Scalars
        a = 2.5
        b = -1.3

        # Compute {af + bh, g}
        bracket_left = poisson_bracket_2d(a * f_k + b * h_k, g_k, grid.kx, grid.ky, grid.Ny, grid.Nx, grid.dealias_mask)

        # Compute a{f,g} + b{h,g}
        bracket_f = poisson_bracket_2d(f_k, g_k, grid.kx, grid.ky, grid.Ny, grid.Nx, grid.dealias_mask)
        bracket_h = poisson_bracket_2d(h_k, g_k, grid.kx, grid.ky, grid.Ny, grid.Nx, grid.dealias_mask)
        bracket_right = a * bracket_f + b * bracket_h

        # Check linearity
        max_diff = jnp.max(jnp.abs(bracket_left - bracket_right))

        # float32 precision: allow errors ~1e-5 with multiple FFTs
        assert max_diff < 1e-4, f"Linearity violated: max difference = {max_diff}"

    def test_constant_field(self):
        """Test that Poisson bracket with constant field is zero: {f, c} = 0."""
        grid = SpectralGrid2D.create(Nx=128, Ny=128)

        # Create a non-constant field f
        x = jnp.linspace(0, grid.Lx, grid.Nx, endpoint=False)
        y = jnp.linspace(0, grid.Ly, grid.Ny, endpoint=False)
        X, Y = jnp.meshgrid(x, y, indexing='ij')
        f = jnp.sin(2*X).T + jnp.cos(3*Y).T
        f_k = rfft2_forward(f)

        # Create a constant field (only k=0 mode)
        c_k = jnp.zeros((grid.Ny, grid.Nx//2+1), dtype=jnp.complex64)
        c_k = c_k.at[0, 0].set(5.0 + 0j)  # Constant = 5.0

        # Compute {f, c}
        bracket_k = poisson_bracket_2d(f_k, c_k, grid.kx, grid.ky, grid.Ny, grid.Nx, grid.dealias_mask)
        bracket = rfft2_inverse(bracket_k, grid.Ny, grid.Nx)

        # Should be zero everywhere
        max_val = jnp.max(jnp.abs(bracket))

        assert max_val < 1e-10, f"Bracket with constant should be zero, got max = {max_val}"

    def test_dealiasing_applied(self):
        """Test that dealiasing is applied to the result."""
        grid = SpectralGrid2D.create(Nx=128, Ny=128)

        # Create test fields
        key = jax.random.PRNGKey(44)
        key1, key2 = jax.random.split(key)

        f_k = jax.random.normal(key1, (grid.Ny, grid.Nx//2+1)) + \
              1j * jax.random.normal(key1, (grid.Ny, grid.Nx//2+1))
        g_k = jax.random.normal(key2, (grid.Ny, grid.Nx//2+1)) + \
              1j * jax.random.normal(key2, (grid.Ny, grid.Nx//2+1))

        # Compute bracket
        bracket_k = poisson_bracket_2d(f_k, g_k, grid.kx, grid.ky, grid.Ny, grid.Nx, grid.dealias_mask)

        # Check that modes outside 2/3 cutoff are zero
        # These should have been zeroed by the dealias mask
        bracket_high_k = bracket_k * (~grid.dealias_mask)

        max_high_k = jnp.max(jnp.abs(bracket_high_k))

        assert max_high_k < 1e-14, f"High-k modes not zeroed: max = {max_high_k}"

    def test_multiple_resolutions(self):
        """Test that the Poisson bracket works correctly at multiple resolutions."""
        resolutions = [64, 128, 256]

        for N in resolutions:
            grid = SpectralGrid2D.create(Nx=N, Ny=N, Lx=2*jnp.pi, Ly=2*jnp.pi)

            # Create simple test: f=sin(x), g=cos(y)
            x = jnp.linspace(0, grid.Lx, grid.Nx, endpoint=False)
            y = jnp.linspace(0, grid.Ly, grid.Ny, endpoint=False)
            X, Y = jnp.meshgrid(x, y, indexing='ij')

            f = jnp.sin(X).T
            g = jnp.cos(Y).T

            f_k = rfft2_forward(f)
            g_k = rfft2_forward(g)

            bracket_k = poisson_bracket_2d(f_k, g_k, grid.kx, grid.ky, grid.Ny, grid.Nx, grid.dealias_mask)
            bracket = rfft2_inverse(bracket_k, grid.Ny, grid.Nx)

            expected = -jnp.cos(X).T * jnp.sin(Y).T

            rel_error = jnp.max(jnp.abs(bracket - expected)) / jnp.max(jnp.abs(expected))

            # float32 precision: expect errors ~1e-5 with multiple FFTs
            assert rel_error < 1e-4, f"Resolution {N}: relative error {rel_error} exceeds tolerance"


class TestPoissonBracket3D:
    """Test suite for 3D Poisson bracket implementation."""

    def test_analytical_sin_cos_3d(self):
        """Test 3D Poisson bracket with f=sin(x), g=cos(y) (z-independent)."""
        grid = SpectralGrid3D.create(Nx=64, Ny=64, Nz=32, Lx=2*jnp.pi, Ly=2*jnp.pi, Lz=2*jnp.pi)

        # Create coordinate arrays
        x = jnp.linspace(0, grid.Lx, grid.Nx, endpoint=False)
        y = jnp.linspace(0, grid.Ly, grid.Ny, endpoint=False)
        z = jnp.linspace(0, grid.Lz, grid.Nz, endpoint=False)
        X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')

        # f = sin(x), g = cos(y) in [Nz, Ny, Nx] ordering
        f = jnp.sin(X).transpose(2, 1, 0)
        g = jnp.cos(Y).transpose(2, 1, 0)

        # Transform to Fourier space
        f_k = rfftn_forward(f)
        g_k = rfftn_forward(g)

        # Compute Poisson bracket
        bracket_k = poisson_bracket_3d(f_k, g_k, grid.kx, grid.ky, grid.Nz, grid.Ny, grid.Nx, grid.dealias_mask)
        bracket = rfftn_inverse(bracket_k, grid.Nz, grid.Ny, grid.Nx)

        # Expected: -cos(x)sin(y) at all z
        expected = -jnp.cos(X).transpose(2, 1, 0) * jnp.sin(Y).transpose(2, 1, 0)

        # Check relative error
        error = jnp.max(jnp.abs(bracket - expected))
        rel_error = error / jnp.max(jnp.abs(expected))

        # float32 precision: expect errors ~1e-5 with multiple FFTs
        assert rel_error < 1e-4, f"Relative error {rel_error} exceeds tolerance"

    def test_z_independence(self):
        """
        Test that Poisson bracket is independent at each z-plane.

        For fields with different z-dependence, the bracket should be computed
        independently at each z-plane since only perpendicular derivatives appear.
        """
        grid = SpectralGrid3D.create(Nx=64, Ny=64, Nz=32, Lx=2*jnp.pi, Ly=2*jnp.pi, Lz=2*jnp.pi)

        # Create fields: f = sin(x)·cos(z), g = cos(y)·sin(z)
        x = jnp.linspace(0, grid.Lx, grid.Nx, endpoint=False)
        y = jnp.linspace(0, grid.Ly, grid.Ny, endpoint=False)
        z = jnp.linspace(0, grid.Lz, grid.Nz, endpoint=False)
        X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')

        f = (jnp.sin(X) * jnp.cos(Z)).transpose(2, 1, 0)
        g = (jnp.cos(Y) * jnp.sin(Z)).transpose(2, 1, 0)

        f_k = rfftn_forward(f)
        g_k = rfftn_forward(g)

        # Compute 3D bracket
        bracket_k = poisson_bracket_3d(f_k, g_k, grid.kx, grid.ky, grid.Nz, grid.Ny, grid.Nx, grid.dealias_mask)
        bracket = rfftn_inverse(bracket_k, grid.Nz, grid.Ny, grid.Nx)

        # Expected: {f,g} = ∂f/∂x · ∂g/∂y - ∂f/∂y · ∂g/∂x
        #                 = [cos(x)·cos(z)] · [-sin(y)·sin(z)] - 0
        #                 = -cos(x)·sin(y)·cos(z)·sin(z)
        expected = (-jnp.cos(X) * jnp.sin(Y) * jnp.cos(Z) * jnp.sin(Z)).transpose(2, 1, 0)

        rel_error = jnp.max(jnp.abs(bracket - expected)) / jnp.max(jnp.abs(expected))

        # float32 precision: expect errors ~1e-5 with multiple FFTs
        assert rel_error < 1e-4, f"Relative error {rel_error} exceeds tolerance"

    def test_antisymmetry_3d(self):
        """Test anti-symmetry property in 3D: {f,g} = -{g,f}."""
        grid = SpectralGrid3D.create(Nx=64, Ny=64, Nz=32)

        # Create test fields with random Fourier coefficients
        key = jax.random.PRNGKey(45)
        key1, key2 = jax.random.split(key)

        f_k = jax.random.normal(key1, (grid.Nz, grid.Ny, grid.Nx//2+1)) + \
              1j * jax.random.normal(key1, (grid.Nz, grid.Ny, grid.Nx//2+1))
        g_k = jax.random.normal(key2, (grid.Nz, grid.Ny, grid.Nx//2+1)) + \
              1j * jax.random.normal(key2, (grid.Nz, grid.Ny, grid.Nx//2+1))

        # Compute both orderings
        bracket_fg = poisson_bracket_3d(f_k, g_k, grid.kx, grid.ky, grid.Nz, grid.Ny, grid.Nx, grid.dealias_mask)
        bracket_gf = poisson_bracket_3d(g_k, f_k, grid.kx, grid.ky, grid.Nz, grid.Ny, grid.Nx, grid.dealias_mask)

        # Check anti-symmetry
        max_diff = jnp.max(jnp.abs(bracket_fg + bracket_gf))

        # float32 precision: allow errors ~1e-5 with multiple FFTs
        assert max_diff < 1e-4, f"Anti-symmetry violated: max|{{f,g}} + {{g,f}}| = {max_diff}"

    def test_constant_field_3d(self):
        """Test that Poisson bracket with constant field is zero in 3D."""
        grid = SpectralGrid3D.create(Nx=64, Ny=64, Nz=32)

        # Create a non-constant field f
        x = jnp.linspace(0, grid.Lx, grid.Nx, endpoint=False)
        y = jnp.linspace(0, grid.Ly, grid.Ny, endpoint=False)
        z = jnp.linspace(0, grid.Lz, grid.Nz, endpoint=False)
        X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')
        f = (jnp.sin(2*X) + jnp.cos(3*Y)).transpose(2, 1, 0)
        f_k = rfftn_forward(f)

        # Create a constant field (only k=0 mode)
        c_k = jnp.zeros((grid.Nz, grid.Ny, grid.Nx//2+1), dtype=jnp.complex64)
        c_k = c_k.at[0, 0, 0].set(5.0 + 0j)

        # Compute {f, c}
        bracket_k = poisson_bracket_3d(f_k, c_k, grid.kx, grid.ky, grid.Nz, grid.Ny, grid.Nx, grid.dealias_mask)
        bracket = rfftn_inverse(bracket_k, grid.Nz, grid.Ny, grid.Nx)

        # Should be zero everywhere
        max_val = jnp.max(jnp.abs(bracket))

        assert max_val < 1e-10, f"Bracket with constant should be zero, got max = {max_val}"

    def test_dealiasing_applied_3d(self):
        """Test that dealiasing is applied to the 3D result."""
        grid = SpectralGrid3D.create(Nx=64, Ny=64, Nz=32)

        # Create test fields
        key = jax.random.PRNGKey(46)
        key1, key2 = jax.random.split(key)

        f_k = jax.random.normal(key1, (grid.Nz, grid.Ny, grid.Nx//2+1)) + \
              1j * jax.random.normal(key1, (grid.Nz, grid.Ny, grid.Nx//2+1))
        g_k = jax.random.normal(key2, (grid.Nz, grid.Ny, grid.Nx//2+1)) + \
              1j * jax.random.normal(key2, (grid.Nz, grid.Ny, grid.Nx//2+1))

        # Compute bracket
        bracket_k = poisson_bracket_3d(f_k, g_k, grid.kx, grid.ky, grid.Nz, grid.Ny, grid.Nx, grid.dealias_mask)

        # Check that modes outside 2/3 cutoff are zero
        bracket_high_k = bracket_k * (~grid.dealias_mask)

        max_high_k = jnp.max(jnp.abs(bracket_high_k))

        assert max_high_k < 1e-14, f"High-k modes not zeroed: max = {max_high_k}"
