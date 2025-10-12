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
        Test analytical solution: f=sin(x), g=cos(y).

        Expected result:
        {f,g} = ∂f/∂x · ∂g/∂y - ∂f/∂y · ∂g/∂x
              = cos(x) · (-sin(y)) - 0 · 0
              = -cos(x)sin(y)
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

        # Create physically valid smooth real fields
        key = jax.random.PRNGKey(42)
        key1, key2 = jax.random.split(key)

        # Two-pass smoothing: random → FFT → dealias → IFFT → FFT
        f = jax.random.normal(key1, (grid.Ny, grid.Nx))
        g = jax.random.normal(key2, (grid.Ny, grid.Nx))

        f_k = rfft2_forward(f)
        g_k = rfft2_forward(g)
        f_k = f_k * grid.dealias_mask
        g_k = g_k * grid.dealias_mask
        f = rfft2_inverse(f_k, grid.Ny, grid.Nx)
        g = rfft2_inverse(g_k, grid.Ny, grid.Nx)

        # Re-transform to Fourier space
        f_k = rfft2_forward(f)
        g_k = rfft2_forward(g)

        # Compute both orderings
        bracket_fg = poisson_bracket_2d(f_k, g_k, grid.kx, grid.ky, grid.Ny, grid.Nx, grid.dealias_mask)
        bracket_gf = poisson_bracket_2d(g_k, f_k, grid.kx, grid.ky, grid.Ny, grid.Nx, grid.dealias_mask)

        # Check anti-symmetry
        max_diff = jnp.max(jnp.abs(bracket_fg + bracket_gf))

        # float32 precision with random fields: expect errors ~0.01-0.05
        # (Multiple FFTs + dealiasing + nonlinear products accumulate error)
        assert max_diff < 0.1, f"Anti-symmetry violated: max|{{f,g}} + {{g,f}}| = {max_diff}"

    def test_linearity_first_argument(self):
        """Test linearity in first argument: {af + bh, g} = a{f,g} + b{h,g}."""
        grid = SpectralGrid2D.create(Nx=128, Ny=128)

        # Create physically valid smooth real fields
        key = jax.random.PRNGKey(43)
        keys = jax.random.split(key, 3)

        # Two-pass smoothing for each field
        f = jax.random.normal(keys[0], (grid.Ny, grid.Nx))
        h = jax.random.normal(keys[1], (grid.Ny, grid.Nx))
        g = jax.random.normal(keys[2], (grid.Ny, grid.Nx))

        f_k = rfft2_forward(f)
        h_k = rfft2_forward(h)
        g_k = rfft2_forward(g)
        f_k = f_k * grid.dealias_mask
        h_k = h_k * grid.dealias_mask
        g_k = g_k * grid.dealias_mask
        f = rfft2_inverse(f_k, grid.Ny, grid.Nx)
        h = rfft2_inverse(h_k, grid.Ny, grid.Nx)
        g = rfft2_inverse(g_k, grid.Ny, grid.Nx)

        # Re-transform to Fourier space
        f_k = rfft2_forward(f)
        h_k = rfft2_forward(h)
        g_k = rfft2_forward(g)

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

        # float32 precision with random fields: expect errors ~0.05-0.1
        # (Linear combinations + multiple bracket evaluations accumulate error)
        assert max_diff < 0.2, f"Linearity violated: max difference = {max_diff}"

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

    def test_mean_preserving(self):
        """
        Test that Poisson bracket preserves spatial mean: bracket_k[0, 0] = 0.

        Since {f,g} involves only derivatives, and derivatives of constants are zero,
        the k=0 mode (spatial mean) of the bracket must be zero.
        This is essential for conservation of total integrated quantities.
        """
        grid = SpectralGrid2D.create(Nx=128, Ny=128)

        # Create smooth random fields with non-zero mean
        key = jax.random.PRNGKey(200)
        key1, key2 = jax.random.split(key)

        # Two-pass smoothing
        f = jax.random.normal(key1, (grid.Ny, grid.Nx))
        g = jax.random.normal(key2, (grid.Ny, grid.Nx))

        f_k = rfft2_forward(f)
        g_k = rfft2_forward(g)
        f_k = f_k * grid.dealias_mask
        g_k = g_k * grid.dealias_mask
        f = rfft2_inverse(f_k, grid.Ny, grid.Nx)
        g = rfft2_inverse(g_k, grid.Ny, grid.Nx)

        # Add constant offsets after smoothing
        f = f + 5.0
        g = g - 3.0

        # Transform to Fourier space
        f_k = rfft2_forward(f)
        g_k = rfft2_forward(g)

        # Compute bracket
        bracket_k = poisson_bracket_2d(f_k, g_k, grid.kx, grid.ky, grid.Ny, grid.Nx, grid.dealias_mask)

        # Check k=0 mode (spatial mean)
        k0_mode = jnp.abs(bracket_k[0, 0])

        # Should be zero (derivatives kill constants)
        # Tolerance accounts for dealiasing and FFT round-off with random fields
        assert k0_mode < 0.01, f"k=0 mode should be zero, got {k0_mode}"

    def test_l2_norm_conservation(self):
        """
        Test that ∫ f · {g, f} dx = 0 (advection conserves L2 norm).

        This is the fundamental property ensuring energy conservation in KRMHD.
        When ∂A∥/∂t = {φ, A∥}, the magnetic energy ∫ A∥² dx must be conserved.

        We compute the integral in real space directly for accuracy.
        """
        grid = SpectralGrid2D.create(Nx=128, Ny=128, Lx=2*jnp.pi, Ly=2*jnp.pi)

        # Create random smooth real fields
        key = jax.random.PRNGKey(100)
        key1, key2 = jax.random.split(key)

        # Create smooth random fields in real space
        f = jax.random.normal(key1, (grid.Ny, grid.Nx))
        g = jax.random.normal(key2, (grid.Ny, grid.Nx))

        # Apply simple smoothing (low-pass filter)
        f_k = rfft2_forward(f)
        g_k = rfft2_forward(g)
        f_k = f_k * grid.dealias_mask
        g_k = g_k * grid.dealias_mask
        f = rfft2_inverse(f_k, grid.Ny, grid.Nx)
        g = rfft2_inverse(g_k, grid.Ny, grid.Nx)

        # Re-transform to Fourier space
        f_k = rfft2_forward(f)
        g_k = rfft2_forward(g)

        # Compute {g, f} (advection of f by g)
        bracket_k = poisson_bracket_2d(g_k, f_k, grid.kx, grid.ky, grid.Ny, grid.Nx, grid.dealias_mask)
        bracket = rfft2_inverse(bracket_k, grid.Ny, grid.Nx)

        # Compute ∫ f · {g, f} dx in real space
        dx = grid.Lx / grid.Nx
        dy = grid.Ly / grid.Ny
        integral = jnp.sum(f * bracket) * dx * dy

        # The integral should be zero (advection conserves L2 norm)
        # Float32 precision with multiple FFTs: expect errors ~1e-4 to 1e-3
        assert jnp.abs(integral) < 1e-2, \
            f"L2 norm conservation violated: ∫ f·{{g,f}} dx = {integral}"


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

        # Create physically valid smooth real fields
        key = jax.random.PRNGKey(45)
        key1, key2 = jax.random.split(key)

        # Two-pass smoothing
        f = jax.random.normal(key1, (grid.Nz, grid.Ny, grid.Nx))
        g = jax.random.normal(key2, (grid.Nz, grid.Ny, grid.Nx))

        f_k = rfftn_forward(f)
        g_k = rfftn_forward(g)
        f_k = f_k * grid.dealias_mask
        g_k = g_k * grid.dealias_mask
        f = rfftn_inverse(f_k, grid.Nz, grid.Ny, grid.Nx)
        g = rfftn_inverse(g_k, grid.Nz, grid.Ny, grid.Nx)

        # Re-transform to Fourier space
        f_k = rfftn_forward(f)
        g_k = rfftn_forward(g)

        # Compute both orderings
        bracket_fg = poisson_bracket_3d(f_k, g_k, grid.kx, grid.ky, grid.Nz, grid.Ny, grid.Nx, grid.dealias_mask)
        bracket_gf = poisson_bracket_3d(g_k, f_k, grid.kx, grid.ky, grid.Nz, grid.Ny, grid.Nx, grid.dealias_mask)

        # Check anti-symmetry
        max_diff = jnp.max(jnp.abs(bracket_fg + bracket_gf))

        # float32 precision with random fields: expect errors ~0.01-0.05
        # (Multiple 3D FFTs + dealiasing + nonlinear products accumulate error)
        assert max_diff < 0.1, f"Anti-symmetry violated: max|{{f,g}} + {{g,f}}| = {max_diff}"

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

    def test_l2_norm_conservation_3d(self):
        """
        Test that ∫ f · {g, f} dx = 0 in 3D (advection conserves L2 norm).

        Same fundamental property as 2D, ensuring energy conservation in KRMHD.
        The Poisson bracket is perpendicular-only, so conservation holds at all z.
        """
        grid = SpectralGrid3D.create(Nx=64, Ny=64, Nz=32, Lx=2*jnp.pi, Ly=2*jnp.pi, Lz=2*jnp.pi)

        # Create random smooth real fields
        key = jax.random.PRNGKey(101)
        key1, key2 = jax.random.split(key)

        # Create smooth random fields in real space
        f = jax.random.normal(key1, (grid.Nz, grid.Ny, grid.Nx))
        g = jax.random.normal(key2, (grid.Nz, grid.Ny, grid.Nx))

        # Apply smoothing (low-pass filter)
        f_k = rfftn_forward(f)
        g_k = rfftn_forward(g)
        f_k = f_k * grid.dealias_mask
        g_k = g_k * grid.dealias_mask
        f = rfftn_inverse(f_k, grid.Nz, grid.Ny, grid.Nx)
        g = rfftn_inverse(g_k, grid.Nz, grid.Ny, grid.Nx)

        # Re-transform to Fourier space
        f_k = rfftn_forward(f)
        g_k = rfftn_forward(g)

        # Compute {g, f}
        bracket_k = poisson_bracket_3d(g_k, f_k, grid.kx, grid.ky, grid.Nz, grid.Ny, grid.Nx, grid.dealias_mask)
        bracket = rfftn_inverse(bracket_k, grid.Nz, grid.Ny, grid.Nx)

        # Compute ∫ f · {g, f} dx in real space
        dx = grid.Lx / grid.Nx
        dy = grid.Ly / grid.Ny
        dz = grid.Lz / grid.Nz
        integral = jnp.sum(f * bracket) * dx * dy * dz

        # Should be zero (L2 norm conservation)
        assert jnp.abs(integral) < 1e-2, \
            f"L2 norm conservation violated in 3D: ∫ f·{{g,f}} dx = {integral}"

    def test_mean_preserving_3d(self):
        """
        Test that Poisson bracket preserves spatial mean in 3D: bracket_k[0, 0, 0] = 0.

        Same principle as 2D: derivatives kill constants, so k=0 mode must be zero.
        """
        grid = SpectralGrid3D.create(Nx=64, Ny=64, Nz=32)

        # Create smooth random fields with non-zero mean
        key = jax.random.PRNGKey(201)
        key1, key2 = jax.random.split(key)

        # Two-pass smoothing
        f = jax.random.normal(key1, (grid.Nz, grid.Ny, grid.Nx))
        g = jax.random.normal(key2, (grid.Nz, grid.Ny, grid.Nx))

        f_k = rfftn_forward(f)
        g_k = rfftn_forward(g)
        f_k = f_k * grid.dealias_mask
        g_k = g_k * grid.dealias_mask
        f = rfftn_inverse(f_k, grid.Nz, grid.Ny, grid.Nx)
        g = rfftn_inverse(g_k, grid.Nz, grid.Ny, grid.Nx)

        # Add constant offsets after smoothing
        f = f + 2.0
        g = g - 4.0

        # Transform to Fourier space
        f_k = rfftn_forward(f)
        g_k = rfftn_forward(g)

        # Compute bracket
        bracket_k = poisson_bracket_3d(f_k, g_k, grid.kx, grid.ky, grid.Nz, grid.Ny, grid.Nx, grid.dealias_mask)

        # Check k=0 mode
        k0_mode = jnp.abs(bracket_k[0, 0, 0])

        # Should be zero (tolerance accounts for dealiasing and FFT round-off)
        assert k0_mode < 1e-4, f"k=0 mode should be zero in 3D, got {k0_mode}"


class TestKRMHDState:
    """Test suite for KRMHD state and initialization functions."""

    def test_state_creation(self):
        """Test basic KRMHDState creation and validation."""
        from krmhd.physics import KRMHDState

        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)
        M = 10

        # Create state with correct shapes
        state = KRMHDState(
            phi=jnp.zeros((grid.Nz, grid.Ny, grid.Nx // 2 + 1), dtype=jnp.complex64),
            A_parallel=jnp.zeros((grid.Nz, grid.Ny, grid.Nx // 2 + 1), dtype=jnp.complex64),
            B_parallel=jnp.zeros((grid.Nz, grid.Ny, grid.Nx // 2 + 1), dtype=jnp.complex64),
            g=jnp.zeros((grid.Nz, grid.Ny, grid.Nx // 2 + 1, M + 1), dtype=jnp.complex64),
            M=M,
            beta_i=1.0,
            v_th=1.0,
            nu=0.01,
            time=0.0,
            grid=grid,
        )

        # Check shapes
        assert state.phi.shape == (grid.Nz, grid.Ny, grid.Nx // 2 + 1)
        assert state.A_parallel.shape == (grid.Nz, grid.Ny, grid.Nx // 2 + 1)
        assert state.B_parallel.shape == (grid.Nz, grid.Ny, grid.Nx // 2 + 1)
        assert state.g.shape == (grid.Nz, grid.Ny, grid.Nx // 2 + 1, M + 1)
        assert state.M == M
        assert state.beta_i == 1.0
        assert state.v_th == 1.0
        assert state.nu == 0.01
        assert state.time == 0.0
        assert state.grid == grid

    def test_state_validation_wrong_dims(self):
        """Test that state creation fails with wrong field dimensions."""
        from krmhd.physics import KRMHDState

        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)

        # Should fail with 2D field
        with pytest.raises(ValueError, match="must be 3D"):
            KRMHDState(
                phi=jnp.zeros((grid.Ny, grid.Nx // 2 + 1), dtype=jnp.complex64),  # 2D instead of 3D
                A_parallel=jnp.zeros((grid.Nz, grid.Ny, grid.Nx // 2 + 1), dtype=jnp.complex64),
                B_parallel=jnp.zeros((grid.Nz, grid.Ny, grid.Nx // 2 + 1), dtype=jnp.complex64),
                g=jnp.zeros((grid.Nz, grid.Ny, grid.Nx // 2 + 1, 11), dtype=jnp.complex64),
                M=10,
                beta_i=1.0,
                v_th=1.0,
                nu=0.01,
                time=0.0,
                grid=grid,
            )

    def test_initialize_hermite_moments(self):
        """Test Hermite moment initialization."""
        from krmhd.physics import initialize_hermite_moments

        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)
        M = 10

        # Test equilibrium (zero perturbation)
        g = initialize_hermite_moments(grid, M, v_th=1.0, perturbation_amplitude=0.0)

        assert g.shape == (grid.Nz, grid.Ny, grid.Nx // 2 + 1, M + 1)
        assert jnp.all(g == 0.0), "Equilibrium should be all zeros in Fourier space"

        # Test with perturbation
        g_pert = initialize_hermite_moments(grid, M, v_th=1.0, perturbation_amplitude=0.1)

        assert g_pert.shape == (grid.Nz, grid.Ny, grid.Nx // 2 + 1, M + 1)
        # Zeroth moment should still be zero
        assert jnp.all(g_pert[:, :, :, 0] == 0.0), "g_0 should be zero"
        # First moment should be non-zero (velocity perturbation)
        assert jnp.any(g_pert[:, :, :, 1] != 0.0), "g_1 should have perturbation"

    def test_initialize_alfven_wave(self):
        """Test Alfvén wave initialization."""
        from krmhd.physics import initialize_alfven_wave

        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)
        M = 10

        state = initialize_alfven_wave(
            grid,
            M,
            kx_mode=1.0,
            ky_mode=0.0,
            kz_mode=1.0,
            amplitude=0.1,
            v_th=1.0,
            beta_i=1.0,
            nu=0.01,
        )

        # Check state type
        from krmhd.physics import KRMHDState
        assert isinstance(state, KRMHDState)

        # Check initialization
        assert state.time == 0.0
        assert state.M == M

        # Check that phi and A_parallel are non-zero (single mode)
        assert jnp.any(state.phi != 0.0), "phi should have wave mode"
        assert jnp.any(state.A_parallel != 0.0), "A_parallel should have wave mode"

        # Check B_parallel is zero (pure Alfvén wave)
        assert jnp.all(state.B_parallel == 0.0), "B_parallel should be zero for Alfvén wave"

        # Check that only one mode is excited (approximately)
        n_nonzero_phi = jnp.sum(jnp.abs(state.phi) > 1e-10)
        n_nonzero_A = jnp.sum(jnp.abs(state.A_parallel) > 1e-10)
        assert n_nonzero_phi == 1, f"Should have 1 mode in phi, got {n_nonzero_phi}"
        assert n_nonzero_A == 1, f"Should have 1 mode in A_parallel, got {n_nonzero_A}"

    def test_initialize_kinetic_alfven_wave(self):
        """Test kinetic Alfvén wave initialization."""
        from krmhd.physics import initialize_kinetic_alfven_wave

        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)
        M = 10

        state = initialize_kinetic_alfven_wave(
            grid,
            M,
            kx_mode=1.0,
            kz_mode=1.0,
            amplitude=0.1,
            v_th=1.0,
            beta_i=1.0,
        )

        # Check state type
        from krmhd.physics import KRMHDState
        assert isinstance(state, KRMHDState)

        # Should be similar to regular Alfvén wave for now (TODO: add kinetic response)
        assert jnp.any(state.phi != 0.0)
        assert jnp.any(state.A_parallel != 0.0)
        assert jnp.all(state.B_parallel == 0.0)

    def test_initialize_random_spectrum(self):
        """Test random turbulent spectrum initialization."""
        from krmhd.physics import initialize_random_spectrum

        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)
        M = 10

        state = initialize_random_spectrum(
            grid,
            M,
            alpha=5.0 / 3.0,
            amplitude=1.0,
            k_min=1.0,
            k_max=5.0,
            v_th=1.0,
            beta_i=1.0,
            nu=0.01,
            seed=42,
        )

        # Check state type
        from krmhd.physics import KRMHDState
        assert isinstance(state, KRMHDState)

        # Check that fields are initialized with random spectrum
        assert jnp.any(state.phi != 0.0), "phi should have random modes"
        assert jnp.any(state.A_parallel != 0.0), "A_parallel should have random modes"

        # Check k=0 mode is zero
        assert state.phi[0, 0, 0] == 0.0, "k=0 mode should be zero"
        assert state.A_parallel[0, 0, 0] == 0.0, "k=0 mode should be zero"

        # Check B_parallel is initially zero
        assert jnp.all(state.B_parallel == 0.0), "B_parallel should start at zero"

        # Check Hermite moments are equilibrium
        assert jnp.all(state.g == 0.0), "Hermite moments should be equilibrium"

        # Check reproducibility with same seed
        state2 = initialize_random_spectrum(
            grid,
            M,
            alpha=5.0 / 3.0,
            amplitude=1.0,
            k_min=1.0,
            k_max=5.0,
            seed=42,
        )
        assert jnp.allclose(state.phi, state2.phi), "Same seed should give same result"

    def test_energy_alfven_wave(self):
        """Test energy calculation for Alfvén wave."""
        from krmhd.physics import initialize_alfven_wave, energy

        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)
        M = 10

        state = initialize_alfven_wave(
            grid,
            M,
            kx_mode=2.0,
            kz_mode=1.0,
            amplitude=0.1,
        )

        E = energy(state)

        # Check structure
        assert "magnetic" in E
        assert "kinetic" in E
        assert "compressive" in E
        assert "total" in E

        # Check all positive
        assert E["magnetic"] >= 0.0
        assert E["kinetic"] >= 0.0
        assert E["compressive"] >= 0.0
        assert E["total"] >= 0.0

        # Check total = sum
        assert jnp.isclose(
            E["total"], E["magnetic"] + E["kinetic"] + E["compressive"]
        ), "Total energy should equal sum of components"

        # For Alfvén wave, expect equipartition E_mag ≈ E_kin
        ratio = E["magnetic"] / (E["kinetic"] + 1e-20)
        assert 0.8 < ratio < 1.2, f"Alfvén wave should have E_mag ≈ E_kin, got ratio {ratio}"

        # Compressive energy should be zero (no B_parallel)
        assert E["compressive"] == 0.0, "Alfvén wave should have zero compressive energy"

    def test_energy_random_spectrum(self):
        """Test energy calculation for random spectrum."""
        from krmhd.physics import initialize_random_spectrum, energy

        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)
        M = 10

        state = initialize_random_spectrum(
            grid,
            M,
            alpha=5.0 / 3.0,
            amplitude=1.0,
            k_min=2.0,
            k_max=5.0,
        )

        E = energy(state)

        # Check all positive
        assert E["magnetic"] > 0.0, "Random spectrum should have magnetic energy"
        assert E["kinetic"] > 0.0, "Random spectrum should have kinetic energy"
        assert E["total"] > 0.0, "Random spectrum should have total energy"

        # Check total = sum
        assert jnp.isclose(
            E["total"], E["magnetic"] + E["kinetic"] + E["compressive"]
        ), "Total energy should equal sum of components"

        # For random initialization, energies should be of similar order
        # (exact ratio depends on random phases)
        assert E["magnetic"] > 0.01 * E["total"], "Magnetic energy should be significant"
        assert E["kinetic"] > 0.01 * E["total"], "Kinetic energy should be significant"

    def test_energy_conservation_scaling(self):
        """Test that energy scales correctly with amplitude."""
        from krmhd.physics import initialize_alfven_wave, energy

        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)
        M = 10

        # Initialize with amplitude A
        state1 = initialize_alfven_wave(grid, M, amplitude=0.1)
        E1 = energy(state1)

        # Initialize with amplitude 2A
        state2 = initialize_alfven_wave(grid, M, amplitude=0.2)
        E2 = energy(state2)

        # Energy should scale as amplitude squared: E ∝ A²
        # So E2/E1 should be ≈ (0.2/0.1)² = 4
        ratio = E2["total"] / E1["total"]
        expected_ratio = (0.2 / 0.1) ** 2

        assert jnp.isclose(
            ratio, expected_ratio, rtol=0.01
        ), f"Energy should scale as amplitude², got ratio {ratio}, expected {expected_ratio}"

    def test_energy_zero_for_zero_fields(self):
        """Test that energy is zero for zero fields."""
        from krmhd.physics import KRMHDState, energy

        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)
        M = 10

        # Create state with all zeros
        state = KRMHDState(
            phi=jnp.zeros((grid.Nz, grid.Ny, grid.Nx // 2 + 1), dtype=jnp.complex64),
            A_parallel=jnp.zeros((grid.Nz, grid.Ny, grid.Nx // 2 + 1), dtype=jnp.complex64),
            B_parallel=jnp.zeros((grid.Nz, grid.Ny, grid.Nx // 2 + 1), dtype=jnp.complex64),
            g=jnp.zeros((grid.Nz, grid.Ny, grid.Nx // 2 + 1, M + 1), dtype=jnp.complex64),
            M=M,
            beta_i=1.0,
            v_th=1.0,
            nu=0.01,
            time=0.0,
            grid=grid,
        )

        E = energy(state)

        assert E["magnetic"] == 0.0
        assert E["kinetic"] == 0.0
        assert E["compressive"] == 0.0
        assert E["total"] == 0.0
