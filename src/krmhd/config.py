"""
Configuration Management for KRMHD Simulations

This module provides a hierarchical configuration system using Pydantic models
for parameter validation and YAML file I/O. It supports:

1. Grid configuration (resolution, domain size)
2. Physics parameters (v_A, resistivity, collision frequency)
3. Initial conditions (turbulent spectrum, waves, Orszag-Tang)
4. Time integration (CFL, number of steps, output intervals)
5. Forcing parameters (for driven turbulence)
6. I/O configuration (output directories, checkpoint intervals)

Example usage:
    # Load from YAML file
    config = SimulationConfig.from_yaml("config.yaml")

    # Validate and access parameters
    grid = config.create_grid()
    state = config.create_initial_state(grid)

    # Save configuration
    config.to_yaml("output/config_used.yaml")
"""

from pathlib import Path
from typing import Literal, Optional
import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
import numpy as np


class GridConfig(BaseModel):
    """Spectral grid configuration."""

    Nx: int = Field(64, ge=8, description="Grid points in x (perpendicular)")
    Ny: int = Field(64, ge=8, description="Grid points in y (perpendicular)")
    Nz: int = Field(32, ge=8, description="Grid points in z (parallel)")

    Lx: float = Field(2 * np.pi, gt=0, description="Domain size in x")
    Ly: float = Field(2 * np.pi, gt=0, description="Domain size in y")
    Lz: float = Field(2 * np.pi, gt=0, description="Domain size in z")

    @field_validator('Nx', 'Ny', 'Nz')
    @classmethod
    def check_power_of_two(cls, v: int) -> int:
        """Recommend power of 2 for FFT efficiency."""
        if v & (v - 1) != 0:
            # Not a power of 2, issue warning but allow
            import warnings
            warnings.warn(
                f"Grid size {v} is not a power of 2. "
                "Powers of 2 are more efficient for FFTs.",
                UserWarning
            )
        return v


class PhysicsConfig(BaseModel):
    """
    Physical parameters for KRMHD.

    Note:
        Hyper-dissipation overflow validation requires knowing dt and grid size,
        which are not available at config creation time. Runtime validation occurs in:
        - gandalf_step() for hyper-collision overflow (timestepping.py:571-594)
        - gandalf_step() for hyper-resistivity overflow (timestepping.py:596-622)
        - run_simulation.py for CFL-based dt validation (lines 131-138)
    """

    v_A: float = Field(1.0, gt=0, description="Alfvén velocity")
    eta: float = Field(0.01, ge=0, description="Resistivity coefficient")
    nu: float = Field(0.01, ge=0, description="Collision frequency")
    beta_i: float = Field(1.0, gt=0, description="Ion plasma beta")

    # Hyper-dissipation parameters (Issue #28)
    # Constraints match timestepping.py:564 and timestepping.py:570
    hyper_r: int = Field(1, description="Hyper-resistivity order (1, 2, 4, or 8)")
    hyper_n: int = Field(1, description="Hyper-collision order (1, 2, or 4)")

    @field_validator('hyper_r')
    @classmethod
    def check_hyper_r(cls, v: int) -> int:
        """Validate hyper_r matches timestepping.py constraints."""
        if v not in {1, 2, 4, 8}:
            raise ValueError(
                f"hyper_r must be 1, 2, 4, or 8 (got {v}). "
                "Use r=1 for standard dissipation, r=2 for typical turbulence studies."
            )
        return v

    @field_validator('hyper_n')
    @classmethod
    def check_hyper_n(cls, v: int) -> int:
        """Validate hyper_n matches timestepping.py constraints."""
        if v not in {1, 2, 4}:
            raise ValueError(
                f"hyper_n must be 1, 2, or 4 (got {v}). "
                "Use n=1 for standard collisions, n=2 for typical turbulence studies."
            )
        return v

    @model_validator(mode='after')
    def check_dissipation_overflow(self):
        """Warn about potential overflow with hyper-dissipation."""
        if self.hyper_r > 2 or self.hyper_n > 2:
            import warnings
            warnings.warn(
                f"Hyper-dissipation orders r={self.hyper_r}, n={self.hyper_n} may "
                "require very small coefficients to avoid overflow. "
                "Recommended: r=2, n=2.",
                UserWarning
            )
        return self


class InitialConditionConfig(BaseModel):
    """Initial condition configuration."""

    type: Literal[
        "random_spectrum",
        "alfven_wave",
        "kinetic_alfven_wave",
        "orszag_tang",
        "zero"
    ] = Field("random_spectrum", description="Type of initial condition")

    # Parameters for random_spectrum
    amplitude: float = Field(1.0, ge=0, description="Initial amplitude")
    alpha: float = Field(5.0/3.0, description="Spectral index (k^-alpha)")
    k_min: float = Field(1.0, gt=0, description="Minimum wavenumber")
    k_max: float = Field(10.0, gt=0, description="Maximum wavenumber")

    # Parameters for wave initial conditions
    k_wave: list[float] = Field(
        [0.0, 0.0, 1.0],
        description="Wave vector [kx, ky, kz]"
    )

    # Hermite moment configuration
    M: int = Field(20, ge=0, description="Number of Hermite moments")

    @field_validator('k_wave')
    @classmethod
    def check_k_wave_length(cls, v: list[float]) -> list[float]:
        """Ensure k_wave has exactly 3 components."""
        if len(v) != 3:
            raise ValueError(f"k_wave must have 3 components, got {len(v)}")
        return v


class ForcingConfig(BaseModel):
    """Forcing configuration for driven turbulence."""

    enabled: bool = Field(False, description="Enable forcing")

    amplitude: float = Field(0.3, ge=0, description="Forcing amplitude")
    k_min: float = Field(2.0, gt=0, description="Minimum forcing wavenumber")
    k_max: float = Field(5.0, gt=0, description="Maximum forcing wavenumber")

    seed: Optional[int] = Field(None, description="Random seed for forcing")

    @model_validator(mode='after')
    def check_k_range(self):
        """Ensure k_max > k_min."""
        if self.enabled and self.k_max <= self.k_min:
            raise ValueError(f"k_max ({self.k_max}) must be > k_min ({self.k_min})")
        return self


class TimeIntegrationConfig(BaseModel):
    """Time integration configuration."""

    n_steps: int = Field(100, ge=1, description="Number of timesteps")
    cfl_safety: float = Field(0.3, gt=0, le=1.0, description="CFL safety factor")

    dt_fixed: Optional[float] = Field(
        None,
        gt=0,
        description="Fixed timestep (overrides CFL if set)"
    )

    save_interval: int = Field(10, ge=1, description="Save diagnostics every N steps")
    checkpoint_interval: Optional[int] = Field(
        None,
        ge=1,
        description="Save checkpoint every N steps (reserved for HDF5 I/O, Issue #13)"
    )


class IOConfig(BaseModel):
    """Input/Output configuration."""

    output_dir: str = Field("output", description="Output directory")

    save_spectra: bool = Field(True, description="Save energy spectra")
    save_energy_history: bool = Field(True, description="Save energy time series")
    save_fields: bool = Field(False, description="Save full field snapshots (reserved for future use)")
    save_final_state: bool = Field(True, description="Save final state")

    overwrite: bool = Field(False, description="Overwrite existing output")


class SimulationConfig(BaseModel):
    """Complete simulation configuration."""

    # Metadata
    name: str = Field("krmhd_simulation", description="Simulation name")
    description: str = Field("", description="Simulation description")

    # Configuration sections
    grid: GridConfig = Field(default_factory=GridConfig)
    physics: PhysicsConfig = Field(default_factory=PhysicsConfig)
    initial_condition: InitialConditionConfig = Field(
        default_factory=InitialConditionConfig
    )
    forcing: ForcingConfig = Field(default_factory=ForcingConfig)
    time_integration: TimeIntegrationConfig = Field(
        default_factory=TimeIntegrationConfig
    )
    io: IOConfig = Field(default_factory=IOConfig)

    @classmethod
    def from_yaml(cls, filepath: str | Path) -> "SimulationConfig":
        """
        Load configuration from YAML file.

        Args:
            filepath: Path to YAML configuration file

        Returns:
            Validated SimulationConfig instance

        Raises:
            FileNotFoundError: If file doesn't exist
            yaml.YAMLError: If YAML is malformed
            pydantic.ValidationError: If configuration is invalid
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Config file not found: {filepath}")

        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)

        return cls(**data)

    def to_yaml(self, filepath: str | Path) -> None:
        """
        Save configuration to YAML file.

        Args:
            filepath: Path to output YAML file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dictionary with clean formatting
        data = self.model_dump(mode='python', exclude_none=True)

        with open(filepath, 'w') as f:
            yaml.safe_dump(
                data,
                f,
                default_flow_style=False,
                sort_keys=False,
                indent=2
            )

    def create_grid(self):
        """
        Create SpectralGrid3D from configuration.

        Returns:
            SpectralGrid3D instance
        """
        from krmhd import SpectralGrid3D

        return SpectralGrid3D.create(
            Nx=self.grid.Nx,
            Ny=self.grid.Ny,
            Nz=self.grid.Nz,
            Lx=self.grid.Lx,
            Ly=self.grid.Ly,
            Lz=self.grid.Lz
        )

    def create_initial_state(self, grid):
        """
        Create initial KRMHDState from configuration.

        Args:
            grid: SpectralGrid3D instance

        Returns:
            KRMHDState instance
        """
        from krmhd import (
            initialize_random_spectrum,
            initialize_alfven_wave,
            initialize_kinetic_alfven_wave,
            KRMHDState
        )
        import jax.numpy as jnp

        ic = self.initial_condition

        if ic.type == "random_spectrum":
            return initialize_random_spectrum(
                grid,
                M=ic.M,
                alpha=ic.alpha,
                amplitude=ic.amplitude,
                k_min=ic.k_min,
                k_max=ic.k_max
            )

        elif ic.type == "alfven_wave":
            kx, ky, kz = ic.k_wave
            return initialize_alfven_wave(
                grid,
                M=ic.M,
                kx_mode=kx,
                ky_mode=ky,
                kz_mode=kz,
                amplitude=ic.amplitude
            )

        elif ic.type == "kinetic_alfven_wave":
            kx, ky, kz = ic.k_wave
            return initialize_kinetic_alfven_wave(
                grid,
                M=ic.M,
                kx_mode=kx,
                ky_mode=ky,
                kz_mode=kz,
                amplitude=ic.amplitude,
                beta_i=self.physics.beta_i
            )

        elif ic.type == "orszag_tang":
            # Use shared Orszag-Tang initialization
            from krmhd.physics import initialize_orszag_tang
            return initialize_orszag_tang(grid, M=ic.M)

        elif ic.type == "zero":
            # Zero state for testing
            nkx = grid.Nx // 2 + 1
            nky = grid.Ny
            nkz = grid.Nz
            M = ic.M

            zeros_3d = jnp.zeros((nkz, nky, nkx), dtype=jnp.complex128)
            zeros_moments = jnp.zeros((nkz, nky, nkx, M+1), dtype=jnp.complex128)

            return KRMHDState(
                z_plus=zeros_3d,
                z_minus=zeros_3d,
                B_parallel=zeros_3d,
                g=zeros_moments,
                M=M,
                beta_i=self.physics.beta_i,
                v_th=1.0,
                nu=self.physics.nu,
                Lambda=1.0,
                time=0.0,
                grid=grid
            )

        else:
            raise ValueError(f"Unknown initial condition type: {ic.type}")

    def get_output_dir(self) -> Path:
        """
        Get output directory as Path, creating if needed.

        Raises:
            FileExistsError: If directory exists and overwrite=False
        """
        output_dir = Path(self.io.output_dir)

        # Check if directory exists and overwrite flag
        if output_dir.exists() and not self.io.overwrite:
            # Check if directory is non-empty
            if any(output_dir.iterdir()):
                raise FileExistsError(
                    f"Output directory exists and is non-empty: {output_dir}\n"
                    f"Use --output-dir to specify a different directory, "
                    f"or set overwrite: true in config to allow overwriting."
                )

        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def summary(self) -> str:
        """Generate human-readable summary of configuration."""
        lines = [
            "=" * 70,
            f"KRMHD Simulation: {self.name}",
            "=" * 70,
        ]

        if self.description:
            lines.extend([
                "",
                self.description,
            ])

        lines.extend([
            "",
            "Grid Configuration:",
            f"  Resolution: {self.grid.Nx} × {self.grid.Ny} × {self.grid.Nz}",
            f"  Domain: [{self.grid.Lx:.2f}, {self.grid.Ly:.2f}, {self.grid.Lz:.2f}]",
            "",
            "Physics Parameters:",
            f"  Alfvén velocity: v_A = {self.physics.v_A}",
            f"  Resistivity: η = {self.physics.eta}",
            f"  Collision frequency: ν = {self.physics.nu}",
            f"  Ion plasma beta: β_i = {self.physics.beta_i}",
        ])

        if self.physics.hyper_r > 1 or self.physics.hyper_n > 1:
            lines.extend([
                f"  Hyper-dissipation: r = {self.physics.hyper_r}, n = {self.physics.hyper_n}",
            ])

        lines.extend([
            "",
            "Initial Condition:",
            f"  Type: {self.initial_condition.type}",
            f"  Hermite moments: M = {self.initial_condition.M}",
        ])

        if self.initial_condition.type == "random_spectrum":
            lines.extend([
                f"  Spectral index: α = {self.initial_condition.alpha:.2f}",
                f"  Amplitude: {self.initial_condition.amplitude}",
                f"  Wavenumber range: k ∈ [{self.initial_condition.k_min}, {self.initial_condition.k_max}]",
            ])

        if self.forcing.enabled:
            lines.extend([
                "",
                "Forcing:",
                f"  Amplitude: {self.forcing.amplitude}",
                f"  Wavenumber range: k ∈ [{self.forcing.k_min}, {self.forcing.k_max}]",
            ])

        lines.extend([
            "",
            "Time Integration:",
            f"  Steps: {self.time_integration.n_steps}",
            f"  CFL safety: {self.time_integration.cfl_safety}",
            f"  Save interval: {self.time_integration.save_interval}",
            "",
            "Output:",
            f"  Directory: {self.io.output_dir}",
            f"  Save spectra: {self.io.save_spectra}",
            f"  Save energy history: {self.io.save_energy_history}",
            "=" * 70,
        ])

        return "\n".join(lines)


# Predefined configuration templates
def decaying_turbulence_config(**kwargs) -> SimulationConfig:
    """
    Create configuration for decaying turbulence simulation.

    Args:
        **kwargs: Override default parameters (top-level keys only)

    Returns:
        SimulationConfig for decaying turbulence

    Note:
        Overrides replace entire configuration objects, not individual fields.
        For example, `grid=GridConfig(Nx=128)` replaces the entire GridConfig,
        so you must specify all fields (Nx, Ny, Nz, Lx, Ly, Lz).

        To override individual fields, dict manipulation is simplest:
        >>> config = decaying_turbulence_config()
        >>> config_dict = config.model_dump()
        >>> config_dict['grid']['Nx'] = 128
        >>> config = SimulationConfig(**config_dict)

        Alternatively, use Pydantic's model_copy():
        >>> config = decaying_turbulence_config()
        >>> config = config.model_copy(update={'grid': config.grid.model_copy(update={'Nx': 128})})
    """
    config = SimulationConfig(
        name="decaying_turbulence",
        description="Decaying turbulence with k^(-5/3) initial spectrum",
        grid=GridConfig(Nx=64, Ny=64, Nz=32),
        physics=PhysicsConfig(v_A=1.0, eta=0.01, nu=0.01, beta_i=1.0),
        initial_condition=InitialConditionConfig(
            type="random_spectrum",
            amplitude=1.0,
            alpha=5.0/3.0,
            k_min=1.0,
            k_max=10.0,
            M=20
        ),
        forcing=ForcingConfig(enabled=False),
        time_integration=TimeIntegrationConfig(
            n_steps=100,
            cfl_safety=0.3,
            save_interval=10
        ),
        io=IOConfig(output_dir="output/decaying_turbulence")
    )

    # Apply overrides
    if kwargs:
        config = SimulationConfig(**{**config.model_dump(), **kwargs})

    return config


def driven_turbulence_config(**kwargs) -> SimulationConfig:
    """
    Create configuration for driven turbulence simulation.

    Args:
        **kwargs: Override default parameters (top-level keys only)

    Returns:
        SimulationConfig for driven turbulence

    Note:
        Overrides replace entire configuration objects. See decaying_turbulence_config()
        docstring for details on override behavior.
    """
    config = SimulationConfig(
        name="driven_turbulence",
        description="Driven turbulence with Gaussian white noise forcing",
        grid=GridConfig(Nx=64, Ny=64, Nz=32),
        physics=PhysicsConfig(v_A=1.0, eta=0.02, nu=0.02, beta_i=1.0),
        initial_condition=InitialConditionConfig(
            type="random_spectrum",
            amplitude=0.1,
            alpha=5.0/3.0,
            k_min=1.0,
            k_max=10.0,
            M=20
        ),
        forcing=ForcingConfig(
            enabled=True,
            amplitude=0.3,
            k_min=2.0,
            k_max=5.0
        ),
        time_integration=TimeIntegrationConfig(
            n_steps=200,
            cfl_safety=0.3,
            save_interval=5
        ),
        io=IOConfig(output_dir="output/driven_turbulence")
    )

    # Apply overrides
    if kwargs:
        config = SimulationConfig(**{**config.model_dump(), **kwargs})

    return config


def orszag_tang_config(**kwargs) -> SimulationConfig:
    """
    Create configuration for Orszag-Tang vortex simulation.

    Args:
        **kwargs: Override default parameters (top-level keys only)

    Returns:
        SimulationConfig for Orszag-Tang vortex

    Note:
        Overrides replace entire configuration objects. See decaying_turbulence_config()
        docstring for details on override behavior.
    """
    config = SimulationConfig(
        name="orszag_tang",
        description="Orszag-Tang vortex benchmark",
        grid=GridConfig(Nx=128, Ny=128, Nz=32),
        physics=PhysicsConfig(v_A=1.0, eta=0.001, nu=0.001, beta_i=1.0),
        initial_condition=InitialConditionConfig(
            type="orszag_tang",
            M=20
        ),
        forcing=ForcingConfig(enabled=False),
        time_integration=TimeIntegrationConfig(
            n_steps=100,
            cfl_safety=0.3,
            save_interval=5
        ),
        io=IOConfig(output_dir="output/orszag_tang")
    )

    # Apply overrides
    if kwargs:
        config = SimulationConfig(**{**config.model_dump(), **kwargs})

    return config
