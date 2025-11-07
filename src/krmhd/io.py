"""
HDF5 I/O for KRMHD checkpoints and diagnostics.

This module provides functions for saving and loading KRMHD simulation state
and diagnostics data to/from HDF5 files. It supports:

1. Checkpoint save/load: Full KRMHDState with all fields and metadata
2. Timeseries save/load: EnergyHistory and other diagnostic time series
3. Metadata preservation: Grid configuration, physical parameters, timestamps

Features:
- Compression: gzip compression for large arrays (level 4)
- Complex arrays: Stored as separate real/imaginary datasets
- Versioning: File format version for future compatibility
- Validation: Automatic shape and dtype checking on load
- Resumability: Save/load simulation state for checkpoint/restart workflows

Example usage - Checkpointing:
    >>> from krmhd import KRMHDState, SpectralGrid3D
    >>> from krmhd.io import save_checkpoint, load_checkpoint
    >>>
    >>> # Save checkpoint
    >>> save_checkpoint(
    ...     state,
    ...     filename="checkpoint_t100.h5",
    ...     metadata={"description": "Turbulence at t=100"}
    ... )
    >>>
    >>> # Load checkpoint
    >>> state_loaded, grid_loaded, meta = load_checkpoint("checkpoint_t100.h5")
    >>> # Resume simulation from loaded state
    >>> state = gandalf_step(state_loaded, dt=0.01, eta=0.01, v_A=1.0)

Example usage - Timeseries:
    >>> from krmhd.diagnostics import EnergyHistory
    >>> from krmhd.io import save_timeseries, load_timeseries
    >>>
    >>> # Save energy history
    >>> history = EnergyHistory()
    >>> # ... run simulation and append states ...
    >>> save_timeseries(history, "energy_history.h5")
    >>>
    >>> # Load energy history
    >>> history_loaded = load_timeseries("energy_history.h5")
    >>> history_loaded.plot()

File format:
    Checkpoint files contain:
    - /state/z_plus: Complex Elsasser field (stored as real + imag)
    - /state/z_minus: Complex Elsasser field (stored as real + imag)
    - /state/B_parallel: Complex parallel magnetic field (stored as real + imag)
    - /state/g: Complex Hermite moments (stored as real + imag)
    - /state attributes: M, beta_i, v_th, nu, Lambda, time
    - /grid attributes: Nx, Ny, Nz, Lx, Ly, Lz
    - /metadata attributes: User-provided metadata, timestamps, version

    Timeseries files contain:
    - /times: Time array
    - /E_magnetic: Magnetic energy time series
    - /E_kinetic: Kinetic energy time series
    - /E_compressive: Compressive energy time series
    - /E_total: Total energy time series
    - attributes: Creation timestamp, version

Physics context:
    Checkpointing is essential for long turbulence runs (t >> τ_nl) where:
    - τ_nl ~ L/v_rms is the nonlinear time
    - Runs may take hours to days on HPC systems
    - System failures or time limits require restart capability
    - Analysis requires intermediate snapshots for convergence studies

References:
    - HDF5 format: https://www.hdfgroup.org/solutions/hdf5/
    - JAX/NumPy interop: JAX arrays convert seamlessly to NumPy for h5py
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Any
from datetime import datetime
import warnings

import h5py
import numpy as np
import jax.numpy as jnp

from krmhd.physics import KRMHDState
from krmhd.spectral import SpectralGrid3D
from krmhd.diagnostics import EnergyHistory, TurbulenceDiagnostics

# File format version for compatibility checking
# Increment when making breaking changes to file structure
IO_FORMAT_VERSION = "1.0.0"

# Compression settings for HDF5 datasets
# level 4 is a good balance between compression ratio and speed
COMPRESSION_LEVEL = 4


def save_checkpoint(
    state: KRMHDState,
    filename: str,
    metadata: Optional[Dict[str, Any]] = None,
    overwrite: bool = False,
) -> None:
    """
    Save KRMHD state to HDF5 checkpoint file.

    Saves complete simulation state including all fields (z±, B∥, g), physical
    parameters (β_i, v_th, ν, Λ), grid configuration, and optional metadata.
    Complex arrays are stored as separate real/imaginary parts.

    Args:
        state: KRMHD state to save
        filename: Output HDF5 file path
        metadata: Optional user metadata dict (arbitrary key-value pairs)
        overwrite: If True, overwrite existing file; otherwise raise error

    Raises:
        FileExistsError: If file exists and overwrite=False
        ValueError: If state validation fails
        OSError: If file write fails

    Example:
        >>> save_checkpoint(
        ...     state,
        ...     "checkpoint_t100.h5",
        ...     metadata={
        ...         "description": "Turbulence simulation",
        ...         "run_id": "turb_001",
        ...         "parameters": {"eta": 0.01, "nu": 0.01}
        ...     }
        ... )

    File structure:
        /state/z_plus_real, z_plus_imag: Elsasser z+ field
        /state/z_minus_real, z_minus_imag: Elsasser z- field
        /state/B_parallel_real, B_parallel_imag: Parallel B field
        /state/g_real, g_imag: Hermite moments
        /state (attributes): M, beta_i, v_th, nu, Lambda, time
        /grid (attributes): Nx, Ny, Nz, Lx, Ly, Lz
        /metadata (attributes): version, timestamp, user metadata

    Note:
        Arrays are stored with gzip compression (level 4) to reduce file size.
        Typical compression ratios are 2-5× for turbulent fields.
    """
    filepath = Path(filename)

    # Check if file exists
    if filepath.exists() and not overwrite:
        raise FileExistsError(
            f"Checkpoint file '{filename}' already exists. "
            "Use overwrite=True to replace it."
        )

    # Create parent directories if needed
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Convert JAX arrays to NumPy for h5py compatibility
    z_plus = np.array(state.z_plus)
    z_minus = np.array(state.z_minus)
    B_parallel = np.array(state.B_parallel)
    g = np.array(state.g)

    # Validate complex dtypes
    if not (z_plus.dtype == np.complex64 or z_plus.dtype == np.complex128):
        raise ValueError(f"Expected complex array for z_plus, got {z_plus.dtype}")

    with h5py.File(filename, 'w') as f:
        # Create groups
        state_group = f.create_group('state')
        grid_group = f.create_group('grid')
        meta_group = f.create_group('metadata')

        # Save state fields (split complex into real + imag)
        # Use float32 to save space while maintaining sufficient precision
        state_group.create_dataset(
            'z_plus_real',
            data=z_plus.real.astype(np.float32),
            compression='gzip',
            compression_opts=COMPRESSION_LEVEL
        )
        state_group.create_dataset(
            'z_plus_imag',
            data=z_plus.imag.astype(np.float32),
            compression='gzip',
            compression_opts=COMPRESSION_LEVEL
        )

        state_group.create_dataset(
            'z_minus_real',
            data=z_minus.real.astype(np.float32),
            compression='gzip',
            compression_opts=COMPRESSION_LEVEL
        )
        state_group.create_dataset(
            'z_minus_imag',
            data=z_minus.imag.astype(np.float32),
            compression='gzip',
            compression_opts=COMPRESSION_LEVEL
        )

        state_group.create_dataset(
            'B_parallel_real',
            data=B_parallel.real.astype(np.float32),
            compression='gzip',
            compression_opts=COMPRESSION_LEVEL
        )
        state_group.create_dataset(
            'B_parallel_imag',
            data=B_parallel.imag.astype(np.float32),
            compression='gzip',
            compression_opts=COMPRESSION_LEVEL
        )

        state_group.create_dataset(
            'g_real',
            data=g.real.astype(np.float32),
            compression='gzip',
            compression_opts=COMPRESSION_LEVEL
        )
        state_group.create_dataset(
            'g_imag',
            data=g.imag.astype(np.float32),
            compression='gzip',
            compression_opts=COMPRESSION_LEVEL
        )

        # Save state scalar parameters as attributes
        state_group.attrs['M'] = state.M
        state_group.attrs['beta_i'] = state.beta_i
        state_group.attrs['v_th'] = state.v_th
        state_group.attrs['nu'] = state.nu
        state_group.attrs['Lambda'] = state.Lambda
        state_group.attrs['time'] = state.time

        # Save grid configuration as attributes (don't save computed arrays)
        grid_group.attrs['Nx'] = state.grid.Nx
        grid_group.attrs['Ny'] = state.grid.Ny
        grid_group.attrs['Nz'] = state.grid.Nz
        grid_group.attrs['Lx'] = state.grid.Lx
        grid_group.attrs['Ly'] = state.grid.Ly
        grid_group.attrs['Lz'] = state.grid.Lz

        # Save metadata
        meta_group.attrs['version'] = IO_FORMAT_VERSION
        meta_group.attrs['timestamp'] = datetime.now().isoformat()

        # Add user metadata if provided
        if metadata is not None:
            for key, value in metadata.items():
                # HDF5 attributes support limited types, so convert to string if needed
                try:
                    meta_group.attrs[key] = value
                except (TypeError, ValueError):
                    # If type not supported, convert to string
                    meta_group.attrs[key] = str(value)


def load_checkpoint(
    filename: str,
    validate_grid: bool = True,
) -> Tuple[KRMHDState, SpectralGrid3D, Dict[str, Any]]:
    """
    Load KRMHD state from HDF5 checkpoint file.

    Loads complete simulation state and reconstructs KRMHDState object with
    all fields, parameters, and grid configuration. Complex arrays are
    reconstructed from real/imaginary parts.

    Args:
        filename: Input HDF5 checkpoint file path
        validate_grid: If True, verify grid dimensions are consistent

    Returns:
        state: Loaded KRMHD state
        grid: Reconstructed SpectralGrid3D
        metadata: Dictionary with file metadata (version, timestamp, user data)

    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
        ValueError: If file format is incompatible or data is invalid
        KeyError: If required datasets/attributes are missing

    Example:
        >>> state, grid, metadata = load_checkpoint("checkpoint_t100.h5")
        >>> print(f"Loaded state at t={state.time}")
        >>> print(f"Grid: {grid.Nx}×{grid.Ny}×{grid.Nz}")
        >>> print(f"Saved at: {metadata['timestamp']}")
        >>>
        >>> # Resume simulation
        >>> for i in range(100):
        ...     state = gandalf_step(state, dt=0.01, eta=0.01, v_A=1.0)

    Note:
        Arrays are loaded as float32 and converted to JAX's default precision
        (float32 on GPU, float32/float64 depending on JAX config on CPU).
    """
    filepath = Path(filename)

    if not filepath.exists():
        raise FileNotFoundError(f"Checkpoint file '{filename}' not found")

    with h5py.File(filename, 'r') as f:
        # Check version compatibility
        version = f['metadata'].attrs.get('version', 'unknown')
        if version != IO_FORMAT_VERSION:
            warnings.warn(
                f"Checkpoint file version ({version}) differs from current "
                f"version ({IO_FORMAT_VERSION}). Loading may fail or produce "
                "unexpected results.",
                UserWarning
            )

        # Load grid configuration
        grid_attrs = f['grid'].attrs
        grid = SpectralGrid3D.create(
            Nx=int(grid_attrs['Nx']),
            Ny=int(grid_attrs['Ny']),
            Nz=int(grid_attrs['Nz']),
            Lx=float(grid_attrs['Lx']),
            Ly=float(grid_attrs['Ly']),
            Lz=float(grid_attrs['Lz']),
        )

        # Load state fields (reconstruct complex from real + imag)
        state_group = f['state']

        # Load raw arrays first
        z_plus_real = jnp.array(state_group['z_plus_real'][:], dtype=jnp.float32)
        z_plus_imag = jnp.array(state_group['z_plus_imag'][:], dtype=jnp.float32)
        z_minus_real = jnp.array(state_group['z_minus_real'][:], dtype=jnp.float32)
        z_minus_imag = jnp.array(state_group['z_minus_imag'][:], dtype=jnp.float32)
        B_parallel_real = jnp.array(state_group['B_parallel_real'][:], dtype=jnp.float32)
        B_parallel_imag = jnp.array(state_group['B_parallel_imag'][:], dtype=jnp.float32)
        g_real = jnp.array(state_group['g_real'][:], dtype=jnp.float32)
        g_imag = jnp.array(state_group['g_imag'][:], dtype=jnp.float32)

        # Validate shapes if requested (before reconstruction to catch errors early)
        if validate_grid:
            expected_shape = (grid.Nz, grid.Ny, grid.Nx // 2 + 1)
            if z_plus_real.shape != expected_shape:
                raise ValueError(
                    f"z_plus shape {z_plus_real.shape} doesn't match grid "
                    f"{expected_shape}"
                )
            if z_minus_real.shape != expected_shape:
                raise ValueError(
                    f"z_minus shape {z_minus_real.shape} doesn't match grid "
                    f"{expected_shape}"
                )
            if B_parallel_real.shape != expected_shape:
                raise ValueError(
                    f"B_parallel shape {B_parallel_real.shape} doesn't match grid "
                    f"{expected_shape}"
                )

            # Validate Hermite moments shape
            M = int(state_group.attrs['M'])
            expected_g_shape = (grid.Nz, grid.Ny, grid.Nx // 2 + 1, M + 1)
            if g_real.shape != expected_g_shape:
                raise ValueError(
                    f"g shape {g_real.shape} doesn't match expected {expected_g_shape}"
                )

        # Reconstruct complex arrays after validation
        z_plus = z_plus_real + 1j * z_plus_imag
        z_minus = z_minus_real + 1j * z_minus_imag
        B_parallel = B_parallel_real + 1j * B_parallel_imag
        g = g_real + 1j * g_imag

        # Load state scalar parameters
        state = KRMHDState(
            z_plus=z_plus,
            z_minus=z_minus,
            B_parallel=B_parallel,
            g=g,
            M=int(state_group.attrs['M']),
            beta_i=float(state_group.attrs['beta_i']),
            v_th=float(state_group.attrs['v_th']),
            nu=float(state_group.attrs['nu']),
            Lambda=float(state_group.attrs['Lambda']),
            time=float(state_group.attrs['time']),
            grid=grid,
        )

        # Load metadata
        metadata = dict(f['metadata'].attrs)

    return state, grid, metadata


def save_timeseries(
    history: EnergyHistory,
    filename: str,
    metadata: Optional[Dict[str, Any]] = None,
    overwrite: bool = False,
) -> None:
    """
    Save EnergyHistory to HDF5 file.

    Saves all time series data (times, energies) for post-processing and
    visualization. Useful for saving diagnostic data separately from
    full checkpoint files.

    Args:
        history: EnergyHistory object to save
        filename: Output HDF5 file path
        metadata: Optional user metadata dict
        overwrite: If True, overwrite existing file; otherwise raise error

    Raises:
        FileExistsError: If file exists and overwrite=False
        ValueError: If history is empty
        OSError: If file write fails

    Example:
        >>> history = EnergyHistory()
        >>> for i in range(1000):
        ...     history.append(state)
        ...     state = gandalf_step(state, dt=0.01, eta=0.01, v_A=1.0)
        >>>
        >>> save_timeseries(history, "energy_history.h5",
        ...                 metadata={"run_id": "turb_001"})

    Note:
        Arrays are stored as float64 to preserve full precision for
        post-processing analysis (dissipation rates, etc.).
    """
    filepath = Path(filename)

    # Check if file exists
    if filepath.exists() and not overwrite:
        raise FileExistsError(
            f"Timeseries file '{filename}' already exists. "
            "Use overwrite=True to replace it."
        )

    # Validate that history has data
    if len(history.times) == 0:
        raise ValueError("Cannot save empty EnergyHistory")

    # Create parent directories if needed
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Convert to numpy arrays
    times = np.array(history.times, dtype=np.float64)
    E_magnetic = np.array(history.E_magnetic, dtype=np.float64)
    E_kinetic = np.array(history.E_kinetic, dtype=np.float64)
    E_compressive = np.array(history.E_compressive, dtype=np.float64)
    E_total = np.array(history.E_total, dtype=np.float64)

    with h5py.File(filename, 'w') as f:
        # Save time series datasets
        f.create_dataset('times', data=times, compression='gzip',
                        compression_opts=COMPRESSION_LEVEL)
        f.create_dataset('E_magnetic', data=E_magnetic, compression='gzip',
                        compression_opts=COMPRESSION_LEVEL)
        f.create_dataset('E_kinetic', data=E_kinetic, compression='gzip',
                        compression_opts=COMPRESSION_LEVEL)
        f.create_dataset('E_compressive', data=E_compressive, compression='gzip',
                        compression_opts=COMPRESSION_LEVEL)
        f.create_dataset('E_total', data=E_total, compression='gzip',
                        compression_opts=COMPRESSION_LEVEL)

        # Save metadata
        f.attrs['version'] = IO_FORMAT_VERSION
        f.attrs['timestamp'] = datetime.now().isoformat()
        f.attrs['n_timesteps'] = len(times)
        f.attrs['t_start'] = float(times[0])
        f.attrs['t_end'] = float(times[-1])

        # Add user metadata if provided
        if metadata is not None:
            for key, value in metadata.items():
                try:
                    f.attrs[key] = value
                except (TypeError, ValueError):
                    f.attrs[key] = str(value)


def load_timeseries(filename: str) -> Tuple[EnergyHistory, Dict[str, Any]]:
    """
    Load EnergyHistory from HDF5 file.

    Loads energy time series data and reconstructs EnergyHistory object
    for analysis and visualization.

    Args:
        filename: Input HDF5 timeseries file path

    Returns:
        history: Loaded EnergyHistory object
        metadata: Dictionary with file metadata

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is incompatible
        KeyError: If required datasets are missing

    Example:
        >>> history, metadata = load_timeseries("energy_history.h5")
        >>> print(f"Loaded {len(history.times)} timesteps")
        >>> print(f"Time range: {metadata['t_start']:.2f} to {metadata['t_end']:.2f}")
        >>>
        >>> # Plot results
        >>> from krmhd.diagnostics import plot_energy_history
        >>> plot_energy_history(history)
    """
    filepath = Path(filename)

    if not filepath.exists():
        raise FileNotFoundError(f"Timeseries file '{filename}' not found")

    with h5py.File(filename, 'r') as f:
        # Check version compatibility
        version = f.attrs.get('version', 'unknown')
        if version != IO_FORMAT_VERSION:
            warnings.warn(
                f"Timeseries file version ({version}) differs from current "
                f"version ({IO_FORMAT_VERSION}). Loading may fail or produce "
                "unexpected results.",
                UserWarning
            )

        # Load time series data
        history = EnergyHistory(
            times=f['times'][:].tolist(),
            E_magnetic=f['E_magnetic'][:].tolist(),
            E_kinetic=f['E_kinetic'][:].tolist(),
            E_compressive=f['E_compressive'][:].tolist(),
            E_total=f['E_total'][:].tolist(),
        )

        # Load metadata
        metadata = dict(f.attrs)

    return history, metadata


def save_turbulence_diagnostics(
    diagnostics_list: list[TurbulenceDiagnostics],
    filename: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save turbulence diagnostics time series to HDF5 file.

    Designed for Issue #82 investigation: stores comprehensive turbulence
    diagnostics computed by compute_turbulence_diagnostics() for post-analysis.

    File format:
        /times: Time array (Alfvén times)
        /max_velocity: Maximum perpendicular velocity |v⊥|
        /cfl_number: CFL number (stability metric)
        /max_nonlinear: Maximum nonlinear term |{z∓, ∇²z±}|
        /energy_highk: Fraction of energy at high-k
        /critical_balance_ratio: Mean τ_nl/τ_A in inertial range
        /energy_total: Total energy
        attributes: Metadata, timestamps, version

    Args:
        diagnostics_list: List of TurbulenceDiagnostics from simulation
        filename: Output HDF5 filename (will be created or overwritten)
        metadata: Optional dictionary of user metadata to store

    Example:
        >>> diagnostics_list = []
        >>> for i in range(n_steps):
        ...     diag = compute_turbulence_diagnostics(state, dt=0.005)
        ...     diagnostics_list.append(diag)
        ...     state = gandalf_step(state, dt=0.005, eta=1.0, v_A=1.0)
        >>> save_turbulence_diagnostics(
        ...     diagnostics_list,
        ...     "turbulence_diag_64cubed.h5",
        ...     metadata={"resolution": 64, "eta": 2.0}
        ... )

    See also:
        load_turbulence_diagnostics: Load diagnostics from file
        compute_turbulence_diagnostics: Compute diagnostics from state
    """
    # Convert list of diagnostics to arrays
    times = np.array([d.time for d in diagnostics_list])
    max_velocity = np.array([d.max_velocity for d in diagnostics_list])
    cfl_number = np.array([d.cfl_number for d in diagnostics_list])
    max_nonlinear = np.array([d.max_nonlinear for d in diagnostics_list])
    energy_highk = np.array([d.energy_highk for d in diagnostics_list])
    critical_balance_ratio = np.array([d.critical_balance_ratio for d in diagnostics_list])
    energy_total = np.array([d.energy_total for d in diagnostics_list])

    # Create HDF5 file
    with h5py.File(filename, 'w') as f:
        # Save time series data with compression
        f.create_dataset('times', data=times, compression='gzip', compression_opts=COMPRESSION_LEVEL)
        f.create_dataset('max_velocity', data=max_velocity, compression='gzip', compression_opts=COMPRESSION_LEVEL)
        f.create_dataset('cfl_number', data=cfl_number, compression='gzip', compression_opts=COMPRESSION_LEVEL)
        f.create_dataset('max_nonlinear', data=max_nonlinear, compression='gzip', compression_opts=COMPRESSION_LEVEL)
        f.create_dataset('energy_highk', data=energy_highk, compression='gzip', compression_opts=COMPRESSION_LEVEL)
        f.create_dataset('critical_balance_ratio', data=critical_balance_ratio, compression='gzip', compression_opts=COMPRESSION_LEVEL)
        f.create_dataset('energy_total', data=energy_total, compression='gzip', compression_opts=COMPRESSION_LEVEL)

        # Save metadata
        f.attrs['format_version'] = IO_FORMAT_VERSION
        f.attrs['created_at'] = datetime.now().isoformat()
        f.attrs['n_samples'] = len(diagnostics_list)
        f.attrs['time_start'] = float(times[0])
        f.attrs['time_end'] = float(times[-1])

        if metadata:
            for key, value in metadata.items():
                f.attrs[key] = value


def load_turbulence_diagnostics(filename: str) -> Tuple[list[TurbulenceDiagnostics], Dict[str, Any]]:
    """
    Load turbulence diagnostics time series from HDF5 file.

    Reads diagnostics saved by save_turbulence_diagnostics() for post-analysis
    and visualization.

    Args:
        filename: HDF5 file to load

    Returns:
        diagnostics_list: List of TurbulenceDiagnostics objects
        metadata: Dictionary of metadata from file attributes

    Example:
        >>> diag_list, meta = load_turbulence_diagnostics("turbulence_diag_64cubed.h5")
        >>> print(f"Loaded {len(diag_list)} samples from t={meta['time_start']:.1f} to t={meta['time_end']:.1f}")
        >>> # Plot CFL number over time
        >>> times = [d.time for d in diag_list]
        >>> cfl = [d.cfl_number for d in diag_list]
        >>> plt.plot(times, cfl)
        >>> plt.xlabel("Time (τ_A)")
        >>> plt.ylabel("CFL number")

    Raises:
        FileNotFoundError: If file doesn't exist
        KeyError: If file is missing required datasets

    See also:
        save_turbulence_diagnostics: Save diagnostics to file
        compute_turbulence_diagnostics: Compute diagnostics from state
    """
    # Check file exists
    if not Path(filename).exists():
        raise FileNotFoundError(f"Turbulence diagnostics file not found: {filename}")

    with h5py.File(filename, 'r') as f:
        # Check version
        version = f.attrs.get('format_version', 'unknown')
        if version != IO_FORMAT_VERSION:
            warnings.warn(
                f"Turbulence diagnostics file version ({version}) differs from current "
                f"version ({IO_FORMAT_VERSION}). Loading may fail or produce "
                "unexpected results.",
                UserWarning
            )

        # Load time series data
        times = f['times'][:]
        max_velocity = f['max_velocity'][:]
        cfl_number = f['cfl_number'][:]
        max_nonlinear = f['max_nonlinear'][:]
        energy_highk = f['energy_highk'][:]
        critical_balance_ratio = f['critical_balance_ratio'][:]
        energy_total = f['energy_total'][:]

        # Convert to list of TurbulenceDiagnostics objects
        diagnostics_list = [
            TurbulenceDiagnostics(
                time=float(times[i]),
                max_velocity=float(max_velocity[i]),
                cfl_number=float(cfl_number[i]),
                max_nonlinear=float(max_nonlinear[i]),
                energy_highk=float(energy_highk[i]),
                critical_balance_ratio=float(critical_balance_ratio[i]),
                energy_total=float(energy_total[i]),
            )
            for i in range(len(times))
        ]

        # Load metadata
        metadata = dict(f.attrs)

    return diagnostics_list, metadata
