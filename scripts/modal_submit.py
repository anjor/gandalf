#!/usr/bin/env python3
"""
Modal Job Submission and Management Script

This script provides a convenient CLI for submitting GANDALF simulations to Modal
and managing results.

Features:
- Submit single simulations with CPU or GPU
- Run parameter sweeps in parallel
- List and download results from Modal volumes
- Monitor running jobs
- Config validation before submission

Usage:
    # Submit a single simulation
    python scripts/modal_submit.py submit configs/driven_turbulence.yaml

    # Submit with GPU
    python scripts/modal_submit.py submit configs/driven_turbulence.yaml --gpu

    # Run a parameter sweep
    python scripts/modal_submit.py sweep configs/driven_turbulence.yaml \\
        --param physics.eta 0.01 0.02 0.05 \\
        --param grid.Nx 64 128 256

    # List results
    python scripts/modal_submit.py list

    # Download results
    python scripts/modal_submit.py download driven_turbulence_20240115_120000 ./results

    # Monitor job status
    python scripts/modal_submit.py status
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional, Dict, List, Any


def sanitize_path_component(path_str: str, allow_absolute: bool = False) -> str:
    """
    Sanitize a path string to prevent path traversal and command injection.

    Args:
        path_str: Path string to sanitize
        allow_absolute: If True, allow absolute paths (for local filesystem paths)

    Returns:
        Sanitized path string

    Raises:
        ValueError: If path contains dangerous patterns
    """
    if not path_str:
        raise ValueError("Path component cannot be empty")

    # Strip leading/trailing whitespace
    sanitized = path_str.strip()

    # Check for path traversal attempts
    if '..' in sanitized:
        raise ValueError(f"Path traversal attempt detected (contains '..'): {path_str}")

    # Check for absolute paths (unless we explicitly allow them for local filesystem)
    if not allow_absolute and sanitized.startswith('/'):
        raise ValueError(f"Absolute paths not allowed: {path_str}")

    # Check for null bytes (command injection)
    if '\0' in sanitized:
        raise ValueError(f"Null byte in path: {path_str}")

    # Check for shell metacharacters that could enable injection
    dangerous_chars = ['&', '|', ';', '`', '$', '(', ')', '<', '>', '"', "'", '\\', '\n', '\r']
    for char in dangerous_chars:
        if char in sanitized:
            raise ValueError(f"Dangerous character '{char}' in path: {path_str}")

    return sanitized


def estimate_cost(config_path: Path, use_gpu: bool, n_jobs: int = 1) -> Dict[str, Any]:
    """
    Estimate Modal cost for a simulation or sweep.

    Args:
        config_path: Path to YAML configuration file
        use_gpu: Whether GPU instances will be used
        n_jobs: Number of parallel jobs (for sweeps)

    Returns:
        Dictionary with cost estimate and details
    """
    import yaml

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception:
        # If can't read config, return unknown cost
        return {
            'estimated_cost': None,
            'warning': 'Could not estimate cost (config unreadable)',
        }

    # Extract parameters
    Nx = config.get('grid', {}).get('Nx', 64)
    n_steps = config.get('time_integration', {}).get('n_steps', 100)

    # Rough runtime estimates (minutes) based on resolution and backend
    # These are conservative estimates from actual runs
    if use_gpu:
        # GPU runtime (faster for high-res)
        if Nx <= 64:
            runtime_min = max(5, n_steps * 0.001)
        elif Nx <= 128:
            runtime_min = max(5, n_steps * 0.003)
        elif Nx <= 256:
            runtime_min = max(10, n_steps * 0.015)
        else:  # 512+
            runtime_min = max(60, n_steps * 0.12)

        # GPU pricing (Modal T4/A10G rates, approximate)
        cost_per_min = 0.10  # ~$0.10/min for T4
        if Nx > 256:
            cost_per_min = 0.15  # A100 needed for 512³
    else:
        # CPU runtime (slower, especially for high-res)
        if Nx <= 64:
            runtime_min = max(5, n_steps * 0.003)
        elif Nx <= 128:
            runtime_min = max(10, n_steps * 0.01)
        else:  # 256+ not recommended on CPU
            runtime_min = max(60, n_steps * 0.08)

        # CPU pricing (Modal CPU rates)
        cost_per_min = 0.02  # ~$0.02/min for 8-core CPU

    # Total cost for all jobs
    total_runtime_min = runtime_min * n_jobs
    total_cost = cost_per_min * total_runtime_min

    return {
        'estimated_cost': total_cost,
        'runtime_min': runtime_min,
        'total_runtime_min': total_runtime_min,
        'resolution': f"{Nx}³",
        'n_steps': n_steps,
        'n_jobs': n_jobs,
        'backend': 'GPU' if use_gpu else 'CPU',
        'warning': None if total_cost < 10 else f'High cost estimate: ${total_cost:.2f}',
    }


def warn_about_cost(config_path: Path, use_gpu: bool, n_jobs: int = 1, force: bool = False) -> bool:
    """
    Display cost estimate and prompt for confirmation if expensive.

    Args:
        config_path: Path to YAML configuration file
        use_gpu: Whether GPU instances will be used
        n_jobs: Number of parallel jobs (for sweeps)
        force: If True, skip confirmation prompt

    Returns:
        True if user confirms or cost is low, False if user cancels
    """
    estimate = estimate_cost(config_path, use_gpu, n_jobs)

    if estimate['estimated_cost'] is None:
        # Could not estimate, warn but allow
        print("⚠️  Warning: Could not estimate cost")
        if not force:
            response = input("Continue anyway? [y/N]: ").strip().lower()
            return response == 'y'
        return True

    cost = estimate['estimated_cost']

    # Always show estimate
    print(f"\n💰 Cost Estimate:")
    print(f"  Resolution: {estimate['resolution']}")
    print(f"  Steps: {estimate['n_steps']}")
    print(f"  Backend: {estimate['backend']}")
    if n_jobs > 1:
        print(f"  Jobs: {n_jobs}")
        print(f"  Runtime/job: ~{estimate['runtime_min']:.1f} min")
        print(f"  Total runtime: ~{estimate['total_runtime_min']:.1f} min")
    else:
        print(f"  Runtime: ~{estimate['runtime_min']:.1f} min")
    print(f"  Estimated cost: ${cost:.2f}")

    # Prompt for confirmation if expensive (>$5 threshold)
    if cost > 5.0 and not force:
        print(f"\n⚠️  This job will cost approximately ${cost:.2f}")
        print("    (Note: Estimate may vary by ±50% depending on actual runtime)")
        response = input("\nProceed with submission? [y/N]: ").strip().lower()
        return response == 'y'

    print()  # Blank line for readability
    return True


def check_modal_installed() -> None:
    """
    Check if Modal is installed and authenticated.

    Raises:
        SystemExit: If Modal is not installed or not authenticated
    """
    try:
        subprocess.run(
            ["modal", "token", "list"],
            capture_output=True,
            text=True,
            check=True  # Raise CalledProcessError on non-zero exit
        )
    except subprocess.CalledProcessError:
        print("Error: Modal is not authenticated.")
        print("Please run: modal token new")
        sys.exit(1)
    except FileNotFoundError:
        print("Error: Modal CLI not found.")
        print("Please install: pip install modal")
        sys.exit(1)


def validate_config(config_path: Path) -> bool:
    """
    Validate YAML configuration before submitting to Modal.

    This prevents costly cloud execution failures due to configuration errors.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        True if config is valid, False otherwise
    """
    try:
        import yaml
        from krmhd.config import SimulationConfig

        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        # Pydantic will validate all fields
        config = SimulationConfig(**config_dict)

        print(f"✓ Config validation passed")
        print(f"  Resolution: {config.grid.Nx}×{config.grid.Ny}×{config.grid.Nz}")
        print(f"  Steps: {config.time_integration.n_steps}")
        print(f"  Hyper-dissipation: r={config.physics.hyper_r}, n={config.physics.hyper_n}")

        return True

    except FileNotFoundError:
        print(f"Error: Config file not found: {config_path}")
        return False
    except Exception as e:
        print(f"Error: Config validation failed: {e}")
        return False


def submit_simulation(
    config_path: str,
    use_gpu: bool = False,
    output_subdir: Optional[str] = None,
    wait: bool = True,
    skip_validation: bool = False
) -> None:
    """
    Submit a simulation to Modal.

    Args:
        config_path: Path to YAML configuration file
        use_gpu: Use GPU instance
        output_subdir: Optional output subdirectory name
        wait: Wait for job to complete
        skip_validation: Skip config validation (not recommended)

    Raises:
        SystemExit: If submission fails
    """
    config_file = Path(config_path)

    # Validate config before submission (unless explicitly skipped)
    if not skip_validation:
        if not validate_config(config_file):
            print("\nTo skip validation (not recommended), use --skip-validation")
            sys.exit(1)
        print()

    # Show cost estimate and confirm if expensive
    if not warn_about_cost(config_file, use_gpu, n_jobs=1):
        print("Submission cancelled by user.")
        sys.exit(0)

    # Sanitize output_subdir if provided (prevent path traversal)
    if output_subdir:
        try:
            output_subdir = sanitize_path_component(output_subdir, allow_absolute=False)
        except ValueError as e:
            print(f"Error: Invalid output_subdir: {e}")
            sys.exit(1)

    # Build command
    cmd = [
        "modal", "run", "modal_app.py",
        "--config-path", str(config_path)
    ]

    if use_gpu:
        cmd.append("--use-gpu")

    if output_subdir:
        cmd.extend(["--output-subdir", output_subdir])

    print(f"Submitting simulation to Modal...")
    print(f"Config: {config_path}")
    print(f"GPU: {use_gpu}")
    print(f"Command: {' '.join(cmd)}\n")

    # Run command
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\nError: Modal job failed with return code {e.returncode}")
        sys.exit(1)


def submit_parameter_sweep(
    config_path: str,
    parameters: Dict[str, List[Any]],
    sweep_name: Optional[str] = None,
    use_gpu: bool = False,
    skip_validation: bool = False
) -> None:
    """
    Submit a parameter sweep to Modal.

    Args:
        config_path: Path to base YAML configuration file
        parameters: Dictionary mapping parameter paths to value lists
        sweep_name: Optional name for the sweep
        use_gpu: Use GPU instances
        skip_validation: Skip config validation (not recommended)

    Raises:
        SystemExit: If submission fails
    """
    config_file = Path(config_path)

    # Validate base config before submission (unless explicitly skipped)
    if not skip_validation:
        if not validate_config(config_file):
            print("\nTo skip validation (not recommended), use --skip-validation")
            sys.exit(1)
        print()

    # Calculate number of jobs for cost estimation
    from itertools import product
    param_values = list(parameters.values())
    n_jobs = len(list(product(*param_values)))

    # Show cost estimate and confirm if expensive
    if not warn_about_cost(config_file, use_gpu, n_jobs=n_jobs):
        print("Sweep cancelled by user.")
        sys.exit(0)

    # Read base config
    with open(config_file, 'r') as f:
        base_config = f.read()

    # Create temporary Python script for sweep
    sweep_script_content = f"""
import modal_app

base_config = '''{base_config}'''

parameters = {parameters!r}

result = modal_app.run_parameter_sweep.remote(
    base_config,
    parameters,
    sweep_name={sweep_name!r},
    use_gpu={use_gpu}
)

print("\\nSweep results:")
for i, r in enumerate(result['successful_results']):
    print(f"  Run {{i}}: {{r['config_name']}} - E_final={{r['final_energy_total']:.6e}}")

if result.get('partial_failures'):
    print(f"\\nPartial failures: {{len(result['partial_failures'])}} (computed but volume not persisted)")
    for pf in result['partial_failures']:
        print(f"  - {{pf['job_metadata']['output_subdir']}}: {{pf.get('volume_commit_error', 'commit error')}}")

if result['failed_jobs']:
    print(f"\\nFailed jobs: {{len(result['failed_jobs'])}}")
    for job in result['failed_jobs']:
        print(f"  - {{job['job_metadata']['output_subdir']}}: {{job.get('error', 'Unknown error')}}")
"""

    # Use context manager for proper cleanup
    temp_script = None
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.py',
            prefix='modal_sweep_',
            delete=False
        ) as f:
            temp_script = Path(f.name)
            f.write(sweep_script_content)

        # Run sweep
        print(f"Submitting parameter sweep to Modal...")
        print(f"Base config: {config_path}")
        print(f"Parameters: {parameters}")
        print(f"GPU: {use_gpu}\n")

        cmd = ["modal", "run", str(temp_script)]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"\nError: Modal sweep failed with return code {e.returncode}")
            sys.exit(1)

    finally:
        # Clean up temp script
        if temp_script and temp_script.exists():
            temp_script.unlink()


def list_results() -> None:
    """
    List all results in Modal volume.

    Raises:
        SystemExit: If listing fails
    """
    print("\nListing results in Modal volume...")

    script_content = """
import modal_app
results = modal_app.list_results.remote()
for r in results:
    print(f"  {r}")
"""

    temp_script = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.py',
            prefix='modal_list_',
            delete=False
        ) as f:
            temp_script = Path(f.name)
            f.write(script_content)

        cmd = ["modal", "run", str(temp_script)]
        subprocess.run(cmd, check=True)

    finally:
        if temp_script and temp_script.exists():
            temp_script.unlink()


def download_results(result_dir: str, local_path: str) -> None:
    """
    Download results from Modal volume.

    Args:
        result_dir: Directory name in Modal volume
        local_path: Local path to save results

    Raises:
        SystemExit: If download fails
    """
    import os

    # Sanitize inputs to prevent path traversal and command injection
    try:
        # result_dir should not contain path traversal or absolute paths
        result_dir = sanitize_path_component(result_dir, allow_absolute=False)
        # local_path can be absolute (it's a local filesystem path)
        local_path = sanitize_path_component(local_path, allow_absolute=True)
    except ValueError as e:
        print(f"Error: Invalid path: {e}")
        sys.exit(1)

    # Get volume name from environment or use default
    volume_name = os.environ.get("MODAL_VOLUME_NAME", "gandalf-results")

    print(f"\nDownloading results from Modal...")
    print(f"Remote: {result_dir}")
    print(f"Local: {local_path}\n")

    cmd = [
        "modal", "volume", "get",
        volume_name,
        result_dir,
        local_path
    ]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\nError: Download failed with return code {e.returncode}")
        sys.exit(1)

    print(f"\n✓ Results downloaded to {local_path}")


def show_status() -> None:
    """
    Show status of Modal jobs.

    Raises:
        SystemExit: If status check fails
    """
    print("\nChecking Modal job status...\n")

    cmd = ["modal", "app", "list"]
    subprocess.run(cmd, check=True)


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Submit and manage GANDALF simulations on Modal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Submit single simulation
  python scripts/modal_submit.py submit configs/driven_turbulence.yaml

  # Submit with GPU
  python scripts/modal_submit.py submit configs/driven_turbulence.yaml --gpu

  # Use custom volume
  python scripts/modal_submit.py submit configs/driven_turbulence.yaml --volume-name my-results

  # Run parameter sweep
  python scripts/modal_submit.py sweep configs/driven_turbulence.yaml \\
      --param physics.eta 0.01 0.02 0.05 \\
      --param grid.Nx 64 128

  # List results
  python scripts/modal_submit.py list

  # Download results
  python scripts/modal_submit.py download driven_turbulence_20240115_120000 ./results
        """
    )

    # Global options
    parser.add_argument(
        '--volume-name',
        help='Modal volume name for results (default: gandalf-results)',
        default=None
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # Submit command
    submit_parser = subparsers.add_parser('submit', help='Submit a single simulation')
    submit_parser.add_argument('config', help='Path to YAML configuration file')
    submit_parser.add_argument('--gpu', action='store_true', help='Use GPU instance')
    submit_parser.add_argument('--output-subdir', help='Output subdirectory name')
    submit_parser.add_argument('--no-wait', action='store_true', help='Do not wait for completion')
    submit_parser.add_argument('--skip-validation', action='store_true',
                              help='Skip config validation (not recommended)')

    # Sweep command
    sweep_parser = subparsers.add_parser('sweep', help='Run a parameter sweep')
    sweep_parser.add_argument('config', help='Path to base YAML configuration file')
    sweep_parser.add_argument(
        '--param',
        action='append',
        nargs='+',
        metavar=('NAME', 'VALUE'),
        help='Parameter to sweep (e.g., --param physics.eta 0.01 0.02 0.05)'
    )
    sweep_parser.add_argument('--name', help='Sweep name')
    sweep_parser.add_argument('--gpu', action='store_true', help='Use GPU instances')
    sweep_parser.add_argument('--skip-validation', action='store_true',
                             help='Skip config validation (not recommended)')

    # List command
    list_parser = subparsers.add_parser('list', help='List results in Modal volume')

    # Download command
    download_parser = subparsers.add_parser('download', help='Download results from Modal')
    download_parser.add_argument('result_dir', help='Result directory name in Modal volume')
    download_parser.add_argument('local_path', help='Local path to save results')

    # Status command
    status_parser = subparsers.add_parser('status', help='Show Modal job status')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    # Set volume name in environment if provided
    if args.volume_name:
        import os
        os.environ['MODAL_VOLUME_NAME'] = args.volume_name
        print(f"Using Modal volume: {args.volume_name}")

    # Check Modal installation
    check_modal_installed()

    # Execute command
    if args.command == 'submit':
        submit_simulation(
            args.config,
            use_gpu=args.gpu,
            output_subdir=args.output_subdir,
            wait=not args.no_wait,
            skip_validation=args.skip_validation
        )

    elif args.command == 'sweep':
        if not args.param:
            print("Error: Must specify at least one --param for sweep")
            sys.exit(1)

        # Parse parameters
        parameters: Dict[str, List[Any]] = {}
        for param_spec in args.param:
            if len(param_spec) < 2:
                print(f"Error: --param must have NAME VALUE [VALUE...], got: {param_spec}")
                sys.exit(1)

            param_name = param_spec[0]
            param_values: List[Any] = []
            for val in param_spec[1:]:
                # Try to parse as number, fall back to string
                try:
                    param_values.append(float(val))
                except ValueError:
                    try:
                        param_values.append(int(val))
                    except ValueError:
                        param_values.append(val)

            parameters[param_name] = param_values

        submit_parameter_sweep(
            args.config,
            parameters,
            sweep_name=args.name,
            use_gpu=args.gpu,
            skip_validation=args.skip_validation
        )

    elif args.command == 'list':
        list_results()

    elif args.command == 'download':
        download_results(args.result_dir, args.local_path)

    elif args.command == 'status':
        show_status()


if __name__ == "__main__":
    main()
