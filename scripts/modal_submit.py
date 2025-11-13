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
from pathlib import Path
from typing import Optional


def check_modal_installed():
    """Check if Modal is installed and authenticated."""
    try:
        result = subprocess.run(
            ["modal", "token", "list"],
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode != 0:
            print("Error: Modal is not authenticated.")
            print("Please run: modal token new")
            sys.exit(1)
    except FileNotFoundError:
        print("Error: Modal CLI not found.")
        print("Please install: pip install modal")
        sys.exit(1)


def submit_simulation(
    config_path: str,
    use_gpu: bool = False,
    output_subdir: Optional[str] = None,
    wait: bool = True
) -> None:
    """
    Submit a simulation to Modal.

    Args:
        config_path: Path to YAML configuration file
        use_gpu: Use GPU instance
        output_subdir: Optional output subdirectory name
        wait: Wait for job to complete
    """
    config_file = Path(config_path)
    if not config_file.exists():
        print(f"Error: Config file not found: {config_path}")
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

    print(f"\nSubmitting simulation to Modal...")
    print(f"Config: {config_path}")
    print(f"GPU: {use_gpu}")
    print(f"Command: {' '.join(cmd)}\n")

    # Run command
    result = subprocess.run(cmd, check=False)

    if result.returncode != 0:
        print(f"\nError: Modal job failed with return code {result.returncode}")
        sys.exit(1)


def submit_parameter_sweep(
    config_path: str,
    parameters: dict[str, list],
    sweep_name: Optional[str] = None,
    use_gpu: bool = False
) -> None:
    """
    Submit a parameter sweep to Modal.

    Args:
        config_path: Path to base YAML configuration file
        parameters: Dictionary mapping parameter paths to value lists
        sweep_name: Optional name for the sweep
        use_gpu: Use GPU instances
    """
    config_file = Path(config_path)
    if not config_file.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    # Read base config
    with open(config_file, 'r') as f:
        base_config = f.read()

    # Create sweep script
    sweep_script = f"""
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
for i, r in enumerate(result):
    print(f"  Run {{i}}: {{r['config_name']}} - E_final={{r['final_energy_total']:.6e}}")
"""

    # Save temporary script
    temp_script = Path(".modal_sweep_temp.py")
    with open(temp_script, 'w') as f:
        f.write(sweep_script)

    try:
        # Run sweep
        print(f"\nSubmitting parameter sweep to Modal...")
        print(f"Base config: {config_path}")
        print(f"Parameters: {parameters}")
        print(f"GPU: {use_gpu}\n")

        cmd = ["modal", "run", str(temp_script)]
        result = subprocess.run(cmd, check=False)

        if result.returncode != 0:
            print(f"\nError: Modal sweep failed with return code {result.returncode}")
            sys.exit(1)

    finally:
        # Clean up temp script
        if temp_script.exists():
            temp_script.unlink()


def list_results() -> None:
    """List all results in Modal volume."""
    print("\nListing results in Modal volume...")

    script = """
import modal_app
results = modal_app.list_results.remote()
for r in results:
    print(f"  {r}")
"""

    temp_script = Path(".modal_list_temp.py")
    with open(temp_script, 'w') as f:
        f.write(script)

    try:
        cmd = ["modal", "run", str(temp_script)]
        subprocess.run(cmd, check=True)
    finally:
        if temp_script.exists():
            temp_script.unlink()


def download_results(result_dir: str, local_path: str) -> None:
    """
    Download results from Modal volume.

    Args:
        result_dir: Directory name in Modal volume
        local_path: Local path to save results
    """
    print(f"\nDownloading results from Modal...")
    print(f"Remote: {result_dir}")
    print(f"Local: {local_path}\n")

    cmd = [
        "modal", "volume", "get",
        "gandalf-results",
        result_dir,
        local_path
    ]

    result = subprocess.run(cmd, check=False)

    if result.returncode != 0:
        print(f"\nError: Download failed with return code {result.returncode}")
        sys.exit(1)

    print(f"\n✓ Results downloaded to {local_path}")


def show_status() -> None:
    """Show status of Modal jobs."""
    print("\nChecking Modal job status...\n")

    cmd = ["modal", "app", "list"]
    subprocess.run(cmd, check=True)


def main():
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

    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # Submit command
    submit_parser = subparsers.add_parser('submit', help='Submit a single simulation')
    submit_parser.add_argument('config', help='Path to YAML configuration file')
    submit_parser.add_argument('--gpu', action='store_true', help='Use GPU instance')
    submit_parser.add_argument('--output-subdir', help='Output subdirectory name')
    submit_parser.add_argument('--no-wait', action='store_true', help='Do not wait for completion')

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

    # Check Modal installation
    check_modal_installed()

    # Execute command
    if args.command == 'submit':
        submit_simulation(
            args.config,
            use_gpu=args.gpu,
            output_subdir=args.output_subdir,
            wait=not args.no_wait
        )

    elif args.command == 'sweep':
        if not args.param:
            print("Error: Must specify at least one --param for sweep")
            sys.exit(1)

        # Parse parameters
        parameters = {}
        for param_spec in args.param:
            if len(param_spec) < 2:
                print(f"Error: --param must have NAME VALUE [VALUE...], got: {param_spec}")
                sys.exit(1)

            param_name = param_spec[0]
            param_values = []
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
            use_gpu=args.gpu
        )

    elif args.command == 'list':
        list_results()

    elif args.command == 'download':
        download_results(args.result_dir, args.local_path)

    elif args.command == 'status':
        show_status()


if __name__ == "__main__":
    main()
