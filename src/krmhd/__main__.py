"""
CLI interface for KRMHD tools.

Usage:
    python -m krmhd validate <config.yaml>
    python -m krmhd suggest --resolution 64 64 32 --type forced
"""

import argparse
import sys
from pathlib import Path

import yaml

from .validation import (
    validate_config_dict,
    suggest_parameters,
    validate_parameters,
)


def cmd_validate(args):
    """Validate parameters from a config file."""
    config_path = Path(args.config)

    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        return 1

    print(f"Validating {config_path}...")
    print("=" * 70)

    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Validate
    result = validate_config_dict(config)

    # Print report
    result.print_report()

    print("=" * 70)

    if result.valid:
        print("✓ Configuration is valid")
        return 0
    else:
        print("❌ Configuration has errors (see above)")
        return 1


def cmd_suggest(args):
    """Suggest parameters for a given resolution."""
    Nx, Ny, Nz = args.resolution
    sim_type = args.type
    cfl = args.cfl

    print(f"Suggesting parameters for {Nx}×{Ny}×{Nz} resolution")
    print(f"Simulation type: {sim_type}")
    print(f"Target CFL: {cfl}")
    print("=" * 70)

    params = suggest_parameters(Nx, Ny, Nz, sim_type, cfl)

    print("\nRecommended parameters:")
    print(f"  dt:              {params['dt']:.4f}")
    print(f"  eta:             {params['eta']:.3f}")
    print(f"  nu:              {params['nu']:.3f}")
    print(f"  v_A:             {params['v_A']:.1f}")
    print(f"  beta_i:          {params['beta_i']:.1f}")
    if sim_type == "forced":
        print(f"  force_amplitude: {params['force_amplitude']:.2f}")
        print(f"  n_force_min:     {params['n_force_min']}")
        print(f"  n_force_max:     {params['n_force_max']}")
    print(f"  cfl_safety:      {params['cfl_safety']:.2f}")

    # Validate suggested parameters
    print("\n" + "=" * 70)
    print("Validating suggested parameters...")
    print("=" * 70)

    result = validate_parameters(
        params['dt'],
        params['eta'],
        params['nu'],
        Nx, Ny, Nz,
        params['v_A'],
        params.get('force_amplitude'),
        hyper_r=1,
        hyper_n=1,
        n_force_max=params.get('n_force_max', 2)
    )

    result.print_report()
    print("=" * 70)

    return 0


def cmd_check(args):
    """Quick parameter check from command line."""
    print("Checking parameters...")
    print("=" * 70)

    result = validate_parameters(
        args.dt,
        args.eta,
        args.nu,
        args.Nx,
        args.Ny,
        args.Nz,
        args.v_A,
        args.force_amplitude,
        args.hyper_r,
        args.hyper_n,
        args.n_force_max
    )

    result.print_report()
    print("=" * 70)

    if result.valid:
        print("✓ Parameters are valid")
        return 0
    else:
        print("❌ Parameters have errors")
        return 1


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="KRMHD parameter validation and suggestion tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate a config file
  python -m krmhd validate configs/driven_turbulence.yaml

  # Suggest parameters for 64³ forced turbulence
  python -m krmhd suggest --resolution 64 64 32 --type forced

  # Quick parameter check
  python -m krmhd check --dt 0.01 --eta 0.5 --nu 0.5 --Nx 64 --Ny 64 --Nz 32

For more information, see:
  - docs/recommended_parameters.md
  - docs/ISSUE82_SUMMARY.md
"""
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Validate command
    parser_validate = subparsers.add_parser(
        "validate",
        help="Validate parameters from a config file"
    )
    parser_validate.add_argument(
        "config",
        help="Path to YAML config file"
    )

    # Suggest command
    parser_suggest = subparsers.add_parser(
        "suggest",
        help="Suggest parameters for a given resolution"
    )
    parser_suggest.add_argument(
        "--resolution",
        nargs=3,
        type=int,
        required=True,
        metavar=("Nx", "Ny", "Nz"),
        help="Grid resolution (Nx Ny Nz)"
    )
    parser_suggest.add_argument(
        "--type",
        choices=["forced", "decaying", "benchmark"],
        default="forced",
        help="Simulation type (default: forced)"
    )
    parser_suggest.add_argument(
        "--cfl",
        type=float,
        default=0.3,
        help="Target CFL safety factor (default: 0.3)"
    )

    # Check command
    parser_check = subparsers.add_parser(
        "check",
        help="Quick parameter validation from command line"
    )
    parser_check.add_argument("--dt", type=float, required=True, help="Timestep")
    parser_check.add_argument("--eta", type=float, required=True, help="Resistivity")
    parser_check.add_argument("--nu", type=float, required=True, help="Collision frequency")
    parser_check.add_argument("--Nx", type=int, required=True, help="Grid resolution X")
    parser_check.add_argument("--Ny", type=int, required=True, help="Grid resolution Y")
    parser_check.add_argument("--Nz", type=int, required=True, help="Grid resolution Z")
    parser_check.add_argument("--v_A", type=float, default=1.0, help="Alfvén velocity")
    parser_check.add_argument("--force_amplitude", type=float, help="Forcing amplitude")
    parser_check.add_argument("--hyper_r", type=int, default=1, help="Hyper-dissipation order")
    parser_check.add_argument("--hyper_n", type=int, default=1, help="Hyper-collision order")
    parser_check.add_argument("--n_force_max", type=int, default=2, help="Max forcing mode")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    if args.command == "validate":
        return cmd_validate(args)
    elif args.command == "suggest":
        return cmd_suggest(args)
    elif args.command == "check":
        return cmd_check(args)


if __name__ == "__main__":
    sys.exit(main())
