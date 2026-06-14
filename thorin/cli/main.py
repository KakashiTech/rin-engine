#!/usr/bin/env python3
"""THORIN CLI — main entry point.

Usage:
    thorin --help
    thorin --version
    thorin <command> [options]

Subcommands:
    run           Run inference and generate text
    bench         Benchmark inference speed
    energy        Profile energy consumption
    inspect       Display model information
    import        Import model from external format
    list-models   List available pre-trained models
"""

import argparse
import sys

from thorin.runtime import ThorEngine

from thorin.cli import run, bench, energy, inspect, import_cmd

VERSION = "1.0.0"

BOLD = "\033[1m"
GREEN = "\033[32m"
CYAN = "\033[36m"
RED = "\033[31m"
YELLOW = "\033[33m"
RESET = "\033[0m"


def get_version() -> str:
    try:
        lib_ver = ThorEngine.version()
    except Exception:
        lib_ver = "unavailable"
    return f"THORIN CLI v{VERSION}  (runtime {lib_ver})"


def main() -> None:
    if "--version" in sys.argv:
        print(get_version())
        sys.exit(0)

    parser = argparse.ArgumentParser(
        prog="thorin",
        description="THORIN Inference Engine — command-line interface",
        epilog=(
            "See 'thorin <command> --help' for detailed help on each subcommand."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    sub = parser.add_subparsers(dest="command", metavar="<command>")

    run.register_subparser(sub)
    bench.register_subparser(sub)
    energy.register_subparser(sub)
    inspect.register_subparser(sub)
    import_cmd.register_subparser(sub)

    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    command_map = {
        "run": run.run,
        "bench": bench.run,
        "energy": energy.run,
        "inspect": inspect.run,
        "import": import_cmd.run,
    }

    try:
        command_map[args.command](args)
    except KeyboardInterrupt:
        print(f"\n{YELLOW}Interrupted.{RESET}", file=sys.stderr)
        sys.exit(130)
    except FileNotFoundError as e:
        print(f"{RED}Error: {e}{RESET}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"{RED}Error: {e}{RESET}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
