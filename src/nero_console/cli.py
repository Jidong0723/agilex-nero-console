"""Stable command-line interface for local development and deployment."""
from __future__ import annotations

import argparse
import runpy
import sys

from .runtime import assert_control_interpreter, doctor_json, project_root


def _serve(arguments: list[str]) -> int:
    root = project_root()
    assert_control_interpreter(root)
    server = root / "scripts" / "nero_control_server.py"
    sys.argv = [str(server), *arguments]
    runpy.run_path(str(server), run_name="__main__")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="nero-console")
    subcommands = parser.add_subparsers(dest="command")
    subcommands.add_parser("doctor", help="report local control and kinematics runtime readiness")
    serve = subcommands.add_parser("serve", help="run the localhost control service")
    serve.add_argument("server_args", nargs=argparse.REMAINDER)
    parsed = parser.parse_args(argv)
    if parsed.command == "doctor":
        print(doctor_json())
        return 0
    if parsed.command == "serve":
        return _serve(list(parsed.server_args))
    parser.print_help()
    return 2

