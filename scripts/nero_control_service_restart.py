"""Restart helper for the local NERO control service.

This process is intentionally independent from the HTTP server it restarts.
It waits for the old PID to disappear, then launches one fresh service.  It
does not import the robot SDK and sends no CAN frames.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        ctypes.windll.kernel32.CloseHandle(handle)
        return True
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _python_process_rows() -> list[dict[str, object]]:
    if os.name != "nt":
        return []
    command = (
        "Get-CimInstance Win32_Process | Where-Object { $_.Name -like 'python*' } | "
        "Select-Object ProcessId,ParentProcessId,CommandLine | ConvertTo-Json"
    )
    try:
        output = subprocess.check_output(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
            text=True, stderr=subprocess.DEVNULL, timeout=3.0,
        ).strip()
    except (subprocess.SubprocessError, OSError):
        return []
    if not output:
        return []
    try:
        payload = json.loads(output)
    except json.JSONDecodeError:
        return []
    rows = payload if isinstance(payload, list) else [payload]
    return [row for row in rows if isinstance(row, dict)]


def matching_python_pids(project_root: Path, script_name: str) -> list[int]:
    if os.name != "nt":
        return []
    rows = _python_process_rows()
    root_text = str(project_root).lower()
    script_text = script_name.lower()
    result: list[int] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        command_line = str(row.get("CommandLine") or "").lower()
        module_text = script_text.replace(".py", "").replace("\\", ".")
        if root_text not in command_line or (script_text not in command_line and module_text not in command_line):
            continue
        if "nero_control_service_restart.py" in command_line:
            continue
        if script_text == "nero_control_server.py" and "nero_control_watchdog.py" in command_line:
            continue
        if script_text == "nero_control_watchdog.py" and "nero_control_server.py --config" in command_line:
            continue
        try:
            result.append(int(row.get("ProcessId")))
        except (TypeError, ValueError):
            pass
    return result


def matching_spawn_workers(project_root: Path) -> list[int]:
    """Find multiprocessing spawn workers whose service ancestor is local.

    On Windows the venv launcher and multiprocessing spawn can detach the
    SDK-owning child from the PID that taskkill was given. These workers have
    no project path in their command line, but retain a parent_pid pointing at
    a project control-server process.
    """
    rows = _python_process_rows()
    by_pid: dict[int, dict[str, object]] = {}
    for row in rows:
        try:
            by_pid[int(row.get("ProcessId", 0))] = row
        except (TypeError, ValueError):
            pass
    root = str(project_root).lower()
    trusted: set[int] = set()
    for pid, row in by_pid.items():
        command = str(row.get("CommandLine") or "").lower()
        if root in command and "nero_control_server.py" in command and "nero_control_service_restart.py" not in command:
            trusted.add(pid)
    workers: set[int] = set()
    changed = True
    while changed:
        changed = False
        for pid, row in by_pid.items():
            command = str(row.get("CommandLine") or "").lower()
            if "spawn_main(parent_pid=" not in command:
                continue
            try:
                parent = int(row.get("ParentProcessId", 0))
            except (TypeError, ValueError):
                continue
            if parent in trusted and pid not in workers:
                workers.add(pid); trusted.add(pid); changed = True
    # A killed venv launcher can leave the spawn child with a dead parent,
    # which is exactly the native CAN-handle leak seen after reset. Such a
    # process is not an active control service and must not survive restart.
    live_pids = set(by_pid)
    for pid, row in by_pid.items():
        command = str(row.get("CommandLine") or "").lower()
        if "spawn_main(parent_pid=" not in command:
            continue
        try:
            parent = int(row.get("ParentProcessId", 0))
        except (TypeError, ValueError):
            continue
        if parent not in live_pids:
            workers.add(pid)
    return sorted(workers)


def terminate_process_tree(pid: int) -> None:
    """Force-stop one old service and every subprocess it created."""
    if pid <= 0 or pid == os.getpid():
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill.exe", "/PID", str(pid), "/T", "/F"],
            capture_output=True,
            check=False,
        )
    else:
        try:
            os.kill(pid, 15)
        except OSError:
            pass


def terminate_old_process(pid: int, grace_s: float) -> None:
    """End the trusted old service PID without depending on WMI discovery."""
    deadline = time.monotonic() + max(0.0, grace_s)
    while pid_alive(pid) and time.monotonic() < deadline:
        time.sleep(0.1)
    if pid_alive(pid):
        terminate_process_tree(pid)


def remove_lock_if_safe(path: Path) -> None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        pid = int(data.get("pid", 0))
        if not pid_alive(pid):
            path.unlink(missing_ok=True)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


def service_pid_is_running(project_root: Path, pid: int) -> bool:
    """Avoid treating a recycled Windows PID as the old control service."""
    return pid > 0 and pid in matching_python_pids(project_root, "nero_control_server.py")


def main() -> int:
    parser = argparse.ArgumentParser(description="Restart the local NERO control service without robot commands.")
    parser.add_argument("--old-pid", type=int, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--service-script", type=Path, required=True)
    parser.add_argument("--service-python", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--wait-timeout", type=float, default=15.0)
    args = parser.parse_args()
    expected_python = args.project_root / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not expected_python.is_file():
        raise SystemExit(f"project control Python is unavailable: {expected_python}")
    expected_python = expected_python.resolve()
    if args.service_python.resolve() != expected_python:
        raise SystemExit(f"restart refuses non-project control Python: {args.service_python}")
    args.service_python = expected_python

    # The watchdog returns its HTTP response before killing the old process.
    # Give that response a short grace period, then make reset unconditional:
    # a stuck SDK call must never veto an operator-requested hard reset.
    # old_pid comes from either the live service itself or its PID lock as
    # observed by the localhost-only watchdog. Do not gate termination on
    # Win32_Process/WMI enumeration: that query can time out while the SDK is
    # wedged, which used to leave the transport thread and USB-CAN handle
    # alive. Killing the complete tree is the reset boundary for native calls.
    terminate_old_process(args.old_pid, min(2.0, max(0.5, args.wait_timeout)))

    # Remove every project-scoped duplicate service. /T also removes Pink or
    # other subprocesses still parented by that service.
    for pid in matching_python_pids(args.project_root, "nero_control_server.py"):
        if pid != os.getpid():
            terminate_process_tree(pid)
    for pid in matching_spawn_workers(args.project_root):
        if pid != os.getpid():
            terminate_process_tree(pid)
    # A solver can be orphaned if its parent was previously killed without /T.
    for pid in matching_python_pids(args.project_root, "osc_kinematics_server.py"):
        if pid != os.getpid():
            terminate_process_tree(pid)

    cleanup_deadline = time.monotonic() + 5.0
    while time.monotonic() < cleanup_deadline:
        remaining = matching_python_pids(args.project_root, "nero_control_server.py")
        if not remaining:
            break
        for pid in remaining:
            if pid != os.getpid():
                terminate_process_tree(pid)
        time.sleep(0.1)
    if matching_python_pids(args.project_root, "nero_control_server.py"):
        return 2

    # No old service remains, so its PID lock is stale by definition.
    try:
        (args.project_root / "runtime" / "nero_control_service.lock").unlink(missing_ok=True)
    except OSError:
        pass

    # Let Windows finish releasing the old USB/CAN and TCP handles before the
    # sole fresh service acquires them.
    time.sleep(0.3)

    command = [
        str(args.service_python),
        str(args.service_script),
        "--config",
        str(args.config),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--no-browser",
    ]
    if os.name == "nt":
        quote_ps = lambda value: "'" + str(value).replace("'", "''") + "'"
        arguments = ",".join(quote_ps(value) for value in command[1:])
        powershell = (
            f"Start-Process -FilePath {quote_ps(command[0])} "
            f"-ArgumentList @({arguments}) -WorkingDirectory {quote_ps(args.project_root)} "
            "-WindowStyle Hidden"
        )
        subprocess.Popen(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-WindowStyle", "Hidden", "-Command", powershell],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    else:
        subprocess.Popen(
            command,
            cwd=str(args.project_root),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            start_new_session=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
