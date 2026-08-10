"""Independent localhost reset agent for the NERO control service.

The watchdog never imports pyAgxArm and never opens CAN.  Its only purpose is
to accept a local reset request when the main HTTP service is unresponsive.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def control_python(project_root: Path) -> Path:
    executable = project_root / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not executable.is_file():
        raise RuntimeError(f"control Python is unavailable: {executable}")
    return executable.resolve()


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


def ps_quote(value: object) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def launch_detached(command: list[str], cwd: Path) -> None:
    if os.name == "nt":
        arguments = ",".join(ps_quote(value) for value in command[1:])
        powershell = (
            f"Start-Process -FilePath {ps_quote(command[0])} "
            f"-ArgumentList @({arguments}) -WorkingDirectory {ps_quote(cwd)} "
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
        return
    subprocess.Popen(
        command,
        cwd=str(cwd),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
        start_new_session=True,
    )


def matching_python_pids(project_root: Path, script_name: str) -> list[int]:
    if os.name != "nt":
        return []
    command = (
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.Name -like 'python*' } | "
        "Select-Object ProcessId,CommandLine | ConvertTo-Json"
    )
    try:
        output = subprocess.check_output(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=3.0,
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


def lock_pid(path: Path) -> int | None:
    try:
        return int(json.loads(path.read_text(encoding="utf-8")).get("pid", 0))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def terminate_process_tree(pid: int) -> None:
    if pid <= 0 or pid == os.getpid():
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill.exe", "/PID", str(pid), "/T", "/F"],
            capture_output=True,
            check=False,
        )
    elif pid_alive(pid):
        try:
            os.kill(pid, 15)
        except OSError:
            pass


def prune_duplicate_processes(project_root: Path, service_lock: Path, watchdog_lock: Path) -> None:
    service_pid = lock_pid(service_lock)
    for pid in matching_python_pids(project_root, "nero_control_server.py"):
        if service_pid is not None and pid != service_pid:
            terminate_process_tree(pid)
    watchdog_pid = lock_pid(watchdog_lock)
    for pid in matching_python_pids(project_root, "nero_control_watchdog.py"):
        if watchdog_pid is not None and pid != watchdog_pid:
            terminate_process_tree(pid)


class ResetHandler(BaseHTTPRequestHandler):
    project_root: Path
    service_script: Path
    config_path: Path
    service_python: Path
    lock_path: Path
    reset_lock = threading.Lock()
    reset_pending = False

    def log_message(self, format: str, *args: object) -> None:
        return None

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "http://127.0.0.1:8765")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        if self.path != "/api/health":
            self._send(HTTPStatus.NOT_FOUND, {"ok": False, "error": "unknown endpoint"})
            return
        service_pids = matching_python_pids(self.project_root, "nero_control_server.py")
        self._send(HTTPStatus.OK, {
            "ok": True,
            "data": {
                "role": "nero-control-reset-watchdog",
                "pid": os.getpid(),
                "reset_pending": self.reset_pending,
                "control_service_pids": service_pids,
                "robot_commands_sent": False,
            },
        })

    def do_POST(self) -> None:
        if self.path != "/api/reset":
            self._send(HTTPStatus.NOT_FOUND, {"ok": False, "error": "unknown endpoint"})
            return
        with self.reset_lock:
            if self.reset_pending:
                self._send(HTTPStatus.ACCEPTED, {"ok": True, "data": {"status": "already_scheduled", "robot_commands_sent": False}})
                return
            self.reset_pending = True
        candidate_pids = matching_python_pids(self.project_root, "nero_control_server.py")
        try:
            payload = json.loads(self.lock_path.read_text(encoding="utf-8"))
            old_pid = int(payload.get("pid", 0))
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            old_pid = candidate_pids[0] if candidate_pids else 0
        helper = self.project_root / "scripts" / "nero_control_service_restart.py"
        command = [
            str(self.service_python),
            str(helper),
            "--old-pid",
            str(old_pid),
            "--project-root",
            str(self.project_root),
            "--service-script",
            str(self.service_script),
            "--service-python",
            str(self.service_python),
            "--config",
            str(self.config_path),
            "--host",
            "127.0.0.1",
            "--port",
            "8765",
        ]
        try:
            launch_detached(command, self.project_root)
            self._send(HTTPStatus.ACCEPTED, {"ok": True, "data": {"status": "restart_scheduled", "robot_commands_sent": False, "retry_after_s": 2}})
        except Exception as exc:
            self._send(HTTPStatus.INTERNAL_SERVER_ERROR, {"ok": False, "error": f"could not start reset helper: {exc}"})
            self.reset_pending = False
            return

        def kill_old_service() -> None:
            time.sleep(0.5)
            targets = set(candidate_pids)
            if old_pid > 0:
                targets.add(old_pid)
            for pid in targets:
                if pid == os.getpid():
                    continue
                if os.name == "nt":
                    terminate_process_tree(pid)
                elif pid_alive(pid):
                    os.kill(pid, 15)

        threading.Thread(target=kill_old_service, name="control-watchdog-reset", daemon=True).start()

        def clear_pending() -> None:
            # This watchdog outlives the control service.  Keep a short
            # debounce window for the current restart, then allow a later
            # operator reset if a different fault occurs.
            time.sleep(3.0)
            with self.reset_lock:
                self.reset_pending = False

        threading.Thread(target=clear_pending, name="control-watchdog-reset-clear", daemon=True).start()

    def _send(self, status: HTTPStatus, payload: dict[str, object]) -> None:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "http://127.0.0.1:8765")
        self.end_headers()
        self.wfile.write(encoded)


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--service-script", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--service-python", type=Path, default=None)
    parser.add_argument("--port", type=int, default=8767)
    args = parser.parse_args()
    if args.port != 8767:
        raise SystemExit("watchdog must use port 8767")
    if str(args.project_root) not in sys.path:
        sys.path.insert(0, str(args.project_root))
    service_python = control_python(args.project_root)
    if args.service_python is not None and args.service_python.resolve() != service_python:
        raise SystemExit(f"watchdog refuses non-project control Python: {args.service_python}")
    from supervisor.instance_lock import InstanceLock

    lock = InstanceLock(args.project_root / "runtime" / "nero_control_watchdog.lock", "nero-control-watchdog")
    try:
        lock.acquire()
        server = ReusableThreadingHTTPServer(("127.0.0.1", args.port), ResetHandler)
    except (OSError, RuntimeError) as exc:
        lock.release()
        print(f"NERO reset watchdog already running or unavailable: {exc}")
        return 0
    ResetHandler.project_root = args.project_root
    ResetHandler.service_script = args.service_script
    ResetHandler.config_path = args.config
    ResetHandler.service_python = service_python
    ResetHandler.lock_path = args.project_root / "runtime" / "nero_control_service.lock"
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        server.server_close()
        lock.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
