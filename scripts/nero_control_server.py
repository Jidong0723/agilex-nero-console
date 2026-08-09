from __future__ import annotations

import argparse
import json
import mimetypes
import multiprocessing
import os
import secrets
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from supervisor.instance_lock import InstanceLock  # noqa: E402

if TYPE_CHECKING:
    from supervisor.control import OperationalSpaceController


WEB_ROOT = PROJECT_ROOT / "web" / "console"
_RESET_LOCK = threading.Lock()
_RESET_PENDING = False


class ControlServiceUnavailable(RuntimeError):
    """The web console is up, but its robot-control backend is not ready."""


class LeaseError(RuntimeError):
    """Local representation of a lease error returned by the backend process."""


def _backend_worker_main(connection: Any, config: str) -> None:
    """Own every SDK/CAN object in a disposable child process."""
    osc = None
    try:
        from supervisor.control import OperationalSpaceController

        osc = OperationalSpaceController(Path(config))
        initial = osc.start()
        connection.send({"kind": "ready", "initial": initial})
        while True:
            request = connection.recv()
            if request.get("method") == "__close__":
                break
            try:
                method = getattr(osc, str(request["method"]))
                result = method(*request.get("args", ()), **request.get("kwargs", {}))
                connection.send({"kind": "result", "id": request["id"], "result": result})
            except BaseException as exc:
                connection.send({
                    "kind": "error",
                    "id": request.get("id"),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                })
    except BaseException as exc:
        try:
            connection.send({"kind": "startup_error", "error_type": type(exc).__name__, "error": str(exc)})
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        if osc is not None:
            try:
                osc.close()
            except BaseException:
                pass
        connection.close()


class BackendProcessProxy:
    """RPC facade for the one process permitted to own the hardware transport."""

    def __init__(self, process: Any, connection: Any, call_timeout_s: float = 30.0) -> None:
        self.process = process
        self.connection = connection
        self.call_timeout_s = call_timeout_s
        self._lock = threading.Lock()
        self._request_id = 0

    def __getattr__(self, method: str) -> Any:
        def invoke(*args: Any, **kwargs: Any) -> Any:
            return self.call(method, *args, **kwargs)
        return invoke

    def call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            if not self.process.is_alive():
                raise ControlServiceUnavailable(
                    f"hardware backend process exited (code={self.process.exitcode})"
                )
            self._request_id += 1
            request_id = self._request_id
            try:
                self.connection.send({
                    "id": request_id,
                    "method": method,
                    "args": args,
                    "kwargs": kwargs,
                })
                if not self.connection.poll(self.call_timeout_s):
                    raise ControlServiceUnavailable(
                        f"hardware backend call {method} exceeded {self.call_timeout_s:.0f}s; use Reset to release it"
                    )
                response = self.connection.recv()
            except (BrokenPipeError, EOFError, OSError) as exc:
                raise ControlServiceUnavailable(
                    f"hardware backend process unavailable (code={self.process.exitcode})"
                ) from exc
            if response.get("kind") == "result" and response.get("id") == request_id:
                return response.get("result")
            error_type = str(response.get("error_type", "RuntimeError"))
            message = str(response.get("error", "backend request failed"))
            if error_type == "PermissionError":
                raise PermissionError(message)
            if error_type == "LeaseError":
                raise LeaseError(message)
            raise RuntimeError(f"{error_type}: {message}")

    def terminate(self) -> None:
        if not self.process.is_alive():
            return
        try:
            self.process.kill()
        except (AttributeError, OSError):
            self.process.terminate()
        self.process.join(timeout=2.0)


class ServiceRuntime:
    """Thread-safe bootstrap state kept independent from USB-CAN startup.

    The HTTP server owns this small object from the moment the port is bound.
    Importing the control stack, constructing the SDK, and connecting hardware
    all happen on a daemon worker, so none of them can delay the web page.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._broker: OperationalSpaceController | None = None
        self._phase = "starting"
        self._error: str | None = None
        self._initial: dict[str, Any] | None = None
        self._started_monotonic = time.monotonic()
        self._backend_process: Any | None = None

    def initialize(self, config: Path, broker_factory: Any | None = None) -> None:
        try:
            if broker_factory is not None:
                # Unit tests can inject a harmless in-process fake. Production
                # always uses the crash boundary below.
                broker = broker_factory(config)
                initial = broker.start()
            else:
                context = multiprocessing.get_context("spawn")
                parent_connection, child_connection = context.Pipe()
                process = context.Process(
                    target=_backend_worker_main,
                    args=(child_connection, str(Path(config).resolve())),
                    name="nero-hardware-backend",
                    daemon=False,
                )
                process.start()
                child_connection.close()
                self._backend_process = process
                # The worker's broker has its own bounded startup timeout. A
                # native SDK crash closes the pipe without taking HTTP down.
                if not parent_connection.poll(25.0):
                    raise RuntimeError("hardware backend did not finish startup within 25s")
                try:
                    response = parent_connection.recv()
                except EOFError as exc:
                    process.join(timeout=0.5)
                    raise RuntimeError(
                        f"hardware backend exited during startup (code={process.exitcode})"
                    ) from exc
                if response.get("kind") != "ready":
                    raise RuntimeError(
                        f"{response.get('error_type', 'RuntimeError')}: "
                        f"{response.get('error', 'hardware backend startup failed')}"
                    )
                initial = response.get("initial")
                broker = BackendProcessProxy(process, parent_connection)
        except BaseException as exc:
            # SystemExit raised by a dependency must not take down the page.
            with self._lock:
                self._phase = "error"
                self._error = f"{type(exc).__name__}: {exc}"
            print(json.dumps({"control_backend_ready": False, "error": self._error}, ensure_ascii=False), flush=True)
            return
        with self._lock:
            self._broker = broker
            self._initial = dict(initial) if isinstance(initial, dict) else {"result": initial}
            self._phase = "ready"
            self._error = None
        print(json.dumps({"control_backend_ready": True, "initial": initial}, ensure_ascii=False, default=str), flush=True)

    def require_broker(self) -> OperationalSpaceController:
        with self._lock:
            if self._broker is not None and self._phase == "ready":
                return self._broker
            detail = self._error or "control backend is still initializing"
            raise ControlServiceUnavailable(detail)

    def health(self) -> dict[str, Any]:
        with self._lock:
            broker = self._broker
            result = {
                "role": "nero-control-service",
                "pid": os.getpid(),
                "http_ready": True,
                "control_backend_phase": self._phase,
                "control_backend_ready": self._phase == "ready" and broker is not None,
                "control_backend_error": self._error,
                "bootstrap_elapsed_s": max(0.0, time.monotonic() - self._started_monotonic),
                "initial": dict(self._initial) if self._initial is not None else None,
            }
        if broker is not None:
            try:
                result.update(broker.health())
            except Exception as exc:
                result["broker_health_error"] = f"{type(exc).__name__}: {exc}"
        return result

    def close(self) -> None:
        with self._lock:
            broker = self._broker
            self._broker = None
            self._phase = "stopped"
        if isinstance(broker, BackendProcessProxy):
            broker.terminate()
        elif broker is not None:
            broker.close()
        elif self._backend_process is not None:
            BackendProcessProxy(self._backend_process, None).terminate()


def control_page_is_healthy(url: str) -> bool:
    """Only open an existing console after proving that it answers locally."""
    try:
        with urlopen(f"{url.rstrip('/')}/api/health", timeout=0.5) as response:
            return response.status == HTTPStatus.OK
    except (OSError, URLError):
        return False


def schedule_service_reset() -> dict[str, Any]:
    """Schedule an independent process restart without touching the robot.

    The HTTP handler returns before the current process exits.  The helper
    waits for this PID to disappear, then starts exactly one replacement
    service.  This remains usable even when an SDK call is stuck in another
    thread.
    """
    global _RESET_PENDING
    with _RESET_LOCK:
        if _RESET_PENDING:
            return {"status": "already_scheduled", "robot_commands_sent": False}
        _RESET_PENDING = True
    command = [
        sys.executable,
        str(Path(__file__).resolve().with_name("nero_control_service_restart.py")),
        "--old-pid",
        str(os.getpid()),
        "--project-root",
        str(PROJECT_ROOT),
        "--service-script",
        str(Path(__file__).resolve()),
        "--service-python",
        sys.executable,
        "--config",
        str(PROJECT_ROOT / "config" / "runtime.json"),
        "--host",
        "127.0.0.1",
        "--port",
        "8765",
    ]
    if os.name == "nt":
        # A direct detached child can still be collected with the service by
        # the Windows process job used by the desktop shell.  PowerShell's
        # Start-Process creates the independent process tree we need here.
        quote_ps = lambda value: "'" + str(value).replace("'", "''") + "'"
        arguments = ",".join(quote_ps(value) for value in command[1:])
        powershell = (
            f"Start-Process -FilePath {quote_ps(command[0])} "
            f"-ArgumentList @({arguments}) -WorkingDirectory {quote_ps(PROJECT_ROOT)} "
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
            cwd=str(PROJECT_ROOT),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            start_new_session=True,
        )

    def terminate_after_response() -> None:
        time.sleep(0.5)
        os._exit(0)

    threading.Thread(target=terminate_after_response, name="control-service-reset", daemon=True).start()
    return {"status": "restart_scheduled", "robot_commands_sent": False, "retry_after_s": 2}


def ensure_reset_watchdog() -> None:
    """Keep a separate localhost reset agent available for this service."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.2)
        if probe.connect_ex(("127.0.0.1", 8767)) == 0:
            return
    command = [
        sys.executable,
        str(Path(__file__).resolve().with_name("nero_control_watchdog.py")),
        "--project-root",
        str(PROJECT_ROOT),
        "--service-script",
        str(Path(__file__).resolve()),
        "--config",
        str(PROJECT_ROOT / "config" / "runtime.json"),
        "--port",
        "8767",
    ]
    if os.name == "nt":
        quote_ps = lambda value: "'" + str(value).replace("'", "''") + "'"
        arguments = ",".join(quote_ps(value) for value in command[1:])
        powershell = (
            f"Start-Process -FilePath {quote_ps(command[0])} "
            f"-ArgumentList @({arguments}) -WorkingDirectory {quote_ps(PROJECT_ROOT)} "
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
            cwd=str(PROJECT_ROOT),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            start_new_session=True,
        )


def _matching_local_processes(script_name: str) -> list[int]:
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
    root_text = str(PROJECT_ROOT).lower()
    script_text = script_name.lower()
    pids: list[int] = []
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
            pids.append(int(row.get("ProcessId")))
        except (TypeError, ValueError):
            pass
    return pids


def _terminate_pid(pid: int) -> None:
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


def _lock_pid(path: Path) -> int | None:
    try:
        return int(json.loads(path.read_text(encoding="utf-8")).get("pid", 0))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def prune_duplicate_local_processes() -> None:
    service_lock_pid = _lock_pid(PROJECT_ROOT / "runtime" / "nero_control_service.lock")
    if service_lock_pid != os.getpid():
        return
    for pid in _matching_local_processes("nero_control_server.py"):
        if pid != os.getpid():
            _terminate_pid(pid)
    watchdog_lock_pid = _lock_pid(PROJECT_ROOT / "runtime" / "nero_control_watchdog.lock")
    for pid in _matching_local_processes("nero_control_watchdog.py"):
        if watchdog_lock_pid is not None and pid != watchdog_lock_pid:
            _terminate_pid(pid)


class ExclusiveThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True

    def server_bind(self) -> None:
        super().server_bind()


class PicoGateway:
    """Small, paired LAN ingress which never exposes the HTTP control API."""

    def __init__(self, runtime: ServiceRuntime, config: dict[str, Any]) -> None:
        self.runtime, self.config = runtime, dict(config)
        self._lock = threading.RLock()
        self._server: Any | None = None
        self._thread: threading.Thread | None = None
        self._pair: dict[str, Any] | None = None
        self._connection_active = False
        self._last_input_monotonic = 0.0
        self.error: str | None = None

    def start(self) -> None:
        if not bool(self.config.get("enabled", True)):
            return
        try:
            from websockets.sync.server import serve
        except ImportError as exc:
            self.error = "websockets dependency is not installed"
            print(json.dumps({"pico_gateway_ready": False, "error": self.error}), flush=True)
            return
        host, port = str(self.config.get("host", "0.0.0.0")), int(self.config.get("port", 8768))
        try:
            self._server = serve(
                self._handle_connection, host, port,
                max_size=int(self.config.get("max_message_bytes", 4096)),
                ping_interval=10, ping_timeout=5,
            )
        except OSError as exc:
            self.error = f"could not bind {host}:{port}: {exc}"
            print(json.dumps({"pico_gateway_ready": False, "error": self.error}), flush=True)
            return
        self._thread = threading.Thread(target=self._server.serve_forever, name="nero-pico-gateway", daemon=True)
        self._thread.start()
        print(json.dumps({"pico_gateway_ready": True, "host": host, "port": port}), flush=True)

    def close(self) -> None:
        server = self._server
        self._server = None
        if server is not None:
            server.shutdown()
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout=1.0)

    def create_pairing(self, session_id: str) -> dict[str, Any]:
        if self._server is None:
            raise RuntimeError(self.error or "PICO WebSocket gateway is unavailable")
        # The operator requested a fixed code so the headset APK does not
        # need a per-session code-entry workflow. Session binding, expiry,
        # single-connection pairing, and invalidation on stop remain active.
        code = "1111"
        expires = time.monotonic() + float(self.config.get("pair_ttl_s", 120.0))
        with self._lock:
            self._pair = {"session_id": session_id, "code": code, "expires_monotonic": expires, "paired": False}
            self._connection_active = False
            self._last_input_monotonic = 0.0
        return self.status()

    def invalidate(self) -> None:
        with self._lock:
            self._pair = None
            self._connection_active = False

    def status(self) -> dict[str, Any]:
        with self._lock:
            pair = dict(self._pair) if self._pair else None
            active = bool(pair and pair.get("paired"))
            host = str(self.config.get("host", "0.0.0.0"))
            advertised_host = str(self.config.get("advertise_host", "")).strip()
            if not advertised_host and host == "0.0.0.0":
                try:
                    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    probe.connect(("192.0.2.1", 9))
                    advertised_host = str(probe.getsockname()[0])
                    probe.close()
                except OSError:
                    advertised_host = "<PC-LAN-IP>"
            if not advertised_host:
                advertised_host = host
            return {
                "enabled": bool(self.config.get("enabled", True)), "ready": self._server is not None,
                "host": host, "port": int(self.config.get("port", 8768)),
                "ws_url": f"ws://{advertised_host}:{int(self.config.get('port', 8768))}",
                "session_id": pair.get("session_id") if pair else None,
                "pair_code": pair.get("code") if pair and not pair.get("paired") else None,
                "paired": active, "connection_active": self._connection_active,
                "last_input_age_s": None if not self._last_input_monotonic else max(0.0, time.monotonic() - self._last_input_monotonic),
                "error": self.error,
            }

    def _send(self, connection: Any, payload: dict[str, Any]) -> None:
        connection.send(json.dumps(payload, ensure_ascii=False))

    def _handle_connection(self, connection: Any) -> None:
        paired_session: str | None = None
        last_sequence = 0
        last_anchor = 0
        try:
            raw = connection.recv(timeout=10)
            message = json.loads(raw)
            if not isinstance(message, dict) or message.get("type") != "pair":
                raise PermissionError("first WebSocket message must be pair")
            with self._lock:
                pair = dict(self._pair) if self._pair else None
                supplied_session = str(message.get("session_id", "")).strip()
                valid = bool(
                    pair
                    and not pair.get("paired")
                    and time.monotonic() <= float(pair["expires_monotonic"])
                    and secrets.compare_digest(str(message.get("code", "")), str(pair["code"]))
                    and (not supplied_session or supplied_session == str(pair["session_id"]))
                )
                if not valid:
                    raise PermissionError("PICO pairing rejected: invalid or expired code")
                self._pair["paired"] = True
                self._connection_active = True
                paired_session = str(pair["session_id"])
            self._send(connection, {"ok": True, "type": "paired", "session_id": paired_session})
            while True:
                raw = connection.recv(timeout=float(self.config.get("message_timeout_s", 0.25)))
                if len(raw.encode("utf-8")) > int(self.config.get("max_message_bytes", 4096)):
                    raise ValueError("PICO message is too large")
                message = json.loads(raw)
                if not isinstance(message, dict) or str(message.get("session_id", paired_session)) != paired_session:
                    raise PermissionError("PICO session id mismatch")
                sequence = int(message.get("sequence", -1))
                if sequence <= last_sequence:
                    raise PermissionError("PICO sequence is not monotonic")
                kind = str(message.get("type", ""))
                event = {"clutch_begin": "clutch_begin", "pose": "pose", "clutch_release": "clutch_release"}.get(kind)
                if event is None:
                    if kind == "heartbeat":
                        self._send(connection, {"ok": True, "type": "heartbeat"})
                        continue
                    raise ValueError("unsupported PICO message type")
                payload: dict[str, Any] = {"event": event, "sequence": sequence, "anchor_id": int(message.get("anchor_id", -1)), "pose_scale": float(message.get("pose_scale", 1.0)), "tracking_valid": bool(message.get("tracking_valid", True))}
                if event == "pose":
                    payload["relative_pose"] = {"position_m": message.get("relative_position_m"), "orientation_xyzw": message.get("relative_orientation_xyzw")}
                result = self.runtime.require_broker().teleop_pico_intent(payload)
                last_sequence = sequence
                last_anchor = int(result.get("anchor_id", payload["anchor_id"]))
                with self._lock:
                    self._last_input_monotonic = time.monotonic()
                self._send(connection, {"ok": True, "type": "ack", "sequence": sequence, "anchor_id": last_anchor, "data": result})
        except TimeoutError:
            pass
        except Exception as exc:
            try:
                self._send(connection, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})
            except Exception:
                pass
        finally:
            with self._lock:
                self._connection_active = False
                if self._pair:
                    self._pair["paired"] = False
            if paired_session:
                try:
                    status = self.runtime.require_broker().teleop_status()
                    if status.get("clutch_active"):
                        self.runtime.require_broker().teleop_pico_intent({"event": "clutch_release", "sequence": max(last_sequence + 1, int(status.get("session", {}).get("sequence", 0)) + 1), "anchor_id": int(status.get("anchor_id", last_anchor)), "pose_scale": 1.0})
                except Exception:
                    pass


class ControlRequestHandler(BaseHTTPRequestHandler):
    runtime: ServiceRuntime
    pico_gateway: PicoGateway | None = None

    @property
    def broker(self) -> RobotControlBroker:
        return self.runtime.require_broker()

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[control-ui] {self.address_string()} {format % args}")

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/status":
                return self._json_ok(self.broker.status())
            if parsed.path.startswith("/api/actions/"):
                return self._json_ok(self.broker.action_status(parsed.path.rsplit("/", 1)[-1]))
            if parsed.path == "/api/health":
                return self._json_ok(self.runtime.health())
            if parsed.path == "/api/teleop/status":
                status = self.broker.teleop_status()
                if self.pico_gateway is not None:
                    status["pico_gateway"] = self.pico_gateway.status()
                return self._json_ok(status)
            if parsed.path == "/api/broker/status":
                return self._json_ok(self.broker.broker_status())
            if parsed.path == "/api/osc/state":
                return self._json_ok(self.broker.osc_state())
            if parsed.path == "/api/teleop/kinematics":
                return self._json_ok(self.broker.teleop_kinematics())
            return self._static(parsed.path)
        except ControlServiceUnavailable as exc:
            self._json_error(exc, HTTPStatus.SERVICE_UNAVAILABLE)
        except Exception as exc:
            self._json_error(exc)

    def do_POST(self) -> None:
        try:
            body = self._read_json()
            if self.path == "/api/control/reset":
                result = schedule_service_reset()
                self._json_accepted(result)
                return
            if self.path == "/api/actions":
                return self._json_accepted(self.broker.submit_action_job(body))
            if self.path == "/api/osc/session/start":
                return self._json_ok(self.broker.osc_start(
                    str(body.get("client_id", "anonymous")),
                    str(body.get("execution_mode", "shadow")),
                ))
            if self.path == "/api/osc/command":
                return self._json_ok(self.broker.osc_command(body))
            if self.path == "/api/osc/session/stop":
                return self._json_ok(self.broker.osc_stop(str(body.get("reason", "OSC session stopped"))))
            if self.path == "/api/osc/session/heartbeat":
                return self._json_ok(self.broker.osc_heartbeat(
                    str(body.get("client_id", "anonymous")),
                    str(body.get("session_id", "")),
                ))
            if self.path == "/api/teleop/session/start":
                result = self.broker.teleop_start(
                        body.get("mode"),
                        bool(body.get("confirm_hardware", False)),
                        str(body.get("client_id", "anonymous")),
                        body.get("execution_mode"),
                        body.get("input_source"),
                    )
                if result.get("session", {}).get("input_source") == "pico":
                    if self.pico_gateway is None:
                        raise RuntimeError("PICO WebSocket gateway is unavailable")
                    result["pico_gateway"] = self.pico_gateway.create_pairing(str(result["session"]["id"]))
                return self._json_ok(result)
            if self.path == "/api/teleop/intent":
                return self._json_ok(self.broker.teleop_intent(body))
            if self.path == "/api/teleop/session/stop":
                result = self.broker.teleop_stop(str(body.get("reason", "teleop stopped")))
                if self.pico_gateway is not None:
                    self.pico_gateway.invalidate()
                return self._json_ok(result)
            if self.path == "/api/teleop/session/heartbeat":
                return self._json_ok(self.broker.teleop_heartbeat(str(body.get("client_id", "anonymous")), str(body.get("session_id", ""))))
            if self.path == "/api/teleop/handoff-to-console":
                return self._json_ok(
                    self.broker.handoff_to_console(
                        str(body.get("reason", "operator returned to the control console"))
                    )
                )
            if self.path == "/api/teleop/session/recenter":
                return self._json_ok(self.broker.teleop_recenter())
            if self.path == "/api/lease/acquire":
                return self._json_ok(self.broker.acquire(str(body.get("owner", "client")), body.get("ttl_s")))
            if self.path == "/api/lease/renew":
                return self._json_ok(self.broker.renew(str(body.get("token", "")), body.get("ttl_s")))
            if self.path == "/api/lease/release":
                return self._json_ok(self.broker.release(str(body.get("token", ""))))
            if self.path == "/api/action":
                return self._json_ok(
                    self.broker.execute(
                        str(body.get("token", "")), dict(body.get("action", {})), body.get("timeout")
                    )
                )
            reason = str(body.get("reason", "operator request from local control page"))
            if self.path == "/api/safety/hold":
                return self._json_ok(self.broker.hold(reason))
            if self.path == "/api/safety/freedrive":
                return self._json_ok(self.broker.freedrive(
                    reason,
                    recover_emergency=bool(body.get("recover_emergency", False)),
                    preserve_gripper=bool(body.get("preserve_gripper", False)),
                ))
            if self.path == "/api/safety/emergency-damping":
                return self._json_ok(self.broker.emergency_damping(reason))
            if self.path == "/api/operator/gripper":
                width = body.get("width_m")
                return self._json_ok(self.broker.command_gripper(
                    mode=str(body.get("mode", "")),
                    width_m=float(width) if width is not None else None,
                    force_n=float(body.get("force_n", 1.0)),
                    preserve_on_freedrive=bool(body.get("preserve_on_freedrive", False)),
                    resume_teleop=bool(body.get("resume_teleop", False)),
                ))
            if self.path == "/api/operator/gripper/clear-hold":
                return self._json_ok(self.broker.clear_gripper_hold())
            if self.path == "/api/operator/gripper/zero-force":
                return self._json_ok(self.broker.release_gripper_zero_force())
            if self.path == "/api/operator/gripper/teaching-params":
                if "teaching_friction" in body:
                    return self._json_ok(self.broker.set_gripper_teaching_friction(
                        int(body["teaching_friction"])
                    ))
                return self._json_ok(self.broker.get_gripper_teaching_params())
            self._json_error(RuntimeError("unknown endpoint"), HTTPStatus.NOT_FOUND)
        except PermissionError as exc:
            self._json_error(exc, HTTPStatus.FORBIDDEN)
        except ControlServiceUnavailable as exc:
            self._json_error(exc, HTTPStatus.SERVICE_UNAVAILABLE)
        except Exception as exc:
            status = HTTPStatus.CONFLICT if type(exc).__name__ == "LeaseError" else HTTPStatus.INTERNAL_SERVER_ERROR
            self._json_error(exc, status)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length == 0:
            return {}
        value = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("JSON request body must be an object")
        return value

    def _static(self, request_path: str) -> None:
        relative = "index.html" if request_path in {"", "/"} else request_path.lstrip("/")
        target = (WEB_ROOT / relative).resolve()
        if WEB_ROOT.resolve() not in target.parents and target != WEB_ROOT.resolve():
            return self.send_error(HTTPStatus.FORBIDDEN)
        if not target.is_file():
            return self.send_error(HTTPStatus.NOT_FOUND)
        payload = target.read_bytes()
        self.send_response(HTTPStatus.OK)
        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        if content_type.startswith("text/") or content_type == "application/javascript":
            content_type += "; charset=utf-8"
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _send_bytes(self, payload: bytes, content_type: str) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _json_ok(self, data: Any) -> None:
        self._send_json(HTTPStatus.OK, {"ok": True, "data": data})

    def _json_accepted(self, data: Any) -> None:
        self._send_json(HTTPStatus.ACCEPTED, {"ok": True, "data": data})

    def _json_error(self, exc: Exception, status: HTTPStatus = HTTPStatus.INTERNAL_SERVER_ERROR) -> None:
        self._send_json(status, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)


def main() -> int:
    parser = argparse.ArgumentParser(description="Local NERO shared-control service and safety console.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "runtime.json")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost"}:
        raise SystemExit("For safety, the first release only listens on 127.0.0.1.")

    url = f"http://127.0.0.1:{args.port}/"
    instance_lock = InstanceLock(PROJECT_ROOT / "runtime" / "nero_control_service.lock", "nero-control-service")
    try:
        instance_lock.acquire()
        server = ExclusiveThreadingHTTPServer(("127.0.0.1", args.port), ControlRequestHandler)
    except (OSError, RuntimeError) as exc:
        instance_lock.release()
        if control_page_is_healthy(url):
            print("NERO control service is already running; opening the active control page.")
            if not args.no_browser:
                webbrowser.open(url)
            return 0
        print(f"NERO control service could not start and no page is listening on {url}: {exc}")
        print("Check the console output above; no browser page was opened because the service is unavailable.")
        return 1

    runtime = ServiceRuntime()
    ControlRequestHandler.runtime = runtime
    try:
        runtime_config = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        runtime_config = {}
        print(f"PICO gateway configuration unavailable: {exc}", flush=True)
    pico_gateway = PicoGateway(runtime, dict(runtime_config.get("pico_gateway", {})))
    ControlRequestHandler.pico_gateway = pico_gateway
    http_thread = threading.Thread(
        target=server.serve_forever,
        kwargs={"poll_interval": 0.1},
        name="nero-control-http",
        daemon=True,
    )
    try:
        # Start accepting static-page, health, and reset-related requests before
        # importing the SDK, starting the PICO gateway, or attempting any
        # hardware operation.  A gateway bind/import must never make the local
        # control page unavailable.
        http_thread.start()
        threading.Thread(target=pico_gateway.start, name="nero-pico-gateway-bootstrap", daemon=True).start()
        threading.Thread(target=ensure_reset_watchdog, name="nero-reset-watchdog-bootstrap", daemon=True).start()
        threading.Thread(
            target=runtime.initialize,
            args=(args.config,),
            name="nero-control-backend-bootstrap",
            daemon=True,
        ).start()
        print(f"NERO control page: {url}", flush=True)
        if not args.no_browser:
            webbrowser.open(url)
        while http_thread.is_alive():
            http_thread.join(timeout=0.5)
    except KeyboardInterrupt:
        print("Stopping control service. The arm state is not automatically reset.")
    finally:
        if http_thread.is_alive():
            server.shutdown()
        server.server_close()
        pico_gateway.close()
        runtime.close()
        instance_lock.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
