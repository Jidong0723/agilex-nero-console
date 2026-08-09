from __future__ import annotations

import json
import os
import time
import sys
import ctypes
from ctypes import wintypes
from pathlib import Path


class InstanceLock:
    """Small Windows-friendly PID lock for local single-owner services."""

    def __init__(self, path: Path, role: str) -> None:
        self.path = Path(path)
        self.role = role
        self.owned = False

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(2):
            try:
                fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    json.dump({"pid": os.getpid(), "role": self.role, "started_at": time.time()}, handle)
                self.owned = True
                return
            except FileExistsError:
                try:
                    data = json.loads(self.path.read_text(encoding="utf-8"))
                    pid = int(data.get("pid", 0))
                    started_at = float(data.get("started_at", 0.0))
                    if not self._pid_matches_lock(pid, started_at):
                        raise ProcessLookupError(pid)
                except (FileNotFoundError, ValueError, OSError, json.JSONDecodeError):
                    try:
                        self.path.unlink()
                    except FileNotFoundError:
                        pass
                    continue
                raise RuntimeError(f"{self.role} already running (pid={pid})")
        raise RuntimeError(f"could not acquire {self.role} instance lock")

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        if pid <= 0:
            return False
        if sys.platform == "win32":
            handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
            if not handle:
                return False
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    @classmethod
    def _pid_matches_lock(cls, pid: int, lock_started_at: float) -> bool:
        """Return whether an existing lock still belongs to its creator.

        Windows can reuse a PID after a crashed process.  A PID-only lock then
        turns a stale file into a permanent startup failure whenever the PID is
        later assigned to an unrelated process.  Compare the Windows process
        creation time to the lock timestamp when it is available.
        """
        if not cls._pid_alive(pid):
            return False
        process_started_at = cls._process_started_at(pid)
        if process_started_at is None or lock_started_at <= 0:
            return True
        # The lock is written immediately after process start.  A small clock
        # tolerance avoids treating a valid lock as stale on coarse clocks.
        return process_started_at <= lock_started_at + 5.0

    @staticmethod
    def _process_started_at(pid: int) -> float | None:
        if sys.platform != "win32":
            return None
        handle = ctypes.windll.kernel32.OpenProcess(0x0400, False, pid)
        if not handle:
            return None
        try:
            creation = wintypes.FILETIME()
            exit_time = wintypes.FILETIME()
            kernel = wintypes.FILETIME()
            user = wintypes.FILETIME()
            if not ctypes.windll.kernel32.GetProcessTimes(handle, ctypes.byref(creation), ctypes.byref(exit_time), ctypes.byref(kernel), ctypes.byref(user)):
                return None
            ticks = (int(creation.dwHighDateTime) << 32) | int(creation.dwLowDateTime)
            return (ticks / 10_000_000) - 11_644_473_600
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)

    def release(self) -> None:
        if not self.owned:
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if int(data.get("pid", -1)) == os.getpid():
                self.path.unlink(missing_ok=True)
        except (FileNotFoundError, ValueError, OSError, json.JSONDecodeError):
            pass
        self.owned = False
