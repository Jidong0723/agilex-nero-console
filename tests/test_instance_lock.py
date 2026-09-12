from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from supervisor.instance_lock import InstanceLock


class InstanceLockTests(unittest.TestCase):
    def test_reclaims_lock_when_pid_has_been_reused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "service.lock"
            path.write_text(json.dumps({"pid": 24680, "started_at": 100.0}), encoding="utf-8")
            lock = InstanceLock(path, "test-service")
            with patch.object(InstanceLock, "_pid_alive", return_value=True), patch.object(
                InstanceLock, "_process_started_at", return_value=200.0
            ):
                lock.acquire()
            self.assertTrue(lock.owned)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["pid"], os.getpid())
            lock.release()

    def test_keeps_lock_for_original_live_process(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "service.lock"
            path.write_text(json.dumps({"pid": 24680, "started_at": 200.0}), encoding="utf-8")
            lock = InstanceLock(path, "test-service")
            with patch.object(InstanceLock, "_pid_alive", return_value=True), patch.object(
                InstanceLock, "_process_started_at", return_value=199.0
            ):
                with self.assertRaisesRegex(RuntimeError, "already running"):
                    lock.acquire()


if __name__ == "__main__":
    unittest.main()
