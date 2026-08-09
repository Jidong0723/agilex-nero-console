from __future__ import annotations

import os
import unittest
from unittest import mock

from scripts import nero_control_service_restart as restart
from scripts import nero_control_watchdog as watchdog


@unittest.skipUnless(os.name == "nt", "Windows process-tree reset contract")
class HardResetTests(unittest.TestCase):
    def test_restart_helper_force_kills_complete_process_tree(self) -> None:
        with mock.patch.object(restart.subprocess, "run") as run:
            restart.terminate_process_tree(43210)
        command = run.call_args.args[0]
        self.assertEqual(command[:3], ["taskkill.exe", "/PID", "43210"])
        self.assertIn("/T", command)
        self.assertIn("/F", command)

    def test_old_service_termination_does_not_depend_on_wmi(self) -> None:
        with (
            mock.patch.object(restart, "pid_alive", return_value=True),
            mock.patch.object(restart, "service_pid_is_running", side_effect=AssertionError("WMI must not gate reset")),
            mock.patch.object(restart.time, "monotonic", side_effect=[0.0, 2.0]),
            mock.patch.object(restart, "terminate_process_tree") as terminate,
        ):
            restart.terminate_old_process(43212, 1.0)
        terminate.assert_called_once_with(43212)

    def test_watchdog_force_kills_complete_process_tree(self) -> None:
        with mock.patch.object(watchdog.subprocess, "run") as run:
            watchdog.terminate_process_tree(43211)
        command = run.call_args.args[0]
        self.assertEqual(command[:3], ["taskkill.exe", "/PID", "43211"])
        self.assertIn("/T", command)
        self.assertIn("/F", command)


if __name__ == "__main__":
    unittest.main()
