from __future__ import annotations

import json
import unittest
from pathlib import Path

from nero_console.runtime import control_python, doctor_report, kinematics_python, project_root


class RuntimeLayoutTests(unittest.TestCase):
    def test_project_runtime_paths_are_local_and_explicit(self) -> None:
        root = project_root()
        self.assertEqual(root, Path(__file__).resolve().parents[1])
        self.assertEqual(control_python(root), (root / ".venv" / "Scripts" / "python.exe").resolve())
        self.assertEqual(kinematics_python(root), (root / ".conda" / "nero-kinematics" / "python.exe").resolve())

    def test_doctor_report_is_json_and_does_not_require_can_hardware(self) -> None:
        report = doctor_report()
        self.assertEqual(report["project_root"], str(project_root()))
        self.assertIn("control_modules", report)
        json.dumps(report)

