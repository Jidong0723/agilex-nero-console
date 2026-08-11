from __future__ import annotations

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    result: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.update(item.name for item in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            result.add(node.module)
    return result


class OscBoundaryTests(unittest.TestCase):
    def test_osc_core_does_not_import_input_adapters_or_cameras(self) -> None:
        forbidden = {"supervisor.pi05_adapter", "supervisor.pico_adapter", "supervisor.camera_resource"}
        for path in (ROOT / "supervisor" / "control.py", ROOT / "motion" / "teleop.py"):
            self.assertFalse(_imports(path) & forbidden)

    def test_osc_servo_has_no_legacy_relative_input_symbols(self) -> None:
        source = (ROOT / "motion" / "teleop.py").read_text(encoding="utf-8")
        for symbol in ("submit_intent", "submit_pico_intent", "clutch_active", "relative_pose", "tcp_anchor", "input_source"):
            self.assertNotIn(symbol, source)

    def test_legacy_teleop_routes_are_absent(self) -> None:
        source = (ROOT / "scripts" / "nero_control_server.py").read_text(encoding="utf-8")
        self.assertNotIn('"/api/teleop/', source)
        self.assertIn('"/api/osc/kinematics"', source)
