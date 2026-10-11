from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from motion.osc_tuning import FIELDS, OscTuning

ROOT = Path(__file__).resolve().parents[1]


class OscTuningTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / "osc.json"
        self.config = json.loads((ROOT / "config/osc.json").read_text(encoding="utf-8-sig"))
        self.path.write_text(json.dumps(self.config), encoding="utf-8")
        self.tuning = OscTuning(self.config, self.path)

    def test_runtime_only_and_saved_restart_values_are_distinct(self):
        before = self.path.read_bytes()
        values = self.tuning.public()["running"]
        values["frame_gain"] = .6
        values["arrival_dwell_s"] = .5
        result = self.tuning.apply(values)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(result["startup"]["frame_gain"], .8)
        self.assertEqual(result["running"]["frame_gain"], .6)
        self.tuning.apply(values, save=True)
        saved_config = json.loads(self.path.read_text(encoding="utf-8"))
        restarted = OscTuning(saved_config, self.path)
        self.assertEqual(restarted.public()["running"], values)
        self.assertEqual(saved_config["limits"], self.config["limits"])
        self.assertEqual(saved_config["tcp"], self.config["tcp"])

    def test_invalid_group_never_partially_applies_or_saves(self):
        before = self.tuning.public()
        disk = self.path.read_bytes()
        for key in FIELDS:
            for bad in (float("nan"), float("inf"), True, "0.5", -1):
                values = dict(before["running"])
                values[key] = bad
                with self.assertRaises(ValueError):
                    self.tuning.apply(values, save=True)
                self.assertEqual(self.tuning.public(), before)
                self.assertEqual(self.path.read_bytes(), disk)
        for values in ({}, {**before["running"], "control_hz": 1}):
            with self.assertRaises(ValueError):
                self.tuning.apply(values)

    def test_disk_error_and_stale_revision_do_not_publish(self):
        values = self.tuning.public()["running"]
        values["frame_gain"] = .5
        with patch.object(Path, "replace", side_effect=OSError("disk unavailable")):
            with self.assertRaises(OSError):
                self.tuning.apply(values, save=True)
        self.assertEqual(self.tuning.snapshot()[0], 0)
        self.tuning.apply(values, expected_revision=0)
        with self.assertRaises(ValueError):
            self.tuning.apply(values, expected_revision=0)
        self.assertEqual(self.tuning.snapshot()[0], 1)

    def test_snapshot_immutable_and_reads_do_not_wait_for_disk(self):
        old = self.tuning.snapshot()
        values = dict(old[1])
        values.update(frame_gain=.5, frame_position_cost=6.)
        entered, release = threading.Event(), threading.Event()
        real_replace = Path.replace
        def slow_replace(path, target):
            entered.set()
            release.wait(3)
            return real_replace(path, target)
        with patch.object(Path, "replace", slow_replace):
            worker = threading.Thread(target=lambda: self.tuning.apply(values, save=True))
            worker.start()
            try:
                self.assertTrue(entered.wait(1))
                self.assertEqual(self.tuning.snapshot(), old)
            finally:
                release.set()
                worker.join(3)
        self.assertFalse(worker.is_alive())
        self.assertEqual(dict(self.tuning.snapshot()[1]), values)
        self.assertEqual(old[1]["frame_gain"], .8)
        with self.assertRaises(TypeError):
            old[1]["frame_gain"] = .5
