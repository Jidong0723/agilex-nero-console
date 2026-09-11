from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from supervisor.dataset_recorder import DatasetRecorder


class DatasetRecorderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.camera_payloads = {"external": b"jpeg-external", "wrist": b"jpeg-wrist"}
        self.cameras = SimpleNamespace(
            snapshot=lambda: {"ready": True, "config": {"external": {"index": 2}, "wrist": {"index": 3}},
                              "sources": {"external": {"available": True, "frame_available": True}, "wrist": {"available": True, "frame_available": True}}},
            frame_timestamp=lambda source, target: time.monotonic_ns(),
            frame_jpeg=lambda source, target: self.camera_payloads.get(source),
        )
        self.command = None
        def osc_state():
            now = time.monotonic_ns()
            command = {"final_joint_target_rad": None, "sent_monotonic_ns": None}
            if self.command is not None:
                command = {"final_joint_target_rad": list(self.command), "sent_monotonic_ns": now}
            return {"execution": {}, "transport": {"hardware_feedback": {"joint_angles_rad": [1, 2, 3, 4, 5, 6, 7], "received_monotonic_ns": now}}, "command": command, "diagnostics": {}, "gripper": {"width_m": 0.05}}
        self.osc = SimpleNamespace(state=osc_state)
        self.pico = SimpleNamespace(snapshot=lambda: {})

    def _record(self, source: str = "pico", duration: float = 0.16):
        folder = tempfile.TemporaryDirectory(); self.addCleanup(folder.cleanup)
        recorder = DatasetRecorder(self.osc, self.cameras, self.pico, Path(folder.name), sample_hz=20)
        recorder.start({"task": "test_task", "description": "test", "control_source": source})
        time.sleep(duration)
        return recorder.stop("completed")

    def test_idle_pico_writes_static_state_and_both_images(self) -> None:
        result = self._record()
        episode = Path(result["episode_dir"])
        rows = [json.loads(line) for line in (episode / "frames.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertGreater(result["frame_count"], 0)
        self.assertEqual(result["static_frame_count"], result["frame_count"])
        self.assertEqual(rows[0]["action"], None)
        self.assertTrue((episode / "images" / "front" / "000000.jpg").exists())
        self.assertTrue((episode / "images" / "wrist" / "000000.jpg").exists())
        self.assertGreater(result["camera_sources"]["external"]["captured_frames"], 0)
        self.assertGreater(result["camera_sources"]["wrist"]["captured_frames"], 0)

    def test_active_pico_command_is_recorded_as_action(self) -> None:
        self.command = [1, 2, 3, 4, 5, 6, 7]
        result = self._record()
        row = json.loads((Path(result["episode_dir"]) / "frames.jsonl").read_text(encoding="utf-8").splitlines()[0])
        self.assertGreater(result["action_frame_count"], 0)
        self.assertEqual(row["action"]["joint_target_rad"], self.command)

    def test_single_missing_camera_does_not_drop_other_image(self) -> None:
        self.camera_payloads["wrist"] = None
        result = self._record()
        episode = Path(result["episode_dir"])
        row = json.loads((episode / "frames.jsonl").read_text(encoding="utf-8").splitlines()[0])
        self.assertIn("front", row["observation"]["images"])
        self.assertIn("wrist", row["observation"]["missing_image_sources"])
        self.assertGreater(result["camera_sources"]["external"]["captured_frames"], 0)
        self.assertGreater(result["camera_sources"]["wrist"]["dropped_frames"], 0)

    def test_invalid_feedback_timestamp_is_rejected_with_reason(self) -> None:
        self.osc = SimpleNamespace(state=lambda: {"transport": {"hardware_feedback": {"joint_angles_rad": [1] * 7, "received_monotonic_ns": []}}, "command": {}, "execution": {}})
        result = self._record()
        self.assertEqual(result["frame_count"], 0)
        self.assertGreater(result["rejection_reasons"]["feedback_timestamp_invalid"], 0)

    def test_web_episode_records_state_without_cameras(self) -> None:
        result = self._record("web")
        row = json.loads((Path(result["episode_dir"]) / "frames.jsonl").read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(result["camera_sources"], {})
        self.assertEqual(row["observation"]["images"], {})
        self.assertFalse((Path(result["episode_dir"]) / "images").exists())

    def test_failed_episode_is_deleted(self) -> None:
        folder = tempfile.TemporaryDirectory(); self.addCleanup(folder.cleanup)
        recorder = DatasetRecorder(self.osc, self.cameras, self.pico, Path(folder.name), sample_hz=20)
        recorder.start({"task": "test_task", "control_source": "pico"})
        episode = Path(recorder.state()["episode_dir"])
        recorder.stop("failed")
        self.assertFalse(episode.exists())


if __name__ == "__main__":
    unittest.main()
