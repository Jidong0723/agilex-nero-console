from __future__ import annotations
import json, tempfile, time, unittest
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
from supervisor.dataset_recorder import DatasetRecorder


class DatasetRecorderTests(unittest.TestCase):
    def setUp(self):
        self.frames = {"external": np.full((480, 640, 3), 30, np.uint8), "wrist": np.full((480, 640, 3), 80, np.uint8)}
        self.command = None
        def camera_frame(source, target):
            return (time.monotonic_ns(), self.frames[source].copy()) if source in self.frames else None
        self.cameras = SimpleNamespace(dataset_frame=camera_frame, snapshot=lambda: {"ready": True, "config": {"model_width": 224, "model_height": 224, "external": {"index": 2, "width": 640, "height": 480}, "wrist": {"index": 3, "width": 640, "height": 480}}, "sources": {key: {"available": True, "frame_available": True, "preview_size": [224, 224], "dataset_size": [640, 480]} for key in ("external", "wrist")}})
        def osc_state():
            now = time.monotonic_ns()
            command = {"final_joint_target_rad": None, "sent_monotonic_ns": None} if self.command is None else {"final_joint_target_rad": self.command, "sent_monotonic_ns": now}
            return {"execution": {}, "transport": {"hardware_feedback": {"joint_angles_rad": [1] * 7, "received_monotonic_ns": now}}, "command": command, "diagnostics": {}, "gripper": {"width_m": .05}}
        self.osc, self.pico = SimpleNamespace(state=osc_state), SimpleNamespace(snapshot=lambda: {"connected": False})

    def _record(self, source="pico", duration=.24):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        recorder = DatasetRecorder(self.osc, self.cameras, self.pico, Path(self.temp.name), sample_hz=20)
        recorder.start({"task": "test", "control_source": source}); time.sleep(duration)
        return recorder.stop()

    @staticmethod
    def _rows(result):
        return [json.loads(line) for line in (Path(result["episode_dir"]) / "frames.jsonl").read_text(encoding="utf-8").splitlines()]

    def test_native_size_and_unconnected_pico_metadata(self):
        result = self._record(); rows = self._rows(result); episode = Path(result["episode_dir"])
        self.assertGreater(result["frame_count"], 0)
        self.assertEqual((result["operator_device"], result["operator_connection"]), (None, "not_connected"))
        self.assertEqual(cv2.imread(str(episode / rows[0]["observation"]["images"]["front"])).shape[:2], (480, 640))
        metadata = json.loads((episode / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["requested_control_source"], "pico")
        self.assertEqual(metadata["camera_sources"]["external"]["preview_width"], 224)
        self.assertEqual(metadata["camera_sources"]["external"]["saved_width"], 640)

    def test_idle_and_active_control_context(self):
        idle = self._record(); self.assertEqual(self._rows(idle)[0]["control_context"], None)
        self.setUp(); self.command = [1] * 7
        active = self._record(); self.assertEqual(self._rows(active)[0]["control_context"]["joint_target_rad"], self.command)

    def test_duplicate_frame_is_not_written_twice(self):
        stamp = time.monotonic_ns(); self.cameras.dataset_frame = lambda source, target: (stamp, self.frames[source].copy())
        result = self._record()
        self.assertEqual(result["frame_count"], 1)
        self.assertGreater(result["camera_sources"]["external"]["duplicate_or_stale_frames"], 0)

    def test_partial_camera_and_invalid_feedback(self):
        self.frames.pop("wrist"); result = self._record(); row = self._rows(result)[0]
        self.assertIn("front", row["observation"]["images"]); self.assertIn("wrist", row["observation"]["missing_image_sources"])
        self.setUp(); self.osc = SimpleNamespace(state=lambda: {"transport": {"hardware_feedback": {"joint_angles_rad": [1] * 7, "received_monotonic_ns": []}}})
        result = self._record(); self.assertEqual(result["frame_count"], 0); self.assertIn("feedback_timestamp_invalid", result["rejection_reasons"])

    def test_web_episode_and_deletion(self):
        result = self._record("web"); self.assertEqual(result["camera_sources"], {}); self.assertEqual(self._rows(result)[0]["observation"]["images"], {})
        self.setUp(); folder = tempfile.TemporaryDirectory(); self.addCleanup(folder.cleanup)
        recorder = DatasetRecorder(self.osc, self.cameras, self.pico, Path(folder.name)); recorder.start({"task": "test", "control_source": "pico"}); path = Path(recorder.state()["episode_dir"]); recorder.stop("failed"); self.assertFalse(path.exists())

    def test_full_encoding_queue_is_reported_without_blocking(self):
        folder = tempfile.TemporaryDirectory(); self.addCleanup(folder.cleanup)
        recorder = DatasetRecorder(self.osc, self.cameras, self.pico, Path(folder.name), sample_hz=20)
        recorder._MAX_PENDING = 0
        recorder.start({"task": "test", "control_source": "pico"}); time.sleep(.12)
        result = recorder.stop()
        self.assertEqual(result["frame_count"], 0)
        self.assertGreater(result["backpressure_drops"], 0)
        self.assertIn("jpeg_queue_full", result["rejection_reasons"])
