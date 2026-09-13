import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from supervisor.dataset_recorder import DatasetRecorder


class CameraStub:
    def __init__(self, ready=True, frozen=False):
        self.ready = ready
        self.frozen = frozen
        self.last = time.monotonic_ns()

    def snapshot(self):
        status = {"available": self.ready, "frame_available": self.ready, "last_frame_age_ms": 0 if self.ready else None}
        return {"ready": self.ready, "config": {"external": {"index": 1, "width": 640, "height": 480}, "wrist": {"index": 2, "width": 640, "height": 480}}, "sources": {"external": status, "wrist": status}}

    def frame_timestamp(self, source):
        if not self.ready:
            return None
        if not self.frozen:
            self.last = max(self.last + 1, time.monotonic_ns())
        return self.last

    def frame_jpeg(self, source, timestamp):
        return b"jpeg-" + source.encode()


class DatasetRecorderTests(unittest.TestCase):
    def setUp(self):
        self.cameras = CameraStub()
        self.osc = SimpleNamespace(state=lambda: {
            "transport": {"hardware_feedback": {"joint_angles_rad": [1, 2, 3, 4, 5, 6, 7], "received_monotonic_ns": time.monotonic_ns(), "gripper_width_m": 0.05}},
            "command": {"final_joint_target_rad": [1, 2, 3, 4, 5, 6, 7], "sent_monotonic_ns": time.monotonic_ns(), "gripper_target_width_m": 0.04},
        })

    def test_completed_episode_writes_separate_rgb_streams(self):
        with tempfile.TemporaryDirectory() as folder:
            recorder = DatasetRecorder(self.osc, self.cameras, None, Path(folder))
            state = recorder.start({"task": "test_task", "description": "test", "control_source": "pico"})
            self.assertTrue(state["recording"])
            self.assertEqual(state["operator_device"], "pico_4_ultra")
            time.sleep(0.2)
            result = recorder.stop("completed")
            episode = Path(result["episode_dir"])
            metadata = json.loads((episode / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["status"], "completed")
            self.assertTrue((episode / "camera_frames.jsonl").read_text(encoding="utf-8").splitlines())
            self.assertTrue((episode / "robot_states.jsonl").read_text(encoding="utf-8").splitlines())
            self.assertTrue((episode / "images" / "front" / "000000.jpg").exists())
            self.assertTrue((episode / "images" / "wrist" / "000000.jpg").exists())
            self.assertEqual(metadata["camera_rate_hz"], 20.0)
            self.assertEqual(metadata["robot_state_rate_hz"], 15.0)
            self.assertEqual(metadata["data_contents"]["images"], "RGB JPEG only")

    def test_failed_episode_is_deleted_and_path_cannot_escape_root(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            recorder = DatasetRecorder(self.osc, self.cameras, None, root)
            recorder.start({"task": "test_task", "control_source": "pico"})
            time.sleep(0.08)
            episode = Path(recorder.state()["episode_dir"])
            recorder.stop("failed")
            self.assertFalse(episode.exists())
            with self.assertRaises(RuntimeError):
                recorder._assert_episode_path(root.parent / "outside")

    def test_camera_not_ready_is_recorded_and_does_not_stop_episode(self):
        with tempfile.TemporaryDirectory() as folder:
            recorder = DatasetRecorder(self.osc, CameraStub(ready=False), None, Path(folder))
            recorder.start({"task": "camera_missing", "control_source": "web"})
            time.sleep(0.08)
            live = recorder.state()
            self.assertTrue(live["recording"])
            result = recorder.stop("completed")
            episode = Path(result["episode_dir"])
            self.assertTrue((episode / "collection_events.jsonl").read_text(encoding="utf-8").strip())
            self.assertIn("camera_unavailable_at_start", (episode / "metadata.json").read_text(encoding="utf-8"))

    def test_repeated_camera_timestamp_is_marked_and_episode_continues(self):
        with tempfile.TemporaryDirectory() as folder:
            recorder = DatasetRecorder(self.osc, CameraStub(frozen=True), None, Path(folder))
            recorder.start({"task": "duplicate", "control_source": "pico"})
            time.sleep(0.35)
            result = recorder.stop("completed")
            self.assertEqual(result["status"], "completed")
            self.assertGreater(result["camera_dropped_frames"], 0)
            self.assertGreater(result["event_count"], 0)

    def test_missing_action_does_not_stop_camera_stream(self):
        with tempfile.TemporaryDirectory() as folder:
            osc = SimpleNamespace(state=lambda: {
                "transport": {"hardware_feedback": {
                    "joint_angles_rad": [1, 2, 3, 4, 5, 6, 7],
                    "received_monotonic_ns": time.monotonic_ns(),
                }},
                "command": {},
            })
            recorder = DatasetRecorder(osc, self.cameras, None, Path(folder))
            recorder.start({"task": "camera_only", "control_source": "web"})
            time.sleep(0.22)
            live = recorder.state()
            self.assertTrue(live["recording"])
            self.assertTrue(live["waiting_for_action"])
            result = recorder.stop("completed")
            self.assertGreaterEqual(result["camera_frame_count"], 3)
            self.assertGreater(result["robot_state_count"], 0)
            self.assertEqual(result["valid_robot_state_count"], 0)
            self.assertGreater(result["state_invalid_events"], 0)
            self.assertEqual(result["status"], "completed")


if __name__ == "__main__":
    unittest.main()
