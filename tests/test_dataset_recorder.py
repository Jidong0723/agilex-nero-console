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
        self.commands = 0
        self.cameras = SimpleNamespace(
            snapshot=lambda: {"ready": True, "config": {"external": {"index": 2}, "wrist": {"index": 3}}},
            frame_jpeg=lambda source: b"jpeg-" + source.encode(),
        )
        self.osc = SimpleNamespace(state=lambda: {
            "session": {"execution_mode": "shadow"},
            "execution": {"measured_joint_state_rad": [1, 2, 3, 4, 5, 6, 7]},
            "transport": {"hardware_feedback": {"joint_angles_rad": [1, 2, 3, 4, 5, 6, 7]}},
            "command": {"final_joint_target_rad": [1, 2, 3, 4, 5, 6, 7]},
            "gripper": {"width_m": 0.05},
        })
        self.pico = SimpleNamespace(snapshot=lambda: {})

    def test_completed_episode_writes_frames_without_robot_commands(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            recorder = DatasetRecorder(self.osc, self.cameras, self.pico, Path(folder), sample_hz=20)
            state = recorder.start({"task": "test_task", "description": "test", "control_source": "pico"})
            self.assertTrue(state["recording"])
            self.assertEqual(state["control_source"], "pico")
            time.sleep(0.15)
            result = recorder.stop("completed")
            episode = Path(result["episode_dir"])
            metadata = json.loads((episode / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["status"], "completed")
            self.assertTrue((episode / "frames.jsonl").read_text(encoding="utf-8").splitlines())
            self.assertTrue((episode / "images" / "front" / "000000.jpg").exists())
            self.assertTrue((episode / "images" / "wrist" / "000000.jpg").exists())
            self.assertEqual(metadata["files"]["images"]["external"], "images/front")
            self.assertEqual(metadata["files"]["images"]["wrist"], "images/wrist")
            self.assertIn("UTF-8 JSON Lines", metadata["data_contents"]["frames"]["format"])
            self.assertEqual(self.commands, 0)

    def test_failed_episode_is_deleted_and_path_cannot_escape_root(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            recorder = DatasetRecorder(self.osc, self.cameras, self.pico, root, sample_hz=20)
            recorder.start({"task": "test_task", "control_source": "pico"})
            time.sleep(0.08)
            episode = Path(recorder.state()["episode_dir"])
            recorder.stop("failed")
            self.assertFalse(episode.exists())
            with self.assertRaises(RuntimeError):
                recorder._assert_episode_path(root.parent / "outside")

    def test_web_episode_records_state_without_cameras(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            recorder = DatasetRecorder(self.osc, self.cameras, self.pico, Path(folder), sample_hz=20)
            state = recorder.start({"task": "web_only", "control_source": "web"})
            self.assertEqual(state["camera_sources"], {})
            self.assertTrue(state["warnings"])
            time.sleep(0.08)
            result = recorder.stop("completed")
            episode = Path(result["episode_dir"])
            row = json.loads((episode / "frames.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(row["observation"]["images"], {})
            self.assertEqual(row["control_source"], "web")
            self.assertFalse((episode / "images").exists())

    def test_missing_camera_frame_does_not_stop_episode(self) -> None:
        self.cameras.frame_jpeg = lambda source: b"jpeg-external" if source == "external" else None
        with tempfile.TemporaryDirectory() as folder:
            recorder = DatasetRecorder(self.osc, self.cameras, self.pico, Path(folder), sample_hz=20)
            recorder.start({"task": "partial", "control_source": "pi05"})
            time.sleep(0.1)
            result = recorder.stop("completed")
            self.assertGreater(result["camera_sources"]["external"]["captured_frames"], 0)
            self.assertGreater(result["camera_sources"]["wrist"]["dropped_frames"], 0)
            self.assertGreater(result["dropped_frames"], 0)


if __name__ == "__main__":
    unittest.main()
