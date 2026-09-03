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
            recorder.start({"task": "test_task", "description": "test"})
            time.sleep(0.15)
            result = recorder.stop("completed")
            episode = Path(result["episode_dir"])
            metadata = json.loads((episode / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["status"], "completed")
            self.assertTrue((episode / "frames.jsonl").read_text(encoding="utf-8").splitlines())
            self.assertTrue((episode / "images" / "front" / "000000.jpg").exists())
            self.assertTrue((episode / "images" / "wrist" / "000000.jpg").exists())
            self.assertEqual(self.commands, 0)

    def test_failed_episode_is_deleted_and_path_cannot_escape_root(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            recorder = DatasetRecorder(self.osc, self.cameras, self.pico, root, sample_hz=20)
            recorder.start({"task": "test_task"})
            time.sleep(0.08)
            episode = Path(recorder.state()["episode_dir"])
            recorder.stop("failed")
            self.assertFalse(episode.exists())
            with self.assertRaises(RuntimeError):
                recorder._assert_episode_path(root.parent / "outside")


if __name__ == "__main__":
    unittest.main()
