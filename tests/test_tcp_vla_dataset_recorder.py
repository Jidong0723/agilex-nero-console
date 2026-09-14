from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts.validate_tcp_vla_episode import validate_episode
from scripts.build_tcp_vla_training_view import build_training_view
from supervisor.tcp_vla_dataset_recorder import TcpVlaDatasetRecorder


class TcpVlaRawDatasetRecorderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.frame = np.full((480, 640, 3), 90, dtype=np.uint8)
        self.sample_id = 0

        def camera_frame(source, target):
            return target, self.frame.copy()

        self.cameras = SimpleNamespace(
            snapshot=lambda: {
                "config": {"external": {"index": 0, "width": 640, "height": 480},
                           "wrist": {"index": 1, "width": 640, "height": 480}},
                "sources": {name: {"available": True, "frame_available": True, "capture_hz": 20.0}
                            for name in ("external", "wrist")},
            },
            dataset_frame=camera_frame,
        )
        self.pico = SimpleNamespace(snapshot=lambda: {"connected": True})

        def osc_state():
            self.sample_id += 1
            now = time.monotonic_ns()
            pose = {"position_m": [0.1, 0.2, 0.3], "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]}
            return {
                "session": {"state": "ACTIVE", "execution_mode": "hardware"},
                "execution": {"sample_id": self.sample_id, "feedback_revision": self.sample_id,
                              "target_generation": 3, "motion_epoch": 2, "sample_monotonic_ns": now,
                              "measured_tcp_pose": pose},
                "transport": {"hardware_feedback": {"joint_angles_rad": [0.0] * 7,
                              "joint_velocity_rad_s": [0.0] * 7, "rx_revision": self.sample_id,
                              "tcp_pose": pose},
                              "feedback_mailbox": {"received_monotonic_ns": now}},
                "command": {"target_generation": 3, "epoch": 2, "target_tcp": pose},
                "diagnostics": {"pink": {"ok": True}, "ruckig": {"enabled": True}},
                "gripper": {"width_m": 0.095},
            }

        self.osc = SimpleNamespace(state=osc_state)

    def recorder(self) -> TcpVlaDatasetRecorder:
        return TcpVlaDatasetRecorder(
            self.osc, self.cameras, self.pico, Path(self.temp.name),
            sample_hz=15, raw_camera_hz=20, raw_robot_state_hz=50,
            camera_sync_limit_s=1.0, feedback_age_limit_s=1.0,
        )

    def test_collects_only_native_rate_raw_streams(self):
        recorder = self.recorder()
        recorder.start({"task": "my_task", "description": "my untouched natural language", "control_source": "pico"})
        time.sleep(1.2)
        result = recorder.stop("completed")
        self.assertEqual(result["status"], "completed", result)
        episode = Path(result["episode_dir"])
        metadata = json.loads((episode / "episode.json").read_text(encoding="utf-8"))
        camera_rows = [json.loads(line) for line in (episode / "raw/camera_frames.jsonl").read_text(encoding="utf-8").splitlines()]
        robot_rows = [json.loads(line) for line in (episode / "raw/robot_states.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(metadata["prompt"], "my untouched natural language")
        self.assertFalse(metadata["training_view_generated"])
        self.assertFalse(metadata["horizon_windows_built"])
        self.assertFalse((episode / "steps.jsonl").exists())
        self.assertFalse((episode / "observations.jsonl").exists())
        self.assertFalse((episode / "videos").exists())
        camera_rows_by_source = {source: [row for row in camera_rows if row["source"] == source]
                                 for source in ("external", "wrist")}
        self.assertTrue(all(len(rows) >= 18 for rows in camera_rows_by_source.values()))
        self.assertGreaterEqual(len(robot_rows), 45)
        self.assertTrue(all((episode / row["image"]).is_file() for row in camera_rows))
        self.assertEqual(metadata["raw_streams"]["camera_frames_by_source"],
                         {source: len(rows) for source, rows in camera_rows_by_source.items()})
        self.assertEqual(metadata["raw_streams"]["camera_pairing"], "deferred_to_postprocessing")
        self.assertTrue(all(len(row["joint_position_rad"]) == 7 for row in robot_rows))
        self.assertTrue(all(isinstance(row["latest_feedback_age_s"], float) for row in robot_rows))
        self.assertTrue(all(isinstance(row["feedback_fresh_perf_counter_ns"], int) for row in robot_rows))
        self.assertTrue(all(isinstance(row["feedback_recording_delay_s"], float) for row in robot_rows))
        self.assertTrue(all(row["prompt"] == "my untouched natural language" for row in robot_rows))
        self.assertTrue(all("measured_tcp_pose" in row and "target_tcp_pose" in row for row in robot_rows))
        validation = validate_episode(episode)
        self.assertTrue(validation["structural_valid"], validation)
        self.assertTrue(validation["raw_valid"], validation)
        self.assertFalse(validation["training_ready"])
        derived = episode / "derived" / "15hz_test"
        built = build_training_view(
            episode, derived,
            lambda joints: {"position_m": [joints[0], 0.2, 0.3],
                            "rotation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]},
            camera_limit_s=0.1, robot_limit_s=0.1,
        )
        self.assertGreaterEqual(built["control_steps"], 16)
        self.assertGreaterEqual(built["h16_windows"], 1)
        self.assertTrue((derived / "observations.jsonl").is_file())
        self.assertTrue((derived / "steps.jsonl").is_file())
        self.assertTrue((derived / "h16.jsonl").is_file())

    def test_producer_drain_starts_after_current_revision_not_state_revision(self):
        pose = {"position_m": [0.1, 0.2, 0.3], "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]}

        class ProducerOsc:
            def __init__(self):
                self.anchor_calls = 0

            def state(self):
                # Deliberately stale: this was the source of the episode_000012
                # pre-history drain and false revision-gap report.
                return {
                    "session": {"state": "ACTIVE", "execution_mode": "hardware"},
                    "execution": {"feedback_revision": 10},
                    "diagnostics": {"ruckig": {"enabled": True}},
                }

            def sensor_sample(self, target_monotonic_ns, wait_s=0.0):
                self.anchor_calls += 1
                return {"feedback_revision": 100, "feedback_monotonic_ns": target_monotonic_ns,
                        "joint_position_rad": [0.0] * 7, "joint_velocity_rad_s": [0.0] * 7,
                        "gripper_width_m": 0.095, "measured_tcp_pose": pose,
                        "latest_feedback_age_s": 0.001}

            def sensor_samples_after(self, revision, wait_s=0.0, max_items=128):
                if revision == 100:
                    now = time.perf_counter_ns()
                    return [
                        {"feedback_revision": 101 + index,
                         "feedback_monotonic_ns": now + index * 20_000_000,
                         "joint_position_rad": [0.0] * 7,
                         "joint_velocity_rad_s": [0.0] * 7,
                         "gripper_width_m": 0.095,
                         "gripper_target_width_m": 0.095,
                         "measured_tcp_pose": pose, "target_tcp_pose": pose,
                         "control_sample_id": index, "target_generation": 1, "motion_epoch": 1}
                        for index in range(2)
                    ]
                time.sleep(min(float(wait_s), 0.01))
                return []

        osc = ProducerOsc()
        recorder = TcpVlaDatasetRecorder(
            osc, self.cameras, self.pico, Path(self.temp.name),
            sample_hz=15, raw_camera_hz=20, raw_robot_state_hz=50,
            camera_sync_limit_s=1.0, feedback_age_limit_s=1.0,
        )
        recorder.start({"task": "test", "description": "manual prompt", "control_source": "pico"})
        time.sleep(0.12)
        result = recorder.stop("completed")
        episode = Path(result["episode_dir"])
        rows = [json.loads(line) for line in (episode / "raw/robot_states.jsonl").read_text(encoding="utf-8").splitlines()]
        metadata = json.loads((episode / "episode.json").read_text(encoding="utf-8"))
        self.assertEqual([row["feedback_revision"] for row in rows], [101, 102])
        self.assertEqual(metadata["raw_streams"]["robot_revision_gaps"], 0)
        self.assertEqual(metadata["raw_streams"]["robot_prestart_ignored"], 0)
        self.assertGreaterEqual(osc.anchor_calls, 1)

    def test_failed_raw_episode_is_archived(self):
        recorder = self.recorder()
        recorder.start({"task": "test", "description": "manual prompt", "control_source": "pico"})
        time.sleep(0.15)
        result = recorder.stop("failed", "operator rejected")
        self.assertEqual(result["status"], "failed")
        self.assertIn("failed_episodes", result["episode_dir"])
        self.assertTrue((Path(result["episode_dir"]) / "episode.json").is_file())

    def test_missing_robot_feedback_preserves_camera_evidence(self):
        original = self.osc.state

        def incomplete_state():
            value = original()
            value["transport"]["hardware_feedback"]["joint_angles_rad"] = None
            return value

        self.osc.state = incomplete_state
        recorder = self.recorder()
        recorder.start({"task": "test", "description": "manual prompt", "control_source": "pico"})
        time.sleep(0.12)
        result = recorder.stop("completed")
        self.assertEqual(result["status"], "failed")
        self.assertTrue(Path(result["episode_dir"]).is_dir())
        self.assertGreater(result["raw_camera_frames"], 0)
        self.assertGreater(result["rejection_reasons"]["robot_missing:joint_position_rad"], 0)
        self.assertIn("missing_required_raw_stream", result["quality"]["reasons"])

    def test_dataset_gripper_reads_receive_cache_when_status_is_missing(self):
        import threading
        from supervisor.control import OperationalSpaceController
        controller = object.__new__(OperationalSpaceController)
        controller._status_lock = threading.Lock()
        controller._status_cache = (time.monotonic(), {"gripper": {}})
        controller._osc = SimpleNamespace(status=lambda: {})
        widths = iter([0.03, 0.08])
        controller._dataset_read_gripper = lambda: SimpleNamespace(width_m=next(widths))
        raw = {"revision": 1, "joints": [0.0] * 7, "velocities": [0.0] * 7,
               "alignment_monotonic_ns": time.perf_counter_ns()}
        first = controller._decorate_osc_sensor_samples([raw])[0]
        second = controller._decorate_osc_sensor_samples([raw])[0]
        self.assertEqual(first["gripper_width_m"], 0.03)
        self.assertEqual(second["gripper_width_m"], 0.08)
        self.assertEqual(second["gripper_feedback_source"], "sdk_receive_cache")

    def test_feedback_age_converts_monotonic_freshness_to_perf_clock(self):
        age, fresh_perf = TcpVlaDatasetRecorder._feedback_age(
            {"feedback_monotonic_ns": 1_250_000_000, "sdk_fresh_monotonic_ns": 1_000_000_000},
            240_000_000,
        )
        self.assertEqual(fresh_perf, 1_240_000_000)
        self.assertAlmostEqual(age, 0.010, places=9)

    def test_start_rejects_incomplete_hardware_feedback_before_creating_episode(self):
        class IncompleteOsc:
            def state(self):
                return {"session": {"state": "ACTIVE", "execution_mode": "hardware"},
                        "diagnostics": {"ruckig": {"enabled": True}}}

            def sensor_sample(self, target_monotonic_ns, wait_s=0.0):
                return {"feedback_revision": 7, "feedback_monotonic_ns": target_monotonic_ns,
                        "joint_position_rad": [0.0] * 7, "joint_velocity_rad_s": [0.0] * 7,
                        "gripper_width_m": None, "measured_tcp_pose": None}

            def sensor_samples_after(self, revision, wait_s=0.0, max_items=128):
                return []

        recorder = TcpVlaDatasetRecorder(IncompleteOsc(), self.cameras, self.pico, Path(self.temp.name))
        with self.assertRaisesRegex(RuntimeError, "夹爪开度.*实测TCP位姿"):
            recorder.start({"task": "test", "description": "manual prompt", "control_source": "pico"})
        self.assertIsNone(recorder.active)
        self.assertEqual(list((Path(self.temp.name) / "staging").iterdir()), [])


if __name__ == "__main__":
    unittest.main()
