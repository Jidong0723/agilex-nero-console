from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from supervisor.osc_input_journal import OscInputJournal
from supervisor.control import OperationalSpaceController
from supervisor.tcp_vla_dataset_recorder import TcpVlaDatasetRecorder
from scripts.validate_tcp_vla_episode import validate_episode
from src.nero_console.application.adapter_runtime import AdapterRuntime


class IngressTests(unittest.TestCase):
    def controller(self):
        controller = object.__new__(OperationalSpaceController)
        controller._osc_inputs = OscInputJournal()
        controller._osc = Mock()
        controller._osc.status.return_value = {"session": {"id": "s", "client_id": "c",
            "state": "ACTIVE", "execution_mode": "shadow"}, "output_mode": "impedance"}
        controller._osc.heartbeat_expired.return_value = False
        return controller

    def test_common_boundary_records_full_payload_acceptance_rejection_and_identity(self):
        controller = self.controller()
        for source in ("web", "pico", "pi05", "custom"):
            controller.osc_bind_input_source("s", "c", source)
            payload = {"target_pose": {"position_m": [.1, .2, .3], "orientation_xyzw": [0, 0, 0, 1]}}
            body = {"session_id": "s", "client_id": "c", "sequence": 7, "type": "track_tcp", "payload": payload}
            controller._execute_osc_command = Mock(return_value={"ok": True})
            controller.osc_command(body)
            payload["target_pose"]["position_m"][0] = 99
            controller._execute_osc_command.side_effect = PermissionError("old session")
            with self.assertRaises(PermissionError):
                controller.osc_command(body)
        rows = controller.osc_input_events(0)["events"]
        self.assertEqual(len(rows), 8)
        self.assertEqual([row["revision"] for row in rows], list(range(1, 9)))
        for row in rows[::2]:
            self.assertTrue(row["accepted"])
            self.assertEqual(row["command"]["payload"]["target_pose"]["position_m"][0], .1)
            self.assertEqual(row["context"]["output_mode"], "impedance")
            self.assertEqual(row["tcp_semantics"], "absolute_pose_robot_base")
        self.assertTrue(all(not row["accepted"] and "old session" in row["error"] for row in rows[1::2]))

    def test_bounded_history_reports_overflow_and_is_copy_isolated(self):
        journal = OscInputJournal(2)
        for i in range(4):
            journal.append({"type": "hold"}, {"control_source": "web"}, time.perf_counter_ns(), accepted=True)
        batch = journal.read(0)
        self.assertEqual(batch["lost_events"], 2)
        batch["events"][0]["command"]["type"] = "mutated"
        self.assertEqual(journal.read(2)["events"][0]["command"]["type"], "hold")

    def test_binding_cannot_change_another_session_and_heartbeat_gates_connection(self):
        controller = self.controller()
        with self.assertRaises(PermissionError):
            controller.osc_bind_input_source("old", "c", "pico")
        self.assertEqual(controller.osc_input_context()["control_source"], "external")
        controller._osc.heartbeat_expired.return_value = True
        self.assertFalse(controller.osc_input_context()["connected"])

    def test_adapters_share_normalized_connection_contract(self):
        runtime = object.__new__(AdapterRuntime)
        context = {"control_source": "web", "connected": True, "session_id": "s", "client_id": "c"}
        runtime.osc = SimpleNamespace(input_context=lambda: dict(context))
        runtime.pico = SimpleNamespace(snapshot=lambda: {"connected": True, "session_id": "s"})
        runtime.pi05 = SimpleNamespace(snapshot=lambda: {"state": "RUNNING", "session_id": "s"})
        for source in ("web", "pico", "pi05", "custom"):
            context["control_source"] = source
            self.assertTrue(runtime.dataset_context()["connected"])
            context["connected"] = False
            self.assertFalse(runtime.dataset_context()["connected"])
            context["connected"] = True
        context["control_source"] = "pico"
        runtime.pico = SimpleNamespace(snapshot=lambda: {"connected": True, "session_id": "old"})
        self.assertFalse(runtime.dataset_context()["connected"])
        context["control_source"] = "pi05"
        runtime.pi05 = SimpleNamespace(snapshot=lambda: {"state": "STOPPED", "session_id": "s"})
        self.assertFalse(runtime.dataset_context()["connected"])


class GenericCollectionTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.journal = OscInputJournal()
        self.context = {"control_source": "web", "session_id": "s", "client_id": "c", "connected": True,
                        "output_mode": "impedance", "execution_mode": "shadow"}
        self.revision = 0
        def state():
            self.revision += 1
            return {"session": {"state": "ACTIVE", "id": "s", "execution_mode": "shadow"},
                "execution": {"measured_joint_state_rad": [0] * 7, "measured_joint_velocity_rad_s": [0] * 7,
                              "feedback_revision": self.revision,
                              "measured_tcp_pose": {"position_m": [.1, .2, .3], "orientation_xyzw": [0, 0, 0, 1]}},
                "gripper": {"width_m": .095}}
        self.osc = SimpleNamespace(state=state, input_context=lambda: dict(self.context), input_events=self.journal.read)
        self.cameras = SimpleNamespace(snapshot=lambda: {}, dataset_frame=lambda *_: None)
        self.recorder = TcpVlaDatasetRecorder(self.osc, self.cameras, self.root)
        self.addCleanup(self.recorder.close)

    def start(self, **extra):
        return self.recorder.start({"task": "test", "description": "move gently", **extra})

    def test_all_sources_start_without_cameras_and_archive_real_provenance(self):
        for source in ("web", "pico", "pi05", "custom"):
            self.context["control_source"] = source
            started = self.start()
            self.assertEqual(started["control_source"], source)
            self.assertTrue(started["warnings"])
            time.sleep(.06)
            result = self.recorder.stop()
            self.assertEqual(result["status"], "completed", result)
            metadata = json.loads((Path(result["episode_dir"]) / "episode.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["control_source"], source)
            self.assertEqual(metadata["output_mode"], "impedance")
            self.assertEqual(metadata["session_id"], "s")
            self.assertEqual(metadata["client_id"], "c")
            self.assertEqual(metadata["enabled_cameras"], [])
            self.assertIn("dual_rgb_unavailable", metadata["quality"]["training_blockers"])
            self.assertFalse(metadata["quality"]["training_eligible"])
            validation = validate_episode(Path(result["episode_dir"]))
            self.assertTrue(validation["structural_valid"], validation)
            self.assertTrue(validation["raw_valid"], validation)

    def test_disconnected_and_spoofed_sources_rejected_before_creating_episode(self):
        for source in ("web", "pico", "pi05", "custom"):
            self.context.update(control_source=source, connected=False)
            with self.assertRaises(RuntimeError):
                self.start()
        self.context.update(control_source="web", connected=True)
        with self.assertRaises(ValueError):
            self.start(control_source="pico")
        self.assertEqual(list(self.root.iterdir()), [])

    def test_input_stream_preserves_every_intermediate_command_and_mode(self):
        self.start()
        for i in range(75):
            context = dict(self.context, output_mode="cpv" if i < 40 else "impedance")
            self.journal.append({"session_id": "s", "client_id": "c", "sequence": i, "type": "track_tcp",
                "payload": {"target_pose": {"position_m": [i / 1000, .2, .3], "orientation_xyzw": [0, 0, 0, 1]}}},
                context, time.perf_counter_ns(), accepted=True)
        self.journal.append({"type": "hold", "payload": {"reason": "released"}}, self.context,
                            time.perf_counter_ns(), accepted=True)
        result = self.recorder.stop()
        episode = Path(result["episode_dir"])
        rows = [json.loads(line) for line in (episode / "raw/osc_inputs.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 76)
        self.assertEqual([row["command"]["sequence"] for row in rows[:-1]], list(range(75)))
        self.assertEqual(rows[-1]["command"]["type"], "hold")
        metadata = json.loads((episode / "episode.json").read_text(encoding="utf-8"))
        self.assertCountEqual(metadata["output_modes"], ["cpv", "impedance"])
        self.assertEqual(metadata["raw_streams"]["osc_inputs"], 76)
        self.assertEqual(metadata["raw_streams"]["osc_input_gaps"], 0)

    def test_input_history_overflow_is_visible_and_invalidates_episode(self):
        self.journal = OscInputJournal(2)
        self.osc.input_events = self.journal.read
        self.start()
        self.recorder.stop_event.set()
        self.recorder.input_thread.join(timeout=1)
        for i in range(5):
            self.journal.append({"type": "hold"}, self.context, time.perf_counter_ns(), accepted=True)
        result = self.recorder.stop()
        self.assertEqual(result["osc_input_gaps"], 3)
        self.assertIn("osc_input_history_overflow", result["quality"]["reasons"])
        self.assertEqual(result["status"], "failed")

    def test_single_camera_is_saved_without_synthesizing_the_missing_camera(self):
        import numpy as np
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.cameras.snapshot = lambda: {"config": {"external": {"width": 640, "height": 480}},
            "sources": {"external": {"available": True, "frame_available": True, "capture_hz": 20}}}
        self.cameras.dataset_frame = lambda source, stamp: (stamp, frame) if source == "external" else None
        result = self.start()
        self.assertEqual(result["enabled_cameras"], ["external"])
        time.sleep(.09)
        result = self.recorder.stop()
        self.assertEqual(result["status"], "completed", result)
        self.assertGreater(result["raw_camera_frames_by_source"]["external"], 0)
        self.assertEqual(result["raw_camera_frames_by_source"]["wrist"], 0)
        self.assertEqual(result["raw_camera_drops"], 0)
        self.assertTrue(validate_episode(Path(result["episode_dir"]))["raw_valid"])

    def test_ending_source_connection_keeps_valid_feedback_and_records_warning(self):
        self.start()
        self.context["connected"] = False
        time.sleep(.25)
        result = self.recorder.stop()
        self.assertEqual(result["status"], "completed", result)
        self.assertFalse(result["input_context"]["connected"])
        self.assertTrue(any("控制源已断开" in warning for warning in result["warnings"]))
        self.assertGreater(result["raw_robot_states"], 0)
