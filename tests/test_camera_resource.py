from __future__ import annotations

import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

from supervisor.camera_resource import CameraPair, SharedCameraResource


class SharedCameraResourceTests(unittest.TestCase):
    def test_capture_preserves_native_rgb_and_letterboxes_model_rgb(self):
        import cv2
        pair = CameraPair.__new__(CameraPair)
        pair.cv2 = cv2; pair.width = pair.height = 224; pair.read_failures = {}
        bgr = np.empty((480,640,3), dtype=np.uint8)
        bgr[:] = [20,40,120]
        _, (model, raw, stamp) = pair._read_one(0, SimpleNamespace(read=lambda: (True,bgr)))
        self.assertGreater(stamp,0)
        self.assertEqual(raw.shape,(480,640,3))
        self.assertEqual(model.shape,(224,224,3))
        np.testing.assert_array_equal(raw[0,0],[120,40,20])
        np.testing.assert_array_equal(model[112,112],raw[0,0])
        self.assertTrue((model[0] == 0).all())

    def test_fixed_capture_order_and_reopen_ignores_legacy_settings(self):
        import cv2
        calls = []
        class Capture:
            def __init__(self, index, backend):
                self.index = index; self.values = {}; calls.append((index, 'open', backend))
            def isOpened(self): return True
            def set(self, prop, value):
                calls.append((self.index, prop, value)); self.values[prop] = value; return True
            def get(self, prop): return self.values.get(prop, 0)
            def getBackendName(self): return 'DSHOW'
        pair = CameraPair.__new__(CameraPair)
        pair.cv2 = SimpleNamespace(**{key: getattr(cv2, key) for key in dir(cv2) if key.startswith('CAP_')},
                                   VideoCapture=Capture, VideoWriter_fourcc=cv2.VideoWriter_fourcc)
        pair.capture_settings = {}
        with patch('cv2_enumerate_cameras.enumerate_cameras', return_value=[]):
            for index in (0, 1, 0, 1):
                capture = pair._open_capture(index, {'fps': 20, 'exposure': -2, 'auto_exposure': True})
                self.assertTrue(pair.capture_settings[index]['configuration_ok'])
                self.assertEqual(capture.get(cv2.CAP_PROP_EXPOSURE), -7)
        expected = [('open', cv2.CAP_DSHOW), (cv2.CAP_PROP_FRAME_WIDTH, 640),
                    (cv2.CAP_PROP_FRAME_HEIGHT, 480),
                    (cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUY2')),
                    (cv2.CAP_PROP_FPS, 30), (cv2.CAP_PROP_AUTO_EXPOSURE, 0),
                    (cv2.CAP_PROP_EXPOSURE, -7)]
        self.assertEqual([(prop, value) for _, prop, value in calls], expected * 4)

    def test_failed_settings_and_unsupported_mode_readback_are_explicit(self):
        import cv2
        class Capture:
            def __init__(self, *args): self.values = {}
            def isOpened(self): return True
            def set(self, prop, value):
                self.values[prop] = value
                return prop != cv2.CAP_PROP_FPS
            def get(self, prop):
                if prop == cv2.CAP_PROP_AUTO_EXPOSURE: return -1
                if prop == cv2.CAP_PROP_FPS: return 20
                return self.values.get(prop, 0)
            def getBackendName(self): return 'DSHOW'
        pair = CameraPair.__new__(CameraPair)
        pair.cv2 = SimpleNamespace(**{key: getattr(cv2, key) for key in dir(cv2) if key.startswith('CAP_')},
                                   VideoCapture=Capture, VideoWriter_fourcc=cv2.VideoWriter_fourcc)
        pair.capture_settings = {}
        with patch('cv2_enumerate_cameras.enumerate_cameras', return_value=[]):
            pair._open_capture(0, {})
        result = pair.capture_settings[0]
        self.assertEqual(result['setting_failures'], ['fps'])
        self.assertEqual(result['negotiation_mismatches'], ['fps'])
        self.assertFalse(result['configuration_ok'])
        self.assertFalse(result['exposure_mode_verified'])
        self.assertEqual(result['exposure_mode'], 'manual_requested')

    def test_read_returns_latest_complete_cached_pair(self):
        """A transient async device read must not leak a missing RGB image."""
        resource = SharedCameraResource({})
        external = np.zeros((224, 224, 3), dtype=np.uint8)
        wrist = np.ones((224, 224, 3), dtype=np.uint8)
        with resource.frame_condition:
            resource.frames = {"external": external, "wrist": wrist}
            resource.frame_condition.notify_all()

        actual_external, actual_wrist = resource.read()

        self.assertIs(actual_external, external)
        self.assertIs(actual_wrist, wrist)
        self.assertEqual(actual_external.shape, (224, 224, 3))
        self.assertEqual(actual_wrist.shape, (224, 224, 3))


if __name__ == "__main__":
    unittest.main()
