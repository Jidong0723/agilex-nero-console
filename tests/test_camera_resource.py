from __future__ import annotations

import unittest

import numpy as np

from supervisor.camera_resource import SharedCameraResource


class SharedCameraResourceTests(unittest.TestCase):
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
