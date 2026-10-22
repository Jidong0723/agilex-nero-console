import time
import unittest
from unittest.mock import patch

from supervisor.inference_gripper_feedback import GripperFeedbackCache, require_fresh


def sample(stamp, width=.03):
    return {'width_m': width, 'raw': {'value': {'timestamp': stamp, 'hz': 50.}}}


class FreshnessTests(unittest.TestCase):
    def read_at(self, cache, stamp, wall, ns, session='s'):
        with patch('supervisor.inference_gripper_feedback.time.time', return_value=wall), \
             patch('supervisor.inference_gripper_feedback.time.perf_counter_ns', return_value=ns):
            return cache.read(lambda: sample(stamp), session)

    def test_first_old_frame_is_not_fresh(self):
        _, d = self.read_at(GripperFeedbackCache(), 999., 1000., 10_000_000_000)
        self.assertIn('error', d)
        with self.assertRaises(RuntimeError): require_fresh(d, 10_000_000_000)

    def test_same_width_new_frame_is_fresh(self):
        c = GripperFeedbackCache()
        self.read_at(c, 1000., 1000., 10_000_000_000)
        _, d = self.read_at(c, 1000.02, 1000.02, 10_020_000_000)
        self.assertTrue(d['new_sdk_feedback'])
        self.assertEqual(d['feedback_revision'], 2)
        require_fresh(d, 10_020_000_000)

    def test_repeated_cache_and_session_change_do_not_renew_age(self):
        c = GripperFeedbackCache()
        _, first = self.read_at(c, 1000., 1000., 10_000_000_000)
        _, d = self.read_at(c, 1000., 1000.2, 10_200_000_000, 'new-session')
        self.assertEqual(d['received_perf_ns'], first['received_perf_ns'])
        self.assertIn('error', d)

    def test_disconnect_error_then_real_feedback_recovers(self):
        c = GripperFeedbackCache()
        self.read_at(c, 1000., 1000., 10_000_000_000)
        _, d = c.read(lambda: (_ for _ in ()).throw(RuntimeError('disconnected')))
        self.assertIn('error', d)
        _, d = self.read_at(c, 1001., 1001., 11_000_000_000)
        require_fresh(d, 11_000_000_000)

    def test_regressed_stale_timestamp_rejected(self):
        c = GripperFeedbackCache()
        self.read_at(c, 1000., 1000., 10_000_000_000)
        _, d = self.read_at(c, 999., 1000.02, 10_020_000_000)
        self.assertIn('error', d)
        self.assertEqual(d['sdk_timestamp'], 1000.)

    def test_read_time_does_not_hide_source_age(self):
        _, d = self.read_at(GripperFeedbackCache(), 999.9, 1000., 10_000_000_000)
        self.assertAlmostEqual(d['feedback_age_ms'], 100., places=3)
        with self.assertRaises(RuntimeError): require_fresh(d, 10_060_000_000)

    def test_clock_jump_requires_new_frame(self):
        c = GripperFeedbackCache()
        self.read_at(c, 1000., 1000., 10_000_000_000)
        _, d = self.read_at(c, 1000., 1001., 10_020_000_000)
        self.assertIn('error', d)
        _, d = self.read_at(c, 1001.02, 1001.02, 10_040_000_000)
        require_fresh(d, 10_040_000_000)

    def test_fresh_first_sample_can_be_consumed_repeatedly(self):
        _, d = self.read_at(GripperFeedbackCache(), 1000., 1000., 10_000_000_000)
        for ns in (10_000_000_000, 10_020_000_000, 10_100_000_000):
            require_fresh(d, ns)


if __name__ == '__main__': unittest.main()
