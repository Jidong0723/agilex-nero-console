import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from inference.rtc.diagnostics import prefix_diagnostics
from inference.rtc.diagnostics import diagnostic_sampler_override
from types import SimpleNamespace


class ResidualTests(unittest.TestCase):
    def test_diagnostic_overrides_restore_production_even_on_error(self):
        policy=SimpleNamespace(steps=20,cap=1000.,prefix_attention_schedule='ones',
                               terminal_noise_time=.001,requires_guided_execution=True)
        with self.assertRaisesRegex(RuntimeError,'test failure'):
            with diagnostic_sampler_override(policy,{'prefix_attention_schedule':'zeros','num_steps':40}):
                self.assertEqual(policy.steps,40)
                self.assertFalse(policy.requires_guided_execution)
                raise RuntimeError('test failure')
        self.assertEqual(policy.steps,20)
        self.assertEqual(policy.prefix_attention_schedule,'ones')
        self.assertTrue(policy.requires_guided_execution)
        with diagnostic_sampler_override(policy, {'prefix_attention_schedule':'linear'}):
            self.assertTrue(policy.requires_guided_execution)
        with diagnostic_sampler_override(policy, {'prefix_attention_schedule':'exp', 'terminal_refinement_steps':4}):
            self.assertFalse(policy.requires_guided_execution)
            self.assertEqual(policy.terminal_refinement_steps,4)
        self.assertEqual(policy.prefix_attention_schedule, 'ones')

    def test_nonzero_rotated_observation_and_real_position_residual(self):
        state = np.array([.2, -.1, .3, .3, -.2, .7, .5])
        actions = np.zeros((20, 7)); actions[:, 6] = .6
        actions[:, :3] = [.01, .02, -.01]; actions[:, 3:6] = [.1, -.05, .03]
        rotations = Rotation.from_rotvec(actions[:14, 3:6]) * Rotation.from_rotvec(state[3:6])
        reference = np.c_[state[:3]+actions[:14, :3], rotations.as_quat(), actions[:14, 6]]
        reference[2, 0] += .002
        reference[3, 3:7] *= -1  # Same quaternion orientation, opposite sign.
        report = prefix_diagnostics(actions, state, reference, np.r_[np.ones(6), np.full(8, .5)], delay=6, config={})
        rows = report['prefix_residuals']
        self.assertAlmostEqual(rows[2]['position_error_m'], .002)
        self.assertAlmostEqual(rows[3]['orientation_error_rad'], 0.)
        self.assertEqual(rows[6]['region'], 'soft')
        self.assertFalse(report['hard_prefix_all_within_1mm'])
        self.assertEqual(len(rows), 14)
        self.assertTrue(all(r['orientation_error_rad'] < 1e-12 for r in rows))


if __name__ == '__main__':
    unittest.main()
