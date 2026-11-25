from dataclasses import replace
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from tron2_pose_compensation import PoseConfig, PoseCorrection
from tron2_timing_calibration import CalibrationReference, fit_effective_lag


class PoseFeedbackTests(unittest.TestCase):
    def test_feedback_removes_static_xyz_and_world_rotation_bias_with_bounds(self):
        c = PoseCorrection()
        nominal_r = Rotation.from_euler('zyx', [70, 20, -10], degrees=True).as_matrix()
        bias_p = np.array([.004, -.003, .002])
        bias_r = Rotation.from_rotvec(np.deg2rad([.6, -.4, .3])).as_matrix()
        history = []
        for step in range(1201):
            now = 1_000_000_000+step*10_000_000
            if step % 5 == 0:
                actual_p = c.translation-bias_p
                actual_r = c.rotation@bias_r.T@nominal_r
                c.observe(step, now-2_000_000, now, now, np.zeros(3), nominal_r, actual_p, actual_r)
            p, r = c.advance(now)
            history.append((p.copy(), r.copy()))
        self.assertLess(np.linalg.norm(c.translation-bias_p), .0006)
        self.assertLess(np.rad2deg(Rotation.from_matrix(c.rotation@bias_r.T).magnitude()), .1)
        positions = np.array([p for p, _ in history])
        self.assertLessEqual(np.linalg.norm(positions, axis=1).max(), c.config.max_translation_m+1e-12)
        self.assertLessEqual(np.linalg.norm(np.diff(positions, axis=0), axis=1).max()/.01, c.config.max_translation_rate_m_s+1e-12)
        rotations = Rotation.from_matrix(np.array([r for _, r in history]))
        self.assertLessEqual(np.rad2deg((rotations[1:]*rotations[:-1].inv()).magnitude()).max()/.01,
                             c.config.max_rotation_rate_deg_s+1e-9)

    def test_stale_duplicate_or_large_error_cannot_accumulate_hidden_correction(self):
        c = PoseCorrection()
        now = 1_000_000_000
        c.observe('a', now-2_000_000, now, now, np.zeros(3), np.eye(3), [-.004, 0, 0], np.eye(3))
        self.assertFalse(c.observe('a', now-2_000_000, now, now, np.zeros(3), np.eye(3), [-.004, 0, 0], np.eye(3)))
        c.advance(now)
        c.observe('b', now+48_000_000, now+50_000_000, now+50_000_000, np.zeros(3), np.eye(3), [-.004, 0, 0], np.eye(3))
        c.advance(now+60_000_000)
        before = c.translation.copy()
        c.advance(now+300_000_000)
        np.testing.assert_array_equal(c.translation, before)
        with self.assertRaises(RuntimeError):
            c.observe('c', now+398_000_000, now+400_000_000, now+400_000_000,
                      np.zeros(3), np.eye(3), [-.1, 0, 0], np.eye(3))

    def test_invalid_rotation_or_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            replace(PoseConfig(), gain_per_s=float('nan'))
        c = PoseCorrection()
        with self.assertRaises(ValueError):
            c.observe(0, 0, 0, 0, np.zeros(3), np.eye(3), np.zeros(3), np.diag([1, 1, -1]))

    def test_persistent_error_saturates_at_translation_and_rotation_bounds(self):
        c = PoseCorrection()
        previous_p, previous_r = np.zeros(3), np.eye(3)
        actual_r = Rotation.from_rotvec(np.deg2rad([6, -4, 3])).as_matrix()
        for step in range(2001):
            now = 1_000_000_000+step*10_000_000
            if step % 5 == 0:
                c.observe(step, now-2_000_000, now, now,
                    np.zeros(3), np.eye(3), [-.035, .015, -.01], actual_r)
            p, r = c.advance(now)
            self.assertLessEqual(np.linalg.norm(p-previous_p), .003*.01+1e-12)
            self.assertLessEqual(Rotation.from_matrix(r@previous_r.T).magnitude(), np.deg2rad(.01)+1e-12)
            previous_p, previous_r = p, r
        self.assertAlmostEqual(np.linalg.norm(p), .01, places=9)
        self.assertAlmostEqual(np.rad2deg(Rotation.from_matrix(r).magnitude()), 2., places=8)

    def test_out_of_order_feedback_freezes_without_changing_applied_pose(self):
        c = PoseCorrection()
        for step in range(20):
            now = 1_000_000_000+step*50_000_000
            c.observe(step, now-2_000_000, now, now, np.zeros(3), np.eye(3), [-.005, 0, 0], np.eye(3))
            c.advance(now)
        before_p, before_r = c.translation.copy(), c.rotation.copy()
        self.assertFalse(c.observe('old', now-100_000_000, now-98_000_000, now,
            np.zeros(3), np.eye(3), [-.005, 0, 0], np.eye(3)))
        c.advance(now+10_000_000)
        np.testing.assert_array_equal(c.translation, before_p)
        np.testing.assert_array_equal(c.rotation, before_r)


class TimingCalibrationTests(unittest.TestCase):
    def samples(self):
        ref = CalibrationReference()
        t = np.linspace(0, 20, 4001)
        poses = [ref.target(v, np.zeros(3), np.eye(3)) for v in t]
        mt = np.arange(.4, 19, .05)
        delayed = [ref.target(v-.137, np.zeros(3), np.eye(3)) for v in mt]
        p, r = np.array([a for a, _ in poses]), np.array([b for _, b in poses])
        mp, mr = np.array([a for a, _ in delayed])+[.004, -.002, .003], np.array([b for _, b in delayed])
        return t, p, r, mt, mp, mr

    def test_independent_window_fit_recovers_delay_without_absorbing_static_bias(self):
        p, _ = fit_effective_lag(*self.samples())
        self.assertAlmostEqual(p['effective_preview_s'], .137, places=3)
        self.assertLess(p['validation_after']['centered_position_RMS_mm'], .01)
        self.assertTrue(p['pose_bias_is_not_applied_to_task_targets'])

    def test_measurement_age_is_identifiably_conflated_without_capture_clock(self):
        t, p, r, mt, mp, mr = self.samples()
        result, _ = fit_effective_lag(t, p, r, mt+.05, mp, mr)
        self.assertAlmostEqual(result['effective_preview_s'], .187, places=3)


if __name__ == '__main__':
    unittest.main()
