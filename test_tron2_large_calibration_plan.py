"""大范围标定规划的纯离线检查；不导入硬件客户端。"""
import json
import math
from pathlib import Path
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from tron2_left_ik import LeftArmIK
from tron2_large_calibration_plan import build_large_plan, _flange, _collision


class LargeCalibrationPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        snapshot = Path(__file__).parent / "validation_results/large_calibration_start_20261027_223148.json"
        cls.start = np.array(json.loads(snapshot.read_text())["joint"]["data"]["q"])
        cls.model = LeftArmIK()
        cls.targets, cls.stages, cls.report = build_large_plan(cls.model, cls.start)

    def test_roles_and_unchanged_other_joints(self):
        r = self.report
        self.assertEqual(r["sample_count"], 21)
        self.assertEqual(r["sample_indices"], list(range(21)))
        self.assertEqual(r["fit_group_indices"], list(range(15)))
        self.assertEqual(r["held_out_group_indices"], list(range(15, 20)))
        self.assertEqual(r["groups"][20]["role"], "return_check")
        np.testing.assert_array_equal(self.targets[-1], self.start)
        for target in self.targets:
            np.testing.assert_array_equal(target[7:], self.start[7:])

    def test_pose_coverage_and_shared_orientation(self):
        p0, r0 = _flange(self.model, self.start)
        for i in self.report["same_orientation_group_indices"]:
            p, r = _flange(self.model, self.targets[i])
            self.assertLess(Rotation.from_matrix(r0 @ r.T).magnitude(), math.radians(.15))
        for i, q in enumerate(self.targets):
            p, r = _flange(self.model, q)
            g = self.report["groups"][i]
            expected_p = p0 + np.array(g["center_offset_m"])
            expected_r = r0 @ Rotation.from_rotvec(np.deg2rad(g["rotation_vector_deg"])).as_matrix()
            self.assertLess(np.linalg.norm(p-expected_p), .0003)
            self.assertLess(Rotation.from_matrix(expected_r @ r.T).magnitude(), math.radians(.15))
        self.assertGreater(self.report["flange_max_pair_distance_m"], .30)
        self.assertGreater(self.report["maximum_pair_orientation_deg"], 60.)

    def test_limits_steps_and_collision_along_all_segments(self):
        lower, upper = self.model.lower.copy(), self.model.upper.copy()
        lower[[5, 6, 12, 13]] = np.maximum(lower[[5, 6, 12, 13]], -.7505)
        upper[[5, 6, 12, 13]] = np.minimum(upper[[5, 6, 12, 13]], .7505)
        previous = self.start.copy()
        for target, segments in zip(self.targets, self.stages):
            for goal in segments:
                self.assertLessEqual(np.max(abs(goal-previous)), math.radians(5)+1e-10)
                np.testing.assert_array_equal(goal[7:], self.start[7:])
                for a in np.linspace(0, 1, 11):
                    q = previous+(goal-previous)*a
                    self.assertTrue(np.all(q >= lower-1e-10))
                    self.assertTrue(np.all(q <= upper+1e-10))
                    self.assertGreaterEqual(np.minimum(q[:7]-lower[:7], upper[:7]-q[:7]).min(), math.radians(5)-1e-10)
                    self.assertIsNone(_collision(self.model, q))
                previous = goal
            np.testing.assert_allclose(previous, target, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
