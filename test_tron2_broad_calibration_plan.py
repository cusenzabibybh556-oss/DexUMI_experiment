"""仅离线验证广域标定规划，包含低起点抬升及原路返回。"""
import json
import math
from pathlib import Path
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from tron2_left_ik import LeftArmIK
from tron2_large_calibration_plan import _flange, _collision
from tron2_broad_calibration_plan import build_broad_plan


class BroadCalibrationPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        snapshot = Path(__file__).parent / "validation_results/broad_calibration_start_20261027_225720.json"
        cls.start = np.array(json.loads(snapshot.read_text())["joint"]["data"]["q"])
        cls.model = LeftArmIK()
        cls.targets, cls.stages, cls.report = build_broad_plan(cls.model, cls.start)

    def test_split_and_return(self):
        r = self.report
        self.assertEqual(r["sample_count"], 36)
        self.assertEqual(len(r["fit_group_indices"]), 24)
        self.assertEqual(len(r["held_out_group_indices"]), 9)
        self.assertEqual(len(r["return_check_group_indices"]), 2)
        self.assertEqual(r["transit_check_group_indices"], [0])
        fit_centers = {r["groups"][i]["center"] for i in r["fit_group_indices"]}
        hold_centers = {r["groups"][i]["center"] for i in r["held_out_group_indices"]}
        self.assertFalse(fit_centers & hold_centers)
        self.assertEqual(len(hold_centers), 3)
        np.testing.assert_array_equal(self.targets[0], self.start)
        np.testing.assert_array_equal(self.targets[-1], self.start)

    def test_full_pose_targets_and_excitation(self):
        p0 = np.array(self.report["calibration_origin_flange_m"])
        r0 = _flange(self.model, self.start)[1]
        for q, group in zip(self.targets, self.report["groups"]):
            p, r = _flange(self.model, q)
            expected_p = p0+group["center_offset_m"]
            expected_r = r0 @ Rotation.from_rotvec(np.deg2rad(group["rotation_vector_deg"])).as_matrix()
            self.assertLess(np.linalg.norm(p-expected_p), .0003)
            self.assertLess(Rotation.from_matrix(r @ expected_r.T).magnitude(), math.radians(.15))
        self.assertGreater(self.report["flange_max_pair_distance_m"], .30)
        self.assertGreater(self.report["maximum_pair_orientation_deg"], 60.)
        self.assertEqual(self.report["fit_rotation_vector_rank"], 3)
        self.assertGreater(min(self.report["fit_pure_rotation_axis_sample_counts"].values()), 0)
        json.dumps(self.report)

    def test_all_actual_segments(self):
        lower, upper = self.model.lower.copy(), self.model.upper.copy()
        lower[[5, 6, 12, 13]] = np.maximum(lower[[5, 6, 12, 13]], -.7505)
        upper[[5, 6, 12, 13]] = np.minimum(upper[[5, 6, 12, 13]], .7505)
        start_p = _flange(self.model, self.start)[0]
        previous = self.start.copy()
        for i, (target, stages) in enumerate(zip(self.targets, self.stages)):
            minimum_z = start_p[2]-.005 if i <= 1 or i == len(self.targets)-1 else -.10
            for goal in stages:
                self.assertLessEqual(np.max(abs(goal-previous)), math.radians(5)+1e-10)
                for alpha in np.linspace(0, 1, 11):
                    q = previous+(goal-previous)*alpha
                    np.testing.assert_array_equal(q[7:], self.start[7:])
                    self.assertGreaterEqual(np.minimum(q[:7]-lower[:7], upper[:7]-q[:7]).min(), math.radians(5)-1e-8)
                    self.assertIsNone(_collision(self.model, q))
                    self.assertGreaterEqual(_flange(self.model, q)[0][2], minimum_z)
                previous = goal
            np.testing.assert_allclose(previous, target, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
