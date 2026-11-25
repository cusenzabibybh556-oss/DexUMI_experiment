"""最终验收轨迹离线测试；不实例化任何设备客户端。"""
import json
import math
from pathlib import Path
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from tron2_left_ik import LeftArmIK
from tron2_large_calibration_plan import _flange, _collision
from tron2_final_calibration_validation_plan import build_validation_plan


class FinalCalibrationValidationPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).parent
        run = root / "tracker_calibrations/broad_20261027_230150_0285f444"
        cls.start = np.array(json.loads((run/"final_readonly_check.json").read_text())["robot_state"]["data"]["q"])
        cls.baseline = json.loads((run/"plan.json").read_text())
        cls.reference = np.array(cls.baseline["start_q_rad"])
        cls.model = LeftArmIK()
        cls.targets, cls.stages, cls.report = build_validation_plan(cls.model, cls.start, cls.reference)

    def test_independent_targets_and_frozen_reference(self):
        self.assertEqual(self.report["validation_target_indices"], list(range(2, 11)))
        self.assertEqual(self.report["sample_indices"], list(range(13)))
        self.assertEqual(self.report["fit_group_indices"], [])
        r0 = _flange(self.model, self.reference)[1]
        old_positions = np.array(self.baseline["flange_positions_m"])
        for i in self.report["validation_target_indices"]:
            g = self.report["groups"][i]
            p, r = _flange(self.model, self.targets[i])
            expected_r = r0 @ Rotation.from_rotvec(np.deg2rad(g["rotation_vector_deg"])).as_matrix()
            self.assertLess(np.linalg.norm(p-g["flange_position_base_m"]), .0003)
            self.assertLess(Rotation.from_matrix(r @ expected_r.T).magnitude(), math.radians(.15))
            self.assertGreater(np.linalg.norm(old_positions-p, axis=1).min(), .01)
            self.assertEqual(np.count_nonzero(g["rotation_vector_deg"]), 3)
        json.dumps(self.report)

    def test_every_actual_segment_and_return(self):
        lo, hi = self.model.lower.copy(), self.model.upper.copy()
        lo[[5, 6, 12, 13]] = np.maximum(lo[[5, 6, 12, 13]], -.7505)
        hi[[5, 6, 12, 13]] = np.minimum(hi[[5, 6, 12, 13]], .7505)
        start_p = _flange(self.model, self.start)[0]
        previous = self.start.copy()
        for i, (target, stages) in enumerate(zip(self.targets, self.stages)):
            minimum_z = start_p[2]-.005 if i <= 1 or i == len(self.targets)-1 else -.10
            for goal in stages:
                self.assertLessEqual(np.max(abs(goal-previous)), math.radians(5)+1e-10)
                for alpha in np.linspace(0, 1, 11):
                    q = previous+(goal-previous)*alpha
                    np.testing.assert_array_equal(q[7:], self.start[7:])
                    self.assertGreaterEqual(np.minimum(q[:7]-lo[:7], hi[:7]-q[:7]).min(), math.radians(5)-1e-8)
                    self.assertIsNone(_collision(self.model, q))
                    self.assertGreaterEqual(_flange(self.model, q)[0][2], minimum_z)
                previous = goal
            np.testing.assert_allclose(previous, target, atol=1e-12)
        np.testing.assert_array_equal(self.targets[-1], self.start)
        self.assertLessEqual(self.report["planned_motion_seconds"], 180)


if __name__ == "__main__":
    unittest.main()
