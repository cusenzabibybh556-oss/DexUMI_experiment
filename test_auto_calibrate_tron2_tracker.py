"""自动标定的计划和反馈门槛测试；不连接硬件。"""
import unittest
from unittest.mock import Mock
import numpy as np
from auto_calibrate_tron2_tracker import build_plan, execute_stages
from tron2_left_ik import LeftArmIK


class AutoCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.start = np.array([-.498,.426,-.288,-1.141,-.549,-.449,-.023,
                               -.056,-.030,.282,-.01,-.006,-.002,.085,.95,.014])

    def test_model_plan_changes_only_left_wrist_and_returns_to_start(self):
        targets, stages, report = build_plan(LeftArmIK(), self.start)
        np.testing.assert_array_equal(targets[-1], self.start)
        fixed = [0,1,2,3,7,8,9,10,11,12,13,14,15]
        np.testing.assert_array_equal(targets[:,fixed], np.tile(self.start[fixed], (16,1)))
        previous = self.start
        for segments in stages:
            for goal in segments:
                self.assertLessEqual(np.max(abs(goal-previous)), np.deg2rad(5)+1e-10)
                previous = goal
        self.assertGreater(report['minimum_wrist_joint_margin_deg'],5)

    def test_movej_document_limit_is_not_overridden_by_model(self):
        self.start[6] = 1.0  # 通用模型允许，但 MoveJ 文档腕 roll 上界为0.7505。
        with self.assertRaisesRegex(ValueError, 'MoveJ'):
            build_plan(LeftArmIK(), self.start)

    def test_independent_validation_small_moves_return_to_start(self):
        from validate_tron2_tracker_mapping import plan
        targets,stages,span=plan(LeftArmIK(),self.start)
        np.testing.assert_array_equal(targets[-1],self.start)
        fixed=[2,4,5,6,7,8,9,10,11,12,13,14,15]
        np.testing.assert_array_equal(targets[:,fixed],np.tile(self.start[fixed],(8,1)))
        self.assertLessEqual(np.max(abs(targets-self.start)),np.deg2rad(2)+1e-12)
        self.assertLessEqual(span,.03)

    def test_no_tracker_prevents_any_move(self):
        session = Mock()
        session.status.return_value = {'tracker':None,'robot':None,'error':''}
        client = Mock()
        with self.assertRaises(RuntimeError):
            execute_stages(client, session, self.start, [self.start], [[self.start]], Mock())
        client.assert_not_called()
        self.assertEqual(client.method_calls, [])


if __name__ == '__main__':
    unittest.main()
