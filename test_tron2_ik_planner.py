"""限位余量择优的离线回归；不连接机器人，不发任何运动指令。"""

from contextlib import redirect_stdout, redirect_stderr
import io
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from tron2_left_ik import LeftArmIK
from tron2_ik_planner import (MarginPlanOptions, joint_path_metrics,
                             choose_margin_candidate, plan_with_joint_margin)
import replay_tron2_fixed_start as fixed
from test_replay_tron2_fixed_start import sample_trajectory


class SelectionTests(unittest.TestCase):
    def candidate(self, identifier, values):
        path = np.asarray(values, dtype=float)[:, None]
        return {"id": identifier, "valid": True,
                **joint_path_metrics(path, np.arange(len(path)), [-1.], [1.])}

    def test_worst_frame_margin_wins_over_average_margin(self):
        # A 平均更靠中间，但某一帧逼近上限；应选全程留余量的 B。
        a = self.candidate(0, [0., 0., .99, 0.])
        b = self.candidate(1, [.5, .5, .5, .5])
        self.assertEqual(choose_margin_candidate([a, b])["id"], 1)
        self.assertEqual(a["critical_frame"], 2)

    def test_equal_margin_prefers_smoother_path(self):
        a = self.candidate(0, [.5, -.5, .5])
        b = self.candidate(1, [.5, .5, .5])
        self.assertEqual(choose_margin_candidate([a, b])["id"], 1)

    def test_failed_or_nonfinite_candidate_cannot_win(self):
        failed = self.candidate(0, [0., 0., 0.]); failed["valid"] = False
        nan = self.candidate(1, [0., 0., 0.]); nan["minimum_margin_deg"] = float('nan')
        valid = self.candidate(2, [.5, .5, .5])
        self.assertEqual(choose_margin_candidate([failed, nan, valid])["id"], 2)
        with self.assertRaisesRegex(RuntimeError, "没有通过整段"):
            choose_margin_candidate([failed, nan])

    def test_invalid_time_and_nonfinite_path_rejected(self):
        for q, times in [([[0], [float('nan')]], [0, 1]), ([[0], [0]], [1, 1])]:
            with self.assertRaises(ValueError):
                joint_path_metrics(q, times, [-1], [1])


class ModelTests(unittest.TestCase):
    def test_official_degree_limits_are_not_widened_by_radian_rounding(self):
        model = LeftArmIK()
        self.assertGreaterEqual(model.lower[6], np.deg2rad(-90))
        self.assertLessEqual(model.upper[6], np.deg2rad(90))
        self.assertLessEqual(model.upper[3], np.deg2rad(15))

    def test_same_cartesian_path_multiple_continuous_joint_candidates(self):
        model = LeftArmIK()
        start = np.r_[fixed.FIXED_LEFT_Q, np.zeros(9)]
        times = np.linspace(0, 2, 81)
        ps = fixed.FIXED_LEFT_POSITION + np.outer(times/2, [.002, -.004, -.002])
        us = np.tile(fixed.FIXED_LEFT_QUATERNION, (len(times), 1))
        expected = ps.copy()
        plan, report = plan_with_joint_margin(model, start, ps, us, times)
        valid = [c for c in report['candidates'] if c['valid']]
        self.assertGreaterEqual(len(valid), 2)
        self.assertEqual(report['minimum_margin_deg'], max(c['minimum_margin_deg'] for c in valid))
        np.testing.assert_array_equal(plan[0], start)
        np.testing.assert_array_equal(plan[:, 7:], np.tile(start[7:], (len(times), 1)))
        np.testing.assert_array_equal(ps, expected)
        self.assertLessEqual(report['max_joint_speed_rad_s'], .5 + 1e-6)
        self.assertLess(report['max_position_error_mm'], .02)
        self.assertLess(report['max_angle_error_deg'], .05)
        self.assertFalse(report['firmware_branch_enforced'])
        # 独立逐帧正解核对，防止只检查优化器的 success 标志。
        for q, p in zip(plan, ps):
            actual, _ = model.pose(q)
            self.assertLess(np.linalg.norm(actual-p), 1e-5)

    def test_180_degree_rotation_is_not_mistaken_for_zero_error(self):
        from replay_tron2_tracker import matrix_to_quaternion, quaternion_to_matrix
        model = LeftArmIK()
        start = np.r_[fixed.FIXED_LEFT_Q, np.zeros(9)]
        rotation = quaternion_to_matrix(fixed.FIXED_LEFT_QUATERNION)
        opposite = matrix_to_quaternion(np.diag([-1., -1., 1.]) @ rotation)
        with self.assertRaisesRegex(RuntimeError, "没有通过整段"):
            plan_with_joint_margin(model, start, np.tile(fixed.FIXED_LEFT_POSITION, (2, 1)),
                                   [fixed.FIXED_LEFT_QUATERNION, opposite], [0., .001],
                                   MarginPlanOptions(motion_weights=(.1,)))

    def test_unreachable_path_does_not_return_a_partial_plan(self):
        model = LeftArmIK()
        with self.assertRaisesRegex(RuntimeError, "没有通过整段"):
            plan_with_joint_margin(model, np.r_[fixed.FIXED_LEFT_Q, np.zeros(9)],
                                   [fixed.FIXED_LEFT_POSITION, [3., 2., 3.]],
                                   np.tile(fixed.FIXED_LEFT_QUATERNION, (2, 1)), [0., 1.],
                                   MarginPlanOptions(motion_weights=(.1,)))

    def test_cannot_improve_margin_by_changing_the_first_joint_pose(self):
        model = LeftArmIK()
        start = np.r_[fixed.FIXED_LEFT_Q, np.zeros(9)]
        # 起点约 32° 的腕旋转余量不能靠偷偷换首帧来宣称全程超过 40°。
        with self.assertRaisesRegex(RuntimeError, "没有通过整段"):
            plan_with_joint_margin(model, start, np.tile(fixed.FIXED_LEFT_POSITION, (2, 1)),
                                   np.tile(fixed.FIXED_LEFT_QUATERNION, (2, 1)), [0., .1],
                                   MarginPlanOptions(min_margin_deg=40, motion_weights=(.1,)))


class OfflineEntryTests(unittest.TestCase):
    def test_planning_and_execution_cannot_be_combined(self):
        for flag in ('--execute', '--check-robot'):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                fixed.parse_args(['--episode', '65', '--plan-ik', flag])

    def test_custom_planning_options_cannot_silently_change_hardware_mode(self):
        args = fixed.parse_args(['--episode', '65', '--execute', '--ik-min-margin-deg', '5'])
        with self.assertRaisesRegex(ValueError, '只用于 --plan-ik'):
            fixed.validate_args(args)

    def test_cli_saves_selected_plan_without_any_robot_or_hand_connection(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'plan.json'
            with patch.object(fixed, 'load_tracker', return_value=sample_trajectory()), \
                    patch.object(fixed, 'Tron2JsonClient') as robot, \
                    patch.object(fixed, 'O6ReplayBridge') as bridge, \
                    redirect_stdout(io.StringIO()):
                code = fixed.main(['--episode', '65', '--plan-ik', '--ik-report', str(path)])
            self.assertEqual(code, 0)
            robot.assert_not_called(); bridge.assert_not_called()
            import json
            report = json.loads(path.read_text())
            self.assertEqual(report['hardware_motion_commands_sent'], 0)
            self.assertFalse(report['firmware_branch_enforced'])
            self.assertTrue(report['first_frame_unchanged'])
            with np.load(path.with_suffix('.npz')) as data:
                self.assertEqual(data['offline_left_q_rad'].shape[1], 7)
                np.testing.assert_array_equal(data['offline_left_q_rad'][0], fixed.FIXED_LEFT_Q)
            # 相同路径拒绝覆盖，并且仍不触碰硬件。
            with patch.object(fixed, 'load_tracker', return_value=sample_trajectory()), \
                    patch.object(fixed, 'Tron2JsonClient') as robot, \
                    redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(fixed.main(['--episode', '65', '--plan-ik', '--ik-report', str(path)]), 2)
            robot.assert_not_called()


if __name__ == '__main__':
    unittest.main()
