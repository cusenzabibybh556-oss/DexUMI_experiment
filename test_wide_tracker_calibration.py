"""大范围标定的规划不变量与全左臂反馈分支；测试不连接硬件。"""
import itertools
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from auto_calibrate_tron2_tracker import execute_stages
from tron2_left_ik import LeftArmIK
from tron2_wide_calibration_plan import build_wide_plan


class WideCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.start = np.array([-.621, .0481, -.2261, -1.8779, -1.6004, .3153, -.0239,
                               .1192, .0635, .2507, -.1006, -.1016, -.3374, .08, .949785, .0141])

    def test_wide_coverage_independent_center_and_fixed_other_targets(self):
        targets, stages, report = build_wide_plan(LeftArmIK(), self.start)
        np.testing.assert_array_equal(targets[:, 7:], np.tile(self.start[7:], (len(targets), 1)))
        np.testing.assert_array_equal(targets[-1], self.start)
        self.assertEqual(report['fit_group_indices'], list(range(24)))
        self.assertEqual(report['held_out_group_indices'], list(range(24, 32)))
        self.assertGreater(report['flange_max_pair_distance_m'], .15)
        self.assertGreater(report['maximum_pair_orientation_deg'], 40)
        previous = self.start
        for segments in stages:
            for goal in segments:
                self.assertLessEqual(np.max(abs(goal-previous)), np.deg2rad(5)+1e-10)
                previous = goal

    def test_all_left_joints_allowed_and_return_not_sampled(self):
        goal = self.start.copy()
        goal[0] += .01
        client, session, log = Mock(), Mock(), Mock()
        session.status.return_value = {
            'tracker': {'age_ms': 1, 'tracking_status': 2, 'mac': 'test'},
            'robot': {'age_ms': 1}, 'error': None}
        feedback = [(self.start, np.zeros(16))] + [(goal, np.zeros(16))]*10
        with patch('auto_calibrate_tron2_tracker.current_joints', side_effect=feedback), \
             patch('auto_calibrate_tron2_tracker.time.sleep'), \
             patch('auto_calibrate_tron2_tracker.time.monotonic', side_effect=itertools.count()):
            execute_stages(client, session, self.start, [goal], [[goal]], log,
                           allowed_joints=tuple(range(7)), sample_indices=set())
        client.request.assert_called_once()
        self.assertEqual(client.request.call_args.args[0], 'request_movej')
        self.assertEqual(len(client.request.call_args.args[1]['joint']), 14)
        session.sample.assert_not_called()

    def test_return_repeatability_sample_is_excluded_from_fit(self):
        from auto_calibrate_tron2_tracker_wide import solve_wide_samples
        plan={'sample_indices':list(range(21)), 'fit_group_indices':list(range(15)),
              'held_out_group_indices':list(range(15,20))}
        groups=[SimpleNamespace(id=i) for i in range(1,22)]
        with patch('solve_tron2_tracker_calibration.load_samples', return_value=({},groups,[],'')), \
             patch('auto_calibrate_tron2_tracker_wide.solve_dexumi', return_value={'status':'failed'}) as solve:
            solve_wide_samples('unused.jsonl',plan)
        self.assertEqual(solve.call_args.kwargs['validation_ids'],set(range(16,21)))
        self.assertEqual(solve.call_args.kwargs['analysis_group_ids'],set(range(1,21)))

    def test_interleaved_validation_centers_keep_transit_out_of_fit(self):
        from auto_calibrate_tron2_tracker_wide import solve_wide_samples
        plan = {'sample_indices': list(range(11)), 'fit_group_indices': [1, 2, 5, 6],
                'held_out_group_indices': [3, 4, 7, 8],
                'transit_check_group_indices': [0], 'return_check_group_indices': [9, 10]}
        # 原始第5组跟踪质量不合格：其余独立组仍只能验证，不能调入拟合。
        groups = [SimpleNamespace(id=i) for i in range(1, 12) if i != 5]
        with patch('solve_tron2_tracker_calibration.load_samples', return_value=({}, groups, [], '')), \
             patch('auto_calibrate_tron2_tracker_wide.solve_dexumi', return_value={'status': 'failed'}) as solve:
            solve_wide_samples('unused.jsonl', plan)
        self.assertEqual(solve.call_args.kwargs['validation_ids'], {4, 8, 9})
        self.assertEqual(solve.call_args.kwargs['analysis_group_ids'], set(range(2, 10)))


if __name__ == '__main__':
    unittest.main()
