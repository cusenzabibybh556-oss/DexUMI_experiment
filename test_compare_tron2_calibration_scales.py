"""纯离线几何与数据关联测试，不导入/初始化机器人或 Tracker SDK。"""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from compare_tron2_calibration_scales import analyze, cross_validate, pair_diagnostic
from solve_tron2_tracker_calibration import PoseGroup, transform, inverse


class ScaleComparisonTests(unittest.TestCase):
    def setUp(self):
        self.C = transform(np.eye(3), [.0338, 0, .2125])
        self.X = transform(Rotation.from_euler('xyz', [12, -25, 37], degrees=True).as_matrix(), [.04, -.02, .07])
        self.Y = transform(Rotation.from_euler('xyz', [-35, 50, 40], degrees=True).as_matrix(), [1, 2, -.3])
        self.calibration = {"T_FT": self.X.tolist(), "T_ET": (self.C @ self.X).tolist(),
                            "T_WB": self.Y.tolist(), "T_BW": inverse(self.Y).tolist(),
                            "tracker_quaternion_convention": "raw-xyzw"}

    def group(self, gid, p, rot_deg=0, drift=None, statuses=None):
        F = transform(Rotation.from_euler('y', rot_deg, degrees=True).as_matrix(), p)
        Q = self.Y @ F @ self.X
        if drift is not None:
            Q[:3, 3] += drift
        return PoseGroup(gid, 'synthetic', F @ inverse(self.C), Q, statuses or [2], 10)

    def test_exact_mapping_and_small_rotation_lever_removal(self):
        a, b = self.group(1, [0, 0, 0]), self.group(2, [.2, .1, .05], 2)
        row = pair_diagnostic(a, b, self.C, self.X, self.Y)
        self.assertAlmostEqual(row['corrected_distance_ratio_tracker_over_robot'], 1, places=12)
        self.assertNotAlmostEqual(row['raw_distance_ratio_tracker_over_robot'], 1, places=6)
        self.assertLess(row['relative_rotation_mapping_error_deg'], 1e-10)
        self.assertLess(row['unexplained_tracker_translation_norm_mm'], 1e-10)

    def test_raw_ratio_direction(self):
        a, b = self.group(1, [0, 0, 0]), self.group(2, [.3, 0, 0])
        b.tracker[:3, 3] = a.tracker[:3, 3] + 1.05 * (b.tracker[:3, 3] - a.tracker[:3, 3])
        row = pair_diagnostic(a, b, self.C, self.X, self.Y)
        self.assertAlmostEqual(row['raw_distance_ratio_tracker_over_robot'], 1.05)
        self.assertAlmostEqual(row['corrected_distance_ratio_tracker_over_robot'], 1.05)

    def test_return_drift_no_zero_baseline_ratio(self):
        a, b = self.group(1, [0, 0, 0]), self.group(2, [.0002, 0, 0], .1, [.003, 0, 0], [18])
        row = pair_diagnostic(a, b, self.C, self.X, self.Y)
        self.assertAlmostEqual(row['unexplained_tracker_translation_norm_mm'], 3)
        self.assertFalse(row['ratio_usable'])
        self.assertIsNone(row['raw_distance_ratio_tracker_over_robot'])
        self.assertEqual(row['raw_tracking_statuses'], [[2], [18]])

    def test_indices_map_by_event_order_and_separate_roles(self):
        groups = [self.group(10, [0, 0, 0]), self.group(30, [.2, 0, 0], statuses=[18]),
                  self.group(50, [.2, .1, 0]), self.group(70, [0, 0, 0])]
        plan = {'sample_count': 4, 'same_orientation_group_indices': [0, 1, 2, 3],
                'groups': [{'center': 0, 'role': 'fit'}, {'center': 1, 'role': 'fit'},
                           {'center': 2, 'role': 'held_out'}, {'center': 0, 'role': 'return_check'}]}
        result = analyze(groups, [10, 30, 50, 70], plan, self.calibration, [-.0338, 0, -.2125])
        self.assertEqual(result['training_center_pairs'][0]['group_ids'], [10, 30])
        self.assertEqual(len(result['independent_center_pairs']), 2)
        self.assertEqual(result['return_checks'][0]['group_ids'], [10, 70])
        self.assertEqual(result['ratio_summaries']['training_center_pairs']['status18_pair_count'], 1)

    def test_incomplete_sample_plan_rejected(self):
        with self.assertRaisesRegex(ValueError, '完成样本数'):
            analyze([], [], {'sample_count': 1, 'groups': [{}]}, self.calibration, [-.0338, 0, -.2125])

    def test_transit_not_classified_as_training_but_can_be_return_reference(self):
        groups = [self.group(1, [0, 0, -.1]), self.group(2, [0, 0, 0]),
                  self.group(3, [.2, 0, 0]), self.group(4, [0, 0, -.1])]
        plan = {'sample_count': 4, 'same_orientation_group_indices': [0, 1, 2, 3],
                'groups': [{'center': -1, 'role': 'transit_check'}, {'center': 0, 'role': 'fit'},
                           {'center': 1, 'role': 'held_out'}, {'center': -1, 'role': 'return_check'}]}
        result = analyze(groups, [1, 2, 3, 4], plan, self.calibration, [-.0338, 0, -.2125])
        self.assertEqual(result['training_center_pairs'], [])
        self.assertEqual(len(result['independent_center_pairs']), 1)
        self.assertEqual(result['independent_center_pairs'][0]['group_ids'], [2, 3])
        self.assertEqual(result['return_checks'][0]['group_ids'], [1, 4])

    def test_inconsistent_tcp_transform_rejected(self):
        invalid = {**self.calibration, 'T_ET': np.eye(4).tolist()}
        with self.assertRaisesRegex(ValueError, 'TCP 转换不一致'):
            analyze([], [], {'groups': []}, invalid, [-.0338, 0, -.2125])

    def test_cross_manifest_requires_continuity_and_separates_status18(self):
        meta = {'arm': 'left', 'robot_id': 'robot', 'tracker_index': 0,
                'tracker_transport': {'mac': 'mac'}, 'robot_kinematics': {
                    'model_sha256': 'model', 'base_frame': 'base', 'wrist_frame': 'wrist', 'tcp_offset_m': [0, 0, 0]}}
        groups = [self.group(1, [0, 0, 0]), self.group(2, [.2, 0, 0], statuses=[18])]
        with tempfile.TemporaryDirectory() as d:
            main = Path(d) / 'main.jsonl'
            manifest = Path(d) / 'cross.json'
            manifest.write_text(json.dumps({'same_map_mount_as': str(main), 'evidence': 'test attestation',
                                            'datasets': [{'input': 'previous.jsonl', 'group_ids': [1, 2]}]}))
            with patch('compare_tron2_calibration_scales.load_diagnostic', return_value=(meta, groups, [], 'hash', [1, 2])):
                result = cross_validate(manifest, main, meta, self.calibration)
            self.assertEqual(result[0]['metrics_status2_only']['count'], 1)
            self.assertEqual(result[0]['metrics_including_status18_diagnostic']['count'], 2)
            manifest.write_text(json.dumps({'same_map_mount_as': str(main), 'datasets': []}))
            with self.assertRaisesRegex(ValueError, '连续性说明'):
                cross_validate(manifest, main, meta, self.calibration)


if __name__ == '__main__':
    unittest.main()
