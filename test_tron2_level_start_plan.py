"""只在本机模型验证所需连杆几何和完整分段路径；不连接机器人。"""
import json
import math
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from tron2_left_ik import LeftArmIK
from tron2_large_calibration_plan import _collision
from tron2_level_start_plan import (
    NEUTRAL_FOREARM_HEADING_RAD, NEUTRAL_FOREARM_ELEVATION_DEG,
    build_level_start_plan, geometry, geometry_angles,
)


class LevelStartPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 固定使用上一轮已保存的实测结果，避免读取持续变化的实时状态或活动常量。
        path = Path(__file__).parent/'validation_results/neutral_start_measured_20261028/fixed_start_profile.json'
        cls.profile = json.loads(path.read_text())
        cls.start = np.asarray(cls.profile['measured_result']['final_q_rad'])
        cls.model = LeftArmIK()
        cls.target, cls.stages, cls.report = build_level_start_plan(cls.model, cls.start)

    def test_forearm_up_five_preserves_upper_vertical_and_neutral_wrist(self):
        before = geometry(self.model, self.start)
        after = geometry(self.model, self.target)
        np.testing.assert_allclose(after['upper_arm_axis'], [0., 0., -1.], atol=1e-7)
        self.assertAlmostEqual(after['forearm_axis'][2], math.sin(math.radians(5.)), places=10)
        self.assertGreaterEqual(after['forearm_axis'][0], .9-1e-7)
        np.testing.assert_allclose(after['hand_extension_axis'], after['forearm_axis'], atol=1e-7)
        down = np.array([0., 0., -1.])
        projected = down-np.dot(down, after['forearm_axis'])*after['forearm_axis']
        np.testing.assert_allclose(after['palm_normal'], projected/np.linalg.norm(projected), atol=1e-7)
        self.assertAlmostEqual(geometry_angles(after)['forearm_yaw_deg'],
                               math.degrees(NEUTRAL_FOREARM_HEADING_RAD), places=8)
        self.assertLess(geometry_angles(after)['hand_forearm_alignment_deg'], 1e-5)
        self.assertAlmostEqual(geometry_angles(after)['forearm_elevation_deg'], 5., places=8)
        self.assertAlmostEqual(geometry_angles(after)['hand_elevation_deg'], 5., places=8)
        self.assertAlmostEqual(geometry_angles(after)['palm_down_error_deg'], 5., places=8)
        self.assertLess(geometry_angles(after)['palm_roll_error_deg'], 1e-5)
        self.assertTrue(geometry_angles(after)['palm_roll_reference_valid'])
        # 改变小臂仰角时整只中立的手随小臂倾斜，而非另弯腕关节。
        error = Rotation.from_matrix(after['rotation']@before['rotation'].T).magnitude()
        self.assertGreater(math.degrees(error), 3.)
        self.assertLess(math.degrees(error), 7.)
        self.assertFalse(self.report['previous_full_flange_orientation_preserved'])

    def test_whole_path_limits_collision_and_held_joints(self):
        previous = self.start
        lower, upper = self.model.lower.copy(), self.model.upper.copy()
        lower[[5, 6, 12, 13]] = np.maximum(lower[[5, 6, 12, 13]], -.7505)
        upper[[5, 6, 12, 13]] = np.minimum(upper[[5, 6, 12, 13]], .7505)
        for target in self.stages:
            np.testing.assert_array_equal(target[7:], self.start[7:])
            self.assertLessEqual(np.max(np.abs(target-previous)), math.radians(5)+1e-10)
            for alpha in np.linspace(0., 1., 11):
                q = previous+(target-previous)*alpha
                self.assertGreaterEqual(np.minimum(q[:7]-lower[:7], upper[:7]-q[:7]).min(), math.radians(5)-1e-8)
                self.assertIsNone(_collision(self.model, q))
            previous = target
        np.testing.assert_array_equal(previous, self.target)

    def test_sdk_pose_and_motion_are_consistent_with_new_neutral_target(self):
        delta = np.asarray(self.report['sdk_position_change_m'])
        np.testing.assert_allclose(np.asarray(self.report['start_sdk_position_m'])+delta,
                                   self.report['target_sdk_position_m'], atol=1e-12)
        np.testing.assert_allclose(self.report['target_sdk_position_m'],
                                   [.44942867072761283, .2530768733774639, -.2661927691193583], atol=1e-8)
        model_position, model_quaternion = self.model.pose(self.target)
        np.testing.assert_allclose(model_position, self.report['target_sdk_position_m'], atol=1e-12)
        np.testing.assert_allclose(model_quaternion, self.report['target_sdk_quaternion_wxyz'], atol=1e-12)
        np.testing.assert_allclose(self.target[5:7], [0., 0.], atol=1e-12)
        self.assertFalse(self.report['hardware_connected'])
        self.assertFalse(self.report['motion_sent'])
        json.dumps(self.report, allow_nan=False)

    def test_exact_neutral_zero_to_five_moves_only_elbow(self):
        neutral = self.start.copy()
        neutral[:7] = self.profile['target_left_q_rad']
        target, stages, report = build_level_start_plan(LeftArmIK(), neutral)
        changed = np.flatnonzero(np.abs(target-neutral) > 1e-10)
        np.testing.assert_array_equal(changed, [3])
        self.assertAlmostEqual(math.degrees(target[3]-neutral[3]), -5., places=8)
        self.assertEqual(len(stages), 1)
        previous = neutral
        for stage in stages:
            self.assertLessEqual(np.max(np.abs(stage-previous)), math.radians(5)+1e-10)
            previous = stage
        self.assertEqual(report['forearm_elevation_deg'], NEUTRAL_FOREARM_ELEVATION_DEG)

    def test_historical_ten_geometry_and_return_to_five_path(self):
        path = Path(__file__).parent/'validation_results/forearm_up10_measured_20261028/fixed_start_profile.json'
        history = json.loads(path.read_text())
        previous_ten = np.asarray(history['measured_result']['final_q_rad']).copy()
        previous_ten[:7] = history['target_left_q_rad']
        model = LeftArmIK()
        old_angles = geometry_angles(geometry(model, previous_ten))
        self.assertAlmostEqual(old_angles['forearm_elevation_deg'], 10., places=8)
        self.assertAlmostEqual(old_angles['palm_down_error_deg'], 10., places=8)
        self.assertLess(old_angles['palm_roll_error_deg'], 1e-5)
        old_position, old_quaternion = model.pose(previous_ten)
        np.testing.assert_allclose(old_position, history['target_position_m'], atol=1e-8)
        self.assertLess((Rotation.from_quat(old_quaternion[[1, 2, 3, 0]])*
                        Rotation.from_quat(np.asarray(history['target_quaternion_wxyz'])[[1, 2, 3, 0]]).inv()).magnitude(), 1e-8)
        target, stages, report = build_level_start_plan(model, previous_ten)
        np.testing.assert_array_equal(np.flatnonzero(np.abs(target-previous_ten)>1e-10), [3])
        self.assertAlmostEqual(math.degrees(target[3]-previous_ten[3]), 5., places=8)
        self.assertAlmostEqual(report['target_angles_deg']['forearm_elevation_deg'], 5., places=8)
        previous = previous_ten
        for stage in stages:
            np.testing.assert_array_equal(stage[4:], previous_ten[4:])
            self.assertLessEqual(np.max(np.abs(stage-previous)), math.radians(5)+1e-10)
            for alpha in np.linspace(0., 1., 11):
                self.assertIsNone(_collision(model, previous+(stage-previous)*alpha))
            previous = stage

    def test_palm_roll_is_separate_from_required_five_degree_tilt(self):
        q = self.target.copy()
        q[4] += math.radians(7.)
        angles = geometry_angles(geometry(LeftArmIK(), q))
        self.assertAlmostEqual(angles['palm_roll_error_deg'], 7., places=8)
        self.assertAlmostEqual(angles['hand_forearm_alignment_deg'], 0., places=5)

    def test_vertical_forearm_has_no_gravity_based_roll_reference(self):
        g = geometry(LeftArmIK(), np.zeros(16))
        angles = geometry_angles(g)
        self.assertFalse(angles['palm_roll_reference_valid'])
        self.assertIsNone(angles['palm_roll_error_deg'])
        json.dumps(angles, allow_nan=False)

    def test_current_hanging_pose_near_limit_still_rejects_planning(self):
        path = Path(__file__).parent/'validation_results/forearm_up5_current_20261028.json'
        hanging = np.asarray(json.loads(path.read_text())['q_rad'])
        with self.assertRaisesRegex(ValueError, '没有保留5°关节余量'):
            build_level_start_plan(LeftArmIK(), hanging)

    def test_wrist_axial_rotation_has_ten_degrees_of_limit_margin(self):
        self.assertAlmostEqual(math.degrees(self.target[4]), -90.)
        # 这是静态中立目标的余量，不能视为整条回放轨迹的限位验收。
        self.assertAlmostEqual(math.degrees(self.target[4]-self.model.lower[4]), 10., places=8)

    def test_new_target_does_not_follow_encoder_heading_drift(self):
        drifting = self.start.copy()
        drifting[2] += .003
        target, _, report = build_level_start_plan(LeftArmIK(), drifting)
        self.assertEqual(target[2], NEUTRAL_FOREARM_HEADING_RAD)
        self.assertAlmostEqual(report['target_angles_deg']['forearm_yaw_deg'],
                               math.degrees(NEUTRAL_FOREARM_HEADING_RAD), places=8)

    def test_model_axis_definition_mismatch_is_rejected(self):
        model = LeftArmIK()
        old = geometry(model, self.start)
        wrong = geometry(model, self.target)
        wrong['hand_extension_axis'] = np.array([0., 0., 1.])
        with patch('tron2_level_start_plan.geometry', side_effect=[old, wrong]):
            with self.assertRaisesRegex(ValueError, '模型连杆轴或法兰定义不一致'):
                build_level_start_plan(model, self.start)

    def test_incorrect_feedback_or_limits_are_rejected(self):
        for bad in (np.zeros(14), np.full(16, np.nan)):
            with self.assertRaises((ValueError, RuntimeError)):
                build_level_start_plan(LeftArmIK(), bad)
        bad = self.start.copy()
        bad[5] = 1.
        with self.assertRaises((ValueError, RuntimeError)):
            build_level_start_plan(LeftArmIK(), bad)
        near_limit = self.start.copy()
        near_limit[4] = math.radians(-98.)
        with self.assertRaisesRegex(ValueError, '没有保留5°关节余量'):
            build_level_start_plan(LeftArmIK(), near_limit)


if __name__ == '__main__':
    unittest.main()
