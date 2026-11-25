"""保留历史离线 IK 核对，并确保它不能再接入真机 ServoJ 回放。"""

import json
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from tron2_left_ik import LeftArmIK
from replay_tron2_fixed_start import (
    FIXED_LEFT_Q, FIXED_LEFT_POSITION, FIXED_LEFT_QUATERNION,
    FIXED_LEFT_REFERENCE_Q, FIXED_LEFT_REFERENCE_POSITION,
    FIXED_LEFT_REFERENCE_QUATERNION, FIXED_LEFT_START_PROFILE,
)
import replay_tron2_tracker as tracker


def q_start():
    return np.r_[FIXED_LEFT_Q, [.0717, -.0344, -.1107, -.18, -.0283, .1779, -.0588, .9503, .0168]]


class RealModelIKTests(unittest.TestCase):
    def setUp(self):
        self.ik = LeftArmIK()

    def test_sdk_tcp_matches_active_profile_and_readonly_reference(self):
        profile = json.loads(FIXED_LEFT_START_PROFILE.read_text())
        np.testing.assert_array_equal(FIXED_LEFT_Q, profile['target_left_q_rad'])
        np.testing.assert_array_equal(FIXED_LEFT_POSITION, profile['target_position_m'])
        np.testing.assert_array_equal(FIXED_LEFT_QUATERNION, profile['target_quaternion_wxyz'])
        self.ik.check_anchor(q_start(), FIXED_LEFT_POSITION, FIXED_LEFT_QUATERNION)
        reference = q_start()
        reference[:7] = FIXED_LEFT_REFERENCE_Q
        p_ref, u_ref = self.ik.pose(reference)
        p_new, u_new = self.ik.pose(q_start())
        np.testing.assert_allclose(p_ref, FIXED_LEFT_REFERENCE_POSITION, atol=1e-6)
        np.testing.assert_allclose(p_new, profile['target_position_m'], atol=1e-6)
        self.assertLess(tracker.quaternion_distance_deg(u_new, profile['target_quaternion_wxyz']), .001)
        self.assertLess(tracker.quaternion_distance_deg(u_ref, FIXED_LEFT_REFERENCE_QUATERNION), .001)
        old = q_start()
        old[:7] = [-.4637, .2639, .1086, -1.1817, -.0158, .1467, -.0234]
        p, _ = self.ik.pose(old)
        np.testing.assert_allclose(p, [.590808, .37168, -.243641], atol=1e-6)

    def test_plan_only_changes_left_joints_and_first_frame_is_exact(self):
        q = q_start()
        t = np.linspace(0, 2, 101)
        ps = FIXED_LEFT_POSITION + np.outer(t/t[-1], [.002, -.004, -.002])
        us = np.tile(FIXED_LEFT_QUATERNION, (len(t), 1))
        plan, stats = self.ik.plan(q, ps, us, t)
        np.testing.assert_array_equal(plan[0], q)
        np.testing.assert_array_equal(plan[:, 7:], np.tile(q[7:], (len(t), 1)))
        self.assertGreater(np.max(np.abs(plan[-1, :7]-q[:7])), 0)
        self.assertLess(stats['max_position_error_mm'], .02)
        self.assertLess(stats['max_joint_speed_rad_s'], .5)

    def test_joint_limit_violation_is_rejected(self):
        q = q_start(); q[0] = 5
        with self.assertRaisesRegex(RuntimeError, '关节越限'):
            self.ik.pose(q)

    def test_robot_fk_mismatch_is_not_silently_calibrated_away(self):
        with self.assertRaisesRegex(RuntimeError, '模型与实机反馈不一致'):
            self.ik.check_anchor(q_start(), FIXED_LEFT_POSITION + [.02, 0, 0], FIXED_LEFT_QUATERNION)

    def test_infeasible_plan_does_not_return_partial_targets(self):
        ps = np.array([FIXED_LEFT_POSITION, FIXED_LEFT_POSITION + [2., 0, 0]])
        with self.assertRaises(RuntimeError):
            self.ik.plan(q_start(), ps, np.tile(FIXED_LEFT_QUATERNION, (2, 1)), [0, 2])

    def test_duplicate_timestamp_is_rejected(self):
        with self.assertRaisesRegex(ValueError, '时间轴无效'):
            self.ik.plan(q_start(), np.tile(FIXED_LEFT_POSITION, (2, 1)),
                         np.tile(FIXED_LEFT_QUATERNION, (2, 1)), [0, 0])


class DisabledJointStreamingTests(unittest.TestCase):
    def test_servoj_is_rejected_before_confirmation_or_any_hardware_operation(self):
        q = q_start()
        anchor = tracker.DualArmPose(
            FIXED_LEFT_POSITION, FIXED_LEFT_QUATERNION,
            np.array([-.0174, -.211, -.7535]), np.array([1., 0., 0., 0.]),
        )
        client = Mock()
        args = SimpleNamespace(arm_control='servoj', countdown=0, feedback_timeout=.5, rate=20)
        hand = Mock()
        with patch('builtins.input') as confirm:
            with self.assertRaisesRegex(ValueError, 'ServoJ.*已停用'):
                tracker.execute_replay(client, anchor, 'left', np.array([0., .01]),
                                       np.zeros((2, 3)), np.tile([1., 0., 0., 0.], (2, 1)),
                                       args, hand_replay=hand)
        confirm.assert_not_called()
        client.assert_not_called()
        self.assertEqual(client.method_calls, [])
        hand.acquire.assert_not_called()
        hand.start_at.assert_not_called()

    def test_servoj_failure_notification_is_not_ignored(self):
        client = object.__new__(tracker.Tron2JsonClient)
        client._condition = threading.Condition()
        client._messages = []
        client._failure = None
        client._pose_probe_guid = None
        client.trace = None
        client._on_message(json.dumps({
            'title': 'notify_servoJ', 'data': {'result': 'fail_invalid_cmd'},
        }))
        self.assertEqual(client._failure, 'fail_invalid_cmd')


if __name__ == '__main__':
    unittest.main()
