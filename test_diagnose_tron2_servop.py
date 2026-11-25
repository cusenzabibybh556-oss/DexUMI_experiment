"""Offline tests only: simulated poses/clients, no robot or Tracker connection."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from scipy.spatial.transform import Rotation

from tools import diagnose_tron2_servop as diagnostic
from replay_tron2_tracker import DualArmPose


def pose(*, left=None, left_quaternion=None, right=None, right_quaternion=None):
    return DualArmPose(
        np.array([.4, .2, -.3] if left is None else left, dtype=float),
        np.array([1., 0., 0., 0.] if left_quaternion is None else left_quaternion, dtype=float),
        np.array([.3, -.4, -.35] if right is None else right, dtype=float),
        np.array([1., 0., 0., 0.] if right_quaternion is None else right_quaternion, dtype=float),
    )


def rotation(axis, angle):
    return Rotation.from_euler(axis, angle, degrees=True).as_quat()[[3, 0, 1, 2]]


class TargetTests(unittest.TestCase):
    def setUp(self):
        self.anchor = pose(left_quaternion=rotation('z', 23), right_quaternion=rotation('x', -17))
        self.times, self.left, self.right, self.phases = diagnostic.make_targets(self.anchor)

    def at(self, seconds):
        return int(np.argmin(abs(self.times-seconds)))

    def test_five_three_second_segments_have_expected_endpoints(self):
        self.assertEqual(len(self.times), 4501)
        self.assertEqual(self.left.shape, (4501, 7))
        self.assertEqual(self.right.shape, (4501, 7))
        np.testing.assert_allclose(np.diff(self.times), 1/300, atol=1e-14, rtol=0)
        for t, dz, phase in (
                (0., 0., 'initial_hold'), (3., 0., 'raise'),
                (6., .02, 'upper_hold'), (9., .02, 'return'),
                (12., 0., 'return_hold'), (15., 0., 'return_hold')):
            with self.subTest(time=t):
                i = self.at(t)
                self.assertAlmostEqual(self.left[i, 2]-self.anchor.left_position[2], dz)
                self.assertEqual(self.phases[i], phase)

    def test_height_stays_inside_two_centimeters_and_holds_are_flat(self):
        dz = self.left[:, 2]-self.anchor.left_position[2]
        self.assertGreaterEqual(float(dz.min()), -1e-12)
        self.assertLessEqual(float(dz.max()), .02+1e-12)
        for first, last, wanted in ((0., 3., 0.), (6., 9., .02), (12., 15., 0.)):
            np.testing.assert_allclose(dz[(self.times >= first) & (self.times <= last)], wanted,
                                       atol=1e-12, rtol=0)
        self.assertTrue(np.all(np.diff(dz[(self.times >= 3) & (self.times <= 6)]) >= 0))
        self.assertTrue(np.all(np.diff(dz[(self.times >= 9) & (self.times <= 12)]) <= 0))

    def test_raise_and_return_have_smooth_endpoints(self):
        for t in (4.5, 10.5):
            self.assertAlmostEqual(self.left[self.at(t), 2]-self.anchor.left_position[2], .01)
        speed = np.diff(self.left[:, 2])/np.diff(self.times)
        for t in (3., 6., 9., 12.):
            i = self.at(t)
            # A finite 1/300s interval next to a zero-derivative cubic endpoint
            # has a small nonzero average speed (about 2.22e-5m/s).
            self.assertLess(abs(speed[i-1]), 3e-5)
            self.assertLess(abs(speed[i]), 3e-5)

    def test_left_xy_and_orientation_and_entire_right_pose_stay_constant(self):
        np.testing.assert_array_equal(self.left[:, :2], np.tile(self.anchor.left_position[:2], (4501, 1)))
        np.testing.assert_array_equal(self.left[:, 3:], np.tile(self.anchor.left_quaternion, (4501, 1)))
        np.testing.assert_array_equal(self.right, np.tile(
            np.r_[self.anchor.right_position, self.anchor.right_quaternion], (4501, 1)))
        np.testing.assert_array_equal(self.anchor.left_position, [.4, .2, -.3])


class PoseGuardTests(unittest.TestCase):
    def setUp(self):
        self.anchor = pose()
        self.target = np.r_[self.anchor.left_position, self.anchor.left_quaternion]

    def test_nominal_and_known_fifteen_mm_sag_are_observable(self):
        diagnostic.guard(self.anchor, self.target, self.anchor)
        diagnostic.guard(self.anchor, self.target, pose(left=[.4, .2, -.315]))

    def test_left_position_guard_uses_xyz_norm(self):
        for left in ([.4, .2, -.325001], [.416, .22, -.3]):
            with self.subTest(left=left), self.assertRaisesRegex(RuntimeError, '左腕误差超限'):
                diagnostic.guard(self.anchor, self.target, pose(left=left))

    def test_left_rotation_exceeding_three_degrees_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, '左腕误差超限'):
            diagnostic.guard(self.anchor, self.target, pose(left_quaternion=rotation('y', 3.01)))

    def test_right_translation_and_rotation_are_guarded_independently(self):
        for actual in (pose(right=[.310001, -.4, -.35]),
                       pose(right_quaternion=rotation('x', 3.01))):
            with self.subTest(actual=actual), self.assertRaisesRegex(RuntimeError, '右腕保持偏移超限'):
                diagnostic.guard(self.anchor, self.target, actual)

    def test_quaternion_sign_changes_do_not_create_false_rotation_errors(self):
        diagnostic.guard(self.anchor, self.target,
                         pose(left_quaternion=[-1., 0., 0., 0.], right_quaternion=[-1., 0., 0., 0.]))


class TelemetryFreshnessTests(unittest.TestCase):
    def setUp(self):
        self.telemetry = diagnostic.Telemetry(Mock(), Mock())
        self.now = 10_000_000_000
        self.pose = pose()
        self.joint = {'q': [0.]*16}

    def test_fresh_pose_and_joint_are_returned(self):
        self.telemetry.latest = (self.now, self.pose)
        self.telemetry.latest_joint = (self.now, self.joint)
        with patch.object(diagnostic.time, 'monotonic_ns', return_value=self.now):
            actual, joint = self.telemetry.require_fresh()
        self.assertIs(actual, self.pose)
        self.assertIs(joint, self.joint)

    def test_missing_or_stale_pose_and_joint_stop_further_commands(self):
        old = self.now-int((diagnostic.POSE_TIMEOUT_S+.001)*1e9)
        for pose_stamp, joint_stamp, message in (
                (None, self.now, '腕位姿反馈失联'), (old, self.now, '腕位姿反馈失联'),
                (self.now, None, '关节反馈失联'), (self.now, old, '关节反馈失联')):
            self.telemetry.latest = None if pose_stamp is None else (pose_stamp, self.pose)
            self.telemetry.latest_joint = None if joint_stamp is None else (joint_stamp, self.joint)
            with self.subTest(pose_stamp=pose_stamp, joint_stamp=joint_stamp), \
                    patch.object(diagnostic.time, 'monotonic_ns', return_value=self.now), \
                    self.assertRaisesRegex(RuntimeError, message):
                self.telemetry.require_fresh()

    def test_worker_failure_is_not_hidden_by_fresh_cached_data(self):
        self.telemetry.latest = (self.now, self.pose)
        self.telemetry.latest_joint = (self.now, self.joint)
        self.telemetry.error = 'SDK query rejected'
        with self.assertRaisesRegex(RuntimeError, '遥测失败.*SDK query rejected'):
            self.telemetry.require_fresh()


class BaselineValidationTests(unittest.TestCase):
    def test_expected_payload_checked_against_complete_left_and_right_configuration(self):
        wanted = [.396, 0., 0., 0., .001, 0., 0., 0.]
        snapshot = {'payload_reply': {'data': {'result': 'success', 'data': wanted.copy()}}}
        diagnostic.require_expected_payload(snapshot, wanted)
        diagnostic.require_expected_payload(snapshot, None)
        for index, value in ((0, .36), (1, .001), (4, .1), (7, .001)):
            changed = wanted.copy()
            changed[index] = value
            with self.subTest(index=index), self.assertRaisesRegex(RuntimeError, '负载配置与本轮预期不符'):
                diagnostic.require_expected_payload(
                    {'payload_reply': {'data': {'result': 'success', 'data': changed}}}, wanted)

    def test_same_anchor_accepts_small_noise_and_equivalent_quaternion_signs(self):
        anchor = pose()
        diagnostic.require_same_anchor(anchor, anchor)
        diagnostic.require_same_anchor(anchor, pose(
            left=[.4005, .2, -.3], left_quaternion=rotation('y', .1),
            right=[.3005, -.4, -.35], right_quaternion=rotation('x', .1)))
        diagnostic.require_same_anchor(anchor, pose(
            left_quaternion=[-1., 0., 0., 0.], right_quaternion=[-1., 0., 0., 0.]))

    def test_same_anchor_rejects_either_arm_translation_or_rotation_before_start(self):
        anchor = pose()
        for latest in (
                pose(left=[.401001, .2, -.3]), pose(right=[.301001, -.4, -.35]),
                pose(left_quaternion=rotation('y', .201)),
                pose(right_quaternion=rotation('x', .201))):
            with self.subTest(latest=latest), self.assertRaisesRegex(RuntimeError, '首帧前基线发生变化'):
                diagnostic.require_same_anchor(anchor, latest)

    def test_payload_validation_preserves_complete_readonly_values(self):
        values = [.36, .001, -.002, .003, .001, 0., 0., 0.]
        reply = {'data': {'result': 'success', 'data': values.copy()}}
        result = diagnostic.validate_payload_reply(reply)
        np.testing.assert_array_equal(result, values)
        self.assertEqual(reply['data']['data'], values)

    def test_payload_validation_rejects_failed_missing_nonfinite_and_negative_mass(self):
        valid = [.36, 0., 0., 0., .001, 0., 0., 0.]
        bad = [
            {}, {'data': {'result': 'fail', 'data': valid}},
            {'data': {'result': 'success'}},
        ]
        for values in (valid[:7], valid+[0.],
                       [float('nan')]+valid[1:], valid[:2]+[float('inf')]+valid[3:],
                       [-.01]+valid[1:], valid[:4]+[-.01]+valid[5:]):
            bad.append({'data': {'result': 'success', 'data': values}})
        for reply in bad:
            with self.subTest(reply=reply), self.assertRaisesRegex(RuntimeError, '负载只读回复无效'):
                diagnostic.validate_payload_reply(reply)


class TelemetryQueryRetryTests(unittest.TestCase):
    def setUp(self):
        self.client = Mock()
        self.trace = Mock()
        self.telemetry = diagnostic.Telemetry(self.client, self.trace)

    def test_timeout_recovery_records_successful_attempt_time_and_guid(self):
        reply = {'guid': 'successful-second-query', 'data': {'result': 'success', 'q': [0.]*16}}
        self.client.request.side_effect = [TimeoutError('first query expired'), reply]
        with patch.object(diagnostic.time, 'monotonic_ns', side_effect=[100, 200, 230]):
            received, actual = self.telemetry.query(
                'request_get_joint_state', 'response_get_joint_state', 'diagnostic_joint')
        self.assertEqual(received, 230)
        self.assertIs(actual, reply)
        self.assertEqual(self.client.request.call_count, 2)
        for call in self.client.request.call_args_list:
            self.assertEqual(call.args, ('request_get_joint_state', {}, 'response_get_joint_state'))
            self.assertEqual(call.kwargs, {'timeout': .1})
        successes = [call for call in self.trace.record.call_args_list if call.args[0] == 'diagnostic_joint']
        self.assertEqual(len(successes), 1)
        self.assertEqual(successes[0].kwargs, {
            'request_guid': 'successful-second-query', 'sent_monotonic_ns': 200,
            'received_monotonic_ns': 230, 'reply': reply, 'attempts': 2,
        })
        retries = [call for call in self.trace.record.call_args_list if call.args[0] == 'telemetry_query_retry']
        self.assertEqual([call.kwargs['attempt'] for call in retries], [1])
        self.client.send.assert_not_called()

    def test_three_timeouts_stop_without_a_fourth_query_or_success_record(self):
        self.client.request.side_effect = TimeoutError('query expired')
        with patch.object(diagnostic.time, 'monotonic_ns', side_effect=[100, 200, 300]), \
                self.assertRaisesRegex(TimeoutError, 'query expired'):
            self.telemetry.query('request_get_joint_state', 'response_get_joint_state', 'diagnostic_joint')
        self.assertEqual(self.client.request.call_count, 3)
        calls = self.trace.record.call_args_list
        self.assertEqual([call.args[0] for call in calls], ['telemetry_query_retry']*3)
        self.assertEqual([call.kwargs['attempt'] for call in calls], [1, 2, 3])
        self.client.send.assert_not_called()

    def test_robot_runtime_error_is_not_retried(self):
        self.client.request.side_effect = RuntimeError('robot rejected query')
        with patch.object(diagnostic.time, 'monotonic_ns', return_value=100), \
                self.assertRaisesRegex(RuntimeError, 'robot rejected query'):
            self.telemetry.query('request_get_joint_state', 'response_get_joint_state', 'diagnostic_joint')
        self.client.request.assert_called_once_with(
            'request_get_joint_state', {}, 'response_get_joint_state', timeout=.1)
        self.trace.record.assert_not_called()
        self.client.send.assert_not_called()


class AbortTests(unittest.TestCase):
    def test_guard_failure_after_first_command_never_sends_return_or_other_actions(self):
        anchor = pose()
        target = np.r_[anchor.left_position, anchor.left_quaternion]
        right = np.r_[anchor.right_position, anchor.right_quaternion]
        raw_pose = {'data': {
            'result': 'success',
            'left_position': anchor.left_position.tolist(), 'left_quat': anchor.left_quaternion.tolist(),
            'right_position': anchor.right_position.tolist(), 'right_quat': anchor.right_quaternion.tolist(),
        }}
        snapshot = {'samples': [{'joint_reply': {'data': {'q': [0.]*16}}, 'pose_reply': raw_pose}]}
        client = Mock(accid='DACH_TRON2A_253')
        client.failure.return_value = None
        telemetry = SimpleNamespace(
            latest=(1, anchor), latest_joint=(1, {'q': [0.]*16}), error=None,
            thread=Mock(), stop=Mock(),
            require_fresh=Mock(side_effect=[(anchor, {'q': [0.]*16}),
                                           (pose(left=[.4, .2, -.326]), {'q': [0.]*16})]),
        )
        report = {'frames': 3, 'height_m': .02}
        prepared = (Mock(), anchor, np.array([0., 1/300, 2/300]),
                    np.tile(target, (3, 1)), np.tile(right, (3, 1)),
                    ['initial_hold']*3, report)
        lock = Mock()
        with TemporaryDirectory() as temp, \
                patch.object(diagnostic, 'Tron2JsonClient', return_value=client), \
                patch.object(diagnostic, 'Telemetry', return_value=telemetry), \
                patch.object(diagnostic, 'baseline', return_value=snapshot), \
                patch.object(diagnostic, 'readonly_configuration', return_value={}), \
                patch.object(diagnostic, 'prepare', return_value=prepared), \
                patch.object(diagnostic, 'TraceRecorder') as trace, \
                patch.object(diagnostic, 'acquire_capture_lock', return_value=lock), \
                patch.object(diagnostic, '_ACTIVE_CLIENTS', []), \
                patch.object(diagnostic.time, 'sleep'), \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            output = Path(temp)/'diagnostic'
            self.assertEqual(diagnostic.main(['--execute', '--output', str(output)]), 2)
            result = json.loads((output/'result.json').read_text())
        self.assertEqual(result['result'], 'aborted')
        self.assertIn('左腕误差超限', result['error'])
        client.send.assert_called_once()
        title, data = client.send.call_args.args
        self.assertEqual(title, 'request_servop')
        self.assertEqual(set(data), {'left_pos', 'right_pos'})
        np.testing.assert_array_equal(data['right_pos'], right)
        final_queries = [call.args[0] for call in client.request.call_args_list]
        self.assertEqual(final_queries, ['request_get_servo_pose', 'request_get_joint_state',
                                         'request_get_payload_param'])
        self.assertIn('diagnostic_abort', [call.args[0] for call in trace.return_value.record.call_args_list])
        telemetry.stop.set.assert_called_once()
        lock.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
