"""模拟 SDK 回调/丢包，验证只读重试不会重复运动或接受其它请求的回复。"""

import json
import threading
import time
import unittest
from unittest.mock import Mock, patch

import replay_tron2_tracker as tracker


def pose_data(x=.5):
    return {"result": "success", "left_position": [x, .2, -.2],
            "right_position": [.4, -.2, -.5], "left_quat": [1, 0, 0, 0],
            "right_quat": [1, 0, 0, 0], "timestamp": 123}


class SnapshotRetryTests(unittest.TestCase):
    def setUp(self):
        self.client = object.__new__(tracker.Tron2JsonClient)
        self.client._condition = threading.Condition()
        self.client.accid = "test_robot"
        self.client._robot_identity_ready = True
        self.client._robot_identity_error = None
        self.client._messages = []
        self.client._failure = None
        self.client._pose_probe_guid = None
        self.client._pose_probe_sent_at = 0.
        self.client._pose_probe_error = None
        self.client._pose_samples = []
        self.client._snapshot_pose_guids = set()
        self.client.trace = Mock()
        self.client.robot = Mock()
        self.sent = []
        self.behavior = lambda message: None

        def publish(payload):
            message = json.loads(payload)
            self.sent.append(message)
            self.behavior(message)
            return True

        self.client.robot.publishJsonMessage.side_effect = publish
        retry = patch.object(tracker, "SNAPSHOT_RETRY_INTERVAL_SECONDS", .01)
        retry.start()
        self.addCleanup(retry.stop)

    def reply(self, message, *, guid=None, data=None, title="response_get_servo_pose"):
        self.client._on_message(json.dumps({
            "title": title, "guid": message['guid'] if guid is None else guid,
            "data": pose_data() if data is None else data,
        }))

    def test_dropped_query_is_retried_with_fresh_guid(self):
        def behavior(message):
            if len(self.sent) == 2:
                self.reply(message)
        self.behavior = behavior
        pose = self.client.current_servo_pose(label="test")
        self.assertEqual(pose.left_position[0], .5)
        self.assertEqual(len(self.sent), 2)
        self.assertNotEqual(self.sent[0]['guid'], self.sent[1]['guid'])
        self.assertTrue(all(m['title'] == 'request_get_servo_pose' for m in self.sent))
        self.assertEqual(self.client._snapshot_pose_guids, set())
        events = [c.args[0] for c in self.client.trace.record.call_args_list]
        self.assertIn('snapshot_query_recovered', events)
        self.assertNotIn('pose_probe_unmatched_reply', events)

    def test_old_or_other_client_guid_cannot_satisfy_a_retry(self):
        def behavior(message):
            if len(self.sent) == 2:
                self.reply(message, guid=self.sent[0]['guid'], data=pose_data(9.))
                self.reply(message, guid="other_client", data=pose_data(8.))
                self.reply(message, data=pose_data(.6))
        self.behavior = behavior
        pose = self.client.current_servo_pose()
        self.assertEqual(pose.left_position[0], .6)

    def test_retries_do_not_extend_total_deadline(self):
        started = time.monotonic()
        with self.assertRaisesRegex(TimeoutError, 'no response_get_servo_pose'):
            self.client.request('request_get_servo_pose', {}, 'response_get_servo_pose', timeout=.045)
        self.assertLess(time.monotonic()-started, .3)
        self.assertGreaterEqual(len(self.sent), 2)
        self.assertEqual(self.client._snapshot_pose_guids, set())

    def test_movej_timeout_never_republishes_the_motion(self):
        with self.assertRaises(TimeoutError):
            self.client.request('request_movej', {'joint': [0.]*14, 'time': 3},
                                'response_movej', timeout=.035)
        self.assertEqual(len(self.sent), 1)
        self.assertEqual(self.sent[0]['title'], 'request_movej')

    def test_robot_rejection_is_not_retried(self):
        self.behavior = lambda m: self.reply(m, data={'result': 'fail_no_data'})
        with self.assertRaisesRegex(RuntimeError, 'fail_no_data'):
            self.client.current_servo_pose()
        self.assertEqual(len(self.sent), 1)

    def test_joint_state_legacy_reply_without_guid_is_supported(self):
        def behavior(message):
            if len(self.sent) == 2:
                self.client._on_message(json.dumps({'title': 'response_get_joint_state',
                    'data': {'result': 'success', 'q': [0.]*16, 'dq': [0.]*16}}))
        self.behavior = behavior
        q, dq = self.client.joint_snapshot(label='test')
        self.assertEqual(q.shape, (16,))
        self.assertEqual(len(self.sent), 2)

    def test_joint_state_with_wrong_guid_is_ignored(self):
        def behavior(message):
            self.reply(message, title='response_get_joint_state', guid='another',
                       data={'result': 'success', 'q': [9.]*16, 'dq': [0.]*16})
            self.reply(message, title='response_get_joint_state',
                       data={'result': 'success', 'q': [0.]*16, 'dq': [0.]*16})
        self.behavior = behavior
        q, _ = self.client.joint_snapshot(label='test')
        self.assertEqual(q[0], 0.)

    def test_malformed_snapshot_does_not_trigger_retries(self):
        self.behavior = lambda m: self.reply(m, data={'result': 'success'})
        with self.assertRaisesRegex(RuntimeError, 'invalid dual-arm pose'):
            self.client.current_servo_pose()
        self.assertEqual(len(self.sent), 1)

    def test_query_with_unexpected_payload_is_not_retried(self):
        with self.assertRaises(TimeoutError):
            self.client.request('request_get_servo_pose', {'unexpected': 1},
                                'response_get_servo_pose', timeout=.025)
        self.assertEqual(len(self.sent), 1)


if __name__ == '__main__':
    unittest.main()
