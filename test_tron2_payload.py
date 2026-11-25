"""负载工具的接口/量纲回归；全部模拟 HeadClient，不连接真实机器人。"""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from websocket import WebSocketTimeoutException

from tools import tron2_payload as payload


class FakeClient:
    def __init__(self):
        self.accid = payload.ROBOT_ID
        self.socket = Mock()
        self.raw = [.1, 0., 0., 0., .001, .000001, -.000002, .000003]
        self.calls = []
        self.dq = [0.] * 16
        self.q = [0.] * 16
        self.set_error = None
        self.selected_mismatch = False
        self.other_changed = False
        self.reject_set = False
        self.bad_ack_arm = False
        self.after_error = False
        self.get_count = 0

    def request(self, title, data, response):
        self.calls.append((title, data, response))
        if title == 'request_get_payload_param':
            self.get_count += 1
            if self.after_error and self.get_count == 2:
                raise TimeoutError('no readback')
            return {'data': {'result': 'success', 'data': self.raw.copy()}}
        if title == 'request_get_joint_state':
            return {'data': {'result': 'success', 'q': self.q.copy(), 'dq': self.dq.copy()}}
        if title == 'request_set_payload_param':
            if self.set_error is not None:
                raise self.set_error
            if self.reject_set:
                return {'data': {'result': 'fail'}}
            offset = 4 * data['arm']
            self.raw[offset:offset + 4] = data['data']
            if self.selected_mismatch:
                self.raw[offset] += .1
            if self.other_changed:
                self.raw[4 - offset] += .1
            return {'data': {'result': 'success', 'arm': 1 - data['arm'] if self.bad_ack_arm else data['arm']}}
        raise AssertionError(f'Unexpected hardware request {title}')


class PayloadCliTests(unittest.TestCase):
    def setUp(self):
        self.client = FakeClient()
        self.lock = Mock()
        self.connect = patch.object(payload, 'HeadClient', return_value=self.client)
        self.connect_mock = self.connect.start()
        self.addCleanup(self.connect.stop)
        self.acquire = patch.object(payload, 'acquire_capture_lock', return_value=self.lock)
        self.acquire_mock = self.acquire.start()
        self.addCleanup(self.acquire.stop)
        self.args = ['--arm', 'left', '--mass-kg', '2.0', '--first-moment-kg-m', '.1', '-.2', '.3']

    def run_cli(self, args):
        with redirect_stdout(io.StringIO()) as out, redirect_stderr(io.StringIO()) as error:
            result = payload.main(args)
        return result, out.getvalue(), error.getvalue()

    def sets(self):
        return [c for c in self.client.calls if c[0] == 'request_set_payload_param']

    def test_default_is_single_read_only_query_without_write_lock(self):
        code, text, _ = self.run_cli([])
        self.assertEqual(code, 0)
        self.assertIn('kg*m', text)
        self.assertEqual([c[0] for c in self.client.calls], ['request_get_payload_param'])
        self.acquire_mock.assert_not_called()
        self.client.socket.close.assert_called_once()

    def test_complete_target_without_execute_only_previews(self):
        code, _, _ = self.run_cli(self.args)
        self.assertEqual(code, 0)
        self.assertFalse(self.sets())
        self.acquire_mock.assert_not_called()

    def test_left_set_uses_raw_first_moments_not_center_of_mass(self):
        original_other = self.client.raw[4:].copy()
        code, _, _ = self.run_cli(self.args + ['--execute'])
        self.assertEqual(code, 0)
        self.assertEqual(self.sets(), [('request_set_payload_param', {'arm': 0, 'data': [2., .1, -.2, .3]},
                                      'response_set_payload_param')])
        self.assertEqual(self.client.raw[4:], original_other)
        self.acquire_mock.assert_called_once()
        self.lock.close.assert_called_once()
        self.assertEqual(self.client.get_count, 2)
        self.assertEqual(sum(c[0] == 'request_get_joint_state' for c in self.client.calls), 2)

    def test_right_set_changes_only_right_arm(self):
        original_left = self.client.raw[:4].copy()
        args = ['--arm', 'right', '--mass-kg', '.5', '--first-moment-kg-m', '.001', '.002', '-.003', '--execute']
        code, _, _ = self.run_cli(args)
        self.assertEqual(code, 0)
        self.assertEqual(self.sets()[0][1], {'arm': 1, 'data': [.5, .001, .002, -.003]})
        self.assertEqual(self.client.raw[:4], original_left)

    def test_execute_requires_explicit_arm_and_all_four_parameters_before_connection(self):
        for args in [[], ['--arm', 'left'], ['--arm', 'left', '--mass-kg', '.5'],
                     ['--mass-kg', '.5', '--first-moment-kg-m', '0', '0', '0'],
                     ['--arm', 'left', '--first-moment-kg-m', '0', '0', '0']]:
            with self.subTest(args=args):
                code, _, _ = self.run_cli(args + ['--execute'])
                self.assertEqual(code, 2)
        self.connect_mock.assert_not_called()
        self.acquire_mock.assert_not_called()

    def test_zero_negative_or_nonfinite_mass_is_rejected_before_connection(self):
        for mass in ['0', '-1', 'nan', 'inf']:
            with self.subTest(mass=mass):
                code, _, _ = self.run_cli(['--arm', 'left', '--mass-kg', mass,
                                           '--first-moment-kg-m', '0', '0', '0', '--execute'])
                self.assertEqual(code, 2)
        self.connect_mock.assert_not_called()

    def test_nonfinite_first_moment_is_rejected_before_connection(self):
        code, _, _ = self.run_cli(['--arm', 'left', '--mass-kg', '.5',
                                   '--first-moment-kg-m', '0', 'nan', '0', '--execute'])
        self.assertEqual(code, 2)
        self.connect_mock.assert_not_called()

    def test_wrong_identity_rejects_before_any_request(self):
        self.client.accid = 'another_robot'
        code, _, _ = self.run_cli(self.args + ['--execute'])
        self.assertEqual(code, 2)
        self.assertEqual(self.client.calls, [])
        self.lock.close.assert_called_once()

    def test_moving_or_invalid_joint_feedback_blocks_set(self):
        for kind in ['moving', 'q_nan', 'dq_short']:
            with self.subTest(kind=kind):
                self.client.dq = [0.] * 16
                self.client.q = [0.] * 16
                if kind == 'moving': self.client.dq[9] = .051
                if kind == 'q_nan': self.client.q[0] = float('nan')
                if kind == 'dq_short': self.client.dq = [0.] * 14
                code, _, _ = self.run_cli(self.args + ['--execute'])
                self.assertEqual(code, 2)
                self.assertFalse(self.sets())

    def test_malformed_or_nonpositive_payload_blocks_set(self):
        variants = [[.1] * 7, [.1] * 9, [.1] * 7 + [float('nan')],
                    [0., 0., 0., 0., .1, 0., 0., 0.], [.1, 0., 0., 0., -1., 0., 0., 0.],
                    [.1, 0., 0., 0., .1, 0., 0., '0'], [.1, 0., 0., 0., .1, 0., 0., True]]
        for raw in variants:
            with self.subTest(raw=raw):
                self.client.raw = raw
                code, _, _ = self.run_cli(self.args + ['--execute'])
                self.assertEqual(code, 2)
                self.assertFalse(self.sets())

    def test_selected_arm_readback_mismatch_is_not_success_or_retried(self):
        self.client.selected_mismatch = True
        code, _, error = self.run_cli(self.args + ['--execute'])
        self.assertEqual(code, 2)
        self.assertIn('目标不符', error)
        self.assertEqual(len(self.sets()), 1)

    def test_other_arm_change_is_not_success_or_rolled_back(self):
        self.client.other_changed = True
        code, _, error = self.run_cli(self.args + ['--execute'])
        self.assertEqual(code, 2)
        self.assertIn('对侧参数', error)
        self.assertEqual(len(self.sets()), 1)

    def test_set_timeout_or_rejection_only_gets_readback_and_never_repeats_set(self):
        for kind in ['timeout', 'rejected', 'wrong_arm_ack']:
            with self.subTest(kind=kind):
                self.client.calls.clear()
                self.client.get_count = 0
                self.client.set_error = TimeoutError('no SET ack') if kind == 'timeout' else None
                self.client.reject_set = kind == 'rejected'
                self.client.bad_ack_arm = kind == 'wrong_arm_ack'
                code, _, error = self.run_cli(self.args + ['--execute'])
                self.assertEqual(code, 2)
                self.assertIn('写入未确认', error)
                self.assertEqual(len(self.sets()), 1)
                self.assertEqual(self.client.get_count, 2)

    def test_failed_readback_never_repeats_set(self):
        self.client.after_error = True
        code, _, _ = self.run_cli(self.args + ['--execute'])
        self.assertEqual(code, 2)
        self.assertEqual(len(self.sets()), 1)

    def test_actual_websocket_timeout_is_recorded_without_repeating_set(self):
        self.client.set_error = WebSocketTimeoutException('SET websocket timed out')
        code, _, error = self.run_cli(self.args + ['--execute'])
        self.assertEqual(code, 2)
        self.assertIn('写入未确认', error)
        self.assertEqual(len(self.sets()), 1)
        self.assertEqual(self.client.get_count, 2)

    def test_nonfinite_device_reply_still_saves_valid_error_evidence(self):
        self.client.raw[-1] = float('nan')
        with TemporaryDirectory() as folder:
            path = Path(folder) / 'evidence.json'
            code, _, _ = self.run_cli(self.args + ['--execute', '--output', str(path)])
            self.assertEqual(code, 2)
            record = json.loads(path.read_text())
        self.assertEqual(record['status'], 'error')
        self.assertEqual(record['before_reply']['data']['data'][-1], 'nonfinite:nan')
        self.assertFalse(self.sets())

    def test_existing_output_is_preserved_and_blocks_connection(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / 'evidence.json'
            path.write_text('preserve original')
            code, _, _ = self.run_cli(self.args + ['--execute', '--output', str(path)])
            self.assertEqual(code, 2)
            self.assertEqual(path.read_text(), 'preserve original')
            self.connect_mock.assert_not_called()

    def test_record_names_units_and_before_after_without_motion_requests(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / 'evidence.json'
            code, _, _ = self.run_cli(self.args + ['--execute', '--output', str(path)])
            self.assertEqual(code, 0)
            report = json.loads(path.read_text())
        self.assertEqual(report['status'], 'set_verified')
        self.assertEqual(report['requested_raw_kg_kgm'], [2., .1, -.2, .3])
        self.assertEqual(report['raw_parameter_fields_per_arm'][1], 'first_moment_x_kg_m')
        self.assertEqual(report['before_payload_raw_kg_kgm'][:4], [.1, 0., 0., 0.])
        self.assertEqual(report['after_payload_raw_kg_kgm'][:4], [2., .1, -.2, .3])
        self.assertFalse(report['motion_or_mode_requests_sent'])
        self.assertEqual(report['set_attempts'], 1)
        self.assertLessEqual({c[0] for c in self.client.calls},
                             {'request_get_payload_param', 'request_set_payload_param', 'request_get_joint_state'})

    def test_float32_transport_rounding_is_accepted(self):
        payload.verify_payload_readback(
            [.1, 0., 0., 0., .001, 0., 0., 0.],
            [.6000000238418579, .0012000000569969416, 0., 0., .001, 0., 0., 0.],
            'left', [.6, .0012, 0., 0.])

    def test_busy_replay_lock_blocks_connection_and_set(self):
        self.acquire_mock.side_effect = RuntimeError('已有恢复、回放或腕部采集任务')
        code, _, _ = self.run_cli(self.args + ['--execute'])
        self.assertEqual(code, 2)
        self.connect_mock.assert_not_called()


if __name__ == '__main__':
    unittest.main()
