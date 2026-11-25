"""纯模拟被动订阅测试，不初始化SDK、不连接机器人。"""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from tools.tron2_motor_observer import MotorObserver, plain_value


class MotorObserverTests(unittest.TestCase):
    def test_start_only_subscribes_and_preserves_original_fields(self):
        trace, robot = Mock(), Mock()
        observer = MotorObserver(trace)
        observer.start(robot)
        callback = robot.subscribeRobotCmd.call_args.args[0]
        callback(SimpleNamespace(q=np.array([.1, -.2]), mode=np.array([0, 2]),
                                 motor_names=['a', 'b'], parallel_solve_required=[True, False],
                                 ignored='not an official field'))
        robot.publishRobotCmd.assert_not_called()
        event = trace.record.call_args
        self.assertEqual(event.args[0], 'motor_command_observed')
        self.assertEqual(event.kwargs['fields'], {
            'q': [.1, -.2], 'mode': [0, 2], 'motor_names': ['a', 'b'],
            'parallel_solve_required': [True, False]})
        self.assertEqual(observer.summary()['streams']['command']['recorded'], 1)

    def test_throttle_and_stop_do_not_publish_or_fabricate_missing_fields(self):
        trace = Mock()
        observer = MotorObserver(trace, rate_hz=50)
        callback = observer.callbacks['state']
        with patch('tools.tron2_motor_observer.time.monotonic_ns',
                   side_effect=[0, 1_000_000, 21_000_000, 44_000_000]):
            callback(SimpleNamespace(q=[]))
            callback(SimpleNamespace(q=[1]))
            callback(SimpleNamespace(q=[2]))
            observer.stop()
            callback(SimpleNamespace(q=[3]))
        stats = observer.summary()['streams']['state']
        self.assertEqual((stats['received'], stats['recorded']), (3, 2))
        self.assertEqual(stats['last_fields'], {'q': [2]})
        self.assertEqual(observer.summary()['control_commands_published'], 0)

    def test_subscription_and_callback_errors_stay_inside_observer(self):
        robot, trace = Mock(), Mock()
        robot.subscribeRobotCmd.side_effect = RuntimeError('unsupported')
        observer = MotorObserver(trace)
        observer.start(robot)
        self.assertIn('error', observer.summary()['subscriptions']['command'])
        trace.record.side_effect = RuntimeError('closed recorder')
        observer.callbacks['state'](SimpleNamespace(q=[0]))
        self.assertEqual(observer.summary()['streams']['state']['errors'], 1)

    def test_numpy_and_nonfinite_values_are_saved_without_guessing_units(self):
        self.assertEqual(plain_value(np.array([1, 2])), [1, 2])
        self.assertEqual(plain_value(np.float32(.5)), .5)
        self.assertEqual(plain_value([float('nan'), float('inf')]), ['nan', 'inf'])
        with self.assertRaises(ValueError):
            MotorObserver(Mock(), rate_hz=float('nan'))


if __name__ == '__main__':
    unittest.main()
