"""ServoP 回放契约回归：使用模拟客户端，绝不连接真实机器人。"""

from contextlib import redirect_stdout
import importlib.util
import io
from pathlib import Path
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

import replay_tron2_tracker as tracker
from replay_tron2_fixed_start import FIXED_LEFT_POSITION, FIXED_LEFT_QUATERNION


def anchor_pose(right=None):
    return tracker.DualArmPose(
        FIXED_LEFT_POSITION.copy(), FIXED_LEFT_QUATERNION.copy(),
        np.array([.32, -.5, -.4] if right is None else right, dtype=float),
        np.array([1., 0., 0., 0.]),
    )


class ServoPReplayTests(unittest.TestCase):
    def setUp(self):
        self.anchor = anchor_pose()
        self.client = Mock()
        self.client.trace = Mock()
        self.client.current_pose.return_value = self.anchor
        self.client.current_servo_pose.return_value = self.anchor
        self.q = np.r_[np.zeros(14), [.9503, .0168]]
        self.client.joint_snapshot.return_value = (self.q, np.zeros(16))
        self.client.failure.return_value = None
        self.client.latest_pose_sample.side_effect = lambda: tracker.PoseSample(
            time.monotonic(), self.anchor,
        )
        self.client.pose_samples.return_value = []
        self.args = SimpleNamespace(arm_control="servop", countdown=0,
                                    feedback_timeout=.5, rate=20)
        self.deltas = np.array([[0., 0., 0.], [.002, -.003, -.001]])

    def run_replay(self, hand=None):
        with patch("builtins.input", return_value="DIRECT"), \
                patch.object(tracker, "FeedbackGuard"), \
                patch.object(tracker, "POST_STOP_OBSERVE_SECONDS", 0), \
                patch.object(tracker, "verify_observed_motion", return_value=True), \
                redirect_stdout(io.StringIO()):
            return tracker.execute_replay(
                self.client, self.anchor, "left", np.linspace(0., .01, len(self.deltas)),
                self.deltas, np.tile([1., 0., 0., 0.], (len(self.deltas), 1)), self.args,
                hand_replay=hand,
            )

    def test_only_left_wrist_target_changes_and_no_head_or_joint_commands_exist(self):
        original_q = self.q.copy()
        self.assertTrue(self.run_replay())
        calls = self.client.send.call_args_list
        self.assertGreater(len(calls), 1)
        right = np.r_[self.anchor.right_position, self.anchor.right_quaternion]
        for call in calls:
            self.assertEqual(call.args[0], "request_servop")
            self.assertEqual(set(call.args[1]), {"left_pos", "right_pos"})
            self.assertEqual(len(call.args[1]["left_pos"]), 7)
            np.testing.assert_array_equal(call.args[1]["right_pos"], right)
        np.testing.assert_array_equal(calls[0].args[1]["left_pos"][:3], self.anchor.left_position)
        np.testing.assert_allclose(calls[-1].args[1]["left_pos"][:3],
                                   self.anchor.left_position + self.deltas[-1])
        np.testing.assert_array_equal(self.q, original_q)

    def test_held_wrist_target_does_not_chase_streaming_feedback(self):
        feedback = anchor_pose(self.anchor.right_position + [.001, 0., 0.])
        self.client.latest_pose_sample.side_effect = lambda: tracker.PoseSample(
            time.monotonic(), feedback,
        )
        self.assertTrue(self.run_replay())
        for call in self.client.send.call_args_list:
            np.testing.assert_array_equal(call.args[1]["right_pos"][:3],
                                          self.anchor.right_position)

    def test_tracker_relative_to_start_is_not_accumulated_a_second_time(self):
        self.deltas = np.array([[0., 0., 0.], [.001, 0., 0.], [.002, 0., 0.]])
        self.assertTrue(self.run_replay())
        commands = self.client.send.call_args_list
        np.testing.assert_allclose(commands[1].args[1]["left_pos"][:3],
                                   self.anchor.left_position + self.deltas[1])
        np.testing.assert_allclose(commands[2].args[1]["left_pos"][:3],
                                   self.anchor.left_position + self.deltas[2])

    def test_o6_and_arm_keep_the_same_integer_monotonic_start(self):
        hand = Mock()
        self.assertTrue(self.run_replay(hand))
        hand.acquire.assert_called_once()
        hand.start_at.assert_called_once()
        start = hand.start_at.call_args.args[0]
        self.assertIsInstance(start, int)
        starts = [call.kwargs["start_monotonic_ns"]
                  for call in self.client.trace.record.call_args_list
                  if call.args[0] == "replay_start"]
        self.assertEqual(starts, [start])

    def test_trace_records_the_training_and_deployment_action_contract(self):
        self.assertTrue(self.run_replay())
        contracts = [call.kwargs for call in self.client.trace.record.call_args_list
                     if call.args[0] == "replay_contract"]
        self.assertEqual(len(contracts), 1)
        contract = contracts[0]
        for key, value in tracker.SERVOP_ACTION_CONTRACT.items():
            self.assertEqual(contract[key], value)
        self.assertEqual(contract["moving_arm"], "left")
        self.assertEqual(contract["held_arm"], "right")
        self.assertEqual(contract["head_start_rad"], self.q[14:16].tolist())
        self.assertEqual(contract["head_control"], "not_commanded")

    def test_no_interpolation_is_logged_without_changing_the_shared_hand_clock(self):
        self.args.interpolation = 'none'
        hand = Mock()
        self.assertTrue(self.run_replay(hand))
        contract = next(c.kwargs for c in self.client.trace.record.call_args_list
                        if c.args[0] == 'replay_contract')
        self.assertEqual(contract['tracker_interpolation'], 'none')
        start = hand.start_at.call_args.args[0]
        arm_start = next(c.kwargs['start_monotonic_ns']
                         for c in self.client.trace.record.call_args_list
                         if c.args[0] == 'replay_start')
        self.assertEqual(start, arm_start)

    def test_hanging_right_arm_preserves_its_startup_target_without_movep_box_check(self):
        self.anchor = anchor_pose([.0988, -.1825, -.7301])
        self.client.current_pose.return_value = self.anchor
        self.client.current_servo_pose.return_value = self.anchor
        self.assertTrue(self.run_replay())
        for call in self.client.send.call_args_list:
            np.testing.assert_array_equal(call.args[1]["right_pos"],
                                          np.r_[self.anchor.right_position,
                                                self.anchor.right_quaternion])

    def test_invalid_held_position_still_blocks_before_confirmation_and_o6_acquisition(self):
        self.anchor = anchor_pose([float("nan"), -.211, -.7535])
        hand = Mock()
        with patch("builtins.input") as confirm:
            with self.assertRaisesRegex(ValueError, "ServoP 保持臂 right"):
                tracker.execute_replay(
                    self.client, self.anchor, "left", np.array([0., .01]),
                    self.deltas, np.tile([1., 0., 0., 0.], (2, 1)),
                    self.args, hand_replay=hand,
                )
        confirm.assert_not_called()
        self.client.send.assert_not_called()
        hand.acquire.assert_not_called()
        hand.start_at.assert_not_called()

    def test_right_feedback_drift_guard_is_not_removed(self):
        self.anchor = anchor_pose([.0988, -.1825, -.7301])
        feedback = anchor_pose(self.anchor.right_position + [tracker.HELD_ARM_DRIFT_M + .01, 0., 0.])
        guard = tracker.FeedbackGuard(self.anchor, "left")
        with self.assertRaisesRegex(RuntimeError, "双臂反馈超限"):
            guard.observe(tracker.PoseSample(time.monotonic(), feedback),
                          self.anchor.left_position, self.anchor.left_quaternion, "trajectory")

    def test_held_position_still_rejects_invalid_shape_and_unknown_side(self):
        for position in ([.1, .2], [.1, .2, .3, .4], [.1, float("inf"), .3]):
            with self.subTest(position=position), self.assertRaises(ValueError):
                tracker.check_servop_hold_position("right", position)
        with self.assertRaises(ValueError):
            tracker.check_servop_hold_position("unknown", np.zeros(3))

    def test_moving_left_workspace_reference_is_unchanged(self):
        self.anchor = tracker.DualArmPose(
            np.array([.02, .18, -.75]), self.anchor.left_quaternion,
            self.anchor.right_position, self.anchor.right_quaternion,
        )
        with patch("builtins.input") as confirm:
            with self.assertRaisesRegex(ValueError, "left target"):
                tracker.execute_replay(
                    self.client, self.anchor, "left", np.array([0., .01]),
                    self.deltas, np.tile([1., 0., 0., 0.], (2, 1)), self.args,
                )
        confirm.assert_not_called()
        self.client.send.assert_not_called()

    def test_14_joint_feedback_does_not_require_head_targets_or_a_local_ik_model(self):
        self.client.joint_snapshot.return_value = (np.zeros(14), np.zeros(14))
        with patch.dict("sys.modules", {"tron2_left_ik": None, "mujoco": None}):
            self.assertTrue(self.run_replay())

    def test_servoj_is_blocked_at_the_lowest_replay_send_entry(self):
        client = object.__new__(tracker.Tron2JsonClient)
        client.robot = Mock()
        with self.assertRaisesRegex(RuntimeError, "ServoJ 回放已停用"):
            client.send("request_servoj", {"q": [0.] * 16, "filter_ratio": 1.0})
        client.robot.publishJsonMessage.assert_not_called()


class WebServoPContractTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).parent / "TransHandR0725/TransHandR/scripts/robot_replay_web.py"
        spec = importlib.util.spec_from_file_location("offline_robot_replay_web", path)
        self.web = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.web)
        self.manager = self.web.RobotReplayManager(Mock(), Mock(), python="offline-python")

    def test_preview_and_execution_explicitly_use_the_same_servop_contract(self):
        options = self.web.replay_options({"with_o6_hand": False})
        for execute in (False, True):
            command = self.manager._command(Path("/tmp/offline.hdf5"), options, execute)
            self.assertEqual(command[command.index("--arm-control") + 1], "servop")
            self.assertEqual(command[command.index("--orientation-scale") + 1], "0")
            self.assertNotIn("servoj", command)
            self.assertEqual("--execute" in command, execute)
        self.assertEqual(self.manager.status()["arm_control"], "servop")

    def test_web_cannot_inject_a_joint_backend(self):
        with self.assertRaisesRegex(ValueError, "不支持的回放参数"):
            self.web.replay_options({"arm_control": "servoj"})


if __name__ == "__main__":
    unittest.main()
