"""固定起点回放入口的离线测试；绝不连接或移动机器人。"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import h5py
import numpy as np

import replay_tron2_fixed_start as fixed
from replay_tron2_tracker import DualArmPose


def captured_pose() -> DualArmPose:
    return DualArmPose(
        fixed.FIXED_LEFT_POSITION.copy(),
        fixed.FIXED_LEFT_QUATERNION.copy(),
        fixed.FIXED_RIGHT_POSITION.copy(),
        fixed.FIXED_RIGHT_QUATERNION.copy(),
    )


def make_episode(path: Path) -> None:
    with h5py.File(path, "w") as handle:
        handle.attrs["hand_type"] = "o6"
        handle.attrs["hand_side"] = "left"
        handle.attrs["control_source"] = "linkerforce"
        handle.create_dataset(
            "timestamps/master_timestamp_ns", data=[0, 33_000_000, 66_000_000]
        )
        handle.create_dataset("action/command_timestamp_ns",
                              data=[0, 32_000_000, 65_000_000])
        handle.create_dataset("action/valid", data=[True, True, True])
        handle.create_dataset("action/command", data=np.tile(
            [255, 100, 200, 200, 200, 200], (3, 1)))
        handle.create_dataset("dexterous_hand/unified_timestamp_ns",
                              data=[0, 33_000_000, 66_000_000])
        handle.create_dataset("dexterous_hand/state_valid",
                              data=[True, True, True])
        handle.create_dataset("dexterous_hand/joint_positions", data=np.tile(
            [240, 120, 150, 150, 150, 150], (3, 1)))
        tracker = handle.create_group("vive_tracker")
        tracker.attrs["position_unit"] = "meter"
        tracker.create_dataset(
            "pos", data=np.array([
                [[0.0, 0.0, 0.0]],
                [[-0.02, 0.01, 0.0]],
                [[-0.01, 0.01, 0.0]],
            ])
        )
        tracker.create_dataset(
            "quat", data=np.tile([1.0, 0.0, 0.0, 0.0], (3, 1, 1))
        )
        tracker.create_dataset("valid", data=np.ones((3, 1), dtype=bool))


class FakeClient:
    def __init__(self, *_args, **_kwargs):
        self.accid = fixed.FIXED_ROBOT_ID
        self.q = fixed.FIXED_Q.copy()
        self.pose = captured_pose()

    def wait_for_identity(self):
        pass

    def joint_snapshot(self, label):
        return self.q.copy(), np.zeros(14)

    def current_servo_pose(self, label):
        return self.pose


class FixedStartTests(unittest.TestCase):
    def test_defaults_are_current_profile_and_requested_replay_settings(self):
        args = fixed.parse_args(["--episode", "65"])
        self.assertEqual(args.episode, 65)
        self.assertEqual(args.scale, 1.0)
        self.assertEqual(args.countdown, 5)
        self.assertEqual(args.position_tolerance, 0.02)
        self.assertEqual(args.axis_map, "y,z,x")
        self.assertEqual(args.speed, 0.2)
        self.assertFalse(args.execute)
        self.assertEqual(fixed.RIGHT_PREPARE_JOINT_TOLERANCE_RAD, 0.015)
        np.testing.assert_allclose(fixed.FIXED_Q[:7], fixed.FIXED_LEFT_Q)
        np.testing.assert_allclose(fixed.FIXED_Q[7:], fixed.FIXED_RIGHT_Q)
        np.testing.assert_allclose(fixed.FIXED_LEFT_POSITION,
                                   [0.590808, 0.371680, -0.243641])
        np.testing.assert_allclose(fixed.FIXED_RIGHT_POSITION,
                                   [0.547606, -0.345544, -0.353572])

    def test_fixed_pose_rejects_moved_wrist(self):
        args = fixed.parse_args(["--episode", "65"])
        pose = captured_pose()
        self.assertTrue(fixed.already_at_fixed_start(fixed.FIXED_Q, pose, args))
        near = DualArmPose(
            pose.left_position + [0.019, 0.0, 0.0], pose.left_quaternion,
            pose.right_position, pose.right_quaternion,
        )
        self.assertTrue(fixed.already_at_fixed_start(fixed.FIXED_Q, near, args))
        moved = DualArmPose(
            pose.left_position + [0.021, 0.0, 0.0], pose.left_quaternion,
            pose.right_position, pose.right_quaternion,
        )
        self.assertFalse(fixed.already_at_fixed_start(fixed.FIXED_Q, moved, args))
        with self.assertRaisesRegex(RuntimeError, "left 未到保存的固定起点"):
            fixed.check_fixed_pose(moved, args.position_tolerance, 3.0)

    def test_restore_function_only_verifies_when_already_at_fixed_start(self):
        client = FakeClient()
        args = fixed.parse_args(["--episode", "65", "--execute"])
        with patch.object(fixed, "move_stage", side_effect=AssertionError(
            "already fixed; no MoveJ expected"
        )), patch.object(fixed, "execute_replay", side_effect=AssertionError(
            "restore must not launch replay"
        )):
            anchor = fixed.restore_fixed_start(
                client, client.q.copy(), np.zeros(14), client.pose, args
            )
        self.assertIs(anchor, client.pose)

    def test_different_episode_can_be_previewed_without_robot(self):
        with tempfile.TemporaryDirectory() as directory:
            data_file = Path(directory) / "episode_000066.hdf5"
            make_episode(data_file)
            with patch.object(fixed, "Tron2JsonClient", side_effect=AssertionError(
                "offline preview must not connect to robot"
            )):
                code = fixed.main([
                    "--episode", "66", "--data-root", directory,
                ])
            self.assertEqual(code, 0)

    def test_fixed_left_replay_rejects_right_tracker_episode(self):
        with tempfile.TemporaryDirectory() as directory:
            data_file = Path(directory) / "episode_000074.hdf5"
            make_episode(data_file)
            with h5py.File(data_file, "r+") as handle:
                handle.attrs["hand_side"] = "right"
            with patch.object(fixed, "Tron2JsonClient", side_effect=AssertionError(
                "wrong-hand episode must not connect to robot"
            )):
                code = fixed.main(["--file", str(data_file), "--execute"])
            self.assertEqual(code, 2)

    def test_hand_option_previews_states_without_ros_or_robot(self):
        with tempfile.TemporaryDirectory() as directory:
            data_file = Path(directory) / "episode_000071.hdf5"
            make_episode(data_file)
            with patch.object(fixed, "Tron2JsonClient", side_effect=AssertionError(
                "offline preview must not connect to robot"
            )), patch.object(fixed.O6HandReplaySession, "acquire", side_effect=AssertionError(
                "offline preview must not acquire O6 bridge"
            )):
                code = fixed.main([
                    "--episode", "71", "--data-root", directory, "--with-o6-hand"
                ])
            self.assertEqual(code, 0)

    def test_o6_can_preview_never_starts_bridge_or_robot(self):
        with tempfile.TemporaryDirectory() as directory:
            data_file = Path(directory) / "episode_000075.hdf5"
            make_episode(data_file)
            with patch.object(fixed, "O6ReplayBridge", side_effect=AssertionError(
                "offline preview must not start O6 bridge"
            )), patch.object(fixed, "Tron2JsonClient", side_effect=AssertionError(
                "offline preview must not connect to robot"
            )):
                code = fixed.main([
                    "--file", str(data_file), "--with-o6-hand", "--o6-can", "can1"
                ])
            self.assertEqual(code, 0)

    def test_o6_can_cannot_be_used_without_hand_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            data_file = Path(directory) / "episode_000077.hdf5"
            make_episode(data_file)
            with patch.object(fixed, "O6ReplayBridge", side_effect=AssertionError(
                "invalid O6 option must not start bridge"
            )):
                code = fixed.main(["--file", str(data_file), "--o6-can", "can1"])
            self.assertEqual(code, 2)

    def test_o6_can_execute_starts_bridge_before_robot_and_closes_on_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            data_file = Path(directory) / "episode_000076.hdf5"
            make_episode(data_file)
            owner = Mock()
            with patch.object(fixed, "O6ReplayBridge", return_value=owner) as bridge_type, \
                 patch.object(fixed, "Tron2JsonClient", side_effect=RuntimeError(
                     "robot unavailable"
                 )) as robot:
                code = fixed.main([
                    "--file", str(data_file), "--with-o6-hand", "--o6-can", "can1",
                    "--trace-file", str(Path(directory) / "trace.jsonl"), "--execute",
                ])
            self.assertEqual(code, 2)
            self.assertEqual(bridge_type.call_args.args[0], "can1")
            owner.start.assert_called_once()
            robot.assert_called_once()
            owner.close.assert_called_once()

    def test_unavailable_o6_bridge_aborts_before_robot_connection(self):
        with tempfile.TemporaryDirectory() as directory:
            data_file = Path(directory) / "episode_000073.hdf5"
            make_episode(data_file)
            with patch.object(fixed.O6HandReplaySession, "probe_bridge",
                              side_effect=RuntimeError("bridge unavailable")), \
                 patch.object(fixed, "Tron2JsonClient", side_effect=AssertionError(
                     "bridge failure must be detected before robot connection"
                 )):
                code = fixed.main([
                    "--episode", "73", "--data-root", directory,
                    "--trace-file", str(Path(directory) / "trace.jsonl"),
                    "--execute", "--with-o6-hand",
                ])
            self.assertEqual(code, 2)

    def test_rejects_displacement_before_any_robot_connection(self):
        with tempfile.TemporaryDirectory() as directory:
            data_file = Path(directory) / "episode_000067.hdf5"
            make_episode(data_file)
            with patch.object(fixed, "Tron2JsonClient", side_effect=AssertionError(
                "invalid plan must not connect to robot"
            )):
                code = fixed.main([
                    "--episode", "67", "--data-root", directory,
                    "--max-displacement", "0.001", "--execute",
                ])
            self.assertEqual(code, 2)

    def test_execute_skips_movej_if_both_arms_are_already_fixed(self):
        with tempfile.TemporaryDirectory() as directory:
            data_file = Path(directory) / "episode_000068.hdf5"
            make_episode(data_file)
            client = FakeClient()
            with patch.object(fixed, "Tron2JsonClient", return_value=client), \
                 patch.object(fixed, "move_stage", side_effect=AssertionError(
                     "already fixed; MoveJ must be skipped"
                 )), \
                 patch.object(fixed, "execute_replay", return_value=True) as replay, \
                 patch("builtins.input", side_effect=AssertionError(
                     "no MoveJ confirmation when already fixed"
                 )):
                code = fixed.main([
                    "--episode", "68", "--data-root", directory,
                    "--trace-file", str(Path(directory) / "trace.jsonl"),
                    "--execute",
                ])
            self.assertEqual(code, 0)
            self.assertEqual(replay.call_count, 1)
            self.assertEqual(replay.call_args.args[2], "left")
            self.assertEqual(replay.call_args.kwargs["confirmation_word"], "REPLAY")

    def test_execute_passes_hand_session_to_existing_replay_and_closes_it(self):
        with tempfile.TemporaryDirectory() as directory:
            data_file = Path(directory) / "episode_000072.hdf5"
            make_episode(data_file)
            client = FakeClient()

            class FakeHandSession:
                states = np.zeros((3, 6))
                due_ns = np.array([0, 10_000_000, 20_000_000])
                trace = None
                closed = False
                probed = False

                def __init__(self, *_args):
                    pass

                def close(self):
                    self.closed = True

                def probe_bridge(self):
                    self.probed = True

            hand = FakeHandSession()
            with patch.object(fixed, "Tron2JsonClient", return_value=client), \
                 patch.object(fixed, "O6HandReplaySession", return_value=hand), \
                 patch.object(fixed, "execute_replay", return_value=True) as replay:
                code = fixed.main([
                    "--episode", "72", "--data-root", directory,
                    "--trace-file", str(Path(directory) / "trace.jsonl"),
                    "--execute", "--with-o6-hand",
                ])
            self.assertEqual(code, 0)
            self.assertIs(replay.call_args.kwargs["hand_replay"], hand)
            self.assertTrue(hand.closed)
            self.assertTrue(hand.probed)

    def test_execute_moves_right_then_left_to_saved_joint_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            data_file = Path(directory) / "episode_000069.hdf5"
            make_episode(data_file)
            client = FakeClient()
            client.q[0] += 0.1
            client.q[7] += 0.1
            client.pose = DualArmPose(
                fixed.FIXED_LEFT_POSITION + [0.02, 0, 0],
                fixed.FIXED_LEFT_QUATERNION,
                fixed.FIXED_RIGHT_POSITION + [0.02, 0, 0],
                fixed.FIXED_RIGHT_QUATERNION,
            )
            stages = []

            def fake_move(_client, _label, side, target, controlled_slice,
                          _duration, tolerance):
                stages.append((side, target.copy(), tolerance))
                client.q[controlled_slice] = target[controlled_slice]
                if side == "right":
                    client.q[7] += 0.003
                    client.pose = DualArmPose(
                        client.pose.left_position, client.pose.left_quaternion,
                        fixed.FIXED_RIGHT_POSITION.copy(),
                        fixed.FIXED_RIGHT_QUATERNION.copy(),
                    )
                else:
                    client.pose = captured_pose()

            with patch.object(fixed, "Tron2JsonClient", return_value=client), \
                 patch.object(fixed, "move_stage", side_effect=fake_move), \
                 patch.object(fixed, "countdown"), \
                 patch.object(fixed, "execute_replay", return_value=True) as replay, \
                 patch("builtins.input", side_effect=AssertionError(
                     "automatic preparation must not ask for PREPARE"
                 )):
                code = fixed.main([
                    "--episode", "69", "--data-root", directory,
                    "--trace-file", str(Path(directory) / "trace.jsonl"),
                    "--execute",
                ])
            self.assertEqual(code, 0)
            self.assertEqual([stage[0] for stage in stages], ["right", "left"])
            self.assertEqual(stages[0][2], fixed.RIGHT_PREPARE_JOINT_TOLERANCE_RAD)
            np.testing.assert_allclose(stages[1][1][7:14], fixed.FIXED_RIGHT_Q)
            np.testing.assert_allclose(client.q, fixed.FIXED_Q, atol=0.0031)
            self.assertEqual(replay.call_count, 1)
            self.assertEqual(replay.call_args.kwargs["confirmation_word"], "REPLAY")

    def test_right_wrist_must_reach_fixed_start_before_left_movej(self):
        with tempfile.TemporaryDirectory() as directory:
            data_file = Path(directory) / "episode_000070.hdf5"
            make_episode(data_file)
            client = FakeClient()
            client.q[7] += 0.026
            client.pose = DualArmPose(
                fixed.FIXED_LEFT_POSITION.copy(), fixed.FIXED_LEFT_QUATERNION.copy(),
                fixed.FIXED_RIGHT_POSITION + [0.0254, 0, 0],
                fixed.FIXED_RIGHT_QUATERNION.copy(),
            )
            stages = []

            def fake_move(_client, _label, side, _target, _controlled_slice,
                          _duration, _tolerance):
                stages.append(side)
                # 模拟 SDK 返回成功，但实机右腕没有发生运动。

            with patch.object(fixed, "Tron2JsonClient", return_value=client), \
                 patch.object(fixed, "move_stage", side_effect=fake_move), \
                 patch.object(fixed, "countdown"), \
                 patch.object(fixed, "execute_replay", side_effect=AssertionError(
                     "failed right preparation must not replay"
                 )):
                code = fixed.main([
                    "--episode", "70", "--data-root", directory,
                    "--trace-file", str(Path(directory) / "trace.jsonl"),
                    "--execute",
                ])
            self.assertEqual(code, 2)
            self.assertEqual(stages, ["right"])


if __name__ == "__main__":
    unittest.main()
