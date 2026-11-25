"""离线回归测试：只准备左臂，不连接机器人或 O6。"""

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

import replay_tron2_fixed_start as fixed
import replay_tron2_tracker as replay
from replay_tron2_tracker import DualArmPose, TrackerTrajectory


def sample_pose(*, left=None, right=None):
    return DualArmPose(
        fixed.FIXED_LEFT_POSITION.copy() if left is None else np.array(left, dtype=float),
        fixed.FIXED_LEFT_QUATERNION.copy(),
        np.array([0.32, -0.5, -0.4]) if right is None else np.array(right, dtype=float),
        np.array([1., 0., 0., 0.]),
    )


def sample_trajectory():
    return TrackerTrajectory(
        source=Path("unused_offline_episode.hdf5"), origin_timestamp_ns=0,
        timestamps_s=np.array([0., .1]), positions=np.zeros((2, 3)),
        quaternions=np.array([[1., 0., 0., 0.], [1., 0., 0., 0.]]),
        hand_side="left", original_frames=2,
    )


class LeftOnlyPreparationTests(unittest.TestCase):
    def setUp(self):
        # 仅在模拟客户端测试中启用；不更改真实脚本的未验证状态。
        self.validation = patch.object(fixed, "FIXED_LEFT_START_VALIDATED", True)
        self.validation.start()
        self.addCleanup(self.validation.stop)
        self.path_validation = patch.object(fixed, "FIXED_LEFT_PREPARE_PATH_VALIDATED", True)
        self.path_validation.start()
        self.addCleanup(self.path_validation.stop)
        self.args = SimpleNamespace(
            joint_tolerance=.04, position_tolerance=.02, angle_tolerance=3.,
            countdown=5, move_duration=10.,
        )
        self.right = np.array([.1, -.7, .3, -.6, .2, -.1, .4])
        self.initial_q = np.concatenate((np.zeros(7), self.right))
        self.final_q = np.concatenate((fixed.FIXED_LEFT_Q, self.right))
        self.dq = np.zeros(14)
        self.start_pose = sample_pose(left=[.3, .25, -.3])
        self.final_pose = sample_pose()
        self.client = Mock()
        self.client.joint_snapshot.side_effect = [
            (self.initial_q.copy(), self.dq.copy()),
            (self.final_q.copy(), self.dq.copy()),
        ]
        self.client.current_servo_pose.side_effect = [self.start_pose, self.final_pose]
        self.move = patch.object(fixed, "move_stage")
        self.move_mock = self.move.start()
        self.addCleanup(self.move.stop)
        self.timer = patch.object(fixed, "countdown")
        self.timer_mock = self.timer.start()
        self.addCleanup(self.timer.stop)

    def run_prepare(self):
        with redirect_stdout(io.StringIO()):
            return fixed.restore_fixed_start(
                self.client, self.initial_q, self.dq, self.start_pose, self.args,
            )

    def test_one_left_move_keeps_measured_right_joint_targets(self):
        original = self.initial_q.copy()
        result = self.run_prepare()
        self.assertIs(result, self.final_pose)
        self.move_mock.assert_called_once()
        values = self.move_mock.call_args.args
        self.assertEqual(values[2], "left")
        np.testing.assert_array_equal(values[3][:7], fixed.FIXED_LEFT_Q)
        np.testing.assert_array_equal(values[3][7:14], self.right)
        self.assertEqual(values[4], slice(0, 7))
        np.testing.assert_array_equal(self.initial_q, original)

    def test_left_already_ready_does_not_home_a_different_right_arm(self):
        self.initial_q = self.final_q.copy()
        self.start_pose = self.final_pose
        self.client.joint_snapshot.side_effect = [(self.final_q.copy(), self.dq.copy())]
        self.client.current_servo_pose.side_effect = [self.final_pose]
        self.run_prepare()
        self.move_mock.assert_not_called()
        self.timer_mock.assert_not_called()

    def test_right_motion_during_countdown_aborts_before_movej(self):
        moved = self.initial_q.copy()
        moved[7] += .1
        self.client.joint_snapshot.side_effect = [(moved, self.dq.copy())]
        with self.assertRaisesRegex(RuntimeError, "右臂偏离本次保持起点"):
            self.run_prepare()
        self.move_mock.assert_not_called()

    def test_right_drift_after_left_move_is_not_corrected_automatically(self):
        self.client.current_servo_pose.side_effect = [
            self.start_pose, sample_pose(right=[.36, -.5, -.4]),
        ]
        with self.assertRaisesRegex(RuntimeError, "不自动纠正右臂"):
            self.run_prepare()
        self.move_mock.assert_called_once()
        self.assertEqual(self.move_mock.call_args.args[2], "left")

    def test_wrong_left_target_is_still_rejected(self):
        wrong = self.final_q.copy()
        wrong[0] += .1
        self.client.joint_snapshot.side_effect = [
            (self.initial_q.copy(), self.dq.copy()), (wrong, self.dq.copy()),
        ]
        with self.assertRaisesRegex(RuntimeError, "左臂固定起点关节最大误差"):
            self.run_prepare()

    def test_unvalidated_start_rejects_direct_helper_without_queries(self):
        with patch.object(fixed, "FIXED_LEFT_START_VALIDATED", False):
            with self.assertRaisesRegex(RuntimeError, "尚未验证 O6"):
                self.run_prepare()
        self.client.joint_snapshot.assert_not_called()
        self.client.current_servo_pose.assert_not_called()
        self.move_mock.assert_not_called()

    def test_right_has_no_historical_fixed_pose(self):
        with self.assertRaisesRegex(ValueError, "右臂不得恢复历史目标"):
            fixed.check_fixed_arm_pose(self.final_pose, "right", .02, 3.)


class FixedStartEntryTests(unittest.TestCase):
    def setUp(self):
        # 模拟真机入口的测试也不在真实日志目录写入伪造的轨迹记录。
        trace = patch.object(fixed, "TraceRecorder")
        self.trace_class = trace.start()
        self.trace_class.return_value.path = Path("/tmp/offline-fixed-start.jsonl")
        self.addCleanup(trace.stop)

    def test_default_and_explicit_control_are_servop(self):
        for flags in ([], ["--arm-control", "servop"]):
            args = fixed.parse_args(["--episode", "65", *flags])
            fixed.validate_args(args)
            self.assertEqual(args.arm_control, "servop")
            self.assertEqual(args.scale, 1.0)
            self.assertEqual(args.countdown, 5)
            self.assertFalse(hasattr(args, "ik_model"))

    def test_old_servoj_cli_is_rejected_without_connecting(self):
        with patch.object(fixed, "Tron2JsonClient") as robot, \
                redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                fixed.main(["--episode", "65", "--arm-control", "servoj", "--execute"])
        self.assertEqual(error.exception.code, 2)
        robot.assert_not_called()

    def test_offline_preview_is_available_without_hardware(self):
        with patch.object(fixed, "load_tracker", return_value=sample_trajectory()), \
                patch.object(fixed, "Tron2JsonClient") as robot, \
                patch.object(fixed, "O6ReplayBridge") as bridge, \
                redirect_stdout(io.StringIO()) as output:
            code = fixed.main(["--episode", "65"])
        self.assertEqual(code, 0)
        self.assertIn("右臂不自动归位", output.getvalue())
        self.assertIn("已在正式起点可跳过恢复", output.getvalue())
        self.assertIn("回放控制：ServoP", output.getvalue())
        self.assertNotIn("离线 IK 通过", output.getvalue())
        robot.assert_not_called()
        bridge.assert_not_called()

    def test_offline_servop_does_not_import_local_ik_or_mujoco(self):
        with patch.dict("sys.modules", {"tron2_left_ik": None, "mujoco": None}), \
                patch.object(fixed, "load_tracker", return_value=sample_trajectory()), \
                patch.object(fixed, "Tron2JsonClient") as robot, \
                redirect_stdout(io.StringIO()):
            self.assertEqual(fixed.main(["--episode", "65"]), 0)
        robot.assert_not_called()

    def test_execute_rejects_old_start_before_robot_or_o6_bridge(self):
        with patch.object(fixed, "FIXED_LEFT_START_VALIDATED", False), \
                patch.object(fixed, "load_tracker", return_value=sample_trajectory()), \
                patch.object(fixed, "acquire_replay_lock", return_value=Mock()), \
                patch.object(fixed, "O6HandReplaySession") as hand, \
                patch.object(fixed, "O6ReplayBridge") as bridge, \
                patch.object(fixed, "Tron2JsonClient") as robot, \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as errors:
            hand.return_value.states = np.zeros((2, 6))
            hand.return_value.due_ns = np.array([0, 500_000_000])
            code = fixed.main([
                "--episode", "65", "--with-o6-hand", "--o6-can", "can1", "--execute",
            ])
        self.assertEqual(code, 2)
        self.assertIn("尚未验证 O6", errors.getvalue())
        robot.assert_not_called()
        bridge.assert_not_called()
        hand.return_value.acquire.assert_not_called()
        hand.return_value.probe_bridge.assert_not_called()

    def test_recorded_pose_does_not_unlock_motion_before_path_validation(self):
        with patch.object(fixed, "FIXED_LEFT_START_VALIDATED", True), \
                patch.object(fixed, "FIXED_LEFT_PREPARE_PATH_VALIDATED", False), \
                patch.object(fixed, "load_tracker", return_value=sample_trajectory()), \
                patch.object(fixed, "acquire_replay_lock", return_value=Mock()), \
                patch.object(fixed, "Tron2JsonClient") as robot, \
                patch.object(fixed, "O6ReplayBridge") as bridge, \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as errors:
            robot.return_value.accid = fixed.FIXED_ROBOT_ID
            robot.return_value.joint_snapshot.return_value = (np.zeros(16), np.zeros(16))
            robot.return_value.current_servo_pose.return_value = sample_pose(left=[.3, .2, -.3])
            code = fixed.main(["--episode", "65", "--execute"])
        self.assertEqual(code, 2)
        self.assertIn("自动恢复路径尚未验证", errors.getvalue())
        robot.assert_called_once()
        bridge.assert_not_called()

    def test_readonly_check_is_available_for_unvalidated_start(self):
        with patch.object(fixed, "load_tracker", return_value=sample_trajectory()), \
                patch.object(fixed, "Tron2JsonClient") as robot, \
                patch.object(fixed, "restore_fixed_start") as prepare, \
                patch.object(fixed, "execute_replay") as replay, \
                redirect_stdout(io.StringIO()):
            robot.return_value.accid = fixed.FIXED_ROBOT_ID
            robot.return_value.joint_snapshot.return_value = (
                np.concatenate((fixed.FIXED_LEFT_Q, np.zeros(9))), np.zeros(16),
            )
            robot.return_value.current_servo_pose.return_value = sample_pose()
            code = fixed.main(["--episode", "65", "--check-robot"])
        self.assertEqual(code, 0)
        prepare.assert_not_called()
        replay.assert_not_called()

    def test_hanging_right_arm_is_reported_readonly_without_motion(self):
        with patch.object(fixed, "load_tracker", return_value=sample_trajectory()), \
                patch.object(fixed, "Tron2JsonClient") as robot, \
                patch.object(fixed, "restore_fixed_start") as prepare, \
                patch.object(fixed, "execute_replay") as replay, \
                redirect_stdout(io.StringIO()) as output:
            robot.return_value.accid = fixed.FIXED_ROBOT_ID
            robot.return_value.joint_snapshot.return_value = (np.zeros(16), np.zeros(16))
            robot.return_value.current_servo_pose.return_value = sample_pose(
                right=[-.0174, -.211, -.7535],
            )
            self.assertEqual(fixed.main(["--episode", "65", "--check-robot"]), 0)
        self.assertNotIn("ServoP 执行条件未满足", output.getvalue())
        self.assertIn("右腕不做 MoveP 工作空间验收", output.getvalue())
        prepare.assert_not_called()
        replay.assert_not_called()

    def test_hanging_right_arm_does_not_bypass_unvalidated_left_prepare_path(self):
        with patch.object(fixed, "load_tracker", return_value=sample_trajectory()), \
                patch.object(fixed, "FIXED_LEFT_PREPARE_PATH_VALIDATED", False), \
                patch.object(fixed, "acquire_replay_lock", return_value=Mock()), \
                patch.object(fixed, "Tron2JsonClient") as robot, \
                patch.object(fixed, "move_stage") as move, \
                patch.object(fixed, "execute_replay") as replay, \
                patch.object(fixed, "O6HandReplaySession") as hand, \
                patch.object(fixed, "O6ReplayBridge") as bridge, \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as errors:
            robot.return_value.accid = fixed.FIXED_ROBOT_ID
            robot.return_value.joint_snapshot.return_value = (np.zeros(16), np.zeros(16))
            robot.return_value.current_servo_pose.return_value = sample_pose(
                right=[-.0174, -.211, -.7535],
            )
            hand.return_value.states = np.zeros((2, 6))
            hand.return_value.due_ns = np.array([0, 500_000_000])
            code = fixed.main([
                "--episode", "65", "--with-o6-hand", "--o6-can", "can1", "--execute",
            ])
        self.assertEqual(code, 2)
        self.assertIn("自动恢复路径尚未验证", errors.getvalue())
        self.assertNotIn("ServoP 保持臂 right", errors.getvalue())
        move.assert_not_called()
        replay.assert_not_called()
        bridge.assert_not_called()
        hand.return_value.probe_bridge.assert_not_called()
        hand.return_value.acquire.assert_not_called()

    def test_hanging_right_arm_allows_already_ready_left_to_reach_replay(self):
        pose = sample_pose(right=[.0988, -.1825, -.7301])
        q = np.r_[fixed.FIXED_LEFT_Q, np.zeros(9)]
        with patch.object(fixed, "load_tracker", return_value=sample_trajectory()), \
                patch.object(fixed, "FIXED_LEFT_PREPARE_PATH_VALIDATED", False), \
                patch.object(fixed, "acquire_replay_lock", return_value=Mock()), \
                patch.object(fixed, "Tron2JsonClient") as robot, \
                patch.object(fixed, "move_stage") as move, \
                patch.object(fixed, "execute_replay", return_value=True) as replay, \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            robot.return_value.accid = fixed.FIXED_ROBOT_ID
            robot.return_value.joint_snapshot.return_value = (q, np.zeros(16))
            robot.return_value.current_servo_pose.return_value = pose
            code = fixed.main(["--episode", "65", "--execute"])
        self.assertEqual(code, 0)
        move.assert_not_called()
        replay.assert_called_once()
        self.assertEqual(replay.call_args.args[2], "left")


class RecordedLeftStartTests(unittest.TestCase):
    def test_reference_constants_preserve_the_readonly_capture(self):
        path = Path(__file__).parent / (
            "camera_alignment_results/top_20261022_204900/left_start_20261022_205119.json"
        )
        report = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(report["robot_id"], fixed.FIXED_ROBOT_ID)
        self.assertTrue(report["read_only"])
        self.assertEqual(report["preparation_arm"], "left")
        self.assertEqual(len(report["samples"]), 6)
        for values, field in (
                (fixed.FIXED_LEFT_REFERENCE_Q, "q_rad"),
                (fixed.FIXED_LEFT_REFERENCE_POSITION, "position_m"),
                (fixed.FIXED_LEFT_REFERENCE_QUATERNION, "quaternion_wxyz")):
            np.testing.assert_allclose(values, report["left_start"][field], atol=1e-8, rtol=0)
        self.assertTrue(fixed.FIXED_LEFT_START_VALIDATED)
        self.assertFalse(fixed.FIXED_LEFT_PREPARE_PATH_VALIDATED)
        self.assertFalse(report["prepare_path_validated"])
        self.assertFalse(report["right_reference_only"]["inside_documented_servop_workspace"])

    def test_active_start_is_horizontal_with_centered_wrist(self):
        profile = json.loads(fixed.FIXED_LEFT_START_PROFILE.read_text())
        np.testing.assert_array_equal(fixed.FIXED_LEFT_Q, profile['target_left_q_rad'])
        np.testing.assert_array_equal(fixed.FIXED_LEFT_POSITION, profile['target_position_m'])
        np.testing.assert_array_equal(fixed.FIXED_LEFT_QUATERNION, profile['target_quaternion_wxyz'])
        np.testing.assert_allclose(
            fixed.FIXED_LEFT_POSITION - fixed.FIXED_LEFT_REFERENCE_POSITION,
            fixed.FIXED_LEFT_POSITION_OFFSET_M, atol=1e-12,
        )
        self.assertEqual(fixed.FIXED_LEFT_HEIGHT_OFFSET_M, fixed.FIXED_LEFT_POSITION_OFFSET_M[2])
        np.testing.assert_allclose(
            fixed.FIXED_LEFT_Q,
            [0., 0., 0.18322281083347552, -np.pi/2, -np.pi/2, 0., 0.],
            atol=1e-12, rtol=0,
        )
        np.testing.assert_allclose(
            fixed.FIXED_LEFT_POSITION, [.450789437, .253329025, -.30589],
            atol=1e-9, rtol=0,
        )
        np.testing.assert_allclose(
            fixed.FIXED_LEFT_QUATERNION,
            [.5436449623382834, .5436449623382833,
             -.4521616468965567, -.4521616468965567],
            atol=1e-12, rtol=0,
        )
        for key in ('upper_arm_vertical_error_deg', 'forearm_elevation_deg',
                    'hand_elevation_deg', 'hand_forearm_alignment_deg',
                    'palm_down_error_deg', 'palm_roll_error_deg'):
            with self.subTest(geometry=key):
                self.assertLess(abs(profile['target_angles_deg'][key]), 1e-5)
        self.assertTrue(profile['target_angles_deg']['palm_roll_reference_valid'])
        self.assertFalse(fixed.FIXED_LEFT_START_MEASURED)
        self.assertFalse(fixed.FIXED_LEFT_PREPARE_PATH_VALIDATED)

    def test_previous_episode8_pose_is_not_accepted_as_horizontal_start(self):
        previous_path = Path(__file__).parent / (
            'validation_results/episode8_height_start_20261028/fixed_start_profile.json'
        )
        previous = json.loads(previous_path.read_text())
        args = SimpleNamespace(joint_tolerance=.04, position_tolerance=.02, angle_tolerance=3.)
        old_q = np.r_[previous['target_left_q_rad'], np.zeros(9)]
        new_q = np.r_[fixed.FIXED_LEFT_Q, np.zeros(9)]
        new_pose = sample_pose()
        old_pose = DualArmPose(
            np.asarray(previous['target_position_m']),
            np.asarray(previous['target_quaternion_wxyz']),
            new_pose.right_position, new_pose.right_quaternion,
        )
        wrong_position = sample_pose(left=previous['target_position_m'])
        wrong_orientation = DualArmPose(
            fixed.FIXED_LEFT_POSITION.copy(), old_pose.left_quaternion,
            new_pose.right_position, new_pose.right_quaternion,
        )
        with redirect_stdout(io.StringIO()):
            self.assertFalse(fixed.already_at_fixed_start(old_q, old_pose, args))
            self.assertFalse(fixed.already_at_fixed_start(new_q, wrong_position, args))
            self.assertFalse(fixed.already_at_fixed_start(new_q, wrong_orientation, args))
            self.assertTrue(fixed.already_at_fixed_start(new_q, new_pose, args))

    def test_active_start_measurement_status_matches_its_own_evidence(self):
        profile = json.loads(fixed.FIXED_LEFT_START_PROFILE.read_text())
        self.assertEqual(fixed.FIXED_LEFT_START_MEASURED, profile['target_measured_within_tolerance'])
        if not fixed.FIXED_LEFT_START_MEASURED:
            self.assertIsNone(profile['measured_result'])
            self.assertEqual(profile['status'], 'planned_not_executed')
            self.assertTrue(Path(profile['source_plan_file']).is_file())
            self.assertFalse(profile['automatic_restore_path_approved'])
            return
        measured = profile['measured_result']
        self.assertEqual(measured['result'], 'completed')
        for key in ('upper_arm_vertical_error_deg', 'forearm_elevation_deg',
                    'hand_forearm_alignment_deg', 'palm_roll_error_deg'):
            self.assertLess(abs(measured['actual_geometry'][key]-profile['target_angles_deg'][key]), 2.)
        self.assertLess(np.linalg.norm(np.array(measured['final_sdk_position_m'])-
                                      fixed.FIXED_LEFT_POSITION), .02)
        self.assertLess(fixed.quaternion_distance_deg(measured['final_sdk_quaternion_wxyz'],
                                                    fixed.FIXED_LEFT_QUATERNION), 3.)
        self.assertFalse(profile['automatic_restore_path_approved'])


class CalibratedReplayCliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.file = self.root / 'episode.hdf5'
        self.calibration = self.root / 'calibration.json'
        self.R = Rotation.from_euler('z', 30, degrees=True).as_matrix()
        world = np.eye(4)
        world[:3, :3] = self.R
        world[:3, 3] = [10., -20., 30.]  # 地图原点不应使首帧跳走。
        mount = np.eye(4)
        mount[:3, 3] = [.07, .03, -.11]  # 真人旧数据不能套用这段机器人安装偏移。
        self.calibration.write_text(json.dumps({
            'position_unit': 'm', 'tracker_quaternion_convention': 'raw-xyzw',
            'session': {'arm': 'left', 'robot_id': fixed.FIXED_ROBOT_ID},
            'T_BW': world.tolist(), 'T_ET': mount.tolist(),
        }))
        correct_q = Rotation.from_euler('x', [10., 30.], degrees=True).as_quat()[:, [3, 0, 1, 2]]
        self.correct_q = correct_q
        with h5py.File(self.file, 'w') as f:
            f.attrs['hand_side'] = 'left'
            f.create_dataset('timestamps/master_timestamp_ns', data=[1000000000, 1100000000])
            g = f.create_group('vive_tracker')
            g.attrs['position_unit'] = 'meter'
            g.attrs['quat_order'] = '[w, x, y, z]'
            g.create_dataset('pos', data=np.array([[[1., 2., 3.]], [[1.01, 2., 3.]]]))
            g.create_dataset('quat', data=correct_q[:, [1, 0, 3, 2]][:, None, :])
            g.create_dataset('valid', data=np.ones((2, 1), dtype=bool))
        self.flags = ['--file', str(self.file), '--tracker-calibration', str(self.calibration),
                      '--tracker-quaternion', 'pyvut-legacy', '--orientation-scale', '.5',
                      '--scale', '1', '--speed', '.2']

    def test_fixed_offline_uses_calibration_and_never_connects(self):
        with patch.object(fixed, 'Tron2JsonClient') as robot, \
                patch.object(fixed, 'O6ReplayBridge') as bridge, \
                redirect_stdout(io.StringIO()) as output:
            self.assertEqual(fixed.main(self.flags), 0)
        self.assertIn('calibrated-tracker-relative', output.getvalue())
        self.assertIn('10.00°', output.getvalue())
        robot.assert_not_called()
        bridge.assert_not_called()

    def test_tracker_entry_also_accepts_matrix_without_axis_map(self):
        with patch.object(replay, 'Tron2JsonClient') as robot, redirect_stdout(io.StringIO()):
            self.assertEqual(replay.main(self.flags), 0)
        robot.assert_not_called()

    def test_both_entry_points_forward_no_interpolation_without_hardware(self):
        for entry in (fixed, replay):
            with self.subTest(entry=entry.__name__), \
                    patch.object(entry, 'sample_relative_trajectory',
                                 wraps=replay.sample_relative_trajectory) as sampler, \
                    patch.object(entry, 'Tron2JsonClient') as robot, \
                    redirect_stdout(io.StringIO()) as output:
                self.assertEqual(entry.main(self.flags + ['--interpolation', 'none']), 0)
            self.assertEqual(sampler.call_args.args[-1], 'none')
            self.assertIn('Tracker目标插值: none', output.getvalue())
            robot.assert_not_called()

    def test_first_pose_and_rotated_target_reach_existing_compose_path(self):
        trajectory = replay.load_tracker(self.file, .15)
        args = fixed.parse_args(self.flags)
        dp, dq, angles = replay.prepare_tracker_motion(
            trajectory, args, 'left', default_axis_map='y,z,x')
        np.testing.assert_allclose(dp[1], self.R @ [.01, 0., 0.], atol=1e-12)
        self.assertAlmostEqual(angles[1], 10.)
        p0, q0 = replay.compose_pose(fixed.FIXED_LEFT_POSITION, fixed.FIXED_LEFT_QUATERNION,
                                    dp[0], dq[0])
        np.testing.assert_allclose(p0, fixed.FIXED_LEFT_POSITION)
        self.assertLess(replay.quaternion_distance_deg(q0, fixed.FIXED_LEFT_QUATERNION), 1e-5)
        expected_delta = self.R @ Rotation.from_euler('x', 10, degrees=True).as_matrix() @ self.R.T
        np.testing.assert_allclose(replay.quaternion_to_matrix(dq[1]), expected_delta, atol=1e-12)
        self.assertFalse(args.tracker_mapping_info['T_ET_applied'])
        self.assertIsNone(args.axis_map)
        self.assertEqual(args.tracker_mapping_info['input_quaternion_format'], 'pyvut-legacy')

    def test_old_default_axes_remain_unchanged_without_calibration(self):
        args = fixed.parse_args(['--file', str(self.file)])
        trajectory = replay.load_tracker(self.file, .15)
        dp, dq, _ = replay.prepare_tracker_motion(trajectory, args, 'left', default_axis_map='y,z,x')
        np.testing.assert_allclose(dp[1], [0., 0., .01], atol=1e-12)
        self.assertEqual(args.axis_map, 'y,z,x')

    def test_double_axis_mapping_rejected_before_hardware_or_data_access(self):
        with patch.object(fixed, 'load_tracker') as read, \
                patch.object(fixed, 'Tron2JsonClient') as robot, \
                redirect_stderr(io.StringIO()):
            self.assertEqual(fixed.main(self.flags + ['--axis-map', 'y,z,x', '--execute']), 2)
        read.assert_not_called()
        robot.assert_not_called()

    def test_undeclared_quaternion_order_rejected_without_guessing(self):
        with h5py.File(self.file, 'r+') as f:
            del f['vive_tracker'].attrs['quat_order']
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as errors:
            self.assertEqual(fixed.main(self.flags), 2)
        self.assertIn('quat_order', errors.getvalue())

    def test_calibration_robot_identity_is_checked(self):
        args = fixed.parse_args(self.flags)
        trajectory = replay.load_tracker(self.file, .15)
        replay.prepare_tracker_motion(trajectory, args, 'left', default_axis_map='y,z,x')
        with self.assertRaisesRegex(RuntimeError, '标定针对机器人'):
            replay.check_tracker_mapping_robot(args, 'different_robot')

    def test_new_mapping_preserves_episode_time_origin_and_hand_clock(self):
        trajectory = replay.load_tracker(self.file, .15)
        args = fixed.parse_args(self.flags)
        dp, dq, _ = replay.prepare_tracker_motion(trajectory, args, 'left', default_axis_map='y,z,x')
        times, sampled_p, sampled_q = replay.sample_relative_trajectory(trajectory, dp, dq, 300, .2)
        self.assertEqual(trajectory.origin_timestamp_ns, 1000000000)
        self.assertAlmostEqual(times[-1], .5)
        np.testing.assert_allclose(sampled_p[-1], dp[-1])
        np.testing.assert_allclose(sampled_q[-1], dq[-1])


class TrackerSamplingTests(unittest.TestCase):
    def setUp(self):
        self.trajectory = TrackerTrajectory(
            source=Path('offline.hdf5'), origin_timestamp_ns=123456789,
            timestamps_s=np.array([0., .1, .25]),
            positions=np.zeros((3, 3)), quaternions=np.tile([1., 0., 0., 0.], (3, 1)),
            hand_side='left', original_frames=3,
        )
        self.positions = np.array([[0., 0., 0.], [.02, -.03, .04], [.01, .05, -.02]])
        self.quats = Rotation.from_euler('z', [0., 40., 80.], degrees=True).as_quat()[:, [3, 0, 1, 2]]

    def sample(self, *, rate=20., speed=1., mode='none'):
        return replay.sample_relative_trajectory(self.trajectory, self.positions,
                                                self.quats, rate, speed, mode)

    def test_raw_frames_are_held_until_due_including_exact_knot_and_endpoint(self):
        original_p, original_q = self.positions.copy(), self.quats.copy()
        times, positions, quats = self.sample()
        np.testing.assert_allclose(times, [0., .05, .1, .15, .2, .25])
        np.testing.assert_array_equal(positions, self.positions[[0, 0, 1, 1, 1, 2]])
        np.testing.assert_array_equal(quats, self.quats[[0, 0, 1, 1, 1, 2]])
        np.testing.assert_array_equal(self.positions, original_p)
        np.testing.assert_array_equal(self.quats, original_q)
        self.assertEqual(self.trajectory.origin_timestamp_ns, 123456789)

    def test_speed_changes_update_times_without_changing_raw_poses(self):
        times, positions, quats = self.sample(speed=.5)
        self.assertEqual(times[-1], .5)
        np.testing.assert_array_equal(positions[:4], np.tile(self.positions[0], (4, 1)))
        np.testing.assert_array_equal(positions[4:-1], np.tile(self.positions[1], (6, 1)))
        np.testing.assert_array_equal(positions[-1], self.positions[-1])
        np.testing.assert_array_equal(quats[4], self.quats[1])

    def test_irregular_frame_updates_are_never_early_or_more_than_one_tick_late(self):
        self.trajectory = replace(self.trajectory, timestamps_s=np.array([0., .073, .191]))
        times, positions, _ = self.sample(rate=100.)
        updates = np.r_[0, np.flatnonzero(np.any(np.diff(positions, axis=0) != 0, axis=1))+1]
        self.assertEqual(len(updates), 3)
        np.testing.assert_array_equal(positions[updates], self.positions)
        lag = times[updates] - self.trajectory.timestamps_s
        self.assertTrue(np.all(lag >= 0))
        self.assertLessEqual(lag.max(), .01)

    def test_none_rejects_skipping_source_frames_when_command_rate_is_insufficient(self):
        with self.assertRaisesRegex(ValueError, '不会静默跳帧'):
            self.sample(rate=1.)

    def test_default_linear_position_and_slerp_are_unchanged(self):
        trajectory = sample_trajectory()
        positions = np.array([[0., 0., 0.], [.02, -.01, .04]])
        quats = Rotation.from_euler('z', [0., 90.], degrees=True).as_quat()[:, [3, 0, 1, 2]]
        implicit = replay.sample_relative_trajectory(trajectory, positions, quats, 20., 1.)
        explicit = replay.sample_relative_trajectory(trajectory, positions, quats, 20., 1., 'linear')
        for a, b in zip(implicit, explicit):
            np.testing.assert_array_equal(a, b)
        np.testing.assert_allclose(implicit[1][1], [.01, -.005, .02])
        mid_q = Rotation.from_euler('z', 45., degrees=True).as_quat()[[3, 0, 1, 2]]
        self.assertLess(replay.quaternion_distance_deg(implicit[2][1], mid_q), 1e-5)

if __name__ == "__main__":
    unittest.main()
