"""恢复路径试验回归：全部使用模拟客户端，不连接或移动机器人。"""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

import replay_tron2_fixed_start as fixed
import restore_tron2_fixed_start as restore
import tron2_restore_trial as trial
from test_replay_tron2_fixed_start import sample_pose, sample_trajectory


class TrialTests(unittest.TestCase):
    def setUp(self):
        self.q = np.r_[fixed.FIXED_LEFT_Q, np.zeros(9)]
        self.q[3] += .1
        self.pose = sample_pose()
        self.args = SimpleNamespace(joint_tolerance=.04, position_tolerance=.02,
                                    angle_tolerance=3., countdown=5,
                                    trial_step_duration=3.)
        first = self.q.copy()
        first[3] -= .05
        final = self.q.copy()
        final[:7] = fixed.FIXED_LEFT_Q
        self.waypoints = np.array([first, final])
        self.client = Mock()
        self.client.accid = fixed.FIXED_ROBOT_ID
        self.actual = self.q.copy()
        self.client.joint_snapshot.side_effect = lambda **kwargs: (self.actual.copy(), np.zeros(16))
        self.client.current_servo_pose.return_value = self.pose
        self.sent = []

        def request(title, data, response, **kwargs):
            self.sent.append((title, data, response))
            self.assertEqual(title, "request_movej")
            self.actual[:14] = data["joint"]
            return {"data": {"result": "success"}}

        self.client.request.side_effect = request
        self.plan = patch.object(trial, "plan_trial", return_value=(
            self.waypoints, {"maximum_planned_joint_step_deg": math.degrees(.05)},
        ))
        self.plan_mock = self.plan.start()
        self.addCleanup(self.plan.stop)
        self.timer = patch.object(fixed, "countdown")
        self.timer_mock = self.timer.start()
        self.addCleanup(self.timer.stop)
        self.wait = patch.object(trial, "wait_for_target")
        self.wait_mock = self.wait.start()
        self.addCleanup(self.wait.stop)

    def run_trial(self, answer="RESTORE", trace=None):
        with patch("builtins.input", return_value=answer), redirect_stdout(io.StringIO()):
            trial.run_trial(self.client, self.q, np.zeros(16), self.pose, self.args, trace)

    def test_cancel_sends_no_motion_and_no_countdown(self):
        with self.assertRaisesRegex(RuntimeError, "cancelled"):
            self.run_trial(answer="no")
        self.client.request.assert_not_called()
        self.timer_mock.assert_not_called()

    def test_eof_sends_no_motion(self):
        with patch("builtins.input", side_effect=EOFError), redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "未收到现场"):
                trial.run_trial(self.client, self.q, np.zeros(16), self.pose, self.args)
        self.client.request.assert_not_called()

    def test_segments_send_only_14_joint_movej_and_do_not_enable_auto_restore(self):
        trace = Mock()
        with patch.object(fixed, "FIXED_LEFT_PREPARE_PATH_VALIDATED", False):
            self.run_trial(trace=trace)
            self.assertFalse(fixed.FIXED_LEFT_PREPARE_PATH_VALIDATED)
        self.assertEqual(len(self.sent), 2)
        previous = self.q[:14].copy()
        for title, data, response in self.sent:
            self.assertEqual(title, "request_movej")
            self.assertEqual(response, "response_movej")
            self.assertEqual(set(data), {"joint", "time"})
            self.assertEqual(len(data["joint"]), 14)
            target = np.array(data["joint"])
            np.testing.assert_array_equal(target[7:14], self.q[7:14])
            self.assertLessEqual(float(np.max(np.abs(target-previous))), trial.MAX_JOINT_STEP_RAD)
            previous = target
        np.testing.assert_array_equal(previous[:7], fixed.FIXED_LEFT_Q)
        self.assertEqual(self.wait_mock.call_count, 2)
        for call in self.wait_mock.call_args_list:
            self.assertLessEqual(call.args[-1], .015)
        endpoint = [c for c in trace.record.call_args_list if c.args[0] == "restore_trial_endpoint_reached"]
        self.assertEqual(len(endpoint), 1)
        self.assertFalse(endpoint[0].kwargs["path_approved"])

    def test_failure_after_first_command_does_not_send_second_command(self):
        self.wait_mock.side_effect = RuntimeError("MoveJ did not settle")
        with self.assertRaisesRegex(RuntimeError, "did not settle"):
            self.run_trial()
        self.assertEqual(len(self.sent), 1)

    def test_snapshot_timeout_before_next_segment_does_not_repeat_or_advance_motion(self):
        def pose(label):
            if label == 'restore_trial_before_2':
                raise TimeoutError('no response_get_servo_pose within 4.0s')
            return self.pose
        self.client.current_servo_pose.side_effect = pose
        with self.assertRaisesRegex(TimeoutError, '第 2/2 段发送前.*尚未发送'):
            self.run_trial()
        self.assertEqual(len(self.sent), 1)

    def test_snapshot_timeout_after_motion_does_not_claim_endpoint_or_advance(self):
        def pose(label):
            if label == 'restore_trial_after_1':
                raise TimeoutError('no response_get_servo_pose within 4.0s')
            return self.pose
        trace = Mock()
        self.client.current_servo_pose.side_effect = pose
        with self.assertRaisesRegex(TimeoutError, 'MoveJ 已发送.*未发送下一段'):
            self.run_trial(trace=trace)
        self.assertEqual(len(self.sent), 1)
        self.assertNotIn('restore_trial_endpoint_reached', [c.args[0] for c in trace.record.call_args_list])

    def test_not_reached_or_unplanned_right_or_head_motion_stops_next_segment(self):
        original = self.client.request.side_effect
        for joint in (0, 7, 14):
            with self.subTest(joint=joint):
                self.actual = self.q.copy()
                self.sent.clear()

                def changed(*args, **kwargs):
                    reply = original(*args, **kwargs)
                    self.actual[joint] += .1
                    return reply

                self.client.request.side_effect = changed
                with self.assertRaises(RuntimeError):
                    self.run_trial()
                self.assertEqual(len(self.sent), 1)

    def test_rejected_movej_is_not_retried(self):
        self.client.request.side_effect = None
        self.client.request.return_value = {"data": {"result": "fail_is_not_develop_mode"}}
        with self.assertRaisesRegex(RuntimeError, "未被接受"):
            self.run_trial()
        self.client.request.assert_called_once()
        self.wait_mock.assert_not_called()

    def test_changed_start_after_confirmation_is_not_reanchored(self):
        self.actual[0] += .02
        with self.assertRaisesRegex(RuntimeError, "确认后机器人姿态已变化"):
            self.run_trial()
        self.client.request.assert_not_called()

    def test_stationary_right_shift_replans_and_holds_latest_right(self):
        # 复现 233906 日志：左臂不变，右第3/5关节静态变化约1°。
        self.actual[9] += .0183
        self.actual[11] -= .0174
        refreshed = self.actual.copy()
        new_waypoints = self.waypoints.copy()
        new_waypoints[:, 7:] = refreshed[7:]
        report = {"maximum_planned_joint_step_deg": math.degrees(.05)}
        self.plan_mock.side_effect = [(self.waypoints, report), (new_waypoints, report)]
        trace = Mock()
        self.run_trial(trace=trace)
        self.assertEqual(self.plan_mock.call_count, 2)
        np.testing.assert_array_equal(self.plan_mock.call_args.args[0], refreshed)
        self.assertEqual(len(self.sent), 2)
        for _, data, _ in self.sent:
            np.testing.assert_array_equal(data['joint'][7:14], refreshed[7:14])
        events = [c for c in trace.record.call_args_list if c.args[0] == 'restore_trial_replan']
        self.assertEqual(len(events), 1)
        np.testing.assert_array_equal(events[0].kwargs['previous_q'], self.q)
        np.testing.assert_array_equal(events[0].kwargs['right_hold_q'], refreshed[7:14])

    def test_replan_collision_prevents_every_movej(self):
        self.actual[9] += .0183
        self.plan_mock.side_effect = [
            (self.waypoints, {"maximum_planned_joint_step_deg": math.degrees(.05)}),
            RuntimeError('恢复试验模型碰撞'),
        ]
        with self.assertRaisesRegex(RuntimeError, '模型碰撞'):
            self.run_trial()
        self.client.request.assert_not_called()

    def test_excessive_right_change_is_not_silently_rebased(self):
        self.actual[9] += self.args.joint_tolerance + .01
        with self.assertRaisesRegex(RuntimeError, '右臂偏离本次保持起点'):
            self.run_trial()
        self.assertEqual(self.plan_mock.call_count, 1)
        self.client.request.assert_not_called()

    def test_pre_start_motion_or_invalid_feedback_cannot_be_rebased(self):
        for kind in ('moving', 'nonfinite', 'shape'):
            with self.subTest(kind=kind):
                def snapshot(label):
                    q = self.q.copy()
                    dq = np.zeros(16)
                    if kind == 'moving':
                        dq[9] = .06
                    elif kind == 'nonfinite':
                        q[9] = float('nan')
                    else:
                        q = q[:14]
                    return q, dq
                self.client.joint_snapshot.side_effect = snapshot
                with self.assertRaisesRegex(RuntimeError, '未停稳或反馈无效'):
                    self.run_trial()
                self.client.request.assert_not_called()

    def test_replan_pose_timeout_sends_no_command(self):
        self.actual[9] += .0183
        self.client.current_servo_pose.side_effect = TimeoutError('no fresh pose')
        with self.assertRaisesRegex(TimeoutError, 'no fresh pose'):
            self.run_trial()
        self.client.request.assert_not_called()

    def test_change_again_after_replan_rejects_stale_first_target(self):
        refreshed = self.q.copy()
        refreshed[9] += .0183
        new_waypoints = self.waypoints.copy()
        new_waypoints[:, 7:] = refreshed[7:]
        report = {"maximum_planned_joint_step_deg": math.degrees(.05)}
        self.plan_mock.side_effect = [(self.waypoints, report), (new_waypoints, report)]

        def snapshot(label):
            q = refreshed.copy()
            if label == 'restore_trial_before_1':
                q[9] += .006
            return q, np.zeros(16)

        self.client.joint_snapshot.side_effect = snapshot
        with self.assertRaisesRegex(RuntimeError, '首段发送前机器人再次变化'):
            self.run_trial()
        self.client.request.assert_not_called()

    def test_small_left_encoder_change_replans_actual_step_sizes(self):
        self.actual[0] += .001
        refreshed = self.actual.copy()
        new_waypoints = self.waypoints.copy()
        new_waypoints[0, 0] = refreshed[0]
        report = {"maximum_planned_joint_step_deg": math.degrees(.05)}
        self.plan_mock.side_effect = [(self.waypoints, report), (new_waypoints, report)]
        self.run_trial()
        np.testing.assert_array_equal(self.plan_mock.call_args.args[0], refreshed)
        np.testing.assert_array_equal(self.sent[0][1]['joint'], new_waypoints[0, :14])

    def test_already_ready_does_not_send_an_unnecessary_movej(self):
        self.q[:7] = fixed.FIXED_LEFT_Q
        self.run_trial()
        self.plan_mock.assert_not_called()
        self.client.request.assert_not_called()

    def test_wrong_robot_or_moving_robot_is_rejected(self):
        self.client.accid = "another_robot"
        with self.assertRaisesRegex(RuntimeError, "机器人身份"):
            self.run_trial()
        self.client.accid = fixed.FIXED_ROBOT_ID
        with self.assertRaisesRegex(RuntimeError, "尚未静止"):
            trial.run_trial(self.client, self.q, np.ones(16), self.pose, self.args)
        self.client.request.assert_not_called()


class TrialEntryTests(unittest.TestCase):
    def test_trial_dispatch_does_not_call_normal_restore_or_enable_it(self):
        with patch.object(fixed, "FIXED_LEFT_PREPARE_PATH_VALIDATED", False), \
                patch.object(restore, "TraceRecorder") as traces, \
                patch.object(restore, "acquire_replay_lock", return_value=Mock()), \
                patch.object(restore, "Tron2JsonClient") as robot, \
                patch.object(restore, "restore_fixed_start") as normal, \
                patch.object(trial, "run_trial") as test_run, \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            traces.return_value.path = Path("/tmp/offline-restore-trial.jsonl")
            robot.return_value.accid = fixed.FIXED_ROBOT_ID
            robot.return_value.joint_snapshot.return_value = (np.zeros(16), np.zeros(16))
            robot.return_value.current_servo_pose.return_value = sample_pose()
            self.assertEqual(restore.main(["--test-path", "--execute"]), 0)
            self.assertFalse(fixed.FIXED_LEFT_PREPARE_PATH_VALIDATED)
        normal.assert_not_called()
        test_run.assert_called_once()
        robot.return_value.send.assert_not_called()

    def test_offline_trial_description_does_not_connect(self):
        with patch.object(restore, "Tron2JsonClient") as robot, redirect_stdout(io.StringIO()):
            self.assertEqual(restore.main(["--test-path"]), 0)
        robot.assert_not_called()

    def test_web_cannot_trigger_the_trial(self):
        with patch.object(restore, "Tron2JsonClient") as robot, \
                redirect_stderr(io.StringIO()):
            self.assertEqual(restore.main(["--test-path", "--execute", "--web-control"]), 2)
        robot.assert_not_called()

    def test_invalid_segment_duration_is_rejected_before_connection(self):
        for duration in ("0", "1", "11", "nan", "inf"):
            with self.subTest(duration=duration), \
                    patch.object(restore, "Tron2JsonClient") as robot, \
                    redirect_stderr(io.StringIO()):
                self.assertEqual(restore.main([
                    "--test-path", "--execute", "--trial-step-duration", duration,
                ]), 2)
            robot.assert_not_called()


class RestoreEntryTests(unittest.TestCase):
    def setUp(self):
        # 避免模拟机器人测试把伪造记录混入真实回放日志目录。
        trace = patch.object(restore, "TraceRecorder")
        self.trace_class = trace.start()
        self.trace_class.return_value.path = Path("/tmp/offline-restore-start.jsonl")
        self.addCleanup(trace.stop)

    def test_offline_description_does_not_connect(self):
        with patch.object(restore, "Tron2JsonClient") as robot, \
                redirect_stdout(io.StringIO()) as output:
            code = restore.main([])
        self.assertEqual(code, 0)
        self.assertIn("仅恢复保存的左臂固定起点", output.getvalue())
        self.assertIn("不自动归位", output.getvalue())
        robot.assert_not_called()

    def test_restore_and_replay_entries_report_the_same_active_target(self):
        with patch.object(restore, "Tron2JsonClient") as restore_robot, \
                patch.object(fixed, "Tron2JsonClient") as replay_robot, \
                patch.object(fixed, "load_tracker", return_value=sample_trajectory()), \
                redirect_stdout(io.StringIO()) as restore_output:
            self.assertEqual(restore.main([]), 0)
            with redirect_stdout(io.StringIO()) as replay_output:
                self.assertEqual(fixed.main(["--episode", "65"]), 0)
        expected_position = f"left 固定腕位置 (m): {fixed.FIXED_LEFT_POSITION.tolist()}"
        for output in (restore_output.getvalue(), replay_output.getvalue()):
            self.assertIn(expected_position, output)
            self.assertIn(fixed.FIXED_LEFT_START_LABEL, output)
            self.assertIn("当前准备目标尚未实测", output)
        self.assertEqual(restore.FIXED_LEFT_START_PROFILE, fixed.FIXED_LEFT_START_PROFILE)
        restore_robot.assert_not_called()
        replay_robot.assert_not_called()

    def test_previous_episode8_pose_cannot_start_either_unvalidated_restore(self):
        previous = json.loads((Path(__file__).parent / (
            "validation_results/episode8_height_start_20261028/fixed_start_profile.json"
        )).read_text())
        new_pose = sample_pose()
        previous_pose = fixed.DualArmPose(
            np.asarray(previous['target_position_m']),
            np.asarray(previous['target_quaternion_wxyz']),
            new_pose.right_position, new_pose.right_quaternion,
        )
        with patch.object(fixed, "FIXED_LEFT_START_VALIDATED", True), \
                patch.object(fixed, "FIXED_LEFT_PREPARE_PATH_VALIDATED", False), \
                patch.object(fixed, "load_tracker", return_value=sample_trajectory()), \
                patch.object(fixed, "TraceRecorder") as replay_trace, \
                patch.object(fixed, "acquire_replay_lock", return_value=Mock()), \
                patch.object(restore, "acquire_replay_lock", return_value=Mock()), \
                patch.object(fixed, "Tron2JsonClient") as replay_robot, \
                patch.object(restore, "Tron2JsonClient") as restore_robot, \
                patch.object(fixed, "move_stage") as move, \
                patch.object(fixed, "execute_replay") as execute:
            replay_trace.return_value.path = Path("/tmp/offline-horizontal-start.jsonl")
            for main, argv, robot in (
                    (restore.main, ["--execute"], restore_robot),
                    (fixed.main, ["--episode", "65", "--execute"], replay_robot)):
                with self.subTest(entry=main.__module__), \
                        redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as errors:
                    robot.return_value.accid = fixed.FIXED_ROBOT_ID
                    robot.return_value.joint_snapshot.return_value = (
                        np.r_[previous['target_left_q_rad'], np.zeros(9)], np.zeros(16),
                    )
                    robot.return_value.current_servo_pose.return_value = previous_pose
                    self.assertEqual(main(argv), 2)
                self.assertIn("自动恢复路径尚未验证", errors.getvalue())
        move.assert_not_called()
        execute.assert_not_called()

    def test_execute_rejects_unvalidated_start_before_connect_or_lock(self):
        with patch.object(fixed, "FIXED_LEFT_START_VALIDATED", False), \
                patch.object(restore, "Tron2JsonClient") as robot, \
                patch.object(restore, "acquire_replay_lock") as lock, \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as errors:
            code = restore.main(["--execute"])
        self.assertEqual(code, 2)
        self.assertIn("尚未验证 O6", errors.getvalue())
        robot.assert_not_called()
        lock.assert_not_called()

    def test_already_at_start_succeeds_without_movej_even_when_path_unvalidated(self):
        with patch.object(fixed, "FIXED_LEFT_PREPARE_PATH_VALIDATED", False), \
                patch.object(restore, "Tron2JsonClient") as robot, \
                patch.object(fixed, "move_stage") as move, \
                patch.object(fixed, "countdown") as timer, \
                redirect_stdout(io.StringIO()) as output:
            robot.return_value.accid = fixed.FIXED_ROBOT_ID
            robot.return_value.joint_snapshot.return_value = (
                np.r_[fixed.FIXED_LEFT_Q, np.zeros(9)], np.zeros(16),
            )
            robot.return_value.current_servo_pose.return_value = sample_pose()
            code = restore.main(["--execute"])
        self.assertEqual(code, 0)
        self.assertIn("跳过 MoveJ", output.getvalue())
        move.assert_not_called()
        timer.assert_not_called()

    def test_new_static_pose_does_not_enable_an_unvalidated_restore_path(self):
        with patch.object(fixed, "FIXED_LEFT_START_VALIDATED", True), \
                patch.object(fixed, "FIXED_LEFT_PREPARE_PATH_VALIDATED", False), \
                patch.object(restore, "Tron2JsonClient") as robot, \
                patch.object(restore, "acquire_replay_lock") as lock, \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as errors:
            robot.return_value.accid = fixed.FIXED_ROBOT_ID
            robot.return_value.joint_snapshot.return_value = (np.zeros(16), np.zeros(16))
            robot.return_value.current_servo_pose.return_value = sample_pose(left=[.3, .2, -.3])
            code = restore.main(["--execute"])
        self.assertEqual(code, 2)
        self.assertIn("自动恢复路径尚未验证", errors.getvalue())
        robot.assert_called_once()
        lock.assert_called_once()

if __name__ == "__main__":
    unittest.main()
