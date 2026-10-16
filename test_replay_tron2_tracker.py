"""Offline protocol and feedback tests; these never connect to a robot."""

import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np

import replay_tron2_tracker as replay


class FakeRobot:
    def __init__(self):
        self.messages = []

    def publishJsonMessage(self, payload):
        self.messages.append(json.loads(payload))
        return None  # The vendor SDK declares this method void.


def fake_client():
    client = replay.Tron2JsonClient.__new__(replay.Tron2JsonClient)
    client.robot = FakeRobot()
    client.accid = "DACH_TEST"
    client._condition = threading.Condition()
    client._robot_identity_ready = True
    client._robot_identity_error = None
    client._messages = []
    client._failure = None
    client._pose_probe_guid = None
    client._pose_probe_sent_at = 0.0
    client._pose_samples = []
    client._pose_probe_error = None
    client.trace = None
    return client


def pose(left_x=0.45):
    return replay.DualArmPose(
        np.array([left_x, 0.17, -0.30]),
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.array([0.45, -0.17, -0.30]),
        np.array([1.0, 0.0, 0.0, 0.0]),
    )


class ReplayProtocolTests(unittest.TestCase):
    def test_trace_records_commands_feedback_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "replay.jsonl"
            trace = replay.TraceRecorder(path)
            client = fake_client()
            client.trace = trace
            payload = replay.servop_data(
                np.array([0.45, 0.17, -0.30, 1, 0, 0, 0]),
                np.array([0.45, -0.17, -0.30, 1, 0, 0, 0]),
            )
            client.send("request_servop", payload)
            client.probe_pose()
            probe = client.robot.messages[-1]
            client._on_message(json.dumps({
                "title": "response_get_servo_pose", "guid": probe["guid"],
                "data": {
                    "result": "success", "timestamp": 123,
                    "left_position": [0.46, 0.17, -0.30],
                    "left_quat": [1, 0, 0, 0],
                    "right_position": [0.45, -0.17, -0.30],
                    "right_quat": [1, 0, 0, 0],
                },
            }))
            trace.close()
            events = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual([event["event"] for event in events], [
                "servop_command", "pose_probe_sent", "pose_feedback"
            ])
            self.assertEqual(events[0]["left_pos"], payload["left_pos"])
            self.assertEqual(events[2]["left_position"], [0.46, 0.17, -0.30])
            self.assertGreater(events[2]["monotonic_ns"], 0)
            with self.assertRaises(FileExistsError):
                replay.TraceRecorder(path)

    def test_commands_wait_for_robot_identity(self):
        client = fake_client()
        client._robot_identity_ready = False
        client.accid = ""
        with self.assertRaisesRegex(RuntimeError, "identity not ready"):
            client.send("request_servop", {"left_pos": [], "right_pos": []})
        client._on_message(json.dumps({
            "title": "notify_robot_info", "accid": "DACH_TEST", "data": {}
        }))
        client.wait_for_identity(timeout=0.1)
        self.assertEqual(client.accid, "DACH_TEST")

    def test_servop_payload_has_two_seven_element_poses(self):
        left = np.array([0.45, 0.17, -0.30, 2.0, 0.0, 0.0, 0.0])
        right = np.array([0.45, -0.17, -0.30, 1.0, 0.0, 0.0, 0.0])
        payload = replay.servop_data(left, right)
        self.assertEqual(set(payload), {"left_pos", "right_pos"})
        self.assertEqual(len(payload["left_pos"]), 7)
        self.assertEqual(len(payload["right_pos"]), 7)
        self.assertEqual(payload["left_pos"][3:], [1.0, 0.0, 0.0, 0.0])
        client = fake_client()
        client.send("request_servop", payload)
        self.assertEqual(client.robot.messages[0]["data"], payload)

    def test_servop_rejects_invalid_pose(self):
        with self.assertRaises(ValueError):
            replay.servop_data(np.zeros(14), np.zeros(7))
        with self.assertRaises(ValueError):
            replay.servop_data(np.array([0, 0, 0, 0, 0, 0, 0]), np.ones(7))

    def test_correlated_feedback_is_recorded_and_failure_is_retained(self):
        client = fake_client()
        client.probe_pose()
        probe = client.robot.messages[0]
        self.assertEqual(probe["title"], "request_get_servo_pose")
        client._on_message(json.dumps({
            "accid": "DACH_TEST",
            "title": "response_get_servo_pose",
            "guid": probe["guid"],
            "data": {
                "result": "success",
                "left_position": [0.46, 0.17, -0.30],
                "left_quat": [1, 0, 0, 0],
                "right_position": [0.45, -0.17, -0.30],
                "right_quat": [1, 0, 0, 0],
            },
        }))
        self.assertEqual(len(client.pose_samples()), 1)
        self.assertAlmostEqual(client.pose_samples()[0].pose.left_position[0], 0.46)
        client._on_message(json.dumps({
            "title": "notify_servop", "data": {"result": "fail_invalid_cmd"}
        }))
        self.assertEqual(client.failure(), "fail_invalid_cmd")

    def test_pose_request_ignores_other_clients_responses(self):
        client = fake_client()

        def publish(payload):
            request = json.loads(payload)
            client._on_message(json.dumps({
                "title": "response_get_move_pose",
                "guid": "another-client",
                "data": {"result": "fail_no_data"},
            }))
            client._on_message(json.dumps({
                "title": "response_get_move_pose",
                "guid": request["guid"],
                "data": {"result": "success"},
            }))

        client.robot.publishJsonMessage = publish
        response = client.request(
            "request_get_move_pose", {}, "response_get_move_pose", timeout=0.1
        )
        self.assertEqual(response["data"]["result"], "success")

    def test_non_pose_request_accepts_vendor_reply_without_matching_guid(self):
        client = fake_client()

        def publish(_payload):
            client._on_message(json.dumps({
                "title": "response_get_joint_state",
                "guid": "vendor-generated-guid",
                "data": {"result": "success"},
            }))

        client.robot.publishJsonMessage = publish
        response = client.request(
            "request_get_joint_state", {}, "response_get_joint_state", timeout=0.1
        )
        self.assertEqual(response["data"]["result"], "success")

    def test_nonzero_target_without_motion_is_not_success(self):
        anchor = pose()
        samples = [replay.PoseSample(float(i), pose()) for i in range(3)]
        deltas = np.array([[0, 0, 0], [0.02, 0, 0], [0, 0, 0]])
        quats = np.tile([1.0, 0.0, 0.0, 0.0], (3, 1))
        with self.assertRaisesRegex(RuntimeError, "不能认定回放正确"):
            replay.verify_observed_motion(anchor, "left", deltas, quats, samples, anchor)
        samples[1] = replay.PoseSample(1.0, pose(0.465))
        self.assertTrue(
            replay.verify_observed_motion(anchor, "left", deltas, quats, samples, anchor)
        )

    def test_overshoot_and_wrong_final_pose_are_not_success(self):
        anchor = pose()
        deltas = np.array([[0, 0, 0], [-0.0095, 0, 0], [-0.0022, 0, 0]])
        quats = np.tile([1.0, 0.0, 0.0, 0.0], (3, 1))
        samples = [
            replay.PoseSample(0.0, anchor),
            replay.PoseSample(1.0, pose(0.4271)),  # 22.9 mm, like the live report
            replay.PoseSample(2.0, pose(0.4344)),
        ]
        with self.assertRaisesRegex(RuntimeError, "超过计划"):
            replay.verify_observed_motion(
                anchor, "left", deltas, quats, samples, pose(0.4344)
            )
        samples[1] = replay.PoseSample(1.0, pose(0.443))
        samples[2] = replay.PoseSample(2.0, pose(0.4478))
        with self.assertRaisesRegex(RuntimeError, "末帧未跟踪目标"):
            replay.verify_observed_motion(
                anchor, "left", deltas, quats, samples, pose(0.4344)
            )

    def test_uncontrolled_arm_drift_is_not_success(self):
        anchor = pose()
        deltas = np.array([[0, 0, 0], [-0.0095, 0, 0], [-0.0022, 0, 0]])
        quats = np.tile([1.0, 0.0, 0.0, 0.0], (3, 1))
        samples = [
            replay.PoseSample(0.0, anchor),
            replay.PoseSample(1.0, pose(0.441)),
            replay.PoseSample(2.0, pose(0.4478)),
        ]
        final = replay.DualArmPose(
            pose(0.4478).left_position,
            anchor.left_quaternion,
            np.array([0.45, -0.17, -0.312]),
            anchor.right_quaternion,
        )
        with self.assertRaisesRegex(RuntimeError, "未指定运动的 right 腕偏移"):
            replay.verify_observed_motion(
                anchor, "left", deltas, quats, samples, final
            )

    def test_countdown_pose_drift_cancels_before_motion(self):
        anchor = pose()
        replay.check_anchor_stable(anchor, pose(0.452))
        with self.assertRaisesRegex(RuntimeError, "取消回放"):
            replay.check_anchor_stable(anchor, pose(0.456))

    def test_missing_feedback_and_zero_motion_are_not_success(self):
        anchor = pose()
        deltas = np.array([[0, 0, 0], [0.02, 0, 0]])
        quats = np.tile([1.0, 0.0, 0.0, 0.0], (2, 1))
        with self.assertRaisesRegex(RuntimeError, "无法验证运动"):
            replay.verify_observed_motion(anchor, "left", deltas, quats, [], anchor)
        self.assertFalse(replay.verify_observed_motion(
            anchor, "left", np.zeros((2, 3)), quats, [], anchor
        ))

    def test_tracker_unit_must_be_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode.hdf5"
            with h5py.File(path, "w") as file:
                file.create_dataset(
                    "timestamps/master_timestamp_ns", data=[0, 33_000_000]
                )
                group = file.create_group("vive_tracker")
                group.create_dataset("pos", data=np.zeros((2, 1, 3)))
                group.create_dataset(
                    "quat", data=np.tile([1, 0, 0, 0], (2, 1, 1))
                )
                group.create_dataset("valid", data=np.ones((2, 1), dtype=bool))
            with self.assertRaisesRegex(ValueError, "position_unit=meter"):
                replay.load_tracker(path, 0.15)
            with h5py.File(path, "a") as file:
                file["vive_tracker"].attrs["position_unit"] = "meter"
            self.assertEqual(replay.load_tracker(path, 0.15).original_frames, 2)

    def test_online_guard_detects_moving_and_held_arm_jump(self):
        anchor = pose()
        target_q = anchor.left_quaternion
        moved_left = pose(0.458)
        # 验证判断逻辑时使用局部严格阈值，不依赖现场已调高的配置。
        with patch.object(replay, "MOVING_ARM_ERROR_M", 0.005):
            with self.assertRaisesRegex(RuntimeError, "trajectory 双臂反馈超限"):
                replay.check_live_feedback(
                    anchor, "left", anchor.left_position, target_q,
                    replay.PoseSample(1.0, moved_left), "trajectory",
                )
        moved_right = replay.DualArmPose(
            anchor.left_position, target_q,
            anchor.right_position + np.array([0.0, 0.011, 0.0]),
            anchor.right_quaternion,
        )
        with patch.object(replay, "HELD_ARM_DRIFT_M", 0.003):
            with self.assertRaisesRegex(RuntimeError, "right 偏移 11.0 mm"):
                replay.check_live_feedback(
                    anchor, "left", anchor.left_position, target_q,
                    replay.PoseSample(1.0, moved_right), "trajectory",
                )

    def test_online_angle_limits_follow_physical_arm(self):
        anchor = pose()

        def rotated(degrees):
            half = np.deg2rad(degrees) / 2
            return np.array([np.cos(half), np.sin(half), 0.0, 0.0])

        def sample(left_degrees=0.0, right_degrees=0.0):
            actual = replay.DualArmPose(
                anchor.left_position, rotated(left_degrees),
                anchor.right_position, rotated(right_degrees),
            )
            return replay.PoseSample(1.0, actual)

        replay.check_live_feedback(
            anchor, "left", anchor.left_position, anchor.left_quaternion,
            sample(left_degrees=29.0), "trajectory",
        )
        with self.assertRaisesRegex(RuntimeError, "trajectory 双臂反馈超限"):
            replay.check_live_feedback(
                anchor, "left", anchor.left_position, anchor.left_quaternion,
                sample(left_degrees=31.0), "trajectory",
            )
        with self.assertRaisesRegex(RuntimeError, "trajectory 双臂反馈超限"):
            replay.check_live_feedback(
                anchor, "left", anchor.left_position, anchor.left_quaternion,
                sample(right_degrees=6.0), "trajectory",
            )
        replay.check_live_feedback(
            anchor, "right", anchor.right_position, anchor.right_quaternion,
            sample(left_degrees=29.0), "trajectory",
        )
        with self.assertRaisesRegex(RuntimeError, "trajectory 双臂反馈超限"):
            replay.check_live_feedback(
                anchor, "right", anchor.right_position, anchor.right_quaternion,
                sample(left_degrees=31.0), "trajectory",
            )

    def test_online_guard_rejects_missing_feedback(self):
        guard = replay.FeedbackGuard(pose(), "left")
        with self.assertRaisesRegex(RuntimeError, "反馈超过"):
            guard.require_fresh(guard.last_feedback_at + 0.6)

    def test_feedback_timeout_can_be_three_seconds(self):
        guard = replay.FeedbackGuard(
            pose(), "left", feedback_timeout_s=3.0,
        )
        guard.require_fresh(guard.last_feedback_at + 2.9)
        with self.assertRaisesRegex(RuntimeError, "超过 3.0s"):
            guard.require_fresh(guard.last_feedback_at + 3.1)

    def test_direct_replay_sends_trajectory_first_and_stops_on_jump(self):
        class FakeClock:
            now = 0.0

            def monotonic(self):
                return self.now

            def monotonic_ns(self):
                return int(self.now * 1e9)

            def sleep(self, duration):
                self.now += max(0.0, duration)

        class FakeLiveClient:
            trace = None

            def __init__(self, clock, jump=False):
                self.clock = clock
                self.current = pose()
                self.commands = []
                self.send_times = []
                self.samples = []
                self.jump = jump

            def current_pose(self, label="snapshot"):
                return self.current

            def current_servo_pose(self, label="servo_snapshot"):
                return self.current

            def joint_snapshot(self, label):
                return np.zeros(14), np.zeros(14)

            def failure(self):
                return None

            def send(self, title, data):
                self.commands.append(data)
                self.send_times.append(self.clock.now)
                if self.jump:
                    self.current = pose(0.459)
                else:
                    self.current = pose(data["left_pos"][0])

            def probe_pose(self):
                return None

            def latest_pose_sample(self):
                sample = replay.PoseSample(self.clock.now, self.current)
                self.samples.append(sample)
                return sample

            def pose_samples(self):
                return self.samples

        args = SimpleNamespace(
            countdown=5, rate=10.0, feedback_timeout=0.5,
        )
        deltas = np.array([[0.0, 0.0, 0.0], [0.002, 0.0, 0.0]])
        quats = np.tile([1.0, 0.0, 0.0, 0.0], (2, 1))
        clock = FakeClock()
        client = FakeLiveClient(clock)
        with patch("builtins.input", return_value="DIRECT"), patch.object(
            replay.time, "monotonic", clock.monotonic
        ), patch.object(
            replay.time, "sleep", clock.sleep
        ), patch.object(replay, "verify_observed_motion", return_value=True):
            self.assertTrue(replay.execute_replay(
                client, pose(), "left", np.array([0.0, 0.1]),
                deltas, quats, args,
            ))
        self.assertEqual(client.commands[0]["left_pos"][:3],
                         pose().left_position.tolist())
        self.assertEqual(client.commands[1]["left_pos"][0], 0.452)
        # 两个轨迹目标后仅重复末帧两次；没有接管保持指令。
        self.assertEqual(len(client.commands), 4)
        self.assertTrue(all(
            command["right_pos"] == client.commands[0]["right_pos"]
            for command in client.commands
        ))
        self.assertEqual(
            client.commands[0]["right_pos"][:3], pose().right_position.tolist(),
        )

        class FakeHand:
            acquired = False
            scheduled_ns = None
            checks = 0

            def acquire(self):
                self.acquired = True

            def start_at(self, start_ns):
                self.scheduled_ns = start_ns

            def require_healthy(self):
                self.checks += 1

        clock = FakeClock()
        client = FakeLiveClient(clock)
        hand = FakeHand()
        with patch("builtins.input", return_value="REPLAY"), patch.object(
            replay.time, "monotonic", clock.monotonic
        ), patch.object(
            replay.time, "monotonic_ns", clock.monotonic_ns
        ), patch.object(
            replay.time, "sleep", clock.sleep
        ), patch.object(replay, "verify_observed_motion", return_value=True):
            self.assertTrue(replay.execute_replay(
                client, pose(), "left", np.array([0.0, 0.1]),
                deltas, quats, args, confirmation_word="REPLAY", hand_replay=hand,
            ))
        self.assertTrue(hand.acquired)
        self.assertGreater(hand.checks, 0)
        self.assertAlmostEqual(client.send_times[0], hand.scheduled_ns / 1e9,
                               places=7)

        # 真实机器人对异步腕位姿查询有延迟。旧实现把同步等待设为
        # 500ms，恰好耗尽 500ms 反馈门限，首条 ServoP 会立刻中止。
        class DelayedFeedbackClient(FakeLiveClient):
            pending_at = None

            def probe_pose(self):
                if self.pending_at is None:
                    self.pending_at = self.clock.now + 0.002

            def latest_pose_sample(self):
                if self.pending_at is not None and self.clock.now >= self.pending_at:
                    self.samples.append(replay.PoseSample(self.clock.now, self.current))
                    self.pending_at = None
                return self.samples[-1] if self.samples else None

        clock = FakeClock()
        client = DelayedFeedbackClient(clock)
        hand = FakeHand()
        with patch("builtins.input", return_value="REPLAY"), patch.object(
            replay.time, "monotonic", clock.monotonic
        ), patch.object(
            replay.time, "monotonic_ns", clock.monotonic_ns
        ), patch.object(
            replay.time, "sleep", clock.sleep
        ), patch.object(replay, "verify_observed_motion", return_value=True):
            self.assertTrue(replay.execute_replay(
                client, pose(), "left", np.array([0.0, 0.1]),
                deltas, quats, args, confirmation_word="REPLAY", hand_replay=hand,
            ))
        self.assertEqual(len(client.commands), 4)
        self.assertAlmostEqual(client.send_times[0], hand.scheduled_ns / 1e9,
                               places=7)

        class MissingFeedbackClient(FakeLiveClient):
            def latest_pose_sample(self):
                return None

        clock = FakeClock()
        client = MissingFeedbackClient(clock)
        hand = FakeHand()
        with patch("builtins.input", return_value="REPLAY"), patch.object(
            replay.time, "monotonic", clock.monotonic
        ), patch.object(
            replay.time, "sleep", clock.sleep
        ), patch.object(replay, "verify_observed_motion", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "首帧前未收到新的"):
                replay.execute_replay(
                    client, pose(), "left", np.array([0.0, 0.1]),
                    deltas, quats, args, confirmation_word="REPLAY", hand_replay=hand,
                )
        self.assertEqual(client.commands, [])
        self.assertIsNone(hand.scheduled_ns)

        clock = FakeClock()
        client = FakeLiveClient(clock, jump=True)
        with patch("builtins.input", return_value="DIRECT"), patch.object(
            replay.time, "monotonic", clock.monotonic
        ), patch.object(
            replay.time, "sleep", clock.sleep
        ), patch.object(replay, "MOVING_ARM_ERROR_M", 0.005):
            with self.assertRaisesRegex(RuntimeError, "trajectory 双臂反馈超限"):
                replay.execute_replay(
                    client, pose(), "left", np.array([0.0, 0.1]),
                    deltas, quats, args,
                )
        self.assertEqual(len(client.commands), 1)

    def test_custom_replay_confirmation_rejects_old_direct_keyword(self):
        args = SimpleNamespace(countdown=5, rate=10.0, feedback_timeout=0.5)
        deltas = np.zeros((2, 3))
        quats = np.tile([1.0, 0.0, 0.0, 0.0], (2, 1))
        with patch("builtins.input", return_value="DIRECT") as ask:
            with self.assertRaisesRegex(RuntimeError, "cancelled by user"):
                replay.execute_replay(
                    SimpleNamespace(), pose(), "left", np.array([0.0, 0.1]),
                    deltas, quats, args, confirmation_word="REPLAY",
                )
        self.assertIn("输入 REPLAY", ask.call_args.args[0])

    def test_pose_probe_retries_before_default_feedback_timeout(self):
        client = fake_client()
        clock = SimpleNamespace(now=0.0)
        with patch.object(replay.time, "monotonic", side_effect=lambda: clock.now):
            client.probe_pose()
            first_guid = client.robot.messages[-1]["guid"]
            clock.now = 0.1
            client.probe_pose()
            self.assertEqual(len(client.robot.messages), 1)
            clock.now = 0.16
            client.probe_pose()
            self.assertEqual(len(client.robot.messages), 2)
            second_guid = client.robot.messages[-1]["guid"]
            self.assertNotEqual(first_guid, second_guid)
            client._on_message(json.dumps({
                "title": "response_get_servo_pose", "guid": second_guid,
                "data": {
                    "result": "success", "timestamp": 123,
                    "left_position": [0.45, 0.17, -0.30],
                    "left_quat": [1, 0, 0, 0],
                    "right_position": [0.45, -0.17, -0.30],
                    "right_quat": [1, 0, 0, 0],
                },
            }))
        self.assertEqual(len(client.pose_samples()), 1)


if __name__ == "__main__":
    unittest.main()
