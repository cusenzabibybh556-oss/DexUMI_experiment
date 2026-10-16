#!/usr/bin/env python3
"""ROS bridge that exclusively owns one LinkerHand O6 CAN connection."""

import json
import os
import sys
import time
from typing import List, Optional

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import SetBool, Trigger

from linkerhand_retarget.command_limits import CommandLimiter


DEFAULT_SDK_PARENT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../../../TransHandR0725/TransHandR")
)
O6_JOINT_NAMES = [
    "thumb_flexion",
    "thumb_abduction",
    "index_flexion",
    "middle_flexion",
    "ring_flexion",
    "pinky_flexion",
]
TACTILE_NAMES = [
    "thumb_matrix",
    "index_matrix",
    "middle_matrix",
    "ring_matrix",
    "little_matrix",
]


class O6BridgeNode(Node):
    def __init__(self):
        super().__init__("o6_bridge_node")

        self.declare_parameter("topic", "/cb_right_hand_control_cmd")
        self.declare_parameter("hand_type", "right")
        self.declare_parameter("can", "can0")
        self.declare_parameter("sdk_parent", DEFAULT_SDK_PARENT)
        self.declare_parameter("speed", [180] * 6)
        self.declare_parameter("torque", [180] * 6)
        self.declare_parameter("dry_run", False)
        self.declare_parameter("min_send_interval", 0.02)
        self.declare_parameter("state_hz", 30.0)
        self.declare_parameter("tactile_hz", 10.0)
        self.declare_parameter("command_min", [0.0] * 6)
        self.declare_parameter("command_max", [255.0] * 6)
        self.declare_parameter("action_applied_topic", "")
        self.declare_parameter("state_topic", "")
        self.declare_parameter("tactile_topic", "")

        self.topic = str(self.get_parameter("topic").value)
        self.hand_type = str(self.get_parameter("hand_type").value)
        self.can = str(self.get_parameter("can").value)
        self.sdk_parent = str(self.get_parameter("sdk_parent").value)
        self.speed = self._six_ints(self.get_parameter("speed").value, "speed")
        self.torque = self._six_ints(
            self.get_parameter("torque").value, "torque"
        )
        self.dry_run = bool(self.get_parameter("dry_run").value)
        self.min_send_interval = max(
            0.005,
            float(self.get_parameter("min_send_interval").value),
        )
        self.state_hz = max(0.1, float(self.get_parameter("state_hz").value))
        self.tactile_hz = max(
            0.1, float(self.get_parameter("tactile_hz").value)
        )
        self.command_limiter = CommandLimiter(
            self.get_parameter("command_min").value,
            self.get_parameter("command_max").value,
        )

        prefix = f"/robot/o6/{self.hand_type}"
        self.action_applied_topic = self._topic_or_default(
            "action_applied_topic", f"{prefix}/action_applied"
        )
        self.state_topic = self._topic_or_default(
            "state_topic", f"{prefix}/joint_states"
        )
        self.tactile_topic = self._topic_or_default(
            "tactile_topic", f"{prefix}/tactile_matrix"
        )
        self.force_feedback_topic = f"/cb_{self.hand_type}_hand_matrix_touch"
        self.replay_topic = f"{prefix}/replay_cmd"
        self.replay_mode_service = f"{prefix}/replay_mode"
        self.resume_live_service = f"{prefix}/resume_live"

        self.hand = None
        self.palm_enabled = False
        self.latest_pose: Optional[List[int]] = None
        self.last_send_time = 0.0
        self.recv_count = 0
        self.sent_count = 0
        self._last_error_log = {}
        # 只有这个桥接进程持有 CAN。手套与回放输入在这里互斥。
        self.control_mode = "live"

        if not self.dry_run:
            self.hand = self._init_hand()
            enable_palm = getattr(self.hand, "enable_palm_touch", None)
            self.palm_enabled = bool(enable_palm()) if callable(enable_palm) else False
            self.get_logger().info(f"O6 B1 palm tactile: {self.palm_enabled}")
            self.hand.set_speed(speed=self.speed)
            self.hand.set_torque(torque=self.torque)
        else:
            self.get_logger().warn(
                "dry_run=True, will not send commands to O6 hardware"
            )

        self.action_publisher = self.create_publisher(
            JointState, self.action_applied_topic, 10
        )
        self.state_publisher = self.create_publisher(
            JointState, self.state_topic, 10
        )
        self.tactile_publisher = self.create_publisher(
            String, self.tactile_topic, 10
        )
        self.feedback_publisher = self.create_publisher(
            String, self.force_feedback_topic, 10
        )
        self.subscription = self.create_subscription(
            JointState, self.topic, self._on_command, 10
        )
        self.replay_subscription = self.create_subscription(
            JointState, self.replay_topic, self._on_replay_command, 10
        )
        self.replay_service = self.create_service(
            SetBool, self.replay_mode_service, self._set_replay_mode
        )
        self.resume_service = self.create_service(
            Trigger, self.resume_live_service, self._resume_live
        )

        self.send_timer = self.create_timer(
            self.min_send_interval, self._send_latest_command
        )
        self.state_timer = self.create_timer(1.0 / self.state_hz, self._poll_state)
        self.tactile_timer = self.create_timer(
            1.0 / self.tactile_hz, self._poll_tactile
        )
        self.get_logger().info(
            "O6 bridge ready: "
            f"input={self.topic}, action={self.action_applied_topic}, "
            f"state={self.state_topic}, tactile={self.tactile_topic}, "
            f"hand_type={self.hand_type}, can={self.can}"
        )

    def _topic_or_default(self, parameter: str, default: str) -> str:
        configured = str(self.get_parameter(parameter).value)
        return configured or default

    def _init_hand(self):
        if not os.path.isdir(self.sdk_parent):
            raise RuntimeError(
                f"SDK parent path does not exist: {self.sdk_parent}"
            )
        if self.sdk_parent not in sys.path:
            sys.path.insert(0, self.sdk_parent)
        try:
            if os.path.isdir(os.path.join(self.sdk_parent, "LinkerHand")):
                from LinkerHand.linker_hand_api import LinkerHandApi
            else:
                from linker_hand_python_sdk.LinkerHand.linker_hand_api import LinkerHandApi
        except ImportError as exc:
            raise RuntimeError(
                "Could not import LinkerHandApi. "
                f"Expected LinkerHand SDK under: {self.sdk_parent}"
            ) from exc
        return LinkerHandApi(
            hand_joint="O6", hand_type=self.hand_type, can=self.can
        )

    def _six_ints(self, values, name: str) -> List[int]:
        data = list(values)
        if len(data) != 6:
            raise ValueError(
                f"{name} must contain exactly 6 values, got {len(data)}"
            )
        return [self._clamp_byte(value) for value in data]

    @staticmethod
    def _clamp_byte(value) -> int:
        return int(max(0, min(255, round(float(value)))))

    def _on_command(self, msg: JointState):
        if self.control_mode != "live":
            return
        self._accept_command(msg)

    def _on_replay_command(self, msg: JointState):
        if self.control_mode != "replay":
            return
        self._accept_command(msg)

    def _accept_command(self, msg: JointState):
        try:
            limited, _ = self.command_limiter.apply(msg.position)
            self.latest_pose = [self._clamp_byte(value) for value in limited]
            self.recv_count += 1
        except (TypeError, ValueError) as exc:
            self._log_error_periodically("command", f"Invalid O6 command: {exc}")

    def _set_replay_mode(self, request, response):
        if request.data:
            if self.control_mode == "replay":
                response.success = False
                response.message = "O6 replay already owns the bridge"
                return response
            self.control_mode = "replay"
            self.get_logger().warn("O6 replay owns command input; glove commands ignored")
        else:
            # 回放结束或中止后保持最后目标；不得自动切回仍在发布的手套。
            self.control_mode = "hold"
            self.get_logger().warn("O6 replay released; holding last hand target")
        response.success = True
        response.message = self.control_mode
        return response

    def _resume_live(self, _request, response):
        if self.control_mode == "replay":
            response.success = False
            response.message = "stop O6 replay before resuming glove control"
            return response
        self.control_mode = "live"
        response.success = True
        response.message = "glove control resumed; next glove command may move O6"
        self.get_logger().warn(response.message)
        return response

    def _joint_state(self, values, timestamp_ns=None) -> JointState:
        msg = JointState()
        stamp_ns = (
            self.get_clock().now().nanoseconds
            if timestamp_ns is None
            else int(timestamp_ns)
        )
        msg.header.stamp.sec = stamp_ns // 1_000_000_000
        msg.header.stamp.nanosec = stamp_ns % 1_000_000_000
        msg.name = list(O6_JOINT_NAMES)
        msg.position = [float(value) for value in values]
        return msg

    def _send_latest_command(self):
        pose = list(self.latest_pose) if self.latest_pose is not None else None
        if pose is None:
            return
        now = time.monotonic()
        if now - self.last_send_time < self.min_send_interval * 0.9:
            return
        try:
            if not self.dry_run:
                self.hand.finger_move(pose=pose)
            applied = self._joint_state(pose)
            applied.header.frame_id = self.control_mode
            self.action_publisher.publish(applied)
            self.sent_count += 1
            if self.sent_count % 100 == 1:
                prefix = "dry_run" if self.dry_run else "sent"
                self.get_logger().info(f"{prefix} pose={pose}")
            self.last_send_time = now
        except Exception as exc:
            self._log_error_periodically("send", f"O6 command send failed: {exc}")

    def _poll_state(self):
        if self.dry_run:
            if self.latest_pose is not None:
                self.state_publisher.publish(self._joint_state(self.latest_pose))
            return
        try:
            read_start_ns = self.get_clock().now().nanoseconds
            state = list(self.hand.get_state())
            read_end_ns = self.get_clock().now().nanoseconds
            if len(state) != 6:
                raise ValueError(f"expected 6 values, got {len(state)}")
            self.state_publisher.publish(
                self._joint_state(state, (read_start_ns + read_end_ns) // 2)
            )
        except Exception as exc:
            self._log_error_periodically("state", f"O6 state read failed: {exc}")

    def _poll_tactile(self):
        if self.dry_run:
            return
        try:
            matrices = self.hand.get_matrix_touch_v2()
            if len(matrices) != len(TACTILE_NAMES):
                raise ValueError(f"expected 5 matrices, got {len(matrices)}")
            payload = {
                "schema": "linkerhand.tactile.v1",
                "hand_type": self.hand_type,
                "timestamp_ns": time.monotonic_ns(),
                "valid_shapes": [],
            }
            for name, matrix in zip(TACTILE_NAMES, matrices):
                values = np.asarray(matrix, dtype=np.int16)
                if values.shape != (12, 6):
                    raise ValueError(f"{name} has shape {values.shape}")
                valid = values >= 0
                if not np.any(valid):
                    raise ValueError(f"{name} has no valid tactile cells")
                valid_rows = np.flatnonzero(np.any(valid, axis=1))
                valid_cols = np.flatnonzero(np.any(valid, axis=0))
                payload["valid_shapes"].append(
                    [int(valid_rows[-1]) + 1, int(valid_cols[-1]) + 1]
                )
                payload[name] = values.tolist()
            if self.palm_enabled:
                palm = self.hand.get_palm_matrix_touch()
                if palm is not None:
                    palm = np.asarray(palm, dtype=np.int16)
                    if palm.shape == (14, 15) and np.any(palm >= 0):
                        payload["palm_matrix"] = palm.tolist()
            message = String()
            message.data = json.dumps(payload, separators=(",", ":"))
            self.tactile_publisher.publish(message)

            # The glove feedback consumer only needs each finger's peak force
            # and treats every non-zero cell as contact.  Hide -1 padding from it.
            feedback_payload = dict(payload)
            for name in TACTILE_NAMES:
                feedback_payload[name] = np.maximum(
                    np.asarray(payload[name], dtype=np.int16), 0
                ).tolist()
            feedback_message = String()
            feedback_message.data = json.dumps(
                feedback_payload, separators=(",", ":")
            )
            self.feedback_publisher.publish(feedback_message)
        except Exception as exc:
            self._log_error_periodically(
                "tactile", f"O6 tactile read failed: {exc}"
            )

    def _log_error_periodically(self, key: str, message: str):
        now = time.monotonic()
        if now - self._last_error_log.get(key, 0.0) >= 2.0:
            self.get_logger().error(message)
            self._last_error_log[key] = now

    def close(self):
        if self.hand is not None:
            close = getattr(self.hand, "close_can", None)
            if callable(close):
                close()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = O6BridgeNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.close()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
