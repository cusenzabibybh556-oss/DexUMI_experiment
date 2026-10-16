#!/usr/bin/env python3
"""Forward profile-selected ROS topics to TransHandR UDP telemetry v2."""

from __future__ import annotations

import argparse
import json
import socket
import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from telemetry_protocol import make_packet


TACTILE_NAMES = [
    "thumb_matrix", "index_matrix", "middle_matrix", "ring_matrix", "little_matrix"
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--hand-side", choices=("left", "right"), default="left")
    parser.add_argument("--robot-model", default="o6")
    parser.add_argument("--operator-topic", default="")
    parser.add_argument("--action-topic", default="")
    parser.add_argument("--state-topic", default="")
    parser.add_argument("--tactile-topic", default="")
    parser.add_argument("--force-topic", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from std_msgs.msg import String

    class RosTelemetryAdapter(Node):
        def __init__(self):
            super().__init__("transhandr_ros_telemetry_adapter")
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.target = (args.host, args.port)
            self.sequence = {kind: 0 for kind in ("operator", "action", "state", "tactile", "palm", "force")}
            side = args.hand_side
            operator_topic = args.operator_topic or f"/operator/linkerforce/{side}/joint_states"
            action_topic = args.action_topic or f"/robot/{args.robot_model}/{side}/action_applied"
            state_topic = args.state_topic or f"/robot/{args.robot_model}/{side}/joint_states"
            self.create_subscription(JointState, operator_topic,
                                     lambda msg: self.send_joint("operator", msg, "linkerforce"), 50)
            self.create_subscription(JointState, action_topic,
                                     lambda msg: self.send_joint(
                                         "action", msg, f"{args.robot_model}_bridge"), 50)
            self.create_subscription(JointState, state_topic,
                                     lambda msg: self.send_joint(
                                         "state", msg, f"{args.robot_model}_sdk"), 50)
            self.tactile_subscription = None
            if args.tactile_topic:
                self.tactile_subscription = self.create_subscription(
                    String, args.tactile_topic, self.send_tactile, 20)
            if args.force_topic:
                self.create_subscription(JointState, args.force_topic, self.send_force, 20)
            self.get_logger().info(
                f"telemetry v2 -> udp://{args.host}:{args.port}, side={side}, model={args.robot_model}"
            )

        @staticmethod
        def ros_timestamp_ns(msg) -> int:
            stamp = msg.header.stamp
            value = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
            return value if value > 0 else time.time_ns()

        def emit(
            self, kind: str, values, source: str, source_timestamp_ns: int,
            unified_timestamp_ns: int | None = None,
        ) -> None:
            received = time.monotonic_ns()
            if unified_timestamp_ns is None:
                wall = time.time_ns()
                unified_timestamp_ns = int(source_timestamp_ns) + (received - wall)
            sequence = self.sequence[kind]
            self.sequence[kind] += 1
            packet = make_packet(kind, values, source, args.robot_model, args.hand_side,
                                 source_timestamp_ns, received, sequence,
                                 unified_timestamp_ns)
            self.sock.sendto(json.dumps(packet, separators=(",", ":")).encode(), self.target)

        def send_joint(self, kind: str, msg, source: str) -> None:
            try:
                source_timestamp_ns = self.ros_timestamp_ns(msg)
                # ROS header stamps may be system-clock, ROS-clock, or a bridge
                # provided stamp.  The collector aligns streams on host
                # monotonic time, so use adapter receive time as the unified
                # timestamp and keep the ROS stamp as source provenance.
                self.emit(
                    kind, msg.position, source, source_timestamp_ns,
                    time.monotonic_ns(),
                )
            except Exception as exc:
                self.get_logger().error(f"failed to forward {kind}: {exc}")

        def send_force(self, msg) -> None:
            try:
                self.emit("force", msg.effort, "inspire_force_act",
                          self.ros_timestamp_ns(msg), time.monotonic_ns())
            except Exception as exc:
                self.get_logger().error(f"failed to forward force: {exc}")

        def send_tactile(self, msg) -> None:
            try:
                payload = json.loads(msg.data)
                values = [payload[name] for name in TACTILE_NAMES]
                if not all(
                    any(float(cell) >= 0 for row in matrix for cell in row)
                    for matrix in values
                ):
                    raise ValueError("tactile payload contains an invalid finger matrix")
                timestamp_ns = int(payload.get("timestamp_ns", time.monotonic_ns()))
                self.emit("tactile", values, "o6_tactile", timestamp_ns,
                          timestamp_ns)
                if "palm_matrix" in payload:
                    self.emit("palm", payload["palm_matrix"], "o6_palm_b1",
                              timestamp_ns, timestamp_ns)
            except Exception as exc:
                self.get_logger().error(f"failed to forward tactile: {exc}")

        def destroy_node(self):
            self.sock.close()
            return super().destroy_node()

    rclpy.init()
    node = RosTelemetryAdapter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
