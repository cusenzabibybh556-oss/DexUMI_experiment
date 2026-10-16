#!/usr/bin/env python3
"""Bridge LinkerForce six-channel ROS commands to an Inspire Hand over Modbus."""

from __future__ import annotations

import math
import sys
import threading
import time
from pathlib import Path
from typing import Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
INSPIRE_SDK = ROOT / "inspire_hands"
JOINT_NAMES = ("little", "ring", "middle", "index", "thumb_bend", "thumb_rotate")
ANGLE_SET_REGISTER = 1486
LINKERFORCE_TO_INSPIRE = (5, 4, 3, 2, 0, 1)
CHANNEL_COUNT = 6


def map_linkerforce_to_inspire(values: Sequence[float]) -> list[int]:
    """Map glove output order to Inspire order and scale 0..255 to 0..1000."""
    if len(values) != 6:
        raise ValueError(f"expected 6 LinkerForce channels, received {len(values)}")
    source = [float(value) for value in values]
    if not all(math.isfinite(value) for value in source):
        raise ValueError("command contains a non-finite value")
    clamped = [min(255.0, max(0.0, value)) for value in source]
    # LinkerForce/O6 order is thumb bend, thumb abduction, index, middle,
    # ring, little. Inspire keeps the four fingers in reverse order but does
    # not swap the two thumb degrees of freedom.
    return [int(round(clamped[index] * 1000.0 / 255.0))
            for index in LINKERFORCE_TO_INSPIRE]


def apply_limits(values: Sequence[int], lower: Iterable[int], upper: Iterable[int]) -> list[int]:
    lows, highs = list(lower), list(upper)
    if len(values) != 6 or len(lows) != 6 or len(highs) != 6:
        raise ValueError("Inspire command limits must contain six values")
    if any(low > high for low, high in zip(lows, highs)):
        raise ValueError("Inspire command minimum exceeds maximum")
    return [int(min(high, max(low, value))) for value, low, high in zip(values, lows, highs)]


def expand_channels(value, name: str, cast) -> list:
    """Accept a legacy scalar or a six-channel parameter value."""
    values = list(value) if isinstance(value, (list, tuple)) else [value] * CHANNEL_COUNT
    if len(values) != CHANNEL_COUNT:
        raise ValueError(f"{name} must be a scalar or contain six values")
    return [cast(item) for item in values]


def commands_differ(first: Sequence[int] | None, second: Sequence[int], epsilon: int) -> bool:
    if first is None or len(first) != len(second):
        return True
    return any(abs(int(a) - int(b)) > epsilon for a, b in zip(first, second))


def write_acknowledged_command(write, publish, command: Sequence[int]) -> None:
    """Publish an Action only after the hardware acknowledges its write."""
    if not write(list(command)):
        raise RuntimeError("Modbus write was not acknowledged")
    publish(list(command))


class CommandStabilizer:
    """Suppress sensor noise while retaining bounded, responsive commands."""

    def __init__(
        self,
        deadband=8,
        alpha: float | None = None,
        max_step=100,
        alpha_min: float = 0.25,
        alpha_max: float = 0.75,
        full_response_error: int = 160,
    ):
        if alpha is not None:
            alpha_min = alpha_max = alpha
        self.deadband = [max(0, value) for value in expand_channels(deadband, "deadband", int)]
        self.max_step = [max(1, value) for value in expand_channels(max_step, "max_step", int)]
        self.alpha_min = min(1.0, max(0.01, float(alpha_min)))
        self.alpha_max = min(1.0, max(self.alpha_min, float(alpha_max)))
        self.full_response_error = max(1, int(full_response_error))
        self.current: list[float] | None = None
        self.accepted: list[float] | None = None
        self.last_deadband_suppressed = 0

    def reset(self, values: Sequence[float]) -> None:
        if len(values) != CHANNEL_COUNT:
            raise ValueError("stabilizer state must contain six values")
        self.current = [float(value) for value in values]
        self.accepted = [float(value) for value in values]
        self.last_deadband_suppressed = 0

    def _alpha(self, error: float, deadband: int) -> float:
        span = max(1.0, float(self.full_response_error - deadband))
        progress = min(1.0, max(0.0, (abs(error) - deadband) / span))
        return self.alpha_min + (self.alpha_max - self.alpha_min) * progress

    def update(self, values: Sequence[int]) -> list[int]:
        if len(values) != CHANNEL_COUNT:
            raise ValueError("stabilizer command must contain six values")
        if self.current is None:
            self.reset(values)
            return [int(round(value)) for value in self.current]

        output = []
        suppressed = 0
        for index, (previous, requested) in enumerate(zip(self.current, values)):
            requested = float(requested)
            if abs(requested - self.accepted[index]) > self.deadband[index]:
                self.accepted[index] = requested
            else:
                suppressed += 1

            error = self.accepted[index] - previous
            if abs(error) <= 0.5:
                current = self.accepted[index]
            else:
                target = previous + self._alpha(error, self.deadband[index]) * error
                delta = min(self.max_step[index], max(-self.max_step[index], target - previous))
                current = previous + delta
            rounded = int(round(current))
            previous_rounded = int(round(previous))
            if rounded == previous_rounded and abs(error) > 0.5:
                rounded += 1 if error > 0 else -1
            output.append(rounded)
        self.current = [float(value) for value in output]
        self.last_deadband_suppressed = suppressed
        return output


class ControlDiagnostics:
    def __init__(self, channels: int = CHANNEL_COUNT):
        self.channels = channels
        self.started_at = time.monotonic()
        self.inputs = 0
        self.writes = 0
        self.deadband_suppressed = 0
        self.duplicate_suppressed = 0
        self.request_min = [math.inf] * channels
        self.request_max = [-math.inf] * channels
        self.state_min = [math.inf] * channels
        self.state_max = [-math.inf] * channels

    @staticmethod
    def _observe(values, lows, highs) -> None:
        for index, value in enumerate(values):
            lows[index] = min(lows[index], float(value))
            highs[index] = max(highs[index], float(value))

    @staticmethod
    def _peak_to_peak(lows, highs) -> list[int]:
        return [0 if not math.isfinite(low) else int(round(high - low))
                for low, high in zip(lows, highs)]

    def note_input(self, values, deadband_suppressed: int) -> None:
        self.inputs += 1
        self.deadband_suppressed += int(deadband_suppressed)
        self._observe(values, self.request_min, self.request_max)

    def note_state(self, values) -> None:
        self._observe(values, self.state_min, self.state_max)

    def snapshot_and_reset(self, now: float | None = None) -> dict:
        now = time.monotonic() if now is None else now
        elapsed = max(1e-6, now - self.started_at)
        result = {
            "input_hz": self.inputs / elapsed,
            "write_hz": self.writes / elapsed,
            "deadband_suppressed": self.deadband_suppressed,
            "duplicate_suppressed": self.duplicate_suppressed,
            "request_p2p": self._peak_to_peak(self.request_min, self.request_max),
            "state_p2p": self._peak_to_peak(self.state_min, self.state_max),
        }
        self.__init__(self.channels)
        self.started_at = now
        return result


def main(args=None) -> None:
    import rclpy
    from rcl_interfaces.msg import ParameterDescriptor
    from rclpy.node import Node
    from sensor_msgs.msg import JointState

    class InspireBridgeNode(Node):
        def __init__(self):
            super().__init__("inspire_bridge_node")
            self.declare_parameter("topic", "/cb_left_hand_control_cmd")
            self.declare_parameter("hand_side", "left")
            self.declare_parameter("serial_port", "/dev/ttyUSB0")
            self.declare_parameter("baudrate", 115200)
            self.declare_parameter("slave_id", 1)
            dynamic = ParameterDescriptor(dynamic_typing=True)
            self.declare_parameter("speed", 800)
            self.declare_parameter("force", 500)
            self.declare_parameter("min_send_interval", 0.02)
            self.declare_parameter("state_hz", 60.0)
            self.declare_parameter("force_hz", 20.0)
            self.declare_parameter(
                "command_deadband", [12, 12, 12, 12, 16, 60], dynamic)
            self.declare_parameter("command_filter_alpha", 0.0)
            self.declare_parameter("command_filter_alpha_min", 0.25)
            self.declare_parameter("command_filter_alpha_max", 0.75)
            self.declare_parameter("command_full_response_error", 160)
            self.declare_parameter("command_max_step", 100, dynamic)
            self.declare_parameter("command_write_epsilon", 5)
            self.declare_parameter("command_min", [0, 0, 0, 0, 0, 0])
            self.declare_parameter("command_max", [1000, 1000, 1000, 1000, 1000, 1000])
            self.declare_parameter("dry_run", False)

            self.side = str(self.get_parameter("hand_side").value).lower()
            if self.side not in {"left", "right"}:
                raise ValueError("hand_side must be left or right")
            self.min_interval = max(0.001, float(self.get_parameter("min_send_interval").value))
            self.state_period = 1.0 / max(0.1, float(self.get_parameter("state_hz").value))
            self.force_period = 1.0 / max(0.1, float(self.get_parameter("force_hz").value))
            self.lower = [int(value) for value in self.get_parameter("command_min").value]
            self.upper = [int(value) for value in self.get_parameter("command_max").value]
            apply_limits([0] * 6, self.lower, self.upper)
            self.dry_run = bool(self.get_parameter("dry_run").value)
            self.hand = None
            self.lock = threading.Lock()
            self.pending = None
            self.pending_sequence = 0
            self.sent_sequence = 0
            self.last_applied = [1000] * 6
            self.has_published_action = False
            legacy_alpha = float(self.get_parameter("command_filter_alpha").value)
            self.stabilizer = CommandStabilizer(
                deadband=self.get_parameter("command_deadband").value,
                alpha=legacy_alpha if legacy_alpha > 0 else None,
                alpha_min=float(self.get_parameter("command_filter_alpha_min").value),
                alpha_max=float(self.get_parameter("command_filter_alpha_max").value),
                full_response_error=int(
                    self.get_parameter("command_full_response_error").value),
                max_step=self.get_parameter("command_max_step").value,
            )
            self.write_epsilon = max(
                0, int(self.get_parameter("command_write_epsilon").value))
            self.diagnostics = ControlDiagnostics()
            self.next_diagnostic = time.monotonic() + 2.0
            self.stopping = threading.Event()
            self.failures = {"write": 0, "state": 0, "force": 0}

            topic = str(self.get_parameter("topic").value)
            action_topic = f"/robot/inspire/{self.side}/action_applied"
            state_topic = f"/robot/inspire/{self.side}/joint_states"
            force_topic = f"/robot/inspire/{self.side}/force_states"
            self.action_publisher = self.create_publisher(JointState, action_topic, 20)
            self.state_publisher = self.create_publisher(JointState, state_topic, 20)
            self.force_publisher = self.create_publisher(JointState, force_topic, 20)
            self.subscription = self.create_subscription(JointState, topic, self.on_command, 50)

            if not self.dry_run:
                if str(INSPIRE_SDK) not in sys.path:
                    sys.path.insert(0, str(INSPIRE_SDK))
                from inspire_hand import InspireHand

                self.hand = InspireHand(
                    port=str(self.get_parameter("serial_port").value),
                    baudrate=int(self.get_parameter("baudrate").value),
                    slave_id=int(self.get_parameter("slave_id").value),
                )
                self.hand.open()
                self.hand.set_all_finger_speeds(int(self.get_parameter("speed").value))
                self.hand.set_all_finger_forces(int(self.get_parameter("force").value))
                initial_state = self.hand.get_finger_angles()
                if initial_state is not None and len(initial_state) == 6:
                    self.last_applied = [int(value) for value in initial_state]
                    self.stabilizer.reset(self.last_applied)
            else:
                self.get_logger().warning("dry_run=True, Inspire hardware writes are disabled")
                self.stabilizer.reset(self.last_applied)

            self.worker = threading.Thread(target=self.run_io, name="inspire-modbus", daemon=True)
            self.worker.start()
            self.get_logger().info(
                f"Inspire bridge ready: topic={topic}, side={self.side}, "
                f"serial={self.get_parameter('serial_port').value}"
            )

        def on_command(self, message) -> None:
            try:
                mapped = apply_limits(
                    map_linkerforce_to_inspire(message.position), self.lower, self.upper)
                command = apply_limits(
                    self.stabilizer.update(mapped), self.lower, self.upper)
            except (TypeError, ValueError) as exc:
                self.get_logger().error(f"rejected command: {exc}")
                return
            with self.lock:
                self.diagnostics.note_input(
                    mapped, self.stabilizer.last_deadband_suppressed)
                if not commands_differ(self.pending, command, self.write_epsilon):
                    self.diagnostics.duplicate_suppressed += 1
                    return
                if (
                    self.pending is None
                    and self.has_published_action
                    and not commands_differ(self.last_applied, command, self.write_epsilon)
                ):
                    self.diagnostics.duplicate_suppressed += 1
                    return
                self.pending = command
                self.pending_sequence += 1

        def log_diagnostics(self, now: float) -> None:
            if now < self.next_diagnostic:
                return
            with self.lock:
                stats = self.diagnostics.snapshot_and_reset(now)
            self.next_diagnostic = now + 2.0
            self.get_logger().info(
                "control stats: input=%.1fHz write=%.1fHz deadband=%d duplicate=%d "
                "request_p2p=%s state_p2p=%s" % (
                    stats["input_hz"], stats["write_hz"],
                    stats["deadband_suppressed"], stats["duplicate_suppressed"],
                    stats["request_p2p"], stats["state_p2p"],
                ))

        def publish_joint_state(
            self, publisher, values: Sequence[int], timestamp_ns: int | None = None
        ) -> None:
            message = JointState()
            stamp_ns = (self.get_clock().now().nanoseconds
                        if timestamp_ns is None else int(timestamp_ns))
            message.header.stamp.sec = stamp_ns // 1_000_000_000
            message.header.stamp.nanosec = stamp_ns % 1_000_000_000
            message.name = list(JOINT_NAMES)
            message.position = [float(value) for value in values]
            publisher.publish(message)

        def run_io(self) -> None:
            next_send = 0.0
            next_state = time.monotonic()
            next_force = time.monotonic()
            while not self.stopping.is_set() and rclpy.ok():
                now = time.monotonic()
                self.log_diagnostics(now)
                with self.lock:
                    sequence = self.pending_sequence
                    command = list(self.pending) if self.pending is not None else None
                if command is not None and sequence != self.sent_sequence and now >= next_send:
                    try:
                        def write(values):
                            return True if self.dry_run else self.hand.modbus.write_multiple_registers(
                                ANGLE_SET_REGISTER, values)

                        def publish(values):
                            acknowledged_ns = self.get_clock().now().nanoseconds
                            self.publish_joint_state(
                                self.action_publisher, values, acknowledged_ns)

                        write_acknowledged_command(write, publish, command)
                        self.last_applied = command
                        self.has_published_action = True
                        self.sent_sequence = sequence
                        with self.lock:
                            self.diagnostics.writes += 1
                        self.failures["write"] = 0
                        next_send = time.monotonic() + self.min_interval
                        continue
                    except Exception as exc:
                        self.handle_io_failure("write", exc)
                        next_send = time.monotonic() + self.min_interval

                now = time.monotonic()
                if now >= next_state:
                    try:
                        read_start_ns = self.get_clock().now().nanoseconds
                        state = self.last_applied if self.dry_run else self.hand.get_finger_angles()
                        read_end_ns = self.get_clock().now().nanoseconds
                        if state is None or len(state) != 6:
                            raise RuntimeError("invalid six-channel state response")
                        with self.lock:
                            self.diagnostics.note_state(state)
                        self.failures["state"] = 0
                        self.publish_joint_state(
                            self.state_publisher, state,
                            (read_start_ns + read_end_ns) // 2,
                        )
                    except Exception as exc:
                        self.handle_io_failure("state", exc)
                    completed = time.monotonic()
                    while next_state <= completed:
                        next_state += self.state_period
                now = time.monotonic()
                if now >= next_force and next_state - now >= 0.006:
                    try:
                        read_start_ns = self.get_clock().now().nanoseconds
                        raw = ([0] * 6 if self.dry_run else self.hand.get_finger_forces())
                        read_end_ns = self.get_clock().now().nanoseconds
                        if raw is None or len(raw) != 6:
                            raise RuntimeError("invalid six-channel force response")
                        forces = [int(value) & 0xFFFF for value in raw]
                        forces = [value - 0x10000 if value & 0x8000 else value
                                  for value in forces]
                        message = JointState()
                        stamp_ns = (read_start_ns + read_end_ns) // 2
                        message.header.stamp.sec = stamp_ns // 1_000_000_000
                        message.header.stamp.nanosec = stamp_ns % 1_000_000_000
                        message.name = list(JOINT_NAMES)
                        message.effort = [float(value) for value in forces]
                        self.force_publisher.publish(message)
                        self.failures["force"] = 0
                    except Exception as exc:
                        self.failures["force"] += 1
                        if self.failures["force"] <= 3:
                            self.get_logger().warning(f"Inspire force read failed: {exc}")
                    next_force = time.monotonic() + self.force_period
                self.stopping.wait(0.002)

        def handle_io_failure(self, operation: str, exc: Exception) -> None:
            self.failures[operation] += 1
            self.get_logger().error(
                f"Inspire {operation} failed ({self.failures[operation]}/3): {exc}")
            if self.failures[operation] >= 3:
                self.get_logger().fatal("Inspire communication failed repeatedly; stopping bridge")
                self.stopping.set()
                if rclpy.ok():
                    rclpy.shutdown()

        def close(self) -> None:
            self.stopping.set()
            if hasattr(self, "worker"):
                self.worker.join(timeout=2.0)
            if self.hand is not None:
                try:
                    self.hand.close()
                except Exception as exc:
                    self.get_logger().warning(f"failed to close Inspire serial port: {exc}")

    rclpy.init(args=args)
    node = None
    try:
        node = InspireBridgeNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.close()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
