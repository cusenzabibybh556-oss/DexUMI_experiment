#!/usr/bin/env python3
"""ROS2 sidecar for replaying recorded O6 left-hand state through o6_bridge.

The parent process sends JSON lines on stdin: {"command":"start", "start_ns":...}
or {"command":"stop"}.  Status is emitted as JSON lines on stdout.  This
process never opens CAN; o6_bridge remains the only device owner.
"""

from __future__ import annotations

import argparse
import json
import math
import select
import sys
import time
from pathlib import Path

import h5py
import numpy as np


STATE_DATASET = "dexterous_hand/joint_positions"


def load_states(path: Path, speed: float, origin_timestamp_ns: int) -> tuple[np.ndarray, np.ndarray]:
    if not math.isfinite(speed) or speed <= 0:
        raise ValueError("hand replay speed must be positive and finite")
    with h5py.File(path, "r") as handle:
        attrs = handle.attrs
        if (attrs.get("hand_type") != "o6" or attrs.get("hand_side") != "left"
                or attrs.get("control_source") != "linkerforce"):
            raise ValueError("episode is not a LinkerForce O6 left-hand recording")
        required = (
            "timestamps/master_timestamp_ns",
            "dexterous_hand/unified_timestamp_ns",
            "dexterous_hand/state_valid",
            STATE_DATASET,
        )
        missing = [name for name in required if name not in handle]
        if missing:
            raise ValueError("O6 recorded state datasets missing: " + ", ".join(missing))
        master = np.asarray(handle["timestamps/master_timestamp_ns"][:], dtype=np.int64)
        stamps = np.asarray(handle["dexterous_hand/unified_timestamp_ns"][:], dtype=np.int64)
        states = np.asarray(handle[STATE_DATASET][:], dtype=np.float64)
        valid = np.asarray(handle["dexterous_hand/state_valid"][:], dtype=bool)
    if (master.ndim != 1 or len(master) == 0 or states.shape != (len(master), 6)
            or stamps.shape != master.shape or valid.shape != master.shape):
        raise ValueError("invalid O6 state shape or timestamp shape")
    if not np.all(np.diff(master) > 0):
        raise ValueError("master timestamps are not strictly increasing")
    if origin_timestamp_ns not in master:
        raise ValueError("Tracker origin timestamp is absent from episode master time")
    states, stamps = states[valid], stamps[valid]
    if len(states) == 0 or not np.all(np.diff(stamps) > 0):
        raise ValueError("O6 has no valid, time-ordered state samples")
    if not np.array_equal(stamps, master[valid]):
        raise ValueError("O6 state timestamps are not aligned to the master timeline")
    if (not np.isfinite(states).all() or np.any(states < 0)
            or np.any(states > 255) or np.any(states[:, 0] < 80)):
        raise ValueError("O6 recorded state exceeds configured hardware range")
    # State is the O6 SDK feedback interpolated onto camera/master frames.
    # Samples before the first valid Tracker frame become a t=0 target.
    due_ns = np.maximum(0, np.rint((stamps - origin_timestamp_ns) / speed)).astype(np.int64)
    if np.any(np.diff(due_ns) < 0):
        raise ValueError("O6 state replay timeline is not monotonic")
    return due_ns, states


def emit(event: str, **fields: object) -> None:
    print(json.dumps({"event": event, "monotonic_ns": time.monotonic_ns(),
                      **fields}, ensure_ascii=False), flush=True)


def run(path: Path, speed: float, origin_timestamp_ns: int) -> int:
    due_ns, states = load_states(path, speed, origin_timestamp_ns)
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from std_srvs.srv import SetBool

    rclpy.init()
    node = Node("o6_left_replay_client")
    service = node.create_client(SetBool, "/robot/o6/left/replay_mode")
    publisher = node.create_publisher(JointState, "/robot/o6/left/replay_cmd", 10)
    applied_count = 0
    matched_applied_count = 0
    last_target = None
    last_target_index = None
    last_matched_index = None
    last_publish_ns = None

    def on_applied(message: JointState) -> None:
        nonlocal applied_count, matched_applied_count, last_matched_index
        if message.header.frame_id == "replay":
            applied_count += 1
            if (last_target is not None and len(message.position) == 6
                    and last_target_index != last_matched_index
                    and np.allclose(message.position, last_target, atol=0.5)):
                matched_applied_count += 1
                last_matched_index = last_target_index
                emit("hand_applied", index=last_target_index,
                     received_monotonic_ns=time.monotonic_ns())

    node.create_subscription(JointState, "/robot/o6/left/action_applied", on_applied, 10)

    def set_mode(enabled: bool) -> None:
        if not service.wait_for_service(timeout_sec=5.0):
            raise RuntimeError("找不到 O6 回放服务；请确认唯一的新版 O6 桥接已启动")
        request = SetBool.Request()
        request.data = enabled
        future = service.call_async(request)
        rclpy.spin_until_future_complete(node, future, timeout_sec=5.0)
        if not future.done() or future.result() is None:
            raise RuntimeError("O6 bridge replay service timed out")
        result = future.result()
        if not result.success:
            raise RuntimeError(f"O6 bridge rejected replay mode: {result.message}")

    acquired = False
    try:
        set_mode(True)
        acquired = True
        emit("ready", samples=len(states), duration_s=float(due_ns[-1]) / 1e9,
             source_dataset=STATE_DATASET)
        start_ns = None
        index = 0
        published = 0
        while True:
            rclpy.spin_once(node, timeout_sec=0.0)
            if (last_publish_ns is not None and matched_applied_count == 0
                    and time.monotonic_ns() - last_publish_ns > 1_000_000_000):
                raise RuntimeError("O6 bridge did not confirm any replay action")
            readable, _, _ = select.select([sys.stdin], [], [], 0.002)
            if readable:
                line = sys.stdin.readline()
                if not line:
                    break
                command = json.loads(line)
                if command.get("command") == "stop":
                    break
                if command.get("command") != "start" or start_ns is not None:
                    raise ValueError("invalid or duplicate O6 replay command")
                start_ns = int(command["start_ns"])
                if start_ns <= time.monotonic_ns():
                    raise ValueError("O6 replay start time must be in the future")
                emit("started", start_ns=start_ns)
            if start_ns is None or index >= len(states):
                continue
            now_ns = time.monotonic_ns()
            latest_due = index - 1
            while latest_due + 1 < len(states) and now_ns >= start_ns + int(due_ns[latest_due + 1]):
                latest_due += 1
            if latest_due < index:
                continue
            # If scheduling was delayed, send the newest due command once;
            # never burst several stale finger targets into the CAN bridge.
            index = latest_due
            message = JointState()
            message.name = ["thumb_flexion", "thumb_abduction", "index_flexion",
                            "middle_flexion", "ring_flexion", "pinky_flexion"]
            message.position = states[index].tolist()
            wall_ns = time.time_ns()
            message.header.stamp.sec = wall_ns // 1_000_000_000
            message.header.stamp.nanosec = wall_ns % 1_000_000_000
            last_target = [round(float(value)) for value in states[index]]
            last_target_index = index
            publisher.publish(message)
            emit("o6_state_target_sent", index=index,
                 planned_monotonic_ns=start_ns + int(due_ns[index]),
                 sent_monotonic_ns=time.monotonic_ns(),
                 source_dataset=STATE_DATASET, target=states[index].tolist())
            if last_publish_ns is None:
                last_publish_ns = time.monotonic_ns()
            published += 1
            index += 1
            if index == len(states):
                emit("completed", published=published, applied_count=applied_count,
                     matched_applied_count=matched_applied_count)
    finally:
        if acquired:
            try:
                set_mode(False)  # Hold last O6 target; never switch glove back automatically.
                emit("holding", applied_count=applied_count,
                     matched_applied_count=matched_applied_count)
            except Exception as exc:
                emit("release_error", message=str(exc))
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


def probe_bridge() -> int:
    """Read-only ROS service discovery; do not acquire O6 or send finger targets."""
    import rclpy
    from rclpy.node import Node
    from std_srvs.srv import SetBool

    rclpy.init()
    node = Node("o6_left_replay_probe")
    try:
        client = node.create_client(SetBool, "/robot/o6/left/replay_mode")
        if not client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError("找不到 O6 回放服务；请确认唯一的新版 O6 桥接已启动")
        emit("probe_ready")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path)
    parser.add_argument("--speed", type=float)
    parser.add_argument("--origin-timestamp-ns", type=int)
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    try:
        if args.probe:
            return probe_bridge()
        if args.file is None or args.speed is None or args.origin_timestamp_ns is None:
            parser.error("--file, --speed and --origin-timestamp-ns are required for replay")
        return run(args.file, args.speed, args.origin_timestamp_ns)
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        emit("error", message=str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
