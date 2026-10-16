#!/usr/bin/env python3
"""将一段 TransHandR VIVE Tracker 轨迹用于 TRON2 腕部回放。

HDF5 中的 Tracker 位姿尚未标定到 TRON2 基座坐标系。本程序以首个有效帧为参考，
把相对运动叠加到机器人当前腕部位姿；默认只做离线预览，不连接机器人。

示例
----
离线预览第 65 段，使用按动作方向推断、尚未完整标定的候选换轴，
平移缩放至 30%（相对先前 3% 示例放大十倍；不连接、不移动机器人）：

    python3 replay_tron2_tracker.py --episode 65 --arm left \
        --axis-map y,z,x --scale 0.3 --orientation-scale 0 \
        --speed 0.2 --max-displacement 0.10

只读检查当前腕位姿和计划目标（不发送运动指令）：

    python3 replay_tron2_tracker.py --episode 65 \
        --arm left --axis-map y,z,x --scale 0.3 --orientation-scale 0 \
        --speed 0.2 --max-displacement 0.10 --check-robot

--execute 在确认和倒计时后直接发送 Tracker 轨迹，不再执行接管停稳和重新锚定。
ServoP 仍会同时发送左右腕目标，曾观察到启动时双腕偏移；软件反馈检查
只能停止后续指令，不能阻止首条指令造成的运动，也不能替代实体急停。
即便只读预检通过，也不能证明实际轨迹跟踪正确。
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
import os
import queue
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol

import h5py
import numpy as np


DEFAULT_DATA_ROOT = Path(
    "/home/zmy/workplace/insert-data/transhandr_demo/o6"
)
DEFAULT_ROBOT_IP = "10.192.1.2"

# TRON2 高层笛卡尔控制的已公布工作空间，单位：米。
WORKSPACE = {
    "left": np.array([[0.250, 0.732], [-0.213, 0.900], [-0.673, 0.500]]),
    "right": np.array([[0.250, 0.732], [-0.900, 0.213], [-0.673, 0.500]]),
}

# 软件限差只能在收到反馈之后停止后续指令，不能阻止首条指令已造成的运动。
POST_STOP_OBSERVE_SECONDS = 0.5  # 秒：停止发送指令后，只读观察双腕反馈的时长。
FEEDBACK_INTERVAL_SECONDS = 0.05  # 秒：发送 ServoP 时，发起腕位姿反馈查询的最短间隔。
FEEDBACK_TIMEOUT_SECONDS = 0.5  # 秒：腕位姿反馈超时即停止后续指令；这不是硬件急停。
POSE_PROBE_RETRY_SECONDS = 0.15  # 秒：腕位姿查询未获回复时，允许重试的等待时间。
MOVING_ARM_ERROR_M = 0.5  # 米：轨迹阶段运动臂实测腕位置相对当前目标的最大误差。
HELD_ARM_DRIFT_M = 0.1 # 米：轨迹阶段，未计划运动臂实测腕位置相对起点的最大偏移。
LEFT_ARM_ANGLE_ERROR_DEG = 30.0  # 度：轨迹阶段，左腕相对目标/起点的在线姿态误差上限。
RIGHT_ARM_ANGLE_ERROR_DEG = 5.0  # 度：轨迹阶段，右腕相对目标/起点的在线姿态误差上限。


@dataclass(frozen=True)
class TrackerTrajectory:
    source: Path
    origin_timestamp_ns: int  # 第一帧有效 Tracker 对应的主时间戳
    timestamps_s: np.ndarray
    positions: np.ndarray
    quaternions: np.ndarray  # 四元数顺序：w、x、y、z
    hand_side: str
    original_frames: int

    @property
    def duration_s(self) -> float:
        return float(self.timestamps_s[-1] - self.timestamps_s[0])


@dataclass(frozen=True)
class DualArmPose:
    left_position: np.ndarray
    left_quaternion: np.ndarray
    right_position: np.ndarray
    right_quaternion: np.ndarray


@dataclass(frozen=True)
class PoseSample:
    received_at: float
    pose: DualArmPose


class HandReplaySession(Protocol):
    def acquire(self) -> None: ...
    def start_at(self, start_ns: int) -> None: ...
    def require_healthy(self) -> None: ...


class TraceRecorder:
    """后台写入 JSONL，避免逐帧磁盘 I/O 阻塞 ServoP 发送。"""

    def __init__(self, path: Path):
        self.path = path.expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("x", encoding="utf-8", buffering=1)
        self._queue: queue.SimpleQueue[dict | None] = queue.SimpleQueue()
        self._closed = False
        self._error: Optional[Exception] = None
        self._thread = threading.Thread(target=self._write_loop, daemon=True)
        self._thread.start()

    def _write_loop(self) -> None:
        try:
            while True:
                item = self._queue.get()
                if item is None:
                    break
                self._file.write(json.dumps(item, ensure_ascii=False) + "\n")
                self._file.flush()
        except Exception as exc:
            self._error = exc

    def record(self, event: str, **fields: object) -> None:
        if self._closed:
            return
        self._queue.put({
            "event": event,
            "monotonic_ns": time.monotonic_ns(),
            "unix_ns": time.time_ns(),
            **fields,
        })

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._queue.put(None)
        self._thread.join(timeout=10.0)
        if self._thread.is_alive():
            raise RuntimeError(f"轨迹日志写入超时，文件可能不完整: {self.path}")
        self._file.close()
        if self._error is not None:
            raise RuntimeError(f"轨迹日志写入失败: {self._error}")


class ReplayInterrupted(Exception):
    """用户请求停止轨迹发送时抛出；这不代表硬件急停。"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--episode", type=int, help="episode number, e.g. 65")
    source.add_argument("--file", type=Path, help="explicit HDF5 episode path")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--arm", choices=("left", "right"), default=None)
    parser.add_argument(
        "--axis-map",
        default=None,
        help=("tracker axes expressed in robot axes, e.g. 'z,-x,-y'; "
              "preview defaults to x,y,z, execution requires an explicit value"),
    )
    parser.add_argument(
        "--scale", type=float, default=0.25,
        help="translation scale applied to tracker displacement (default: 0.25)",
    )
    parser.add_argument(
        "--orientation-scale", type=float, default=0.0,
        help=("fraction of relative tracker rotation to apply, 0..1 "
              "(safe default: 0; use 1 only after axis calibration)"),
    )
    parser.add_argument(
        "--rate", type=float, default=300.0,
        help="ServoP output rate after interpolation (default: 300 Hz)",
    )
    parser.add_argument(
        "--speed", type=float, default=1.0,
        help="replay speed multiplier (default: 1.0)",
    )
    parser.add_argument(
        "--max-gap", type=float, default=0.15,
        help="maximum allowed gap between valid tracker frames in seconds",
    )
    parser.add_argument(
        "--max-displacement", type=float, default=0.20,
        help="maximum target displacement from the initial wrist pose in metres",
    )
    parser.add_argument(
        "--max-angle", type=float, default=90.0,
        help="maximum relative wrist rotation in degrees",
    )
    parser.add_argument("--robot-ip", default=DEFAULT_ROBOT_IP)
    live_mode = parser.add_mutually_exclusive_group()
    live_mode.add_argument(
        "--check-robot", action="store_true",
        help="read the current wrist poses and validate targets without moving",
    )
    live_mode.add_argument(
        "--execute", action="store_true",
        help="确认后直接向真机发送 Tracker ServoP 轨迹",
    )
    parser.add_argument(
        "--direct-replay", action="store_true",
        help="兼容旧命令的别名；--execute 已默认直接回放",
    )
    parser.add_argument(
        "--feedback-timeout", type=float, default=FEEDBACK_TIMEOUT_SECONDS,
        metavar="SECONDS",
        help="腕位姿反馈失联多久停止后续指令（0.5 至 3 秒，默认 0.5）",
    )
    live_mode.add_argument(
        "--record-idle", type=float, metavar="SECONDS",
        help="只读记录双腕静止位姿，不发送运动指令（1 至 60 秒）",
    )
    parser.add_argument(
        "--trace-file", type=Path,
        help="将带时间戳的指令与反馈保存为 JSONL；文件不得已存在",
    )
    parser.add_argument(
        "--countdown", type=int, default=15,
        help="seconds after confirmation before ServoP starts (default: 15)",
    )
    return parser.parse_args()


def episode_path(args: argparse.Namespace) -> Path:
    if args.file is not None:
        return args.file.expanduser().resolve()
    return (args.data_root / f"episode_{args.episode:06d}.hdf5").resolve()


def normalized_quaternion(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if not np.isfinite(q).all() or not np.isfinite(norm) or norm < 1e-9:
        raise ValueError("invalid zero/non-finite quaternion")
    return q / norm


def load_tracker(path: Path, max_gap_s: float) -> TrackerTrajectory:
    if not path.is_file():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as f:
        required = (
            "timestamps/master_timestamp_ns",
            "vive_tracker/pos",
            "vive_tracker/quat",
            "vive_tracker/valid",
        )
        missing = [name for name in required if name not in f]
        if missing:
            raise ValueError(f"missing HDF5 datasets: {', '.join(missing)}")
        position_unit = f["vive_tracker"].attrs.get("position_unit")
        if isinstance(position_unit, bytes):
            position_unit = position_unit.decode("utf-8")
        if position_unit not in ("meter", "meters", "m"):
            raise ValueError(
                "vive_tracker/pos must declare position_unit=meter; "
                f"found {position_unit!r}. Refusing to guess the distance unit."
            )

        timestamps_ns = np.asarray(
            f["timestamps/master_timestamp_ns"][:], dtype=np.int64
        )
        positions = np.asarray(f["vive_tracker/pos"][:, 0], dtype=np.float64)
        quaternions = np.asarray(f["vive_tracker/quat"][:, 0], dtype=np.float64)
        valid = np.asarray(f["vive_tracker/valid"][:, 0], dtype=bool)
        hand_side = str(f.attrs.get("hand_side", "left")).lower()

    lengths = {len(timestamps_ns), len(positions), len(quaternions), len(valid)}
    if len(lengths) != 1 or not len(timestamps_ns):
        raise ValueError("tracker arrays are empty or have inconsistent lengths")
    finite = np.isfinite(positions).all(axis=1) & np.isfinite(quaternions).all(axis=1)
    norms = np.linalg.norm(quaternions, axis=1)
    valid &= finite & (norms > 1e-6)
    indices = np.flatnonzero(valid)
    if len(indices) < 2:
        raise ValueError("episode has fewer than two valid tracker frames")

    timestamps_s = (timestamps_ns[indices] - timestamps_ns[indices[0]]) / 1e9
    if np.any(np.diff(timestamps_s) <= 0):
        raise ValueError("tracker timestamps are not strictly increasing")
    largest_gap = float(np.max(np.diff(timestamps_s)))
    if largest_gap > max_gap_s:
        raise ValueError(
            f"valid tracker gap {largest_gap:.3f}s exceeds --max-gap {max_gap_s:.3f}s"
        )
    positions = positions[indices]
    quaternions = quaternions[indices] / norms[indices, None]
    # q 与 -q 表示同一旋转；统一符号，避免相邻帧插值时发生跳变。
    for index in range(1, len(quaternions)):
        if float(np.dot(quaternions[index - 1], quaternions[index])) < 0:
            quaternions[index] *= -1

    return TrackerTrajectory(
        source=path,
        origin_timestamp_ns=int(timestamps_ns[indices[0]]),
        timestamps_s=timestamps_s,
        positions=positions,
        quaternions=quaternions,
        hand_side=hand_side if hand_side in {"left", "right"} else "left",
        original_frames=len(valid),
    )


def parse_axis_map(spec: str) -> np.ndarray:
    tokens = [token.strip().lower() for token in spec.split(",")]
    if len(tokens) != 3:
        raise ValueError("--axis-map must contain three entries, e.g. x,-z,y")
    basis = {"x": 0, "y": 1, "z": 2}
    result = np.zeros((3, 3), dtype=np.float64)
    used = set()
    for robot_axis, token in enumerate(tokens):
        sign = -1.0 if token.startswith("-") else 1.0
        name = token[1:] if token[:1] in "+-" else token
        if name not in basis or name in used:
            raise ValueError("--axis-map must be a signed permutation of x,y,z")
        result[robot_axis, basis[name]] = sign
        used.add(name)
    return result


def quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = normalized_quaternion(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    m = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        q = np.array([0.25 * s, (m[2, 1] - m[1, 2]) / s,
                      (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    else:
        i = int(np.argmax(np.diag(m)))
        if i == 0:
            s = math.sqrt(max(0.0, 1 + m[0, 0] - m[1, 1] - m[2, 2])) * 2
            q = np.array([(m[2, 1] - m[1, 2]) / s, 0.25 * s,
                          (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
        elif i == 1:
            s = math.sqrt(max(0.0, 1 + m[1, 1] - m[0, 0] - m[2, 2])) * 2
            q = np.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s,
                          0.25 * s, (m[1, 2] + m[2, 1]) / s])
        else:
            s = math.sqrt(max(0.0, 1 + m[2, 2] - m[0, 0] - m[1, 1])) * 2
            q = np.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s,
                          (m[1, 2] + m[2, 1]) / s, 0.25 * s])
    return normalized_quaternion(q)


def quaternion_slerp(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    q0 = normalized_quaternion(q0)
    q1 = normalized_quaternion(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0:
        q1 = -q1
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 0.9995:
        return normalized_quaternion(q0 + alpha * (q1 - q0))
    theta = math.acos(dot)
    return normalized_quaternion(
        math.sin((1 - alpha) * theta) / math.sin(theta) * q0
        + math.sin(alpha * theta) / math.sin(theta) * q1
    )


def rotation_angle_deg(matrix: np.ndarray) -> float:
    cosine = min(1.0, max(-1.0, (float(np.trace(matrix)) - 1.0) / 2.0))
    return math.degrees(math.acos(cosine))


def relative_motion(
    trajectory: TrackerTrajectory,
    axis_map: np.ndarray,
    translation_scale: float,
    orientation_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if translation_scale < 0:
        raise ValueError("--scale must be non-negative")
    if not 0 <= orientation_scale <= 1:
        raise ValueError("--orientation-scale must be between 0 and 1")
    delta_positions = (trajectory.positions - trajectory.positions[0]) @ axis_map.T
    delta_positions *= translation_scale

    first_rotation = quaternion_to_matrix(trajectory.quaternions[0])
    delta_quaternions = []
    angles = []
    identity_q = np.array([1.0, 0.0, 0.0, 0.0])
    for q in trajectory.quaternions:
        # 用首帧抵消固定的旋转参考；安装位置偏移造成的平移仍需单独标定。
        delta_tracker = quaternion_to_matrix(q) @ first_rotation.T
        delta_robot = axis_map @ delta_tracker @ axis_map.T
        delta_q = matrix_to_quaternion(delta_robot)
        delta_quaternions.append(
            quaternion_slerp(identity_q, delta_q, orientation_scale)
        )
        angles.append(rotation_angle_deg(delta_robot) * orientation_scale)
    return delta_positions, np.asarray(delta_quaternions), np.asarray(angles)


def sample_relative_trajectory(
    trajectory: TrackerTrajectory,
    delta_positions: np.ndarray,
    delta_quaternions: np.ndarray,
    output_rate: float,
    speed: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if output_rate <= 0 or speed <= 0:
        raise ValueError("--rate and --speed must be positive")
    replay_duration = trajectory.duration_s / speed
    count = max(2, int(math.ceil(replay_duration * output_rate)) + 1)
    replay_times = np.linspace(0.0, replay_duration, count)
    source_times = replay_times * speed
    output_positions = np.column_stack(
        [np.interp(source_times, trajectory.timestamps_s, delta_positions[:, i])
         for i in range(3)]
    )
    output_quaternions = np.empty((count, 4), dtype=np.float64)
    right_indices = np.searchsorted(trajectory.timestamps_s, source_times, side="right")
    right_indices = np.clip(right_indices, 1, len(trajectory.timestamps_s) - 1)
    for output_index, right in enumerate(right_indices):
        left = right - 1
        span = trajectory.timestamps_s[right] - trajectory.timestamps_s[left]
        alpha = 0.0 if span <= 0 else (
            source_times[output_index] - trajectory.timestamps_s[left]
        ) / span
        output_quaternions[output_index] = quaternion_slerp(
            delta_quaternions[left], delta_quaternions[right], float(alpha)
        )
    output_quaternions[0] = delta_quaternions[0]
    output_quaternions[-1] = delta_quaternions[-1]
    return replay_times, output_positions, output_quaternions


def compose_pose(
    anchor_position: np.ndarray,
    anchor_quaternion: np.ndarray,
    delta_position: np.ndarray,
    delta_quaternion: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    position = np.asarray(anchor_position) + np.asarray(delta_position)
    rotation = quaternion_to_matrix(delta_quaternion) @ quaternion_to_matrix(
        anchor_quaternion
    )
    return position, matrix_to_quaternion(rotation)


def parse_dual_arm_pose(message: dict) -> DualArmPose:
    data = message.get("data", {})
    if not isinstance(data, dict) or data.get("result") != "success":
        result = data.get("result", "missing result") if isinstance(data, dict) else data
        raise RuntimeError(f"robot did not provide dual-arm poses: {result}")
    try:
        left_position = np.asarray(data["left_position"], dtype=np.float64)
        right_position = np.asarray(data["right_position"], dtype=np.float64)
        left_quaternion = normalized_quaternion(data["left_quat"])
        right_quaternion = normalized_quaternion(data["right_quat"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"invalid dual-arm pose payload: {data}") from exc
    if (left_position.shape != (3,) or right_position.shape != (3,)
            or not np.isfinite(left_position).all()
            or not np.isfinite(right_position).all()
            or left_quaternion.shape != (4,)
            or right_quaternion.shape != (4,)):
        raise RuntimeError(f"invalid dual-arm pose payload: {data}")
    return DualArmPose(
        left_position, left_quaternion, right_position, right_quaternion
    )


def pose_for_trace(pose: DualArmPose) -> dict[str, list[float]]:
    """把双腕位姿转换为可写入 JSONL 的普通列表。"""
    return {
        "left_position": pose.left_position.tolist(),
        "left_quaternion": pose.left_quaternion.tolist(),
        "right_position": pose.right_position.tolist(),
        "right_quaternion": pose.right_quaternion.tolist(),
    }


class Tron2JsonClient:
    def __init__(self, robot_ip: str, trace: Optional[TraceRecorder] = None):
        import limxsdk.robot.Robot as Robot
        import limxsdk.robot.RobotType as RobotType

        self.robot = Robot(RobotType.Tron2)
        if not self.robot.init(robot_ip):
            raise RuntimeError(f"cannot connect to TRON2 at {robot_ip}")
        self.accid = ""
        self.trace = trace
        self._condition = threading.Condition()
        self._robot_identity_ready = False
        self._robot_identity_error: Optional[str] = None
        self._messages: list[dict] = []
        self._failure: Optional[str] = None
        self._pose_probe_guid: Optional[str] = None
        self._pose_probe_sent_at = 0.0
        self._pose_samples: list[PoseSample] = []
        self._pose_probe_error: Optional[str] = None
        self.robot.subscribeJsonMessage(self._on_message)

    def _on_message(self, payload: str) -> None:
        try:
            message = json.loads(payload)
        except (TypeError, json.JSONDecodeError):
            return
        with self._condition:
            title = str(message.get("title", ""))
            if title == "notify_robot_info":
                incoming_accid = message.get("accid")
                if isinstance(incoming_accid, str) and incoming_accid:
                    if self._robot_identity_ready and self.accid != incoming_accid:
                        self._robot_identity_error = (
                            f"robot identity changed: {self.accid} -> {incoming_accid}"
                        )
                    else:
                        self.accid = incoming_accid
                        self._robot_identity_ready = True
            self._messages.append(message)
            if title.lower() == "notify_servop":
                data = message.get("data", {})
                result = data.get("result", "unknown failure") if isinstance(data, dict) else data
                if result != "success":
                    self._failure = str(result)
            if (title == "response_get_servo_pose"
                    and message.get("guid") == self._pose_probe_guid
                    and self._pose_probe_guid is not None):
                self._pose_probe_guid = None
                try:
                    pose = parse_dual_arm_pose(message)
                except RuntimeError as exc:
                    self._pose_probe_error = str(exc)
                else:
                    self._pose_samples.append(PoseSample(time.monotonic(), pose))
                    if self.trace is not None:
                        self.trace.record(
                            "pose_feedback", request_guid=message.get("guid"),
                            robot_timestamp_ms=message.get("data", {}).get("timestamp"),
                            **pose_for_trace(pose),
                        )
            elif title == "response_get_servo_pose" and self.trace is not None:
                reply_data = message.get("data")
                self.trace.record(
                    "pose_probe_unmatched_reply",
                    reply_guid=message.get("guid"),
                    pending_guid=self._pose_probe_guid,
                    result=(reply_data.get("result")
                            if isinstance(reply_data, dict) else None),
                )
            self._condition.notify_all()

    def wait_for_identity(self, timeout: float = 5.0) -> None:
        with self._condition:
            deadline = time.monotonic() + timeout
            while not self._robot_identity_ready:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("no notify_robot_info/accid after connecting")
                self._condition.wait(remaining)
            if self._robot_identity_error:
                raise RuntimeError(self._robot_identity_error)

    def send(self, title: str, data: object, *, guid: Optional[str] = None) -> str:
        if self._robot_identity_error:
            raise RuntimeError(self._robot_identity_error)
        if not self._robot_identity_ready:
            raise RuntimeError("robot identity not ready; refusing to send command")
        guid = guid or uuid.uuid4().hex
        message = {
            "accid": self.accid,
            "title": title,
            "timestamp": int(time.time() * 1000),
            "guid": guid,
            "data": data,
        }
        if title == "request_servop" and self.trace is not None:
            # 先记目标和发送前时间；机器人响应与实际反馈另行记录。
            self.trace.record(
                "servop_command", guid=guid,
                left_pos=data["left_pos"], right_pos=data["right_pos"],
            )
        sent = self.robot.publishJsonMessage(
            json.dumps(message, separators=(",", ":"))
        )
        if sent is not None and not sent:
            raise RuntimeError(f"local SDK failed to publish {title}")
        return guid

    def request(
        self, title: str, data: dict, response_title: str, timeout: float = 3.0
    ) -> dict:
        with self._condition:
            start = len(self._messages)
            guid = self.send(title, data)
            # 其他客户端的腕部位姿回复也会被广播，因此按 GUID 匹配；
            # 厂商的其他回复不一定回传 GUID，仍按消息类型匹配。
            correlate_guid = response_title in {
                "response_get_move_pose", "response_get_servo_pose"
            }
            deadline = time.monotonic() + timeout
            while True:
                for message in self._messages[start:]:
                    if (message.get("title") == response_title
                            and (not correlate_guid or message.get("guid") == guid)):
                        return message
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"no {response_title} within {timeout:.1f}s")
                self._condition.wait(remaining)

    def current_pose(self, label: str = "snapshot") -> DualArmPose:
        response = self.request(
            "request_get_move_pose", {}, "response_get_move_pose", timeout=4.0
        )
        pose = parse_dual_arm_pose(response)
        if self.trace is not None:
            self.trace.record(
                "pose_snapshot", label=label,
                robot_timestamp_ms=response.get("data", {}).get("timestamp"),
                **pose_for_trace(pose),
            )
        return pose

    def current_servo_pose(self, label: str = "servo_snapshot") -> DualArmPose:
        """读取与 ServoP 对应的双腕实测位姿，禁止以旧缓存作回放起点。"""
        response = self.request(
            "request_get_servo_pose", {}, "response_get_servo_pose", timeout=4.0
        )
        pose = parse_dual_arm_pose(response)
        if self.trace is not None:
            self.trace.record(
                "pose_snapshot", label=label,
                robot_timestamp_ms=response.get("data", {}).get("timestamp"),
                **pose_for_trace(pose),
            )
        return pose

    def joint_snapshot(self, label: str) -> tuple[np.ndarray, np.ndarray]:
        """只读记录双臂关节反馈，以便区分腕位姿跳变与 IK 关节跳变。"""
        response = self.request(
            "request_get_joint_state", {}, "response_get_joint_state", timeout=4.0
        )
        data = response.get("data", {})
        if not isinstance(data, dict) or data.get("result") != "success":
            raise RuntimeError(f"robot did not provide joint state: {data}")
        q = np.asarray(data.get("q", []), dtype=np.float64)
        dq = np.asarray(data.get("dq", []), dtype=np.float64)
        if (q.size < 14 or dq.size < 14 or not np.isfinite(q).all()
                or not np.isfinite(dq).all()):
            raise RuntimeError("invalid joint state in read-only snapshot")
        if self.trace is not None:
            self.trace.record(
                "joint_snapshot", label=label,
                robot_timestamp_ms=data.get("timestamp"),
                q=q.tolist(), dq=dq.tolist(),
            )
        return q, dq

    def probe_pose(self) -> None:
        """异步请求 ServoP 腕位姿反馈，不阻塞 300 Hz 发送循环。"""
        with self._condition:
            if self._pose_probe_error:
                raise RuntimeError(self._pose_probe_error)
            now = time.monotonic()
            if self._pose_probe_guid is not None:
                if now - self._pose_probe_sent_at < POSE_PROBE_RETRY_SECONDS:
                    return
                if self.trace is not None:
                    self.trace.record(
                        "pose_probe_retry", expired_guid=self._pose_probe_guid,
                        age_s=now - self._pose_probe_sent_at,
                    )
                self._pose_probe_guid = None
            guid = uuid.uuid4().hex
            self._pose_probe_guid = guid
            self._pose_probe_sent_at = now
            if self.trace is not None:
                self.trace.record("pose_probe_sent", guid=guid)
        try:
            self.send("request_get_servo_pose", {}, guid=guid)
        except Exception:
            with self._condition:
                if self._pose_probe_guid == guid:
                    self._pose_probe_guid = None
            raise

    def pose_samples(self) -> list[PoseSample]:
        with self._condition:
            if self._pose_probe_error:
                raise RuntimeError(self._pose_probe_error)
            return list(self._pose_samples)

    def latest_pose_sample(self) -> Optional[PoseSample]:
        with self._condition:
            if self._pose_probe_error:
                raise RuntimeError(self._pose_probe_error)
            return self._pose_samples[-1] if self._pose_samples else None

    def failure(self) -> Optional[str]:
        with self._condition:
            return self._failure


def check_workspace(side: str, positions: np.ndarray) -> None:
    limits = WORKSPACE[side]
    outside = np.any((positions < limits[:, 0]) | (positions > limits[:, 1]), axis=1)
    if np.any(outside):
        index = int(np.flatnonzero(outside)[0])
        raise ValueError(
            f"{side} target {index} outside documented workspace: "
            f"{positions[index].round(4).tolist()}"
        )


def print_summary(
    trajectory: TrackerTrajectory,
    side: str,
    replay_times: np.ndarray,
    delta_positions: np.ndarray,
    angles_deg: np.ndarray,
    args: argparse.Namespace,
) -> None:
    displacement_norms = np.linalg.norm(delta_positions, axis=1)
    print(f"数据文件: {trajectory.source}")
    print(
        f"Tracker: {len(trajectory.timestamps_s)}/{trajectory.original_frames} 有效帧, "
        f"原始时长 {trajectory.duration_s:.3f}s"
    )
    print("Tracker 位置单位: m（已校验 HDF5 position_unit）")
    print(
        f"重放: {side} 臂, {len(replay_times)} 帧 @ {args.rate:g}Hz, "
        f"时长 {replay_times[-1]:.3f}s, 速度 {args.speed:g}x"
    )
    print(
        "相对平移范围 (m): min="
        f"{delta_positions.min(axis=0).round(4).tolist()}, max="
        f"{delta_positions.max(axis=0).round(4).tolist()}"
    )
    print(
        f"最大位移 {displacement_norms.max():.4f}m; "
        f"最大相对转角 {angles_deg.max():.2f}deg; axis-map={args.axis_map}"
    )
    if args.execute:
        mode = (
            "零位移发送流程测试（不会验证机械臂运动）"
            if args.scale == 0 and args.orientation_scale == 0
            else "真机轨迹执行"
        )
    elif args.check_robot:
        mode = "机器人只读预检（不发送运动指令）"
    elif args.record_idle is not None:
        mode = "机器人只读静止记录（不发送运动指令）"
    else:
        mode = "仅预览（未连接、未控制机器人）"
    print("模式: " + mode)
    if args.execute:
        print(f"腕位姿反馈超时: {args.feedback_timeout:g}s"
              "（失联期间可能继续发送 ServoP；这不是急停）")
        print(
            f"在线腕姿态误差上限: left {LEFT_ARM_ANGLE_ERROR_DEG:g}° / "
            f"right {RIGHT_ARM_ANGLE_ERROR_DEG:g}°"
        )
    if args.execute:
        print("轨迹启动: 直接发送首帧，不进行接管停稳或重新锚定；"
              "首帧仍可能引发双臂跳变")


def default_trace_path() -> Path:
    """自动生成不覆盖既有文件的轨迹日志路径。"""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return (Path(__file__).resolve().parent / "tron2_replay_traces"
            / f"tron2_{stamp}_{uuid.uuid4().hex[:8]}.jsonl")


def record_idle(
    client: Tron2JsonClient, anchor: DualArmPose, duration_s: float
) -> None:
    """只读采样静止时的双腕位置，以首帧为基准计算最大偏移。"""
    start = time.monotonic()
    count = 0
    max_left = 0.0
    max_right = 0.0
    while time.monotonic() - start < duration_s:
        time.sleep(min(0.1, max(0.0, start + duration_s - time.monotonic())))
        if time.monotonic() - start >= duration_s:
            break
        pose = client.current_pose(label="idle")
        count += 1
        max_left = max(max_left, float(np.linalg.norm(
            pose.left_position - anchor.left_position
        )))
        max_right = max(max_right, float(np.linalg.norm(
            pose.right_position - anchor.right_position
        )))
    print(
        f"静止记录完成：{count} 次双腕采样；相对首帧最大偏移 "
        f"left={max_left * 1000:.1f} mm, right={max_right * 1000:.1f} mm。"
    )


def servop_data(left: np.ndarray, right: np.ndarray) -> dict[str, list[float]]:
    """按厂商协议构造左右臂成对的 XYZ+WXYZ ServoP 数据。"""
    poses = {}
    for name, value in (("left_pos", left), ("right_pos", right)):
        pose = np.asarray(value, dtype=np.float64)
        if pose.shape != (7,) or not np.isfinite(pose).all():
            raise ValueError(f"{name} must contain 7 finite XYZ+WXYZ values")
        pose = pose.copy()
        pose[3:] = normalized_quaternion(pose[3:])
        poses[name] = pose.tolist()
    return poses


def quaternion_distance_deg(first: np.ndarray, second: np.ndarray) -> float:
    dot = abs(float(np.dot(normalized_quaternion(first), normalized_quaternion(second))))
    return math.degrees(2.0 * math.acos(min(1.0, max(0.0, dot))))


def check_anchor_stable(anchor: DualArmPose, latest: DualArmPose) -> None:
    """若确认或倒计时期间任一腕位姿变化，禁止从过期起点启动 ServoP。"""
    for side in ("left", "right"):
        position_error = float(np.linalg.norm(
            getattr(latest, f"{side}_position") - getattr(anchor, f"{side}_position")
        ))
        angle_error = quaternion_distance_deg(
            getattr(latest, f"{side}_quaternion"),
            getattr(anchor, f"{side}_quaternion"),
        )
        if position_error > 0.003 or angle_error > 1.0:
            raise RuntimeError(
                f"{side} 腕在倒计时期间变化了 {position_error * 1000:.1f} mm / "
                f"{angle_error:.1f}°，取消回放；请检查是否有其他控制器或姿态漂移。"
            )


def check_live_feedback(
    anchor: DualArmPose,
    side: str,
    target_position: np.ndarray,
    target_quaternion: np.ndarray,
    sample: PoseSample,
    phase: str,
) -> None:
    """检查每次实测反馈；超限后调用方立即停止后续指令。"""
    held_side = "right" if side == "left" else "left"
    actual_position = getattr(sample.pose, f"{side}_position")
    actual_quaternion = getattr(sample.pose, f"{side}_quaternion")
    held_position = getattr(sample.pose, f"{held_side}_position")
    held_quaternion = getattr(sample.pose, f"{held_side}_quaternion")
    held_anchor_position = getattr(anchor, f"{held_side}_position")
    held_anchor_quaternion = getattr(anchor, f"{held_side}_quaternion")
    moving_error = float(np.linalg.norm(actual_position - target_position))
    held_drift = float(np.linalg.norm(held_position - held_anchor_position))
    moving_angle = quaternion_distance_deg(actual_quaternion, target_quaternion)
    held_angle = quaternion_distance_deg(held_quaternion, held_anchor_quaternion)
    moving_angle_limit = (LEFT_ARM_ANGLE_ERROR_DEG if side == "left"
                          else RIGHT_ARM_ANGLE_ERROR_DEG)
    held_angle_limit = (LEFT_ARM_ANGLE_ERROR_DEG if held_side == "left"
                        else RIGHT_ARM_ANGLE_ERROR_DEG)
    if (moving_error > MOVING_ARM_ERROR_M or held_drift > HELD_ARM_DRIFT_M
            or moving_angle > moving_angle_limit
            or held_angle > held_angle_limit):
        raise RuntimeError(
            f"{phase} 双臂反馈超限：{side} 目标误差 "
            f"{moving_error * 1000:.1f} mm / {moving_angle:.1f}°，"
            f"{held_side} 偏移 {held_drift * 1000:.1f} mm / "
            f"{held_angle:.1f}°；停止发送后续 ServoP，"
            "这不是硬件急停"
        )


class FeedbackGuard:
    """跟踪新鲜的 ServoP 反馈，避免无反馈时继续盲发轨迹。"""

    def __init__(
        self, anchor: DualArmPose, side: str,
        feedback_timeout_s: float = FEEDBACK_TIMEOUT_SECONDS,
    ):
        self.anchor = anchor
        self.side = side
        self.feedback_timeout_s = feedback_timeout_s
        self.last_checked_at = -math.inf
        self.last_feedback_at = time.monotonic()

    def observe(
        self,
        sample: Optional[PoseSample],
        target_position: np.ndarray,
        target_quaternion: np.ndarray,
        phase: str,
    ) -> None:
        if sample is None or sample.received_at <= self.last_checked_at:
            return
        check_live_feedback(
            self.anchor, self.side, target_position, target_quaternion,
            sample, phase,
        )
        self.last_checked_at = sample.received_at
        self.last_feedback_at = sample.received_at

    def require_fresh(self, now: float) -> None:
        if now - self.last_feedback_at > self.feedback_timeout_s:
            raise RuntimeError(
                f"ServoP 腕位姿反馈超过 {self.feedback_timeout_s:.1f}s 未更新；"
                "停止发送后续指令，这不是硬件急停"
            )


def verify_observed_motion(
    anchor: DualArmPose,
    side: str,
    delta_positions: np.ndarray,
    delta_quaternions: np.ndarray,
    samples: list[PoseSample],
    final_pose: DualArmPose,
) -> bool:
    """比较实际最大位移、末帧误差与未控制臂偏移，不能只看是否发生运动。"""
    expected_distance = float(np.linalg.norm(delta_positions, axis=1).max())
    expected_angle = max(
        quaternion_distance_deg(np.array([1.0, 0.0, 0.0, 0.0]), q)
        for q in delta_quaternions
    )
    anchor_position = getattr(anchor, f"{side}_position")
    anchor_quaternion = getattr(anchor, f"{side}_quaternion")
    final_position = getattr(final_pose, f"{side}_position")
    final_quaternion = getattr(final_pose, f"{side}_quaternion")
    held_side = "right" if side == "left" else "left"
    held_anchor_position = getattr(anchor, f"{held_side}_position")
    held_final_position = getattr(final_pose, f"{held_side}_position")
    held_distance = max(
        [float(np.linalg.norm(held_final_position - held_anchor_position))]
        + [float(np.linalg.norm(
            getattr(sample.pose, f"{held_side}_position") - held_anchor_position
        )) for sample in samples]
    )
    final_target_position, final_target_quaternion = compose_pose(
        anchor_position, anchor_quaternion,
        delta_positions[-1], delta_quaternions[-1],
    )
    final_position_error = float(np.linalg.norm(final_position - final_target_position))
    final_angle_error = quaternion_distance_deg(
        final_quaternion, final_target_quaternion
    )
    if len(samples) < 3:
        if expected_distance < 0.005 and expected_angle < 3.0:
            print("零位移发送完成，但反馈不足，无法验证机器人是否保持静止。")
            return False
        raise RuntimeError(
            f"回放期间只有 {len(samples)} 次有效腕部反馈，无法验证运动。"
        )
    observed_distance = max(
        float(np.linalg.norm(getattr(sample.pose, f"{side}_position") - anchor_position))
        for sample in samples
    )
    observed_angle = max(
        quaternion_distance_deg(
            anchor_quaternion, getattr(sample.pose, f"{side}_quaternion")
        )
        for sample in samples
    )
    print(
        f"机器人反馈 {side} 腕: {len(samples)} 次, 最大平移 "
        f"{observed_distance * 1000:.1f} mm, 最大转角 {observed_angle:.1f}°; "
        f"末帧位置误差 {final_position_error * 1000:.1f} mm, "
        f"姿态误差 {final_angle_error:.1f}°; "
        f"非控制臂最大偏移 {held_distance * 1000:.1f} mm"
    )
    position_tolerance = max(0.005, expected_distance * 0.3)
    angle_tolerance = max(2.0, expected_angle * 0.3)
    if observed_distance > expected_distance + position_tolerance:
        raise RuntimeError(
            f"腕部最大位移 {observed_distance * 1000:.1f} mm 超过计划 "
            f"{expected_distance * 1000:.1f} mm 的允许范围；不能认定回放正确。"
        )
    if observed_angle > expected_angle + angle_tolerance:
        raise RuntimeError(
            f"腕部最大转角 {observed_angle:.1f}° 超过计划 "
            f"{expected_angle:.1f}° 的允许范围；不能认定回放正确。"
        )
    if final_position_error > position_tolerance or final_angle_error > angle_tolerance:
        raise RuntimeError(
            f"末帧未跟踪目标：位置误差 {final_position_error * 1000:.1f} mm，"
            f"姿态误差 {final_angle_error:.1f}°；不能认定回放正确。"
        )
    if held_distance > 0.005:
        raise RuntimeError(
            f"未指定运动的 {held_side} 腕偏移 {held_distance * 1000:.1f} mm；"
            "不能认定单臂回放安全或正确。"
        )
    if expected_distance < 0.005 and expected_angle < 3.0:
        print("目标变化小于 5 mm 且 3°；只能确认机器人基本保持静止。")
        return False
    if (expected_distance >= 0.005
            and observed_distance < max(0.002, expected_distance * 0.3)):
        raise RuntimeError(
            f"计划位移 {expected_distance * 1000:.1f} mm，机器人仅反馈 "
            f"{observed_distance * 1000:.1f} mm；不能认定回放正确。"
        )
    if (expected_angle >= 3.0
            and observed_angle < max(1.0, expected_angle * 0.3)):
        raise RuntimeError(
            f"计划转角 {expected_angle:.1f}°，机器人仅反馈 "
            f"{observed_angle:.1f}°；不能认定回放正确。"
        )
    return True


def execute_replay(
    client: Tron2JsonClient,
    anchor: DualArmPose,
    side: str,
    replay_times: np.ndarray,
    delta_positions: np.ndarray,
    delta_quaternions: np.ndarray,
    args: argparse.Namespace,
    *,
    confirmation_word: str = "DIRECT",
    hand_replay: HandReplaySession | None = None,
) -> bool:
    if side == "left":
        anchor_position, anchor_quaternion = anchor.left_position, anchor.left_quaternion
    else:
        anchor_position, anchor_quaternion = anchor.right_position, anchor.right_quaternion

    target_positions = np.asarray(
        [anchor_position + delta for delta in delta_positions], dtype=np.float64
    )
    check_workspace(side, target_positions)

    answer = input(
        "将向 TRON2 双臂发送 ServoP，直接回放 Tracker 轨迹"
        + ("，并同步回放 O6 左手" if hand_replay is not None else "") + "。"
        + f"输入 {confirmation_word} 继续: "
    )
    if answer.strip() != confirmation_word:
        raise RuntimeError("cancelled by user")

    if hand_replay is not None:
        hand_replay.acquire()

    print(
        "提示：Ctrl+C 和软件超限只会停止后续指令，不等同于硬件急停。"
        "紧急情况请使用机器人实体急停或遥控器急停。"
    )
    for remaining in range(args.countdown, 0, -1):
        print(
            f"\r{remaining:2d}s 后开始 ServoP 直接轨迹；"
            "请离开电脑并拿好遥控器急停...",
            end="",
            flush=True,
        )
        time.sleep(1.0)
    pre_start_move = client.current_pose(label="pre_start_move")
    check_anchor_stable(anchor, pre_start_move)
    servo_anchor = client.current_servo_pose(label="pre_start_servo")
    check_anchor_stable(pre_start_move, servo_anchor)
    _, pre_start_dq = client.joint_snapshot(label="pre_replay")
    if float(np.max(np.abs(pre_start_dq[:14]))) > 0.05:
        raise RuntimeError("双臂关节仍在运动，取消直接回放")
    anchor = servo_anchor
    anchor_position = getattr(anchor, f"{side}_position")
    anchor_quaternion = getattr(anchor, f"{side}_quaternion")
    held_side = "right" if side == "left" else "left"
    target_positions = anchor_position[None, :] + delta_positions
    check_workspace(side, target_positions)
    check_workspace(
        held_side, getattr(anchor, f"{held_side}_position")[None, :]
    )
    guard = FeedbackGuard(
        anchor, side, feedback_timeout_s=args.feedback_timeout,
    )
    next_probe_at = 0.0
    phase = "trajectory"

    def send_monitored(
        moving_position: np.ndarray,
        moving_quaternion: np.ndarray,
        current_phase: str,
    ) -> None:
        nonlocal next_probe_at
        if hand_replay is not None:
            hand_replay.require_healthy()
        failure = client.failure()
        if failure:
            raise RuntimeError(f"TRON2 ServoP rejected the trajectory: {failure}")
        moving = np.concatenate((moving_position, moving_quaternion))
        held = np.concatenate((
            getattr(anchor, f"{held_side}_position"),
            getattr(anchor, f"{held_side}_quaternion"),
        ))
        left, right = (moving, held) if side == "left" else (held, moving)
        client.send("request_servop", servop_data(left, right))
        now = time.monotonic()
        if now >= next_probe_at:
            client.probe_pose()
            next_probe_at = now + FEEDBACK_INTERVAL_SECONDS
        guard.observe(
            client.latest_pose_sample(), moving_position, moving_quaternion,
            current_phase,
        )
        guard.require_fresh(now)

    def observe_after_stop(
        target_position: np.ndarray,
        target_quaternion: np.ndarray,
        current_phase: str,
    ) -> None:
        """停止发送后只读观察控制器释放行为，绝不把它当作急停。"""
        deadline = time.monotonic() + POST_STOP_OBSERVE_SECONDS
        while time.monotonic() < deadline:
            time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
            latest = client.current_servo_pose(label="post_stop")
            guard.observe(
                PoseSample(time.monotonic(), latest),
                target_position, target_quaternion, current_phase,
            )

    try:
        if hand_replay is not None:
            # 首帧前先取得一次新的腕反馈，再安排共同起点。不能在等待 O6
            # 同步启动的时间里耗尽 0.5s 反馈超时预算。
            probe_at = time.monotonic()
            probe_deadline = probe_at + args.feedback_timeout
            client.probe_pose()
            while True:
                sample = client.latest_pose_sample()
                if sample is not None and sample.received_at >= probe_at:
                    check_anchor_stable(anchor, sample.pose)
                    guard.observe(
                        sample, anchor_position, anchor_quaternion, "pre_start",
                    )
                    break
                if time.monotonic() >= probe_deadline:
                    raise RuntimeError(
                        "联合回放首帧前未收到新的 ServoP 腕位姿反馈；"
                        "未安排 O6 动作，也未发送轨迹"
                    )
                time.sleep(min(0.01, probe_deadline - time.monotonic()))
                client.probe_pose()
            # 两个进程仍使用同一个 CLOCK_MONOTONIC 起点。250ms 足够
            # sidecar 接收计划，也小于默认 500ms 反馈新鲜度门限。
            start_ns = time.monotonic_ns() + 250_000_000
            hand_replay.start_at(start_ns)
            start = start_ns / 1e9
        else:
            start = time.monotonic()
        if client.trace is not None:
            client.trace.record("replay_start", side=side,
                                start_monotonic_ns=int(start * 1e9))
        if getattr(args, "web_control", False):
            print('ROBOT_REPLAY_EVENT {"phase":"replaying"}', flush=True)
        print("开始直接回放 Tracker 轨迹。请持续观察机器人。")
        for index, due in enumerate(replay_times):
            while True:
                remaining = start + float(due) - time.monotonic()
                if remaining <= 0:
                    break
                time.sleep(min(remaining, 0.001))
            moving_position, moving_quaternion = compose_pose(
                anchor_position,
                anchor_quaternion,
                delta_positions[index],
                delta_quaternions[index],
            )
            send_monitored(moving_position, moving_quaternion, "trajectory")

        # 末帧仍在实时监控下短暂发送；异常时不盲目继续保持目标。
        phase = "final_hold"
        hold_start = time.monotonic()
        hold_count = max(1, int(math.ceil(0.2 * args.rate)))
        for index in range(hold_count):
            remaining = hold_start + index / args.rate - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            send_monitored(moving_position, moving_quaternion, "final_hold")
    except KeyboardInterrupt as exc:
        raise ReplayInterrupted(
            "收到 Ctrl+C，已停止发送后续目标；机器人端保持行为未知，"
            "这不是硬件急停"
        ) from exc
    except (RuntimeError, TimeoutError, ValueError) as exc:
        if client.trace is not None:
            client.trace.record("control_abort", phase=phase, message=str(exc))
        print(
            f"已停止发送后续 ServoP：{exc}。如机器人仍异常运动，请立即使用实体急停。",
            file=sys.stderr, flush=True,
        )
        # 只读记录中止后的状态；记录失败不能掩盖原始超限原因。
        for snapshot in (
            lambda: client.current_servo_pose(label="abort"),
            lambda: client.joint_snapshot(label="abort"),
        ):
            try:
                snapshot()
            except (RuntimeError, TimeoutError):
                pass
        raise

    try:
        failure = client.failure()
        if failure:
            raise RuntimeError(f"TRON2 ServoP rejected the trajectory: {failure}")
        observe_after_stop(moving_position, moving_quaternion, "trajectory")
        final_pose = client.current_servo_pose(label="final")
        _, final_dq = client.joint_snapshot(label="post_replay")
        if float(np.max(np.abs(final_dq[:14]))) > 0.05:
            raise RuntimeError("停止发送后双臂关节仍在运动，不能认定回放完成")
    except (RuntimeError, TimeoutError) as exc:
        if client.trace is not None:
            client.trace.record("control_abort", phase="post_stop", message=str(exc))
        print(
            f"停止发送后反馈异常：{exc}。如机器人仍异常运动，请立即使用实体急停。",
            file=sys.stderr, flush=True,
        )
        raise
    return verify_observed_motion(
        anchor, side, delta_positions, delta_quaternions,
        [sample for sample in client.pose_samples()
         if sample.received_at >= start], final_pose,
    )


def main() -> int:
    args = parse_args()
    trace: Optional[TraceRecorder] = None
    try:
        if not math.isfinite(args.feedback_timeout) or not (
            FEEDBACK_TIMEOUT_SECONDS <= args.feedback_timeout <= 3.0
        ):
            raise ValueError("--feedback-timeout 必须在 0.5 至 3 秒之间")
        trajectory = load_tracker(episode_path(args), args.max_gap)
        live_mode = (
            args.execute or args.check_robot or args.record_idle is not None
        )
        if args.direct_replay and not args.execute:
            raise ValueError("--direct-replay 必须配合 --execute")
        if args.execute and args.countdown < 5:
            raise ValueError("--countdown must be at least 5 seconds for motor control")
        if args.record_idle is not None and not 1 <= args.record_idle <= 60:
            raise ValueError("--record-idle must be between 1 and 60 seconds")
        if live_mode and args.arm is None:
            raise ValueError(
                "live robot modes require an explicit --arm left or right"
            )
        side = args.arm or trajectory.hand_side
        if live_mode and args.axis_map is None:
            raise ValueError(
                "live robot modes require an explicit --axis-map; "
                "calibrate tracker axes first"
            )
        args.axis_map = args.axis_map or "x,y,z"
        axis_map = parse_axis_map(args.axis_map)
        delta_positions, delta_quaternions, angles_deg = relative_motion(
            trajectory,
            axis_map,
            args.scale,
            args.orientation_scale,
        )
        max_displacement = float(np.linalg.norm(delta_positions, axis=1).max())
        if max_displacement > args.max_displacement:
            raise ValueError(
                f"scaled tracker displacement {max_displacement:.3f}m exceeds "
                f"--max-displacement {args.max_displacement:.3f}m"
            )
        if float(angles_deg.max()) > args.max_angle:
            raise ValueError(
                f"relative rotation {angles_deg.max():.1f}deg exceeds "
                f"--max-angle {args.max_angle:.1f}deg"
            )
        replay_times, sampled_positions, sampled_quaternions = (
            sample_relative_trajectory(
                trajectory,
                delta_positions,
                delta_quaternions,
                args.rate,
                args.speed,
            )
        )
        print_summary(
            trajectory, side, replay_times, delta_positions, angles_deg, args
        )
        if not live_mode:
            return 0

        if (args.execute or args.record_idle is not None
                or args.trace_file is not None):
            trace = TraceRecorder(args.trace_file or default_trace_path())
            trace.record(
                "plan", episode=str(trajectory.source), side=side,
                axis_map=args.axis_map, scale=args.scale,
                orientation_scale=args.orientation_scale,
                speed=args.speed, rate=args.rate,
                max_displacement_m=max_displacement,
                expected_final_delta_m=sampled_positions[-1].tolist(),
                direct_replay=args.execute,
                feedback_timeout_s=args.feedback_timeout,
                mode=("execute" if args.execute else
                      "record_idle" if args.record_idle is not None else
                      "check_robot"),
            )
            print(f"记录文件: {trace.path}")
        client = Tron2JsonClient(args.robot_ip, trace=trace)
        client.wait_for_identity()
        anchor = client.current_pose(label="anchor")
        anchor_position = (
            anchor.left_position if side == "left" else anchor.right_position
        )
        absolute_targets = anchor_position[None, :] + sampled_positions
        check_workspace(side, absolute_targets)
        # 同时显示两侧腕部位置，便于发现未控制臂的静态偏移。
        print("当前双腕位置 (m):")
        print(f"  left : {anchor.left_position.round(4).tolist()}")
        print(f"  right: {anchor.right_position.round(4).tolist()}")
        print(
            "计划绝对位置范围 (m): min="
            f"{absolute_targets.min(axis=0).round(4).tolist()}, max="
            f"{absolute_targets.max(axis=0).round(4).tolist()}"
        )
        if args.check_robot:
            print("机器人只读预检通过；没有发送 ServoP 或其他运动指令。")
            return 0
        if args.record_idle is not None:
            record_idle(client, anchor, args.record_idle)
            print("只读记录完成；没有发送 ServoP 或其他运动指令。")
            return 0
        verified = execute_replay(
            client,
            anchor,
            side,
            replay_times,
            sampled_positions,
            sampled_quaternions,
            args,
        )
        if args.scale == 0 and args.orientation_scale == 0:
            print("零位移发送流程完成；未验证 ServoP 能驱动机械臂。")
        elif verified:
            print("轨迹指令发送完成；腕部反馈通过幅度和末帧验收。")
        else:
            print("轨迹指令发送完成；运动幅度过小，未能验证机械臂执行。")
        sys.stdout.flush()
        return 0
    except ReplayInterrupted as exc:
        if trace is not None:
            trace.record("interrupted", message=str(exc))
        print(f"已中止: {exc}", file=sys.stderr)
        return 130
    except KeyboardInterrupt:
        if trace is not None:
            trace.record("interrupted", message="KeyboardInterrupt")
        print("已取消；若运动已开始，请用硬件急停处理异常运动。", file=sys.stderr)
        return 130
    except (FileNotFoundError, ValueError, RuntimeError, TimeoutError) as exc:
        if trace is not None:
            trace.record("error", message=str(exc))
        print(f"错误: {exc}", file=sys.stderr)
        return 2
    finally:
        if trace is not None:
            try:
                trace.close()
            except RuntimeError as exc:
                print(f"警告: {exc}", file=sys.stderr)


if __name__ == "__main__":
    code = main()
    # 厂商 Python SDK 在真机连接后的原生资源清理阶段可能崩溃。
    # 仅真机连接模式使用 os._exit；离线预览仍走正常的 Python 退出流程。
    if ("--execute" in sys.argv or "--check-robot" in sys.argv
            or "--record-idle" in sys.argv):
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
    raise SystemExit(code)
