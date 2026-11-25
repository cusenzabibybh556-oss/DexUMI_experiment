#!/usr/bin/env python3
"""将一段 TransHandR VIVE Tracker 轨迹用于 TRON2 腕部回放。

本程序以首个有效帧为参考，把相对运动叠加到机器人当前腕部位姿。
--tracker-calibration 使用标定文件的 R_BW 转换相对平移和旋转，并按文件约定
解释 Tracker 四元数；旧真人腕部数据不套用机器人法兰的 T_ET 安装平移。
采集地图与标定地图的坐标轴须一致。未指定标定文件时保留 --axis-map 旧行为。
默认只做离线预览，不连接机器人。

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

# MoveP 文档给出的工作空间，单位：米。运动臂暂保守复用，不能据此
# 宣称已确认当前固件的 ServoP 可达边界。左臂回放时，右腕仅保持启动
# 实测位姿，不用此范围验收右腕，也不自动移动右臂以满足该范围。
WORKSPACE = {
    "left": np.array([[0.250, 0.732], [-0.213, 0.900], [-0.673, 0.500]]),
    "right": np.array([[0.250, 0.732], [-0.900, 0.213], [-0.673, 0.500]]),
}

# 回放、训练数据转换及部署应共用这套动作语义。这里约定的是解码后
# 送给机器人控制器的目标，不规定 ACT 内部必须使用何种旋转表示。
SERVOP_ACTION_CONTRACT = {
    "version": 1,
    "arm_control": "servop",
    "position_frame": "robot_base",
    "position_unit": "m",
    "orientation_order": "wxyz",
    "pose_reference": "robot_sdk_end_effector",
    "target_type": "absolute_wrist_pose",
    "tracker_reference": "first_valid_frame",
    "translation_reference": "startup_wrist_pose",
    "rotation_composition": "delta_rotation_times_anchor_rotation",
    "held_arm_target": "startup_measured_pose",
    "head_control": "not_commanded",
    "o6_source": "dexterous_hand/joint_positions",
}

# 软件限差只能在收到反馈之后停止后续指令，不能阻止首条指令已造成的运动。
POST_STOP_OBSERVE_SECONDS = 0.5  # 秒：停止发送指令后，只读观察双腕反馈的时长。
FEEDBACK_INTERVAL_SECONDS = 0.05  # 秒：发送 ServoP 时，发起腕位姿反馈查询的最短间隔。
FEEDBACK_TIMEOUT_SECONDS = 0.5  # 秒：腕位姿反馈超时即停止后续指令；这不是硬件急停。
POSE_PROBE_RETRY_SECONDS = 0.15  # 秒：腕位姿查询未获回复时，允许重试的等待时间。
SNAPSHOT_RETRY_INTERVAL_SECONDS = 0.5  # 秒：同步只读快照的重试间隔；总超时仍由单次 request 限定。
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
    quaternion_order: str = "wxyz"  # HDF5 声明的存储顺序；不等于 HID 原始字段已正确解释。

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


def add_tracker_mapping_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--tracker-calibration", type=Path,
        help="标定 JSON：使用 R_BW 转换相对位置/旋转；与 --axis-map 互斥，旧真人数据不套用 T_ET",
    )
    parser.add_argument(
        "--tracker-quaternion", choices=("pyvut-legacy", "wxyz"),
        help="标定模式的输入解释；默认遵循标定约定。pyvut-legacy 纠正旧保存排列，wxyz 表示已正确解码",
    )


def validate_tracker_mapping_arguments(args: argparse.Namespace) -> None:
    calibration = getattr(args, "tracker_calibration", None)
    if calibration is not None and args.axis_map is not None:
        raise ValueError("--tracker-calibration 与 --axis-map 不能同时使用，避免重复转换坐标轴")
    if getattr(args, "tracker_quaternion", None) is not None and calibration is None:
        raise ValueError("--tracker-quaternion 仅用于 --tracker-calibration 模式")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
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
    add_tracker_mapping_arguments(parser)
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
        help="ServoP command output rate (default: 300 Hz)",
    )
    parser.add_argument(
        "--interpolation", choices=("linear", "none"), default="linear",
        help="linear：位置线性/旋转SLERP；none：原帧到期后更新目标，帧间重复上一帧",
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
    parser.add_argument('--z-compensation', action='store_true',
                        help='显式启用有界Z位置修正；不是关节力矩或重力前馈')
    parser.add_argument('--z-adaptation', choices=('stationary', 'continuous'), default='stationary',
                        help='默认仅静止更新；continuous实验模式允许自由空间运动中更新，仍限幅限速')
    parser.add_argument('--z-max-correction-mm', type=float, default=20.,
                        help='Z位置修正上限，5至20mm，默认20')
    parser.add_argument('--settle-handover', action='store_true',
                        help='配合--z-compensation，先恢复接管Z再开始轨迹；不能消除首瞬下沉')
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
    return parser.parse_args(argv)


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
        quaternion_order = f["vive_tracker"].attrs.get("quat_order", "unknown")
        if isinstance(quaternion_order, bytes):
            quaternion_order = quaternion_order.decode("utf-8")
        quaternion_order = "".join(c for c in str(quaternion_order).lower() if c.isalpha())
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
        quaternion_order=quaternion_order,
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


def prepare_tracker_motion(
    trajectory: TrackerTrajectory, args: argparse.Namespace, side: str,
    *, default_axis_map: str, require_explicit_axes: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """离线、只读生成位移与空间相对旋转；首帧始终为零位移/单位旋转。"""
    validate_tracker_mapping_arguments(args)
    if getattr(args, "tracker_calibration", None) is None:
        if require_explicit_axes and args.axis_map is None:
            raise ValueError("live robot modes require an explicit --axis-map or --tracker-calibration")
        args.axis_map = args.axis_map or default_axis_map
        args.tracker_mapping_info = {"mode": "axis-map", "axis_map": args.axis_map}
        return relative_motion(trajectory, parse_axis_map(args.axis_map),
                               args.scale, args.orientation_scale)

    from tracker_replay_mapping import load_calibration, calibrated_relative_motion

    if trajectory.quaternion_order != "wxyz":
        raise ValueError(
            "标定回放要求 HDF5 vive_tracker.quat_order 声明 WXYZ；"
            f"实际为 {trajectory.quaternion_order!r}，不能猜测存储顺序"
        )
    if trajectory.hand_side != side:
        raise ValueError("标定回放的目标手侧与采集数据不一致")
    calibration = load_calibration(args.tracker_calibration, arm=side)
    quaternion_format = args.tracker_quaternion or (
        "pyvut-legacy" if calibration["tracker_quaternion_convention"] == "raw-xyzw" else "wxyz"
    )
    delta = calibrated_relative_motion(
        trajectory, calibration, args.scale, args.orientation_scale,
        mode="tracker-relative", quaternion_format=quaternion_format,
    )
    args.tracker_mapping_info = {
        "mode": "calibrated-tracker-relative",
        "calibration_file": str(Path(args.tracker_calibration).expanduser().resolve()),
        "calibration_sha256": calibration["sha256"],
        "robot_id": calibration.get("robot_id"),
        "R_BW": np.asarray(calibration["T_BW"])[:3, :3].tolist(),
        "input_quaternion_format": quaternion_format,
        "stored_quaternion_order": trajectory.quaternion_order,
        "position_reference": "tracker_origin_displacement_from_first_valid_frame",
        "orientation_reference": "spatial_delta_from_first_valid_frame",
        "anchor": "startup_wrist_pose",
        "T_ET_applied": False,
        "T_BW_translation_applied": False,
        "source_frame_requirement": "recording and calibration world axes must agree",
    }
    print(f"Tracker 标定文件: {args.tracker_mapping_info['calibration_file']}")
    print(f"标定 SHA256: {calibration['sha256']}")
    print(f"Tracker 四元数解释: {quaternion_format}；HDF5 原始数据保持不变。")
    print("映射: 新标定 R_BW 同时转换相对平移与旋转；首帧锚定准备/启动腕位姿。")
    print("采集地图与标定地图的坐标轴须一致；真人腕部旧数据不套用机器人 T_ET 安装平移。")
    return delta


def check_tracker_mapping_robot(args: argparse.Namespace, robot_id: str) -> None:
    expected = getattr(args, "tracker_mapping_info", {}).get("robot_id")
    if expected and expected != robot_id:
        raise RuntimeError(f"标定针对机器人 {expected}，当前连接 {robot_id}，不发送运动")


def sample_relative_trajectory(
    trajectory: TrackerTrajectory,
    delta_positions: np.ndarray,
    delta_quaternions: np.ndarray,
    output_rate: float,
    speed: float,
    interpolation: str = "linear",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """保持发送时钟；none仅选择已到期的原帧，不生成帧间位姿。"""
    if output_rate <= 0 or speed <= 0:
        raise ValueError("--rate and --speed must be positive")
    if interpolation not in ("linear", "none"):
        raise ValueError("--interpolation must be linear or none")
    replay_duration = trajectory.duration_s / speed
    count = max(2, int(math.ceil(replay_duration * output_rate)) + 1)
    replay_times = np.linspace(0.0, replay_duration, count)
    if interpolation == "none":
        # 原帧在到期后的首个发送周期生效，不提前采用下一帧；末帧不受浮点误差影响。
        source_due = trajectory.timestamps_s / speed
        indices = np.searchsorted(source_due, replay_times, side="right") - 1
        indices = np.clip(indices, 0, len(source_due) - 1)
        indices[-1] = len(source_due) - 1
        if len(np.unique(indices)) != len(source_due):
            raise ValueError("不插值模式的发送频率不足以覆盖全部原始Tracker帧；"
                             "请降低 --speed 或提高 --rate，不会静默跳帧")
        return (replay_times, delta_positions[indices].copy(),
                delta_quaternions[indices].copy())
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
        # 同步位姿查询与异步 ServoP 探测分开登记，避免正常快照被误记为未匹配回复。
        self._snapshot_pose_guids: set[str] = set()
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
            if title.lower() in {"notify_servop", "notify_servoj"}:
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
            elif (title == "response_get_servo_pose" and self.trace is not None
                  and message.get("guid") not in getattr(self, "_snapshot_pose_guids", set())):
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
        # 已观察到 ServoJ 发送值与真机电机目标不一致；本回放客户端禁止
        # 再走该入口，不能仅改变默认参数而留下可误触的真机发送分支。
        if title == "request_servoj":
            raise RuntimeError("ServoJ 回放已停用：真机目标限幅异常尚未解决；仅支持 ServoP")
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
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("request timeout must be finite and positive")
        # 只对白名单里的无副作用查询重试。MoveJ/MoveH/ServoP 及其它命令
        # 即使超时也绝不自动重发，否则可能重复触发已经被机器人接受的运动。
        retry_read = not data and {
            "request_get_servo_pose": "response_get_servo_pose",
            "request_get_move_pose": "response_get_move_pose",
            "request_get_joint_state": "response_get_joint_state",
        }.get(title) == response_title
        with self._condition:
            start = len(self._messages)
            # 其他客户端的腕部位姿回复也会被广播，因此按 GUID 匹配；
            # joint_state 若带 GUID 也核对；旧固件不带 GUID 时保留类型匹配。
            correlate_guid = response_title in {
                "response_get_move_pose", "response_get_servo_pose"
            }
            deadline = time.monotonic() + timeout
            if not hasattr(self, "_snapshot_pose_guids"):
                self._snapshot_pose_guids = set()
            guid = None
            attempts = 0
            next_query_at = time.monotonic()
            try:
                while True:
                    now = time.monotonic()
                    if now >= deadline:
                        if retry_read and self.trace is not None:
                            self.trace.record("snapshot_query_timeout", request=title,
                                              attempts=attempts, timeout_s=timeout)
                        raise TimeoutError(f"no {response_title} within {timeout:.1f}s")
                    for message in self._messages[start:]:
                        if message.get("title") != response_title:
                            continue
                        strict_guid = correlate_guid or (retry_read and bool(message.get("guid")))
                        if not strict_guid or message.get("guid") == guid:
                            if retry_read and attempts > 1 and self.trace is not None:
                                self.trace.record("snapshot_query_recovered", request=title,
                                                  attempts=attempts, request_guid=guid)
                            return message
                    start = len(self._messages)
                    if attempts == 0 or (retry_read and now >= next_query_at):
                        if guid is not None:
                            self._snapshot_pose_guids.discard(guid)
                        guid = uuid.uuid4().hex
                        if correlate_guid:
                            self._snapshot_pose_guids.add(guid)
                        attempts += 1
                        if retry_read and self.trace is not None:
                            self.trace.record("snapshot_query_sent", request=title, guid=guid,
                                              attempt=attempts,
                                              remaining_s=max(0., deadline - now))
                        self.send(title, data, guid=guid)
                        next_query_at = time.monotonic() + SNAPSHOT_RETRY_INTERVAL_SECONDS
                        # 回调可能在发送调用内已经到达，先扫描回复再等待。
                        continue
                    remaining = deadline - time.monotonic()
                    if retry_read:
                        remaining = min(remaining, next_query_at - time.monotonic())
                    if remaining > 0:
                        self._condition.wait(remaining)
            finally:
                if guid is not None:
                    self._snapshot_pose_guids.discard(guid)

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
            f"{side} target {index} outside conservative workspace (MoveP reference): "
            f"{positions[index].round(4).tolist()}"
        )


def check_servop_hold_position(side: str, position: np.ndarray) -> None:
    """右保持臂只核对有效坐标；不将 MoveP 参考范围作为左臂执行条件。"""
    if side not in WORKSPACE:
        raise ValueError(f"unknown held arm: {side}")
    position = np.asarray(position, dtype=float)
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError(f"ServoP 保持臂 {side} 的腕位置必须为三个有限坐标")
    if side == "right":
        # 用户仅规划左臂；右腕目标仍来自本次启动读数。这里只取消并未
        # 验证适用于 ServoP 的右腕 MoveP 包围盒检查，不修改真实反馈
        # 偏移/姿态/超时检查，不意味着此保持姿态已通过真机验证。
        return
    try:
        check_workspace(side, position[None, :])
    except ValueError as exc:
        raise ValueError(
            f"ServoP 保持臂 {side} 的当前腕位置未通过保守工作空间预检："
            f"{np.asarray(position).round(4).tolist()} m。"
            "该参考范围取自 MoveP 文档，并不证明 ServoP 必然不支持此姿态；"
            "但当前固件在此位置的保持行为尚未验证，因此不发送轨迹，"
            "不会自动抬臂、归位或切换 ServoJ"
        ) from exc


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
    print(f"Tracker目标插值: {args.interpolation}；原始有效帧 {len(trajectory.timestamps_s)}")
    if args.interpolation == "none":
        print("不生成帧间位姿；原帧按时间戳到期更新，发送周期内重复上一帧。")
        print(f"原帧最大平移目标阶跃: {np.linalg.norm(np.diff(delta_positions, axis=0), axis=1).max()*1000:.2f}mm")
    print(
        "相对平移范围 (m): min="
        f"{delta_positions.min(axis=0).round(4).tolist()}, max="
        f"{delta_positions.max(axis=0).round(4).tolist()}"
    )
    print(
        f"最大位移 {displacement_norms.max():.4f}m; "
        f"最大相对转角 {angles_deg.max():.2f}deg; "
        f"mapping={getattr(args, 'tracker_mapping_info', {}).get('mode', args.axis_map)}"
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
    if getattr(args,'pose_compensation',False):
        from tron2_pose_replay import execute_pose_compensated_replay
        return execute_pose_compensated_replay(client,anchor,side,replay_times,delta_positions,delta_quaternions,args,
            confirmation_word=confirmation_word,hand_replay=hand_replay)
    if getattr(args, 'settle_handover', False) and not getattr(args, 'z_compensation', False):
        raise ValueError('--settle-handover 必须显式配合 --z-compensation')
    if getattr(args, 'z_adaptation', 'stationary') != 'stationary' and not getattr(args, 'z_compensation', False):
        raise ValueError('--z-adaptation continuous 必须显式配合 --z-compensation')
    if getattr(args, 'z_compensation', False):
        from tron2_z_replay import execute_z_compensated_replay
        return execute_z_compensated_replay(
            client, anchor, side, replay_times, delta_positions, delta_quaternions, args,
            confirmation_word=confirmation_word, hand_replay=hand_replay,
        )
    control = getattr(args, "arm_control", "servop")
    if control != "servop":
        raise ValueError("回放仅支持 ServoP；ServoJ 因真机目标限幅异常已停用")
    control_label = "ServoP 双腕轨迹"
    if side == "left":
        anchor_position, anchor_quaternion = anchor.left_position, anchor.left_quaternion
    else:
        anchor_position, anchor_quaternion = anchor.right_position, anchor.right_quaternion

    target_positions = np.asarray(
        [anchor_position + delta for delta in delta_positions], dtype=np.float64
    )
    check_workspace(side, target_positions)
    held_side = "right" if side == "left" else "left"
    # 在用户确认、O6 接管和倒计时前检查，不能到发送首帧时才发现保持臂未验证。
    check_servop_hold_position(held_side, getattr(anchor, f"{held_side}_position"))

    answer = input(
        f"将向 TRON2 发送 {control_label}，回放 Tracker 轨迹"
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
            f"\r{remaining:2d}s 后开始 {control_label}；"
            "请离开电脑并拿好遥控器急停...",
            end="",
            flush=True,
        )
        time.sleep(1.0)
    pre_start_move = client.current_pose(label="pre_start_move")
    check_anchor_stable(anchor, pre_start_move)
    servo_anchor = client.current_servo_pose(label="pre_start_servo")
    check_anchor_stable(pre_start_move, servo_anchor)
    pre_start_q, pre_start_dq = client.joint_snapshot(label="pre_replay")
    if float(np.max(np.abs(pre_start_dq[:14]))) > 0.05:
        raise RuntimeError("双臂关节仍在运动，取消直接回放")
    anchor = servo_anchor
    anchor_position = getattr(anchor, f"{side}_position")
    anchor_quaternion = getattr(anchor, f"{side}_quaternion")
    target_positions = anchor_position[None, :] + delta_positions
    check_workspace(side, target_positions)
    check_servop_hold_position(held_side, getattr(anchor, f"{held_side}_position"))
    if client.trace is not None:
        client.trace.record("replay_contract", **SERVOP_ACTION_CONTRACT,
                            moving_arm=side, held_arm=held_side,
                            tracker_interpolation=getattr(args, "interpolation", "linear"),
                            tracker_mapping=getattr(args, "tracker_mapping_info", None),
                            with_o6_hand=hand_replay is not None,
                            head_start_rad=(pre_start_q[14:16].tolist()
                                            if len(pre_start_q) >= 16 else None),
                            **pose_for_trace(anchor))
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
            raise RuntimeError(f"TRON2 {control.upper()} rejected the trajectory: {failure}")
        moving = np.concatenate((moving_position, moving_quaternion))
        held = np.concatenate((
            getattr(anchor, f"{held_side}_position"),
            getattr(anchor, f"{held_side}_quaternion"),
        ))
        left, right = (moving, held) if side == "left" else (held, moving)
        # 只包含双腕 XYZ+WXYZ；头部既不进入模型动作，也不进入该运动报文。
        client.send("request_servop", servop_data(left, right))
        now = time.monotonic()
        if now >= next_probe_at:
            client.probe_pose()
            next_probe_at = now + FEEDBACK_INTERVAL_SECONDS
        latest = client.latest_pose_sample()
        guard.observe(
            latest, moving_position, moving_quaternion,
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
            start_ns = time.monotonic_ns()
            start = start_ns / 1e9
        if client.trace is not None:
            client.trace.record("replay_start", side=side,
                                start_monotonic_ns=start_ns, arm_control="servop")
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
            f"已停止发送后续 {control.upper()}：{exc}。如机器人仍异常运动，请立即使用实体急停。",
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
            raise RuntimeError(f"TRON2 {control.upper()} rejected the trajectory: {failure}")
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


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    trace: Optional[TraceRecorder] = None
    try:
        validate_tracker_mapping_arguments(args)
        if args.settle_handover and not args.z_compensation:
            raise ValueError('--settle-handover 必须显式配合 --z-compensation')
        if args.z_adaptation != 'stationary' and not args.z_compensation:
            raise ValueError('--z-adaptation continuous 必须显式配合 --z-compensation')
        if (not math.isfinite(args.z_max_correction_mm)
                or not 5 <= args.z_max_correction_mm <= 20):
            raise ValueError('--z-max-correction-mm 必须为5至20之间的有限数值')
        if not args.z_compensation and args.z_max_correction_mm != 20.:
            raise ValueError('修改Z修正上限时须同时指定 --z-compensation')
        if args.z_compensation and (args.rate != 300. or args.arm == 'right'):
            raise ValueError('Z位置修正仅支持300Hz左臂ServoP')
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
        delta_positions, delta_quaternions, angles_deg = prepare_tracker_motion(
            trajectory, args, side, default_axis_map="x,y,z",
            require_explicit_axes=live_mode,
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
                args.interpolation,
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
                tracker_mapping=args.tracker_mapping_info,
                orientation_scale=args.orientation_scale,
                speed=args.speed, rate=args.rate,
                tracker_interpolation=args.interpolation,
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
        check_tracker_mapping_robot(args, client.accid)
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
