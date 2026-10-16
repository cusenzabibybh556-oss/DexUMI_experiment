#!/usr/bin/env python3
"""从 2026-10-15 记录的双臂固定姿态准备，并回放左腕 Tracker 轨迹。

默认只做离线预检。加 --execute 后，若不在固定起点就倒计时自动 MoveJ；
准备到位后等待输入 REPLAY，再倒计时发送 ServoP 轨迹。
加 --with-o6-hand 时同步回放本 episode 的 O6 左手实测状态反馈；若再指定
--o6-can can1，则由本任务独立启动并关闭回放专用桥接，无需启动手套采集。
未指定 --o6-can 时仍须已有支持回放的 O6 桥接。回放结束后不自动切回
手套；专用桥接关闭后硬件是否继续持位须现场确认。未加 --with-o6-hand 时
仍只回放机械臂。
此配置记录的是机器人 DACH_TRON2A_253 的一次实测姿态，不是通用安全姿态。
MoveJ 与 ServoP 接管仍可能引起意外运动，必须在现场观察并备好实体急停。

示例：
    python3 replay_tron2_fixed_start.py --episode 65
    python3 replay_tron2_fixed_start.py --episode 65 --execute
    python3 replay_tron2_fixed_start.py --episode 65 --with-o6-hand --execute
    python3 replay_tron2_fixed_start.py --episode 65 --with-o6-hand --o6-can can1 --execute
    python3 replay_tron2_fixed_start.py --episode 66 --scale 1.0 --execute
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

from o6_hand_replay import O6HandReplaySession
from o6_replay_bridge import O6ReplayBridge, validate_o6_can
from prepare_tron2_arms import get_joint_state, move_stage
from replay_tron2_tracker import (
    DEFAULT_DATA_ROOT,
    DEFAULT_ROBOT_IP,
    DualArmPose,
    FEEDBACK_TIMEOUT_SECONDS,
    ReplayInterrupted,
    TraceRecorder,
    Tron2JsonClient,
    check_workspace,
    default_trace_path,
    execute_replay,
    load_tracker,
    parse_axis_map,
    quaternion_distance_deg,
    relative_motion,
    sample_relative_trajectory,
)


# 2026-10-15 21:07:57（北京时间）只读查询 DACH_TRON2A_253：
# 连续 4 次反馈确认停稳后，记录左右臂 q、ServoP 腕位姿。
# 保存两个独立的 7 关节目标；旧 prepare_tron2_arms.py 的 READY_ARM_Q 不会被改动。
FIXED_ROBOT_ID = "DACH_TRON2A_253"
FIXED_LEFT_Q = np.array([
    -0.463700, 0.263900, 0.108600, -1.181700,
    -0.015800, 0.146700, -0.023400,
], dtype=np.float64)
FIXED_RIGHT_Q = np.array([
    -0.392900, -0.183300, -0.121800, -1.232200,
    -0.017900, 0.555500, -0.054314,
], dtype=np.float64)
FIXED_LEFT_POSITION = np.array([0.590808, 0.371680, -0.243641])
FIXED_RIGHT_POSITION = np.array([0.547606, -0.345544, -0.353572])
FIXED_LEFT_QUATERNION = np.array([0.722473, 0.148792, -0.674935, 0.018889])
FIXED_RIGHT_QUATERNION = np.array([0.849163, -0.119934, -0.507944, -0.080816])
FIXED_Q = np.concatenate((FIXED_LEFT_Q, FIXED_RIGHT_Q))
# 右臂 MoveJ 阶段必须实际接近保存关节角；0.04 rad 的最终容差会把
# 本次日志中 0.026 rad 的未运动偏差误判为“已到位”。
# 这里采用 0.015 rad，并在下一步独立核验右腕位姿。
RIGHT_PREPARE_JOINT_TOLERANCE_RAD = 0.015


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--episode", type=int, help="episode 编号，如 65")
    source.add_argument("--file", type=Path, help="直接指定 HDF5 数据文件")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--axis-map", default="y,z,x",
                        help="Tracker 到机器人坐标轴的候选映射，默认 y,z,x")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="平移倍率，默认 1.0")
    parser.add_argument("--orientation-scale", type=float, default=0.0,
                        help="相对姿态倍率，默认 0（只回放平移）")
    parser.add_argument("--speed", type=float, default=0.2,
                        help="回放速度倍率，默认 0.2")
    parser.add_argument("--rate", type=float, default=300.0,
                        help="ServoP 发送频率，默认 300 Hz")
    parser.add_argument("--max-gap", type=float, default=0.15)
    parser.add_argument("--max-displacement", type=float, default=0.35,
                        help="相对起点最大位移，默认 0.35 m")
    parser.add_argument("--max-angle", type=float, default=90.0)
    parser.add_argument("--robot-ip", default=DEFAULT_ROBOT_IP)
    parser.add_argument("--move-duration", type=float, default=10.0,
                        help="每只手臂的 MoveJ 时间，默认 10 秒")
    parser.add_argument("--countdown", type=int, default=5,
                        help="自动准备运动和确认回放后的倒计时各 5 秒")
    parser.add_argument("--joint-tolerance", type=float, default=0.04,
                        help="固定关节角验收误差，默认 0.04 rad")
    parser.add_argument("--position-tolerance", type=float, default=0.02,
                        help="固定腕位置验收误差，默认 0.02 m（20 mm）")
    parser.add_argument("--angle-tolerance", type=float, default=3.0,
                        help="固定腕姿态验收误差，默认 3°")
    parser.add_argument("--feedback-timeout", type=float,
                        default=FEEDBACK_TIMEOUT_SECONDS)
    parser.add_argument("--trace-file", type=Path,
                        help="真机日志路径；默认自动创建且不覆盖旧日志")
    parser.add_argument("--with-o6-hand", action="store_true",
                        help="同步回放 O6 左手六通道实测状态；可配合 --o6-can 独立启动桥接")
    parser.add_argument("--o6-can",
                        help="独立启动回放专用 O6 桥接的 CAN 接口，如 can1；不启动手套采集")
    parser.add_argument("--web-control", action="store_true", help=argparse.SUPPRESS)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true",
                      help="若不在固定起点，倒计时后自动 MoveJ；到位后输入 REPLAY 回放")
    mode.add_argument("--check-robot", action="store_true",
                      help="只读查看机器人当前状态与固定起点的距离")
    return parser.parse_args(argv)


def data_path(args: argparse.Namespace) -> Path:
    if args.file is not None:
        return args.file.expanduser().resolve()
    return (args.data_root / f"episode_{args.episode:06d}.hdf5").resolve()


def validate_args(args: argparse.Namespace) -> None:
    if args.web_control and not args.execute:
        raise ValueError("--web-control 仅能与 --execute 一同使用")
    if args.o6_can is not None:
        validate_o6_can(args.o6_can)
        if not args.with_o6_hand:
            raise ValueError("--o6-can 必须与 --with-o6-hand 一同使用")
        if args.check_robot:
            raise ValueError("只读 --check-robot 不会启动 O6 桥接；请移除 --o6-can")
    for name in ("scale", "orientation_scale", "speed", "rate", "max_gap",
                 "max_displacement", "max_angle", "move_duration",
                 "joint_tolerance", "position_tolerance", "angle_tolerance",
                 "feedback_timeout"):
        if not math.isfinite(getattr(args, name)):
            raise ValueError(f"--{name.replace('_', '-')} 必须是有限数值")
    if args.episode is not None and args.episode < 0:
        raise ValueError("--episode 不得为负数")
    if args.scale < 0 or not 0 <= args.orientation_scale <= 1:
        raise ValueError("--scale 不得为负；--orientation-scale 必须在 0 至 1 之间")
    if min(args.speed, args.rate, args.max_gap, args.max_displacement) <= 0:
        raise ValueError("速度、频率、最大数据间隔与最大位移必须大于零")
    if args.max_angle < 0:
        raise ValueError("--max-angle 不得为负")
    if args.move_duration < 5 or args.countdown < 5:
        raise ValueError("MoveJ 时间和倒计时均不得短于 5 秒")
    if not 0.005 <= args.joint_tolerance <= 0.1:
        raise ValueError("--joint-tolerance 必须在 0.005 至 0.1 rad 之间")
    if not 0.001 <= args.position_tolerance <= 0.02:
        raise ValueError("--position-tolerance 必须在 0.001 至 0.02 m 之间")
    if not 0.1 <= args.angle_tolerance <= 5.0:
        raise ValueError("--angle-tolerance 必须在 0.1 至 5° 之间")
    if not FEEDBACK_TIMEOUT_SECONDS <= args.feedback_timeout <= 3.0:
        raise ValueError("--feedback-timeout 必须在 0.5 至 3 秒之间")
    parse_axis_map(args.axis_map)


def check_fixed_arm_pose(pose, side: str, position_tolerance: float,
                         angle_tolerance: float) -> None:
    wanted_position = FIXED_LEFT_POSITION if side == "left" else FIXED_RIGHT_POSITION
    wanted_quaternion = (
        FIXED_LEFT_QUATERNION if side == "left" else FIXED_RIGHT_QUATERNION
    )
    actual_position = getattr(pose, f"{side}_position")
    actual_quaternion = getattr(pose, f"{side}_quaternion")
    position_error = float(np.linalg.norm(actual_position - wanted_position))
    angle_error = quaternion_distance_deg(actual_quaternion, wanted_quaternion)
    print(f"  {side}: 固定起点误差 {position_error * 1000:.1f} mm / "
          f"{angle_error:.2f}°")
    if position_error > position_tolerance or angle_error > angle_tolerance:
        raise RuntimeError(
            f"{side} 未到保存的固定起点；误差 "
            f"{position_error * 1000:.1f} mm / {angle_error:.2f}°"
        )


def check_fixed_pose(pose, position_tolerance: float,
                     angle_tolerance: float) -> None:
    """固定起点以保存的双腕位姿为准，不从本次漂移位置静默重锚定。"""
    for side in ("left", "right"):
        check_fixed_arm_pose(pose, side, position_tolerance, angle_tolerance)


def already_at_fixed_start(q: np.ndarray, pose, args: argparse.Namespace) -> bool:
    if float(np.max(np.abs(q[:14] - FIXED_Q))) > args.joint_tolerance:
        return False
    try:
        check_fixed_pose(pose, args.position_tolerance, args.angle_tolerance)
    except RuntimeError:
        return False
    return True


def countdown(seconds: int) -> None:
    for remaining in range(seconds, 0, -1):
        print(f"\r{remaining:2d}s 后开始 MoveJ；请远离运动范围并备好实体急停...",
              end="", flush=True)
        time.sleep(1.0)
    print("\r开始固定起点准备。请持续观察机器人。                     ")


def web_event(args: argparse.Namespace, phase: str) -> None:
    if args.web_control:
        print("ROBOT_REPLAY_EVENT " + json.dumps({"phase": phase}), flush=True)


def acquire_replay_lock():
    """固定起点恢复和回放共用的进程互斥锁。"""
    lock = Path(__file__).with_name(".tron2_replay.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock.close()
        raise RuntimeError("已有固定起点恢复或回放进程，拒绝并发控制机器人") from exc
    return lock


def restore_fixed_start(client: Tron2JsonClient, q: np.ndarray, dq: np.ndarray,
                        pose: DualArmPose, args: argparse.Namespace,
                        trace: TraceRecorder | None = None) -> DualArmPose:
    """将双臂恢复到保存的关节目标并验证腕位姿；只负责准备，不启动回放。"""
    if float(np.max(np.abs(dq[:14]))) > 0.05:
        raise RuntimeError("双臂关节仍在运动，不开始准备阶段")

    if already_at_fixed_start(q, pose, args):
        print("双臂已在保存的固定起点附近，跳过两次 MoveJ。")
        if trace is not None:
            trace.record("prepare_skipped", reason="already_at_fixed_start")
    else:
        print("已指定 --execute：倒计时后自动准备固定起点，"
              "先右臂 MoveJ，再左臂 MoveJ。准备前不再要求输入。")
        print("请立即清空双臂运动范围并备好实体急停；"
              "MoveJ 期间 Ctrl+C 不保证机器人停止。")
        countdown(args.countdown)

        first_target = q[:14].copy()
        first_target[7:14] = FIXED_RIGHT_Q
        if trace is not None:
            trace.record("prepare_stage", side="right", target_q=first_target.tolist())
        move_stage(client, "固定准备 1/2", "right", first_target, slice(7, 14),
                   args.move_duration, RIGHT_PREPARE_JOINT_TOLERANCE_RAD)

        after_right_q, after_right_dq = client.joint_snapshot(label="after_right")
        if float(np.max(np.abs(after_right_dq[:14]))) > 0.05:
            raise RuntimeError("右臂准备后双臂仍在运动，不开始左臂准备")
        right_pose = client.current_servo_pose(label="after_right")
        print("右臂 MoveJ 后与保存的固定起点的腕位姿误差：")
        check_fixed_arm_pose(right_pose, "right", args.position_tolerance,
                             args.angle_tolerance)
        second_target = after_right_q[:14].copy()
        second_target[:7] = FIXED_LEFT_Q
        # MoveJ 消息包含双臂 14 个目标；右臂必须保持保存的固定起点，
        # 不能把一次未到位的实测关节角作为第二阶段的新目标。
        second_target[7:14] = FIXED_RIGHT_Q
        if trace is not None:
            trace.record("prepare_stage", side="left", target_q=second_target.tolist())
        move_stage(client, "固定准备 2/2", "left", second_target, slice(0, 7),
                   args.move_duration, args.joint_tolerance)

    final_q, final_dq = client.joint_snapshot(label="after_prepare")
    joint_error = float(np.max(np.abs(final_q[:14] - FIXED_Q)))
    if joint_error > args.joint_tolerance:
        raise RuntimeError(
            f"固定起点关节最大误差 {joint_error:.4f} rad 超过 "
            f"{args.joint_tolerance:.4f} rad"
        )
    if float(np.max(np.abs(final_dq[:14]))) > 0.05:
        raise RuntimeError("准备后双臂关节尚未停稳")
    anchor = client.current_servo_pose(label="fixed_start")
    print("准备后与保存的固定起点的双腕误差：")
    check_fixed_pose(anchor, args.position_tolerance, args.angle_tolerance)
    if trace is not None:
        trace.record("fixed_start_verified", joint_max_error_rad=joint_error)
    return anchor


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    trace = None
    hand_replay = None
    o6_bridge = None
    replay_lock = None
    try:
        validate_args(args)
        if args.execute:
            # CLI 与网页使用同一把进程锁，防止两个入口同时发真机指令。
            replay_lock = acquire_replay_lock()
        trajectory = load_tracker(data_path(args), args.max_gap)
        if trajectory.hand_side != "left":
            raise ValueError("固定起点脚本只回放左腕；该 episode 的 Tracker 手侧不是 left")
        axis_map = parse_axis_map(args.axis_map)
        delta_positions, delta_quaternions, angles = relative_motion(
            trajectory, axis_map, args.scale, args.orientation_scale
        )
        max_displacement = float(np.linalg.norm(delta_positions, axis=1).max())
        if max_displacement > args.max_displacement:
            raise ValueError(
                f"轨迹最大位移 {max_displacement:.4f} m 超过上限 "
                f"{args.max_displacement:.4f} m"
            )
        if float(angles.max()) > args.max_angle:
            raise ValueError("轨迹相对姿态变化超过 --max-angle")
        replay_times, sampled_positions, sampled_quaternions = (
            sample_relative_trajectory(
                trajectory, delta_positions, delta_quaternions,
                args.rate, args.speed,
            )
        )
        if args.with_o6_hand:
            hand_replay = O6HandReplaySession(
                trajectory.source, args.speed, trajectory.origin_timestamp_ns
            )
        targets = FIXED_LEFT_POSITION[None, :] + sampled_positions
        check_workspace("left", targets)
        check_workspace("right", FIXED_RIGHT_POSITION[None, :])
        print(f"数据文件: {trajectory.source}")
        print(f"目标机器人: {FIXED_ROBOT_ID}; 固定起点: 2026-10-15 实测双臂姿态")
        print(f"left 固定腕位置 (m): {FIXED_LEFT_POSITION.tolist()}")
        print(f"right 固定腕位置 (m): {FIXED_RIGHT_POSITION.tolist()}")
        print(f"计划: {len(replay_times)} 帧 @ {args.rate:g} Hz, "
              f"{replay_times[-1]:.3f} s, scale={args.scale:g}, "
              f"speed={args.speed:g}, axis-map={args.axis_map}")
        print(f"轨迹最大相对位移: {max_displacement:.4f} m")
        print("左腕目标范围 (m): "
              f"min={targets.min(axis=0).round(4).tolist()}, "
              f"max={targets.max(axis=0).round(4).tolist()}")
        print("注意：工作空间包围盒检查不等于关节可达性或碰撞检查。")
        if hand_replay is not None:
            print(f"O6 左手实测状态: {len(hand_replay.states)} 帧，"
                  f"最后一帧时间 {hand_replay.due_ns[-1] / 1e9:.3f}s；"
                  "与 Tracker 使用同一回放时钟。")
        if not args.execute and not args.check_robot:
            print("离线预检完成；未连接机器人、未发送运动指令。")
            return 0

        if args.execute:
            trace = TraceRecorder(args.trace_file or default_trace_path())
            if hand_replay is not None:
                hand_replay.trace = trace
            trace.record(
                "fixed_start_plan", episode=str(trajectory.source),
                robot_id=FIXED_ROBOT_ID, fixed_q=FIXED_Q.tolist(),
                fixed_left_position=FIXED_LEFT_POSITION.tolist(),
                fixed_right_position=FIXED_RIGHT_POSITION.tolist(),
                scale=args.scale, axis_map=args.axis_map,
                speed=args.speed, with_o6_hand=bool(hand_replay),
                o6_can=args.o6_can,
                expected_final_delta_m=sampled_positions[-1].tolist(),
            )
            print(f"记录文件: {trace.path}")
            if hand_replay is not None:
                if args.o6_can is not None:
                    print(f"正在检查 {args.o6_can} 并启动回放专用 O6 桥接；"
                          "尚未连接 TRON2 或发送轨迹。")
                    o6_bridge = O6ReplayBridge(
                        args.o6_can, trace.path.with_suffix(".o6_bridge.log"), trace
                    )
                    o6_bridge.start(hand_replay)
                else:
                    hand_replay.probe_bridge()
                    print("O6 回放桥接服务已发现；尚未接管手指或发送运动指令。")
        client = Tron2JsonClient(args.robot_ip, trace=trace)
        client.wait_for_identity()
        if client.accid != FIXED_ROBOT_ID:
            raise RuntimeError(
                f"连接的是 {client.accid}，固定姿态只针对 {FIXED_ROBOT_ID}；"
                "拒绝移动"
            )
        q, dq = client.joint_snapshot(label="before_prepare")
        pose = client.current_servo_pose(label="before_prepare")
        print("当前关节角 (rad):")
        print(f"  left : {q[:7].round(4).tolist()}")
        print(f"  right: {q[7:14].round(4).tolist()}")
        print("当前腕位置 (m):")
        print(f"  left : {pose.left_position.round(4).tolist()}")
        print(f"  right: {pose.right_position.round(4).tolist()}")
        if args.check_robot:
            print("当前与固定起点的双腕误差：")
            try:
                check_fixed_pose(pose, args.position_tolerance,
                                 args.angle_tolerance)
            except RuntimeError as exc:
                print(f"提示：{exc}")
            print("只读预检结束；没有发送 MoveJ 或 ServoP。")
            return 0
        anchor = restore_fixed_start(client, q, dq, pose, args, trace)
        print("固定起点已验证。回放将再要求输入 REPLAY 并倒计时；"
              "ServoP 接管仍可能使双臂跳变。")
        web_event(args, "awaiting_replay")
        verified = execute_replay(
            client, anchor, "left", replay_times,
            sampled_positions, sampled_quaternions, args,
            confirmation_word="REPLAY",
            hand_replay=hand_replay,
        )
        if hand_replay is not None:
            hand_replay.close()
            hand_replay = None
        if verified:
            print("轨迹发送完成，现有实机反馈验收通过。")
        else:
            print("轨迹发送完成，但反馈不足以证明机器人正确跟踪。")
        web_event(args, "completed")
        return 0
    except ReplayInterrupted as exc:
        if trace is not None:
            trace.record("interrupted", message=str(exc))
        print(f"已中止: {exc}", file=sys.stderr)
        return 130
    except KeyboardInterrupt:
        if trace is not None:
            trace.record("interrupted", message="KeyboardInterrupt")
        print("已取消后续指令；这不是硬件急停。", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError, TimeoutError) as exc:
        if trace is not None:
            trace.record("error", message=str(exc))
        print(f"错误: {exc}", file=sys.stderr)
        return 2
    finally:
        if hand_replay is not None:
            try:
                hand_replay.close()
            except RuntimeError as exc:
                print(f"O6 回放收尾异常：{exc}", file=sys.stderr)
        if o6_bridge is not None:
            o6_bridge.close()
        if trace is not None:
            try:
                trace.close()
            except RuntimeError as exc:
                print(f"警告: {exc}", file=sys.stderr)
        if replay_lock is not None:
            replay_lock.close()


if __name__ == "__main__":
    code = main()
    if "--execute" in sys.argv or "--check-robot" in sys.argv:
        # 与原脚本一致：避免厂商 SDK 在真机连接后的原生析构崩溃。
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
    raise SystemExit(code)
