#!/usr/bin/env python3
"""仅准备 TRON2 左臂固定起点，并回放左腕 Tracker 轨迹。

默认只做离线轨迹预检。加 --execute 后，已在正式起点时跳过 MoveJ；
若不在起点，必须先验证恢复路径才允许 MoveJ。到位后输入 REPLAY，再倒计时回放。
右臂没有固定起点：准备指令保持启动前实测关节角，不执行右臂归位。
固定使用 ServoP：左腕发送轨迹，右腕保持启动时实测位姿，不发送头部目标。
指定 --tracker-calibration 后，使用标定轴向与四元数约定生成相对轨迹；
真人腕部数据不套用机器人法兰安装平移；不与 --axis-map 叠加。
ServoJ 因真机目标裁剪异常已停用；切回 ServoP 不代表该异常已修复。
右腕不做 MoveP 参考范围验收，不自动抬臂、归位或修改负载；保留反馈有效性、
异常偏移及超时检查。这不代表右臂完全不受控制或其保持行为已通过真机验证。
当前目标为大臂竖直、小臂水平、手与小臂对直且掌心向下的中立腕姿态；
腕 pitch/roll 为零，完整目标 XYZ/朝向共同更新，本轮尚未实测到位。
原 episode8 高度目标、5° 与 10 月 22 日参考独立保留。自动恢复路径
尚未批准；需要恢复时先单独运行分段恢复试验，不从任意位置自动恢复。
加 --with-o6-hand 时同步回放本 episode 的 O6 左手实测状态反馈；若再指定
--o6-can can1，则由本任务独立启动并关闭回放专用桥接，无需启动手套采集。
未指定 --o6-can 时仍须已有支持回放的 O6 桥接。回放结束后不自动切回
手套；专用桥接关闭后硬件是否继续持位须现场确认。未加 --with-o6-hand 时
仍只回放机械臂。
此配置源于机器人 DACH_TRON2A_253 的实测参考，不是通用安全姿态。
MoveJ/ServoP 接管仍可能引起意外运动，必须在现场观察并备好实体急停。

示例：
    python3 replay_tron2_fixed_start.py --episode 65
    python3 replay_tron2_fixed_start.py --episode 65 --check-robot
    python3 replay_tron2_fixed_start.py --episode 65 --scale 0.1 --max-displacement 0.04
    python3 replay_tron2_fixed_start.py --episode 65 --with-o6-hand --o6-can can1
执行模板和尚未完成的真机条件见 tron2_tracker_replay_commands.md。
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
from prepare_tron2_arms import move_stage
from replay_tron2_tracker import (
    DEFAULT_DATA_ROOT,
    DEFAULT_ROBOT_IP,
    DualArmPose,
    FEEDBACK_TIMEOUT_SECONDS,
    ReplayInterrupted,
    SERVOP_ACTION_CONTRACT,
    TraceRecorder,
    Tron2JsonClient,
    check_workspace,
    check_servop_hold_position,
    default_trace_path,
    execute_replay,
    add_tracker_mapping_arguments,
    validate_tracker_mapping_arguments,
    prepare_tracker_motion,
    check_tracker_mapping_robot,
    load_tracker,
    parse_axis_map,
    quaternion_distance_deg,
    sample_relative_trajectory,
)


# 2026-10-22 20:51:19（北京时间）只读查询 DACH_TRON2A_253：
# 六次采样均停稳，头部真实画面和用户确认掌心向下、四指大致向前。
# 腕位姿取 request_get_servo_pose；关节角取 request_get_joint_state。
# 原始记录：camera_alignment_results/top_20261022_204900/left_start_20261022_205119.json
# 原始参考保持不变。2026-10-30 恢复水平小臂、中立腕的完整几何约束，
# 不再固定旧 episode8 高度目标的 XYZ/朝向。中立指手与小臂对直且掌心朝下；
# 按模型轴定义腕 yaw=-90°、pitch/roll=0°。本轮规划不继承历史到位验收。
# ServoP运行时不做本机IK；恢复与回放共用以下同一组目标。
FIXED_ROBOT_ID = "DACH_TRON2A_253"
FIXED_LEFT_REFERENCE_LABEL = "2026-10-22 20:51:19 O6 左臂掌心向下实测参考"
FIXED_LEFT_REFERENCE_Q = np.array([
    -0.139100090, 0.183299780, -0.068099856, -1.405000091,
    -1.249800086, -0.204900086, 0.193499804,
], dtype=np.float64)
FIXED_LEFT_REFERENCE_POSITION = np.array([0.497985154, 0.186801851, -0.266951889])
FIXED_LEFT_REFERENCE_QUATERNION = np.array([
    -0.438100371, -0.444070641, 0.627195482, 0.466363761,
])
FIXED_LEFT_START_PROFILE = Path(__file__).parent / (
    "validation_results/neutral_start_20261030/fixed_start_profile.json"
)
FIXED_LEFT_START_LABEL = "2026-10-30 水平小臂、中立腕、掌心朝下准备目标（待本轮实测）"
FIXED_LEFT_START_VALIDATED = True  # 用户要求的完整几何目标已核对，非本轮实测到位。
FIXED_LEFT_START_MEASURED = False  # 本轮尚未发送动作，历史同几何目标不替代本轮实测。
FIXED_LEFT_PREPARE_PATH_VALIDATED = False  # 仅阻止尚未验证的 MoveJ 恢复，不拦截已到位分支。
FIXED_LEFT_Q = np.array([
    0.0, 0.0, 0.18322281083347552, -1.5707963267948966,
    -1.5707963267948966, 0.0, 0.0,
], dtype=np.float64)
FIXED_LEFT_POSITION = np.array([0.45078943693329554, 0.2533290247458302, -0.3058899999999999])
FIXED_LEFT_QUATERNION = np.array([
    0.5436449623382834, 0.5436449623382833, -0.4521616468965567, -0.4521616468965567,
])
FIXED_LEFT_POSITION_OFFSET_M = FIXED_LEFT_POSITION - FIXED_LEFT_REFERENCE_POSITION
FIXED_LEFT_HEIGHT_OFFSET_M = float(FIXED_LEFT_POSITION_OFFSET_M[2])  # 仅为新目标相对旧参考的Z差，不代表纯Z平移。


def require_validated_left_start() -> None:
    """连接前检查静态起点；是否需要准备运动须读取当前位置后判断。"""
    if not FIXED_LEFT_START_VALIDATED:
        raise RuntimeError(
            "左臂固定起点尚未验证 O6 掌心朝下、四指向前及准备路径；"
            "当前历史值仅供离线/只读核对，不会发送准备或回放运动。"
            "请先确认新的左臂起点，再启用真机执行"
        )


def require_validated_prepare_path() -> None:
    """仅在确实需要 MoveJ 时调用；已在起点不能被不存在的恢复运动拦截。"""
    if not FIXED_LEFT_PREPARE_PATH_VALIDATED:
        raise RuntimeError(
            "当前左臂不在正式准备起点，且自动恢复路径尚未验证；"
            "未发送 MoveJ。请先手动恢复到已记录起点，或完成恢复路径验证"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--episode", type=int, help="episode 编号，如 65")
    source.add_argument("--file", type=Path, help="直接指定 HDF5 数据文件")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--axis-map", default=None,
                        help="旧候选换轴；未指定标定文件时默认 y,z,x，与 --tracker-calibration 互斥")
    add_tracker_mapping_arguments(parser)
    parser.add_argument("--scale", type=float, default=1.0,
                        help="平移倍率，默认 1.0")
    parser.add_argument("--orientation-scale", type=float, default=0.0,
                        help="相对姿态倍率，默认 0（只回放平移）")
    parser.add_argument("--speed", type=float, default=0.2,
                        help="回放速度倍率，默认 0.2")
    parser.add_argument("--rate", type=float, default=300.0,
                        help="机械臂发送频率，默认 300 Hz")
    parser.add_argument("--interpolation", choices=("linear", "none"), default="linear",
                        help="linear：位置线性/旋转SLERP；none：原帧到期后更新目标，帧间重复上一帧")
    parser.add_argument("--arm-control", choices=("servop",), default="servop",
                        help="固定使用 ServoP；右腕保持实测位姿，不发送头部目标；ServoJ 已停用")
    parser.add_argument("--max-gap", type=float, default=0.15)
    parser.add_argument("--max-displacement", type=float, default=0.35,
                        help="相对起点最大位移，默认 0.35 m")
    parser.add_argument("--max-angle", type=float, default=90.0)
    parser.add_argument("--robot-ip", default=DEFAULT_ROBOT_IP)
    parser.add_argument("--move-duration", type=float, default=10.0,
                        help="左臂准备 MoveJ 时间，默认 10 秒；右臂不归位")
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
    parser.add_argument("--z-compensation", action="store_true",
                        help="有界Z位置修正，默认直接回放；可配合--settle-handover先恢复接管Z，不是力控制")
    parser.add_argument('--z-adaptation', choices=('stationary', 'continuous'), default='stationary',
                        help='Z学习策略：默认stationary仅静止更新；continuous为实验自由空间连续反馈，仍限20mm和1mm/s')
    parser.add_argument('--settle-handover', action='store_true',
                        help='配合--z-compensation，接管Z恢复后再开始轨迹；不是力矩连续修复')
    parser.add_argument("--z-max-correction-mm", type=float, default=20.,
                        help="Z位置修正硬上限，5–20mm，默认20；不会修改原始轨迹或标定")
    parser.add_argument("--trace-file", type=Path,
                        help="真机日志路径；默认自动创建且不覆盖旧日志")
    parser.add_argument("--with-o6-hand", action="store_true",
                        help="同步回放 O6 左手六通道实测状态；可配合 --o6-can 独立启动桥接")
    parser.add_argument("--o6-can",
                        help="独立启动回放专用 O6 桥接的 CAN 接口，如 can1；不启动手套采集")
    parser.add_argument("--web-control", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--ik-report", type=Path,
                        help="--plan-ik 的 JSON 报告路径；同时保存同名 NPZ 完整离线轨迹，不覆盖旧文件")
    parser.add_argument("--ik-min-margin-deg", type=float, default=0.0,
                        help="离线 IK 候选至少保留的关节限位余量，默认 0°；择优仍最大化整段余量")
    parser.add_argument("--ik-max-joint-speed", type=float, default=0.5,
                        help="离线 IK 连续性检查速度上限，默认 0.5 rad/s；不是厂家速度限制")
    from tron2_pose_replay import add_pose_arguments
    add_pose_arguments(parser)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan-ik", action="store_true",
                      help="离线比较整段 IK 候选，优先选关节限位余量最大的路径；不连接或控制真机")
    mode.add_argument("--execute", action="store_true",
                      help="若不在固定起点，倒计时后自动 MoveJ；到位后输入 REPLAY 回放")
    mode.add_argument("--check-robot", action="store_true",
                      help="只读查看机器人状态与左臂固定起点的距离")
    return parser.parse_args(argv)


def data_path(args: argparse.Namespace) -> Path:
    if args.file is not None:
        return args.file.expanduser().resolve()
    return (args.data_root / f"episode_{args.episode:06d}.hdf5").resolve()


def validate_args(args: argparse.Namespace) -> None:
    validate_tracker_mapping_arguments(args)
    from tron2_pose_replay import validate_pose_options
    validate_pose_options(args)
    if getattr(args, 'settle_handover', False) and not args.z_compensation:
        raise ValueError('--settle-handover 必须显式配合 --z-compensation')
    if getattr(args, 'z_adaptation', 'stationary') != 'stationary' and not args.z_compensation:
        raise ValueError('--z-adaptation continuous 必须显式配合 --z-compensation')
    if args.arm_control != "servop":
        raise ValueError("固定起点回放仅支持 ServoP；ServoJ 已停用")
    if (not math.isfinite(args.z_max_correction_mm)
            or not 5. <= args.z_max_correction_mm <= 20.):
        raise ValueError("--z-max-correction-mm 必须为5至20之间的有限数值")
    if not args.z_compensation and args.z_max_correction_mm != 20.:
        raise ValueError("修改Z修正上限时须同时指定 --z-compensation")
    if args.z_compensation and (args.rate != 300. or args.plan_ik):
        raise ValueError("Z补偿直接回放使用300Hz；--plan-ik 为独立离线工具，不与Z补偿叠加")
    if args.web_control and not args.execute:
        raise ValueError("--web-control 仅能与 --execute 一同使用")
    if args.ik_report is not None and not args.plan_ik:
        raise ValueError("--ik-report 必须与 --plan-ik 一同使用")
    if (not math.isfinite(args.ik_min_margin_deg) or args.ik_min_margin_deg < 0
            or not math.isfinite(args.ik_max_joint_speed) or args.ik_max_joint_speed <= 0):
        raise ValueError("离线 IK 余量须为非负数，速度上限须为正数，且均须有限")
    if not args.plan_ik and (args.ik_min_margin_deg != 0 or args.ik_max_joint_speed != .5):
        raise ValueError("--ik-min-margin-deg / --ik-max-joint-speed 只用于 --plan-ik，不会改变 ServoP 固件")
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
    if args.axis_map is not None:
        parse_axis_map(args.axis_map)


def check_fixed_arm_pose(pose, side: str, position_tolerance: float,
                         angle_tolerance: float) -> None:
    if side != "left":
        raise ValueError("只有左臂有固定起点；右臂不得恢复历史目标")
    wanted_position = FIXED_LEFT_POSITION
    wanted_quaternion = FIXED_LEFT_QUATERNION
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
    """只核验保存的左腕目标；右腕不参与历史固定起点验收。"""
    check_fixed_arm_pose(pose, "left", position_tolerance, angle_tolerance)


def already_at_fixed_start(q: np.ndarray, pose, args: argparse.Namespace) -> bool:
    if float(np.max(np.abs(q[:7] - FIXED_LEFT_Q))) > args.joint_tolerance:
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


def check_right_hold(q: np.ndarray, pose: DualArmPose,
                     held_q: np.ndarray, held_pose: DualArmPose,
                     args: argparse.Namespace) -> float:
    """只观察右臂相对本次起点的偏移；超限时不发送右臂纠正或归位指令。"""
    joint_error = float(np.max(np.abs(q[7:14] - held_q)))
    position_error = float(np.linalg.norm(
        pose.right_position - held_pose.right_position
    ))
    angle_error = quaternion_distance_deg(
        pose.right_quaternion, held_pose.right_quaternion
    )
    if (joint_error > args.joint_tolerance
            or position_error > args.position_tolerance
            or angle_error > args.angle_tolerance):
        raise RuntimeError(
            f"准备期间右臂偏离本次保持起点：关节 {joint_error:.4f} rad，"
            f"腕位移 {position_error * 1000:.1f} mm / {angle_error:.2f}°；"
            "不再发送后续运动指令，也不自动纠正右臂"
        )
    return joint_error


def restore_fixed_start(client: Tron2JsonClient, q: np.ndarray, dq: np.ndarray,
                        pose: DualArmPose, args: argparse.Namespace,
                        trace: TraceRecorder | None = None) -> DualArmPose:
    """只恢复左臂；MoveJ 的右臂部分保持本次实测关节角，不启动回放。"""
    require_validated_left_start()
    if float(np.max(np.abs(dq[:14]))) > 0.05:
        raise RuntimeError("双臂关节仍在运动，不开始准备阶段")
    held_right_q = q[7:14].copy()

    if already_at_fixed_start(q, pose, args):
        print("左臂已在保存的固定起点附近，跳过 MoveJ；右臂不归位。")
        if trace is not None:
            trace.record("prepare_skipped", reason="already_at_fixed_start")
    else:
        require_validated_prepare_path()
        print("已指定 --execute：倒计时后自动准备固定起点，"
              "仅左臂 MoveJ；右臂保持启动前实测关节角，不归位。")
        print("请立即清空双臂运动范围并备好实体急停；"
              "MoveJ 期间 Ctrl+C 不保证机器人停止。")
        countdown(args.countdown)

        # 倒计时后重新只读确认，防止发送基于过期右臂位置的保持目标。
        current_q, current_dq = client.joint_snapshot(label="pre_left_prepare")
        current_pose = client.current_servo_pose(label="pre_left_prepare")
        if float(np.max(np.abs(current_dq[:14]))) > 0.05:
            raise RuntimeError("准备倒计时后双臂仍在运动，不发送 MoveJ")
        check_right_hold(current_q, current_pose, held_right_q, pose, args)
        target = current_q[:14].copy()
        target[:7] = FIXED_LEFT_Q
        # 厂商 MoveJ 需要 14 个关节目标；保留本次右臂读数不是硬件锁死。
        target[7:14] = held_right_q
        if trace is not None:
            trace.record("prepare_stage", side="left", target_q=target.tolist(),
                         right_hold_q=held_right_q.tolist())
        move_stage(client, "左臂固定准备", "left", target, slice(0, 7),
                   args.move_duration, args.joint_tolerance)

    final_q, final_dq = client.joint_snapshot(label="after_prepare")
    joint_error = float(np.max(np.abs(final_q[:7] - FIXED_LEFT_Q)))
    if joint_error > args.joint_tolerance:
        raise RuntimeError(
            f"左臂固定起点关节最大误差 {joint_error:.4f} rad 超过 "
            f"{args.joint_tolerance:.4f} rad"
        )
    if float(np.max(np.abs(final_dq[:14]))) > 0.05:
        raise RuntimeError("准备后双臂关节尚未停稳")
    anchor = client.current_servo_pose(label="fixed_start")
    print("准备后与保存的左臂固定起点的误差：")
    check_fixed_pose(anchor, args.position_tolerance, args.angle_tolerance)
    right_error = check_right_hold(final_q, anchor, held_right_q, pose, args)
    if trace is not None:
        trace.record("fixed_start_verified", joint_max_error_rad=joint_error,
                     right_hold_joint_error_rad=right_error, preparation_arm="left")
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
        delta_positions, delta_quaternions, angles = prepare_tracker_motion(
            trajectory, args, "left", default_axis_map="y,z,x"
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
                args.interpolation,
            )
        )
        if args.with_o6_hand:
            hand_replay = O6HandReplaySession(
                trajectory.source, args.speed, trajectory.origin_timestamp_ns
            )
        targets = FIXED_LEFT_POSITION[None, :] + sampled_positions
        check_workspace("left", targets)
        if args.pose_compensation:
            from tron2_pose_reference import VERSION
            print(f'将使用最新实验控制器：{VERSION}；不是旧连续Z版本。')
        if args.z_compensation:
            print(f'Z学习策略：{args.z_adaptation}；连续模式仅用于已确认无接触的自由空间。')
            upper_targets = targets.copy()
            upper_targets[:, 2] += args.z_max_correction_mm/1000.
            check_workspace('left', upper_targets)
        print(f"数据文件: {trajectory.source}")
        print(f"目标机器人: {FIXED_ROBOT_ID}; 固定起点: {FIXED_LEFT_START_LABEL}")
        print(f"left 固定腕位置 (m): {FIXED_LEFT_POSITION.tolist()}")
        if not FIXED_LEFT_START_MEASURED:
            print("当前准备目标尚未实测，需要先单独恢复试验及现场核对。")
        print("右臂不自动归位；准备时保持本次实测关节角。")
        print("回放控制：ServoP；左腕目标变化，右腕保持本次启动实测位姿；不发送头部目标。")
        print("右腕不做 MoveP 工作空间验收；仍检查反馈有效性、异常偏移及超时。")
        if args.z_compensation:
            if args.settle_handover:
                print(f"接管Z恢复：修正上限{args.z_max_correction_mm:g}mm；Z误差连续1秒不超过1mm后才启动轨迹，最多30秒。")
                print("这是有界位置补偿，不能修复底层力矩切换或阻止首瞬下沉；名义锚点不变。")
            else:
                print(f"Z闭环直接回放：上限{args.z_max_correction_mm:g}mm；无整段IK包络预检、无预热。")
                print("修正从0开始；初次接管下沉不会被提前补偿。")
        if not FIXED_LEFT_START_VALIDATED:
            print("注意：旧左臂起点尚未验证 O6 朝向，离线结果仅供参考；真机执行暂停。")
        elif not FIXED_LEFT_PREPARE_PATH_VALIDATED:
            print("注意：已在正式起点可跳过恢复；偏离起点时，未验证的 MoveJ 恢复仍被拦截。")
        print(f"计划: {len(replay_times)} 帧 @ {args.rate:g} Hz, "
              f"{replay_times[-1]:.3f} s, scale={args.scale:g}, "
              f"speed={args.speed:g}, mapping={args.tracker_mapping_info['mode']}")
        print(f"Tracker目标插值: {args.interpolation}；原始有效帧 {len(trajectory.timestamps_s)}")
        if args.interpolation == "none":
            print("不生成帧间位姿；原帧按时间戳到期更新，发送周期内重复上一帧。")
            max_step = np.linalg.norm(np.diff(delta_positions, axis=0), axis=1).max()
            max_turn = max(quaternion_distance_deg(a, b) for a, b in
                           zip(delta_quaternions[:-1], delta_quaternions[1:]))
            print(f"原帧最大目标阶跃: {max_step*1000:.2f}mm / {max_turn:.2f}°")
        print(f"轨迹最大相对位移: {max_displacement:.4f} m")
        print(f"轨迹最大相对转角: {float(angles.max()):.2f}°，orientation-scale={args.orientation_scale:g}")
        print("左腕目标范围 (m): "
              f"min={targets.min(axis=0).round(4).tolist()}, "
              f"max={targets.max(axis=0).round(4).tolist()}")
        print("注意：保守参考范围取自 MoveP 文档，不是已经验证的 ServoP 边界；"
              "仅用于运动左臂；离线检查不等于 IK 可达性、碰撞或真机保持验证。")
        if hand_replay is not None:
            print(f"O6 左手实测状态: {len(hand_replay.states)} 帧，"
                  f"最后一帧时间 {hand_replay.due_ns[-1] / 1e9:.3f}s；"
                  "与 Tracker 使用同一回放时钟。")
        if args.plan_ik:
            from tron2_ik_report import write_ik_plan
            write_ik_plan(args, trajectory.source, replay_times,
                          sampled_positions, sampled_quaternions,
                          FIXED_LEFT_Q, FIXED_LEFT_POSITION, FIXED_LEFT_QUATERNION)
            return 0
        if args.pose_compensation and not args.execute and not args.check_robot:
            from tron2_pose_replay import prepare_pose_replay
            offline_anchor = DualArmPose(FIXED_LEFT_POSITION.copy(),FIXED_LEFT_QUATERNION.copy(),
                                        np.zeros(3),np.array([1.,0.,0.,0.]))
            prepare_pose_replay(offline_anchor,np.r_[FIXED_LEFT_Q,np.zeros(9)],replay_times,
                                sampled_positions,sampled_quaternions,args)
        if not args.execute and not args.check_robot:
            print("全位姿名义/模型修正参考整段预检完成；未连接机器人或发送运动指令。" if args.pose_compensation else
                  "轨迹信息读取完成；未执行整段IK预检，未连接机器人或发送运动指令。")
            return 0

        if args.execute:
            require_validated_left_start()
            trace = TraceRecorder(args.trace_file or default_trace_path())
            if hand_replay is not None:
                hand_replay.trace = trace
            trace.record(
                "fixed_start_plan", episode=str(trajectory.source),
                robot_id=FIXED_ROBOT_ID, fixed_left_q=FIXED_LEFT_Q.tolist(),
                fixed_left_position=FIXED_LEFT_POSITION.tolist(),
                fixed_left_reference_position=FIXED_LEFT_REFERENCE_POSITION.tolist(),
                fixed_left_position_offset_m=FIXED_LEFT_POSITION_OFFSET_M.tolist(),
                fixed_left_start_profile=str(FIXED_LEFT_START_PROFILE),
                fixed_left_height_offset_m=FIXED_LEFT_HEIGHT_OFFSET_M,
                fixed_left_start_measured=FIXED_LEFT_START_MEASURED,
                preparation_arm="left", right_hold="measured_at_start",
                right_movep_workspace_check=False,
                scale=args.scale, orientation_scale=args.orientation_scale, axis_map=args.axis_map,
                tracker_mapping=args.tracker_mapping_info,
                speed=args.speed, with_o6_hand=bool(hand_replay),
                tracker_interpolation=args.interpolation,
                z_compensation=args.z_compensation,
                pose_compensation=args.pose_compensation,
                controller_version=('pose-reference-v1-experimental-20261105' if args.pose_compensation
                                    else 'legacy-continuous-z' if args.z_compensation and args.z_adaptation=='continuous'
                                    else 'legacy-stationary-z' if args.z_compensation else 'legacy-direct-servop'),
                z_adaptation=args.z_adaptation,
                z_max_correction_mm=args.z_max_correction_mm if args.z_compensation else None,
                o6_can=args.o6_can, arm_control=args.arm_control,
                action_contract=SERVOP_ACTION_CONTRACT,
                expected_final_delta_m=sampled_positions[-1].tolist(),
            )
            print(f"记录文件: {trace.path}")
        client = Tron2JsonClient(args.robot_ip, trace=trace)
        client.wait_for_identity()
        check_tracker_mapping_robot(args, client.accid)
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
        if q.size >= 16:
            print(f"当前头部 pitch/yaw（只读，rad）: {q[14:16].round(4).tolist()}")
        if args.check_robot:
            print("当前与左臂固定起点的误差（右臂不要求回历史位置）：")
            try:
                check_fixed_pose(pose, args.position_tolerance,
                                 args.angle_tolerance)
            except RuntimeError as exc:
                print(f"提示：{exc}")
            try:
                check_workspace("left", pose.left_position[None, :] + sampled_positions)
                check_servop_hold_position("right", pose.right_position)
            except ValueError as exc:
                print(f"ServoP 执行条件未满足：{exc}")
            print("只读预检结束；没有发送 MoveJ 或 ServoP。")
            return 0
        # 右腕只核验有效坐标，不因 MoveP 参考包围盒而拦截左臂准备。
        check_servop_hold_position("right", pose.right_position)
        anchor = restore_fixed_start(client, q, dq, pose, args, trace)
        check_servop_hold_position("right", anchor.right_position)
        if args.pose_compensation:
            from tron2_pose_replay import prepare_pose_replay
            live_q,live_dq=client.joint_snapshot(label='pose_reference_preflight')
            if np.max(abs(live_dq))>.05:
                raise RuntimeError('全位姿参考预检前关节未停稳')
            prepare_pose_replay(anchor,live_q,replay_times,sampled_positions,sampled_quaternions,args,trace)
        if hand_replay is not None:
            if args.o6_can is not None:
                o6_bridge = O6ReplayBridge(
                    args.o6_can, trace.path.with_suffix(".o6_bridge.log"), trace
                )
                o6_bridge.start(hand_replay)
            else:
                hand_replay.probe_bridge()
            print("O6 回放桥接已就绪；尚未发送手指轨迹。")
        print("固定起点已验证。回放将再要求输入 REPLAY 并倒计时；"
              "首帧接管仍需现场观察，固定右腕目标不等于关节锁死；头部不参与回放。")
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
