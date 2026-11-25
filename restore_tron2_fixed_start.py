#!/usr/bin/env python3
"""仅恢复 TRON2 左臂固定起点；右臂保持本次关节角，不回放 episode。

--test-path --execute 是需要现场输入 RESTORE 的独立分段试验，
不修改正式恢复路径验证标志，不允许由网页后台启动。
"""

from __future__ import annotations

import argparse
import math
import os
import sys

from replay_tron2_fixed_start import (
    FIXED_LEFT_POSITION,
    FIXED_LEFT_START_LABEL,
    FIXED_LEFT_START_MEASURED,
    FIXED_LEFT_HEIGHT_OFFSET_M,
    FIXED_LEFT_POSITION_OFFSET_M,
    FIXED_LEFT_START_PROFILE,
    FIXED_ROBOT_ID,
    acquire_replay_lock,
    require_validated_left_start,
    restore_fixed_start,
)
from replay_tron2_tracker import (
    DEFAULT_ROBOT_IP,
    TraceRecorder,
    Tron2JsonClient,
    default_trace_path,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true",
                        help="倒计时后仅执行左臂固定起点 MoveJ，右臂不归位")
    parser.add_argument("--test-path", action="store_true",
                        help="独立现场恢复试验：输入 RESTORE 后分段 MoveJ，不批准正式自动恢复")
    parser.add_argument("--trial-step-duration", type=float, default=3.0,
                        help="--test-path 每段 MoveJ 秒数（2 至 10 秒，默认 3）；不使用 --move-duration")
    parser.add_argument("--countdown", type=int, default=5)
    parser.add_argument("--move-duration", type=float, default=10.0)
    parser.add_argument("--joint-tolerance", type=float, default=0.04)
    parser.add_argument("--position-tolerance", type=float, default=0.02)
    parser.add_argument("--angle-tolerance", type=float, default=3.0)
    parser.add_argument("--robot-ip", default=DEFAULT_ROBOT_IP)
    parser.add_argument("--web-control", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    trace = None
    replay_lock = None
    try:
        if not math.isfinite(args.move_duration) or not 5 <= args.move_duration <= 30:
            raise ValueError("--move-duration 必须在 5 至 30 秒之间")
        if not 5 <= args.countdown <= 30:
            raise ValueError("--countdown 必须在 5 至 30 秒之间")
        if not 0.005 <= args.joint_tolerance <= 0.1:
            raise ValueError("--joint-tolerance 必须在 0.005 至 0.1 rad 之间")
        if not 0.001 <= args.position_tolerance <= 0.02:
            raise ValueError("--position-tolerance 必须在 0.001 至 0.02 m 之间")
        if not 0.1 <= args.angle_tolerance <= 5:
            raise ValueError("--angle-tolerance 必须在 0.1 至 5 度之间")
        if args.web_control and not args.execute:
            raise ValueError("--web-control 必须与 --execute 一同使用")
        if args.test_path and args.web_control:
            raise ValueError("恢复路径试验仅供现场交互终端，不能由网页后台触发")
        if (not math.isfinite(args.trial_step_duration)
                or not 2 <= args.trial_step_duration <= 10):
            raise ValueError("--trial-step-duration 必须在 2 至 10 秒之间")
        print(f"目标机器人: {FIXED_ROBOT_ID}；仅恢复保存的左臂固定起点")
        print(f"left 固定腕位置 (m): {FIXED_LEFT_POSITION.tolist()}")
        print("右臂保持启动前实测关节角，不自动归位。")
        print(f"固定起点: {FIXED_LEFT_START_LABEL}；静态姿态与自动恢复路径分别验收。")
        if not FIXED_LEFT_START_MEASURED:
            print("当前准备目标尚未实测；需单独恢复试验及现场确认。")
        print("不读取 episode，不发送 Tracker、ServoP 或 O6 手指轨迹。")
        if args.test_path:
            print("模式：独立恢复路径试验；每段计划关节变化不超过 5°，逐段验收。"
                  "仍需现场输入 RESTORE；不会修改正式路径验证标志。")
        if not args.execute:
            print("离线查看结束；未连接机器人、未发送运动指令。")
            return 0

        require_validated_left_start()
        replay_lock = acquire_replay_lock()
        trace = TraceRecorder(default_trace_path())
        trace.record("restore_only_plan", robot_id=FIXED_ROBOT_ID,
                     countdown_s=args.countdown,
                     move_duration_s=None if args.test_path else args.move_duration,
                     trial_step_duration_s=args.trial_step_duration if args.test_path else None,
                     preparation_arm="left",
                     fixed_left_position=FIXED_LEFT_POSITION.tolist(),
                     fixed_left_start_measured=FIXED_LEFT_START_MEASURED,
                     fixed_left_height_offset_m=FIXED_LEFT_HEIGHT_OFFSET_M,
                     fixed_left_position_offset_m=FIXED_LEFT_POSITION_OFFSET_M.tolist(),
                     fixed_left_start_profile=str(FIXED_LEFT_START_PROFILE),
                     right_hold="measured_at_start", path_trial=args.test_path)
        print(f"记录文件: {trace.path}")
        client = Tron2JsonClient(args.robot_ip, trace=trace)
        client.wait_for_identity()
        if client.accid != FIXED_ROBOT_ID:
            raise RuntimeError(
                f"连接的是 {client.accid}，保存姿态只针对 {FIXED_ROBOT_ID}；拒绝移动"
            )
        q, dq = client.joint_snapshot(label="before_restore_only")
        pose = client.current_servo_pose(label="before_restore_only")
        print("当前腕位置 (m):")
        print(f"  left : {pose.left_position.round(4).tolist()}")
        print(f"  right: {pose.right_position.round(4).tolist()}")
        if args.test_path:
            from tron2_restore_trial import run_trial
            run_trial(client, q, dq, pose, args, trace)
        else:
            restore_fixed_start(client, q, dq, pose, args, trace)
        print("左臂固定起点恢复完成；未发送 Tracker、ServoP 或 O6 轨迹。")
        return 0
    except KeyboardInterrupt:
        if trace is not None:
            trace.record("interrupted", message="KeyboardInterrupt")
        print("已停止后续软件指令；MoveJ 可能仍在机器人端继续，这不是实体急停。",
              file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError, TimeoutError) as exc:
        if trace is not None:
            trace.record("error", message=str(exc))
        print(f"错误: {exc}", file=sys.stderr)
        return 2
    finally:
        if trace is not None:
            trace.close()
        if replay_lock is not None:
            replay_lock.close()


if __name__ == "__main__":
    code = main()
    if "--execute" in sys.argv:
        # 与现有回放脚本一致，绕开厂商 SDK 的原生析构崩溃。
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
    raise SystemExit(code)
