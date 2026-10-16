#!/usr/bin/env python3
"""逐臂将 TRON2 移动到预设的 MoveJ 准备姿态。

这是独立的真机运动操作，不会启动 Tracker 数据回放。程序先移动右臂，再移动左臂。
倒计时及运动期间必须持续观察机器人，并确保实体急停随时可用。
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

from replay_tron2_tracker import Tron2JsonClient, WORKSPACE


ROBOT_IP = "10.192.1.2"
READY_ARM_Q = np.array([0.0, 0.0, 0.0, -1.6, 0.0, 0.0, 0.0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-ip", default=ROBOT_IP)
    parser.add_argument(
        "--duration", type=float, default=10.0,
        help="seconds for each arm movement (default: 10)",
    )
    parser.add_argument(
        "--countdown", type=int, default=15,
        help="seconds between confirmation and the first movement (default: 15)",
    )
    parser.add_argument(
        "--joint-tolerance", type=float, default=0.04,
        help="maximum final joint error in radians (default: 0.04)",
    )
    return parser.parse_args()


def get_joint_state(client: Tron2JsonClient) -> tuple[np.ndarray, np.ndarray]:
    response = client.request(
        "request_get_joint_state", {}, "response_get_joint_state", timeout=4.0
    )
    data = response.get("data", {})
    if data.get("result") != "success":
        raise RuntimeError(
            "robot did not provide joint state: "
            f"{data.get('result', 'missing result')}"
        )
    q = np.asarray(data.get("q", []), dtype=np.float64)
    dq = np.asarray(data.get("dq", []), dtype=np.float64)
    if q.shape[0] < 14 or dq.shape[0] < 14:
        raise RuntimeError(
            f"expected at least 14 arm joints, received q={len(q)}, dq={len(dq)}"
        )
    if not np.isfinite(q[:14]).all() or not np.isfinite(dq[:14]).all():
        raise RuntimeError("joint state contains non-finite values")
    return q, dq


def inside_workspace(side: str, position: np.ndarray) -> bool:
    limits = WORKSPACE[side]
    return bool(np.all((position >= limits[:, 0]) & (position <= limits[:, 1])))


def wait_for_target(
    client: Tron2JsonClient,
    target: np.ndarray,
    controlled_slice: slice,
    timeout_s: float,
    tolerance: float,
) -> np.ndarray:
    deadline = time.monotonic() + timeout_s
    settled = 0
    last_q: np.ndarray | None = None
    read_timeouts = 0
    while time.monotonic() < deadline:
        try:
            q, dq = get_joint_state(client)
        except TimeoutError as exc:
            # MoveJ 已被机器人接受；这里只重试只读反馈，绝不重发运动命令。
            # 运动期间控制器可能暂时不回复 joint_state，单次 4s 超时
            # 不能说明 MoveJ 已停止或失败，仍受整个准备阶段截止时间约束。
            read_timeouts += 1
            trace = getattr(client, "trace", None)
            if trace is not None:
                trace.record(
                    "movej_joint_feedback_timeout", count=read_timeouts,
                    remaining_s=max(0.0, deadline - time.monotonic()),
                    message=str(exc),
                )
            continue
        last_q = q
        error = float(np.max(np.abs(q[:14][controlled_slice] - target[controlled_slice])))
        speed = float(np.max(np.abs(dq[:14][controlled_slice])))
        if error <= tolerance and speed <= 0.05:
            settled += 1
            if settled >= 3:
                return q
        else:
            settled = 0
        time.sleep(0.2)
    if last_q is None:
        raise RuntimeError(
            "MoveJ 后始终未收到关节状态反馈；运动是否完成未知，"
            "不能继续下一阶段或回放"
        )
    error = float(
        np.max(np.abs(last_q[:14][controlled_slice] - target[controlled_slice]))
    )
    raise RuntimeError(
        f"MoveJ did not settle before timeout; maximum controlled-joint error "
        f"is {error:.3f} rad"
    )


def move_stage(
    client: Tron2JsonClient,
    label: str,
    side: str,
    target: np.ndarray,
    controlled_slice: slice,
    duration_s: float,
    tolerance: float,
) -> None:
    print(f"\n开始 {label}：{side} 臂，计划用时 {duration_s:.1f}s")
    sys.stdout.flush()
    response = client.request(
        "request_movej",
        {"time": duration_s, "joint": target.tolist()},
        "response_movej",
        timeout=duration_s + 5.0,
    )
    result = response.get("data", {}).get("result")
    if result != "success":
        raise RuntimeError(f"{label} request_movej failed: {result}")

    wait_for_target(
        client,
        target,
        controlled_slice,
        timeout_s=duration_s + 5.0,
        tolerance=tolerance,
    )
    pose = client.current_pose()
    position = pose.right_position if side == "right" else pose.left_position
    print(f"{label}完成；{side} 腕位置 (m): {position.round(4).tolist()}")
    if not inside_workspace(side, position):
        raise RuntimeError(
            f"{label}完成但 {side} 腕仍在 ServoP 工作空间外；不执行下一阶段"
        )


def main() -> int:
    args = parse_args()
    try:
        if args.duration < 5.0:
            raise ValueError("--duration must be at least 5 seconds")
        if args.countdown < 5:
            raise ValueError("--countdown must be at least 5 seconds")
        if not 0.005 <= args.joint_tolerance <= 0.1:
            raise ValueError("--joint-tolerance must be between 0.005 and 0.1 rad")

        client = Tron2JsonClient(args.robot_ip)
        client.wait_for_identity()
        pose = client.current_pose()
        q, dq = get_joint_state(client)
        if float(np.max(np.abs(dq[:14]))) > 0.05:
            raise RuntimeError("arms are currently moving; wait until they stop")

        print("当前双腕位置 (m):")
        print(f"  left : {pose.left_position.round(4).tolist()}")
        print(f"  right: {pose.right_position.round(4).tolist()}")
        print("\n自动准备流程：先右臂，验收通过后再左臂；不会自动回放数据。")
        print("MoveJ 运动中 Ctrl+C 不能保证停止机器人，必须依靠实体急停。")
        answer = input("确认周围清空且急停在手边，输入 PREPARE 继续: ")
        if answer.strip() != "PREPARE":
            raise RuntimeError("cancelled by user")

        for remaining in range(args.countdown, 0, -1):
            print(
                f"\r{remaining:2d}s 后开始移动；请离开电脑并拿好遥控器急停...",
                end="",
                flush=True,
            )
            time.sleep(1.0)
        print("\r开始自动准备。请持续观察机器人。                    ")

        # 阶段 1：左臂保持实测关节角，仅移动右臂。
        first_target = q[:14].copy()
        first_target[7:14] = READY_ARM_Q
        move_stage(
            client,
            "阶段 1/2",
            "right",
            first_target,
            slice(7, 14),
            args.duration,
            args.joint_tolerance,
        )

        # 重新读取关节角，阶段 2 用右臂稳定后的实测位置保持右臂，仅移动左臂。
        q_after_right, dq_after_right = get_joint_state(client)
        if float(np.max(np.abs(dq_after_right[:14]))) > 0.05:
            raise RuntimeError("right arm did not become stationary")
        second_target = q_after_right[:14].copy()
        second_target[0:7] = READY_ARM_Q
        move_stage(
            client,
            "阶段 2/2",
            "left",
            second_target,
            slice(0, 7),
            args.duration,
            args.joint_tolerance,
        )

        final_pose = client.current_pose()
        if not inside_workspace("left", final_pose.left_position):
            raise RuntimeError("final left wrist pose is outside the ServoP workspace")
        if not inside_workspace("right", final_pose.right_position):
            raise RuntimeError("final right wrist pose is outside the ServoP workspace")
        print("\n准备姿态完成；左右腕均在 ServoP 工作空间内。")
        print("本程序没有启动 Tracker 数据回放。")
        sys.stdout.flush()
        return 0
    except KeyboardInterrupt:
        print(
            "\n收到 Ctrl+C；客户端已退出，但 MoveJ 可能仍在执行。"
            "如需立即停止，请使用实体急停。",
            file=sys.stderr,
        )
        return 130
    except (ValueError, RuntimeError, TimeoutError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    # 厂商 SDK 在真机连接后的原生资源清理阶段可能崩溃。
    os._exit(exit_code)
