#!/usr/bin/env python3
"""Simple demo that prints live 6DoF poses using pyvut's UltimateTrackerAPI."""

import argparse
import time
from math import asin, atan2, pi

from pyvut import TrackerPose, UltimateTrackerAPI  # noqa: E402

def quat_to_euler_deg(quat) -> tuple:
    """将四元数 [w, x, y, z] 转换为欧拉角 (roll, pitch, yaw), 单位: 度.

    使用 ZYX 内旋顺序 (yaw → pitch → roll), 与航空航天惯例一致:
        - roll  (滚转角): 绕 X 轴旋转, 范围 [-180°, 180°]
        - pitch (俯仰角): 绕 Y 轴旋转, 范围 [ -90°,  90°]
        - yaw   (偏航角): 绕 Z 轴旋转, 范围 [-180°, 180°]

    Args:
        quat: 长度为4的可迭代对象, 标准顺序 [w, x, y, z].

    Returns:
        tuple (roll, pitch, yaw), 每个值单位 度.
    """
    w, x, y, z = quat

    # roll (X轴旋转)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = atan2(t0, t1)

    # pitch (Y轴旋转), 限制在 [-1, 1] 避免 asin 数值越界
    t2 = +2.0 * (w * y - z * x)
    t2 = max(min(t2, 1.0), -1.0)
    pitch = asin(t2)

    # yaw (Z轴旋转)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = atan2(t3, t4)

    return tuple(angle * 180.0 / pi for angle in (roll, pitch, yaw))


def format_pose(pose: TrackerPose) -> str:
    """将 TrackerPose 格式化为可读的单行字符串.

    输出示例:
        tracker=0 mac=34:7B:02:75:8D:BF status=2
        pos=( 0.123, -0.456,  1.789) quat=( 0.999, -0.001,  0.002, -0.003)
        euler_deg=(  -1.23,   0.45,  89.00) buttons=0x0000 timestamp_ms=12345678

    Args:
        pose: TrackerPose 对象.

    Returns:
        格式化的单行字符串, 包含位置(米)、四元数、欧拉角(度)及状态.
    """
    pos = ", ".join(f"{axis: .3f}" for axis in pose.position)
    rot = ", ".join(f"{axis: .3f}" for axis in pose.rotation)
    euler = ", ".join(f"{angle: .2f}" for angle in quat_to_euler_deg(pose.rotation))
    return (
        f"tracker={pose.tracker_index} mac={pose.mac} status={pose.tracking_status} "
        f"pos=({pos}) quat=({rot}) euler_deg=({euler}) buttons={pose.buttons:#06x}"
        f" timestamp_ms={pose.timestamp_ms}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["DONGLE_USB", "TRACKER_USB"],
        default="DONGLE_USB",
        help="Transport to use for tracker communication.",
    )
    parser.add_argument(
        "--wifi-info",
        dest="wifi_info",
        help="Optional path to a wifi_info.json file (defaults to pyvut/wifi_info.json).",
    )
    args = parser.parse_args()

    def on_pose(pose: TrackerPose) -> None:
        print(format_pose(pose))

    print(
        "Starting UltimateTrackerAPI… Rotations are reported as quaternions (w,x,y,z) and Euler angles (roll, pitch, yaw in degrees)."
        " Trackers emit raw (w,z,y,x) order but pyvut normalizes this for you. Press Ctrl+C to stop."
    )
    try:
        with UltimateTrackerAPI(mode=args.mode, wifi_info_path=args.wifi_info) as api:
            api.add_pose_callback(on_pose)
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping…")


if __name__ == "__main__":
    main()
