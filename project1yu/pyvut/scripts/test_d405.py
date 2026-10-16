#!/usr/bin/env python3
"""
RealSense D405 摄像头测试与调试脚本.

功能:
  - 显示摄像头基本信息 (序列号/固件/USB)
  - 实时 RGB 预览 + 深度图彩色显示
  - 显示 FPS
  - 按键控制:
      q/ESC : 退出
      d     : 切换深度彩色映射 (JET / HOT / VIRIDIS)
      c     : 切换深度着色 (COLORMAP / 伪彩色 / 灰度)
      r     : 切换分辨率 (640x480 / 848x480 / 1280x720)
      s     : 保存当前帧截图
      i     : 打印当前帧中心点深度值

用法:
    python3 scripts/test_d405.py
"""

import time
import os
import sys

import numpy as np
import cv2
import pyrealsense2 as rs


# ---- 预设分辨率 (D405 支持) ----
RESOLUTIONS = [
    (640, 480),
    (848, 480),
    (1280, 720),
]

# ---- 深度图着色模式 ----
COLORMAP_MODES = {
    "JET": cv2.COLORMAP_JET,
    "HOT": cv2.COLORMAP_HOT,
    "VIRIDIS": cv2.COLORMAP_VIRIDIS,
    "BONE": cv2.COLORMAP_BONE,
    "TURBO": cv2.COLORMAP_TURBO,
}


def print_device_info(device):
    """打印设备详细信息."""
    print("=" * 60)
    print("  RealSense D405 设备信息")
    print("=" * 60)
    info_map = {
        "名称": rs.camera_info.name,
        "序列号": rs.camera_info.serial_number,
        "固件版本": rs.camera_info.firmware_version,
        "USB类型": rs.camera_info.usb_type_descriptor,
        "物理端口": rs.camera_info.physical_port,
        "产品ID": rs.camera_info.product_id,
    }
    for label, info_key in info_map.items():
        try:
            val = device.get_info(info_key)
            print(f"  {label:12s}: {val}")
        except Exception:
            print(f"  {label:12s}: N/A")

    # 传感器信息
    print("\n  传感器列表:")
    for sensor in device.sensors:
        try:
            name = sensor.get_info(rs.camera_info.name)
            print(f"    - {name}")
            for profile in sensor.get_stream_profiles():
                try:
                    vp = profile.as_video_stream_profile()
                    fmt = vp.format()
                    w, h = vp.width(), vp.height()
                    fps = vp.fps()
                    print(f"        {w}x{h} @ {fps}fps  {fmt}")
                except Exception:
                    pass
        except Exception:
            pass
    print("=" * 60)


def colorize_depth(depth_frame, colormap=cv2.COLORMAP_JET, min_depth=0.0, max_depth=1.5):
    """
    将深度帧转为彩色图像以便可视化.
    D405 推荐测量范围: 7cm ~ 50cm (近距离高精度).
    """
    depth_image = np.asanyarray(depth_frame.get_data())
    # 裁剪到有效范围
    depth_clipped = np.clip(depth_image.astype(np.float32) * 0.001, min_depth, max_depth)
    # 归一化到 0-255
    depth_normalized = ((depth_clipped - min_depth) / (max_depth - min_depth) * 255).astype(np.uint8)
    # 应用颜色映射
    depth_colored = cv2.applyColorMap(depth_normalized, colormap)
    return depth_colored, depth_image


def put_fps(img, fps):
    """在图像上绘制 FPS."""
    text = f"FPS: {fps:.1f}"
    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return img


def put_help(img):
    """在图像上绘制帮助信息."""
    helps = [
        "q/ESC: 退出 | d: 切换色彩 | c: 切换着色 | r: 切换分辨率",
        "s: 截图 | i: 深度值"
    ]
    h, w = img.shape[:2]
    overlay = img.copy()
    y0 = h - 50
    cv2.rectangle(overlay, (0, y0), (w, h), (0, 0, 0), -1)
    img = cv2.addWeighted(img, 1, overlay, 0.4, 0)
    for j, line in enumerate(helps):
        cv2.putText(img, line, (10, y0 + 18 + j * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    return img


def main():
    print("正在初始化 RealSense D405...")

    # 创建 Pipeline
    pipeline = rs.pipeline()
    config = rs.config()

    # 查找 D405 设备
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        print("[错误] 未检测到 RealSense 设备! 请检查 USB 连接.")
        sys.exit(1)

    device = devices[0]
    serial = device.get_info(rs.camera_info.serial_number)
    config.enable_device(serial)
    print_device_info(device)

    # 初始分辨率
    res_idx = 0
    w, h = RESOLUTIONS[res_idx]
    config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, w, h, rs.format.z16, 30)

    # 启动流
    profile = pipeline.start(config)

    # 获取深度传感器参数
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print(f"\n深度比例因子: {depth_scale:.6f} (值*scale = 米)")
    print(f"深度单位: {depth_sensor.get_option(rs.option.depth_units):.6f}")

    # 深度预设置 (D405 推荐近距离高精度)
    try:
        depth_sensor.set_option(rs.option.visual_preset, 5)  # High Accuracy
        print("深度预设: 高精度模式")
    except Exception:
        print("深度预设: 默认模式")

    # 对齐到彩色帧
    align = rs.align(rs.stream.color)

    # 状态变量
    colormap_idx = 0
    colormap_names = list(COLORMAP_MODES.keys())
    colorize_mode = 0  # 0: colormap, 1: pseudo RGB, 2: grayscale
    frame_count = 0
    fps = 0
    fps_timer = time.time()
    save_dir = os.path.join(os.path.dirname(__file__), "..", "d405_screenshots")
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n截图保存至: {save_dir}")
    print("开始实时预览, 按 q 退出...\n")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            # 转换为 numpy 数组
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            # 深度可视化
            if colorize_mode == 0:
                depth_colored, _ = colorize_depth(depth_frame, COLORMAP_MODES[colormap_names[colormap_idx]])
            elif colorize_mode == 1:
                # 伪彩色: 直接用深度值生成 RGB
                d_norm = np.clip(depth_image.astype(np.float32) / 1500.0, 0, 1)
                depth_colored = (d_norm * 255).astype(np.uint8)
                depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_GRAY2BGR)
            else:
                # 灰度
                d_gray = np.clip(depth_image.astype(np.float32) / 1500.0 * 255, 0, 255).astype(np.uint8)
                depth_colored = cv2.cvtColor(d_gray, cv2.COLOR_GRAY2BGR)

            # 确保两张图一样大
            if color_image.shape[:2] != depth_colored.shape[:2]:
                depth_colored = cv2.resize(depth_colored, (color_image.shape[1], color_image.shape[0]))

            # FPS 计算
            frame_count += 1
            if time.time() - fps_timer >= 1.0:
                fps = frame_count / (time.time() - fps_timer)
                frame_count = 0
                fps_timer = time.time()

            # 添加 FPS
            put_fps(color_image, fps)
            put_fps(depth_colored, fps)

            # 添加中心点深度文本
            cy, cx = color_image.shape[0] // 2, color_image.shape[1] // 2
            center_depth = depth_image[cy, cx] * depth_scale
            cv2.circle(color_image, (cx, cy), 5, (0, 255, 255), 2)
            cv2.putText(color_image, f"Center: {center_depth:.3f}m",
                        (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            # 添加模式提示
            mode_text = f"CM: {colormap_names[colormap_idx]} | {RESOLUTIONS[res_idx][0]}x{RESOLUTIONS[res_idx][1]}"
            cv2.putText(color_image, mode_text, (10, color_image.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

            # 合并显示
            combined = np.hstack((color_image, depth_colored))
            put_help(combined)

            cv2.imshow("D405 | RGB (左) / Depth (右)", combined)

            # 按键处理
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # q or ESC
                print("退出预览.")
                break
            elif key == ord('d'):
                colormap_idx = (colormap_idx + 1) % len(colormap_names)
                print(f"切换深度色彩: {colormap_names[colormap_idx]}")
            elif key == ord('c'):
                colorize_mode = (colorize_mode + 1) % 3
                modes = ["COLORMAP", "伪彩色", "灰度"]
                print(f"切换着色模式: {modes[colorize_mode]}")
            elif key == ord('r'):
                pipeline.stop()
                res_idx = (res_idx + 1) % len(RESOLUTIONS)
                w, h = RESOLUTIONS[res_idx]
                config = rs.config()
                config.enable_device(serial)
                config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, 30)
                config.enable_stream(rs.stream.depth, w, h, rs.format.z16, 30)
                pipeline.start(config)
                print(f"切换分辨率: {w}x{h}")
            elif key == ord('s'):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                rgb_path = os.path.join(save_dir, f"rgb_{timestamp}.png")
                depth_path = os.path.join(save_dir, f"depth_{timestamp}.png")
                cv2.imwrite(rgb_path, color_image)
                cv2.imwrite(depth_path, depth_colored)
                print(f"截图已保存: {rgb_path}, {depth_path}")
            elif key == ord('i'):
                cy, cx = color_image.shape[0] // 2, color_image.shape[1] // 2
                center_depth = depth_image[cy, cx] * depth_scale
                # 打印 3x3 区域均值
                patch = depth_image[cy-2:cy+3, cx-2:cx+3].astype(np.float32)
                mean_d = np.mean(patch) * depth_scale
                print(f"中心深度: {center_depth:.4f}m | 3x3均值: {mean_d:.4f}m")

    except KeyboardInterrupt:
        print("\n用户中断.")
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("相机已停止.")


if __name__ == "__main__":
    main()
