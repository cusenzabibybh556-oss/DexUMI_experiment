#!/usr/bin/env python3
"""
D405 → CNN Demo: 搞懂图像张量排布 + RGBD 送入卷积网络.

核心要点:
  1. 张量排布: NHWC   NCHW, uint8    float32, BGR   RGB
  2. RGBD 组合: 4 通道输入 (R, G, B, D)
  3. 归一化策略: RGB→[0,1], Depth→Z-score 或 min-max
  4. 完整 pipeline: 相机取帧 → 预处理 → CNN 前向

用法:
    python3 trush-bin/D405_test.py
"""

import time
import sys

import numpy as np
import cv2
import pyrealsense2 as rs
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. 定义一个简单的 CNN（用于演示 RGBD 四通道输入）
# ============================================================
class SimpleRGBDNet(nn.Module):
    """4 通道输入 → 卷积 → 全连接 → 输出"""

    def __init__(self, in_channels=4, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        """
        x: (B, 4, H, W)  NCHW, float32
        通道顺序: [R, G, B, Depth]
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.gap(x)           # (B, 256, 1, 1)
        x = x.view(x.size(0), -1)  # (B, 256)
        x = self.fc(x)             # (B, num_classes)
        return x


# ============================================================
# 2. 预处理: D405 raw → CNN-ready tensor (展示每步排布变化)
# ============================================================
class D405Preprocessor:
    """把 D405 的原始帧转成 CNN 可接受的张量，并展示每步的排布变化."""

    def __init__(self, target_size=(224, 224), depth_norm="zscore"):
        self.H, self.W = target_size
        self.depth_norm = depth_norm

    def process(self, color_bgr_uint8, depth_raw_uint16, depth_scale, verbose=True):
        """
        完整预处理流水线，每一步都打印张量排布。

        输入:
          color_bgr_uint8  : (H_cam, W_cam, 3)  uint8    BGR
          depth_raw_uint16 : (H_cam, W_cam)     uint16   raw (×scale = 米)
          depth_scale      : float, raw × scale = meters

        输出:
          rgbd_tensor      : (1, 4, H, W) float32 NCHW  CNN-Ready
        """
        if verbose:
            print("=" * 60)
            print("  预处理流水线: 张量排布变化")
            print("=" * 60)

        # --- Step 1: BGR → RGB ---
        color_rgb_uint8 = cv2.cvtColor(color_bgr_uint8, cv2.COLOR_BGR2RGB)
        if verbose:
            print(f"\n  [Step 1] BGR → RGB")
            print(f"    shape: {color_bgr_uint8.shape}  →  {color_rgb_uint8.shape}")
            print(f"    dtype: {color_bgr_uint8.dtype}     →  {color_rgb_uint8.dtype}")
            print(f"    注意: OpenCV 默认 BGR, PyTorch/TF 模型通常用 RGB")

        # --- Step 2: Resize ---
        color_rgb_uint8 = cv2.resize(color_rgb_uint8, (self.W, self.H))
        depth_raw_uint16 = cv2.resize(depth_raw_uint16, (self.W, self.H),
                                       interpolation=cv2.INTER_NEAREST)
        if verbose:
            print(f"\n  [Step 2] Resize → ({self.H}, {self.W})")
            print(f"    RGB:   {color_rgb_uint8.shape}  {color_rgb_uint8.dtype}")
            print(f"    Depth: {depth_raw_uint16.shape}  {depth_raw_uint16.dtype}")

        # --- Step 3: uint → float32 ---
        rgb_f32 = color_rgb_uint8.astype(np.float32)
        depth_m = depth_raw_uint16.astype(np.float32) * depth_scale
        if verbose:
            print(f"\n  [Step 3] uint → float32")
            print(f"    RGB:   {rgb_f32.shape}  {rgb_f32.dtype}  "
                  f"range [{rgb_f32.min():.0f}, {rgb_f32.max():.0f}]")
            print(f"    Depth: {depth_m.shape}  {depth_m.dtype}  "
                  f"range [{depth_m.min():.3f}, {depth_m.max():.3f}] m")

        # --- Step 4: 归一化 ---
        rgb_norm = rgb_f32 / 255.0  # [0, 1]

        if self.depth_norm == "zscore":
            valid_mask = depth_m > 0.01
            if valid_mask.sum() > 100:
                d_mean = depth_m[valid_mask].mean()
                d_std = depth_m[valid_mask].std() + 1e-8
                depth_norm = (depth_m - d_mean) / d_std
            else:
                depth_norm = depth_m
        else:
            d_min = depth_m[depth_m > 0].min()
            d_max = depth_m.max()
            depth_norm = (depth_m - d_min) / (d_max - d_min + 1e-8)

        if verbose:
            print(f"\n  [Step 4] 归一化 (depth={self.depth_norm})")
            print(f"    RGB:   range [{rgb_norm.min():.3f}, {rgb_norm.max():.3f}]")
            print(f"    Depth: range [{depth_norm.min():.3f}, {depth_norm.max():.3f}]")

        # --- Step 5: concat RGB + Depth → 4 通道 NHWC ---
        depth_ch = depth_norm[:, :, np.newaxis]          # (H,W) → (H,W,1)
        rgbd_nhwc = np.concatenate([rgb_norm, depth_ch], axis=-1)  # (H,W,4)
        if verbose:
            print(f"\n  [Step 5] concat RGB + Depth → 4 通道")
            print(f"    NHWC: {rgbd_nhwc.shape}  dtype={rgbd_nhwc.dtype}")
            print(f"    通道顺序: [R, G, B, Depth]")

        # --- Step 6: NHWC → NCHW + batch dim ---
        rgbd_nchw = np.transpose(rgbd_nhwc, (2, 0, 1))      # (4, H, W)
        rgbd_tensor = torch.from_numpy(rgbd_nchw).unsqueeze(0)  # (1, 4, H, W)
        if verbose:
            print(f"\n  [Step 6] NHWC → NCHW + batch dim")
            print(f"    transpose:  {rgbd_nhwc.shape} → {rgbd_nchw.shape}")
            print(f"    unsqueeze:  {rgbd_nchw.shape} → {rgbd_tensor.shape}")
            print(f"    最终: {rgbd_tensor.shape}  {rgbd_tensor.dtype}  "
                  f"device={rgbd_tensor.device}")
            print(f"    内存连续: {rgbd_tensor.is_contiguous()}")
            print("=" * 60 + "\n")

        return rgbd_tensor


# ============================================================
# 3. 主循环: 取帧 → 预处理 → CNN 推理 → 显示
# ============================================================

def main():
    CAM_W, CAM_H = 640, 480
    CNN_SIZE = (224, 224)
    PRINT_EVERY = 60

    # ---- 初始化 D405 ----
    print("正在初始化 D405 ...")
    pipeline = rs.pipeline()
    config = rs.config()
    ctx = rs.context()
    dev = ctx.query_devices()[0]
    serial = dev.get_info(rs.camera_info.serial_number)
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, CAM_W, CAM_H, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, CAM_W, CAM_H, rs.format.z16, 30)

    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

    print(f"设备: {dev.get_info(rs.camera_info.name)} (SN: {serial})")
    print(f"深度比例: {depth_scale}  (raw × {depth_scale} = 米)")
    print(f"CNN 输入: {CNN_SIZE[1]}×{CNN_SIZE[0]}  4通道 NCHW\n")

    # ---- 初始化预处理 & CNN ----
    preprocessor = D405Preprocessor(target_size=CNN_SIZE, depth_norm="zscore")
    model = SimpleRGBDNet(in_channels=4, num_classes=10)
    model.eval()
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"\n{'-'*40}")
    print("  按键: q=退出  a=打印预处理流水线  p=打印模型结构")
    print(f"  每 {PRINT_EVERY} 帧自动打印一次排布")
    print(f"{'-'*40}\n")

    frame_cnt = 0

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color_np = np.asanyarray(color_frame.get_data())
            depth_np = np.asanyarray(depth_frame.get_data())
            frame_cnt += 1

            # ---- 预处理 (verbose=True 时打印每步排布) ----
            verbose = (frame_cnt == 1) or (frame_cnt % PRINT_EVERY == 0)
            rgbd_tensor = preprocessor.process(color_np, depth_np, depth_scale,
                                                verbose=verbose)

            # ---- CNN 前向 ----
            with torch.no_grad():
                output = model(rgbd_tensor)
                prob = F.softmax(output, dim=-1).squeeze(0)

            if verbose:
                print(f"  [CNN out] shape={output.shape}  "
                      f"range=[{output.min():.3f}, {output.max():.3f}]")
                print(f"  [Softmax] sum={prob.sum():.4f}  "
                      f"top3 indices: {prob.topk(3).indices.tolist()}  "
                      f"vals: {[f'{v:.3f}' for v in prob.topk(3).values.tolist()]}\n")

            # ---- 可视化 ----
            depth_viz = cv2.applyColorMap(
                (np.clip(depth_np.astype(np.float32) / 1500.0, 0, 1) * 255).astype(np.uint8),
                cv2.COLORMAP_JET
            )
            color_disp = cv2.resize(color_np, CNN_SIZE[::-1])
            depth_viz = cv2.resize(depth_viz, CNN_SIZE[::-1])

            # 叠加张量信息
            info = [
                f"RGB: {color_np.shape} {color_np.dtype}",
                f"Depth: {depth_np.shape} {depth_np.dtype}",
                f"CNN in: {tuple(rgbd_tensor.shape)} (N,C,H,W) {rgbd_tensor.dtype}",
                f"CNN out top3: {prob.topk(3).indices.tolist()}",
            ]
            for i, line in enumerate(info):
                cv2.putText(color_disp, line, (5, 18 + i * 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            combined = np.hstack((color_disp, depth_viz))
            cv2.imshow("D405 → CNN | Left: RGB  Right: Depth", combined)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('a'):
                _ = preprocessor.process(color_np, depth_np, depth_scale, verbose=True)
                with torch.no_grad():
                    out = model(rgbd_tensor)
                    print(f"  [CNN] output: {out.squeeze(0).tolist()}\n")
            elif key == ord('p'):
                print("\n" + str(model) + "\n")

    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("D405 已停止.")


if __name__ == "__main__":
    main()
