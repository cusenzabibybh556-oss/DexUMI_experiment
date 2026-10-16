# pyright: reportMissingImports=false
import os
import select
import signal
import sys
import termios
import time
import tty

import h5py
import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None

from config import CAMERAS, CAMERA_FPS, DATA_SAVE_DIR, EPISODE_IDX, IMAGE_SIZE


H, W = IMAGE_SIZE
CAPTURE_HZ = min(CAMERA_FPS, 15)
CAPTURE_DT = 1.0 / CAPTURE_HZ
USE_RS_ALIGN = False
RS_WARMUP_FRAMES = 15


class RealSenseCamera:
    def __init__(self, name, serial_number, fps, use_align=USE_RS_ALIGN, warmup_frames=RS_WARMUP_FRAMES):
        self.name = name
        self.serial_number = serial_number
        self.fps = fps
        self.use_align = use_align
        self.warmup_frames = warmup_frames
        self.pipeline = None
        self.align = None

    def start(self):
        if rs is None:
            raise ImportError("缺少 pyrealsense2，请先安装 RealSense Python SDK")

        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(self.serial_number)
        config.enable_stream(rs.stream.depth, W, H, rs.format.z16, self.fps)
        config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, self.fps)
        self.pipeline.start(config)
        self.align = rs.align(rs.stream.color) if self.use_align else None
        print(
            f"{self.name} 已启动，序列号: {self.serial_number}，"
            f"align={'on' if self.use_align else 'off'}，warmup={self.warmup_frames}"
        )
        self.warmup()

    def warmup(self):
        if self.pipeline is None:
            raise RuntimeError(f"{self.name} 尚未启动，无法进行预热")

        for _ in range(self.warmup_frames):
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)
            if self.use_align:
                frames = self.align.process(frames)
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                raise RuntimeError(f"{self.name} 预热阶段获取图像帧失败")
        print(f"{self.name} 预热完成")

    def read(self):
        if self.pipeline is None:
            raise RuntimeError(f"{self.name} 尚未启动")

        frames = self.pipeline.wait_for_frames(timeout_ms=1000)
        if self.use_align:
            frames = self.align.process(frames)

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            raise RuntimeError(f"{self.name} 获取图像帧失败")

        color_bgr = np.asanyarray(color_frame.get_data())
        color_rgb = np.ascontiguousarray(color_bgr[:, :, ::-1])
        depth = np.ascontiguousarray(np.asanyarray(depth_frame.get_data()))
        return color_rgb, depth

    def stop(self):
        if self.pipeline is not None:
            self.pipeline.stop()
            self.pipeline = None


def create_episode_buffer():
    return {
        "timestamp": [],
        "wrist_rgb": [],
        "front_rgb": [],
        "wrist_depth": [],
        "front_depth": [],
    }


def get_buffer_frame_count(buffer):
    return len(buffer["timestamp"])


def build_save_path(episode_idx):
    os.makedirs(DATA_SAVE_DIR, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return os.path.join(DATA_SAVE_DIR, f"dual_camera_episode_{episode_idx}_{timestamp}.hdf5")


def save_episode(buffer, save_path, episode_idx):
    frame_count = get_buffer_frame_count(buffer)
    if frame_count <= 0:
        print("当前 episode 未采集到图像，未生成 HDF5 文件。")
        return

    timestamps = np.array(buffer["timestamp"], dtype=np.float64)
    wrist_rgb = np.stack(buffer["wrist_rgb"]).astype(np.uint8)
    front_rgb = np.stack(buffer["front_rgb"]).astype(np.uint8)
    wrist_depth = np.stack(buffer["wrist_depth"]).astype(np.uint16)
    front_depth = np.stack(buffer["front_depth"]).astype(np.uint16)

    with h5py.File(save_path, "w") as f:
        f.attrs["episode_idx"] = episode_idx
        f.attrs["image_height"] = H
        f.attrs["image_width"] = W
        f.attrs["camera_fps"] = CAMERA_FPS
        f.attrs["wrist_camera_name"] = "realsenseD305"
        f.attrs["wrist_camera_serial"] = CAMERAS["realsenseD305"]["serial_number"]
        f.attrs["front_camera_name"] = "realsenseD435"
        f.attrs["front_camera_serial"] = CAMERAS["realsenseD435"]["serial_number"]

        f.create_dataset("timestamps", data=timestamps, dtype="float64")

        obs = f.create_group("observations")
        images = obs.create_group("images")
        depth = obs.create_group("depth")

        images.create_dataset("wrist_RGB", data=wrist_rgb, dtype="uint8")
        images.create_dataset("front_RGB", data=front_rgb, dtype="uint8")
        depth.create_dataset("wrist_depth", data=wrist_depth, dtype="uint16")
        depth.create_dataset("front_depth", data=front_depth, dtype="uint16")

    print(f"当前 episode 采集完成，共 {frame_count} 帧，已保存到: {save_path}")


def read_key_nonblocking(fd, timeout=0.01):
    readable, _, _ = select.select([fd], [], [], timeout)
    if readable:
        return sys.stdin.read(1).lower()
    return None


def main():
    wrist_camera = RealSenseCamera(
        name="wrist_RGB",
        serial_number=CAMERAS["realsenseD305"]["serial_number"],
        fps=CAMERAS["realsenseD305"].get("fps", CAMERA_FPS),
        use_align=USE_RS_ALIGN,
    )
    front_camera = RealSenseCamera(
        name="front_RGB",
        serial_number=CAMERAS["realsenseD435"]["serial_number"],
        fps=CAMERAS["realsenseD435"].get("fps", CAMERA_FPS),
        use_align=USE_RS_ALIGN,
    )

    print("正在启动双相机...")
    wrist_camera.start()
    front_camera.start()

    episode_buffer = create_episode_buffer()
    current_episode_idx = EPISODE_IDX
    current_save_path = None
    is_recording = False
    next_capture_time = time.time()
    cleaned_up = False

    def finalize_current_episode():
        nonlocal episode_buffer, current_save_path, is_recording, current_episode_idx
        if not is_recording or current_save_path is None:
            return
        save_episode(episode_buffer, current_save_path, current_episode_idx)
        episode_buffer = create_episode_buffer()
        current_save_path = None
        is_recording = False
        current_episode_idx += 1

    def cleanup_and_exit():
        nonlocal cleaned_up
        if cleaned_up:
            return
        cleaned_up = True
        if is_recording:
            print("\n收到退出信号，正在保存当前 episode...")
            finalize_current_episode()
        wrist_camera.stop()
        front_camera.stop()

    def signal_handler(*_):
        cleanup_and_exit()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("双相机测试已启动。按 R 开始录制，按 S 结束当前录制并保存。按 Ctrl+C 退出。")

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    try:
        tty.setcbreak(fd)
        while True:
            key = read_key_nonblocking(fd, timeout=0.01)
            if key == "r":
                if is_recording:
                    print("当前正在录制，R 键已忽略。")
                else:
                    episode_buffer = create_episode_buffer()
                    current_save_path = build_save_path(current_episode_idx)
                    is_recording = True
                    next_capture_time = time.time()
                    print(f"开始录制 episode {current_episode_idx} -> {current_save_path}")
                continue

            if key == "s":
                if is_recording:
                    print(f"停止录制 episode {current_episode_idx}，正在保存...")
                    finalize_current_episode()
                else:
                    print("当前未在录制中，请按 R 开始新的 episode。")
                continue

            if not is_recording:
                time.sleep(0.01)
                continue

            now = time.time()
            if now < next_capture_time:
                time.sleep(min(0.005, next_capture_time - now))
                continue

            try:
                wrist_rgb, wrist_depth = wrist_camera.read()
                front_rgb, front_depth = front_camera.read()
            except Exception as e:
                print(f"双相机采集失败，本次采样跳过: {e}")
                next_capture_time = time.time() + CAPTURE_DT
                continue

            episode_buffer["timestamp"].append(time.time())
            episode_buffer["wrist_rgb"].append(wrist_rgb)
            episode_buffer["front_rgb"].append(front_rgb)
            episode_buffer["wrist_depth"].append(wrist_depth)
            episode_buffer["front_depth"].append(front_depth)
            next_capture_time = time.time() + CAPTURE_DT
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        cleanup_and_exit()
        print("双相机测试已结束。")


if __name__ == "__main__":
    main()
