# pyright: reportMissingImports=false
import multiprocessing as mp
import os
import re
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

from config import (
    CAMERAS,
    CAMERA_FPS,
    CONTROL_DT,
    DATA_SAVE_DIR,
    EPISODE_IDX,
    HAND_CONTROLLER_TYPE,
    IMAGE_SIZE,
    TELEOP_EPISODE_PREFIX,
)
from joystick_process import joystick_process
from robot_interface import RM65Controller, create_hand_controller
from shared_memory import SharedData
from teleop_utils import map_grip_to_inspire, map_grip_to_l10


H, W = IMAGE_SIZE
CAPTURE_HZ = min(CAMERA_FPS, 15)
CAPTURE_DT = 1.0 / CAPTURE_HZ
USE_RS_ALIGN = False
RS_WARMUP_FRAMES = 15
ENABLE_WRIST_CAMERA = True
ENABLE_FRONT_CAMERA = True
SAVE_ACTUAL_STATES = True
INSPIRE_SEND_INTERVAL = 15


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

        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)
        except Exception as e:
            raise RuntimeError(f"{self.name} wait_for_frames 失败: {e}") from e

        if self.use_align:
            try:
                frames = self.align.process(frames)
            except Exception as e:
                raise RuntimeError(f"{self.name} align.process 失败: {e}") from e

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


def build_save_path(episode_idx):
    os.makedirs(DATA_SAVE_DIR, exist_ok=True)
    return os.path.join(
        DATA_SAVE_DIR,
        f"{TELEOP_EPISODE_PREFIX}{episode_idx}.hdf5",
    )


def get_next_episode_idx():
    os.makedirs(DATA_SAVE_DIR, exist_ok=True)
    pattern = re.compile(rf"^{re.escape(TELEOP_EPISODE_PREFIX)}(\d+)\.hdf5$")
    max_idx = EPISODE_IDX - 1
    for file_name in os.listdir(DATA_SAVE_DIR):
        match = pattern.match(file_name)
        if match:
            max_idx = max(max_idx, int(match.group(1)))
    return max_idx + 1


def create_hdf5_file(save_path, episode_idx, hand_joint_dim):
    with h5py.File(save_path, "w") as f:
        f.attrs["episode_idx"] = episode_idx
        f.attrs["hand_type"] = HAND_CONTROLLER_TYPE
        f.attrs["hand_joint_dim"] = hand_joint_dim
        f.attrs["image_height"] = H
        f.attrs["image_width"] = W
        f.attrs["camera_fps"] = CAMERA_FPS
        f.attrs["wrist_camera_name"] = "realsenseD305"
        f.attrs["wrist_camera_serial"] = CAMERAS["realsenseD305"]["serial_number"]
        f.attrs["front_camera_name"] = "realsenseD435"
        f.attrs["front_camera_serial"] = CAMERAS["realsenseD435"]["serial_number"]
        f.create_dataset("timestamps", (0,), maxshape=(None,), dtype="float64")

        f.create_dataset(
            "observations/arm_joints",
            (0, 6),
            maxshape=(None, 6),
            dtype="float64",
        )
        f.create_dataset(
            "observations/hand_joints",
            (0, hand_joint_dim),
            maxshape=(None, hand_joint_dim),
            dtype="int32",
        )
        f.create_dataset(
            "observations/images/wrist_RGB",
            (0, H, W, 3),
            maxshape=(None, H, W, 3),
            dtype="uint8",
        )
        f.create_dataset(
            "observations/images/front_RGB",
            (0, H, W, 3),
            maxshape=(None, H, W, 3),
            dtype="uint8",
        )
        f.create_dataset(
            "observations/depth/wrist_depth",
            (0, H, W),
            maxshape=(None, H, W),
            dtype="uint16",
        )
        f.create_dataset(
            "observations/depth/front_depth",
            (0, H, W),
            maxshape=(None, H, W),
            dtype="uint16",
        )


def create_episode_buffer():
    return {
        "timestamp": [],
        "arm_joints": [],
        "hand_joints": [],
        "wrist_rgb": [],
        "front_rgb": [],
        "wrist_depth": [],
        "front_depth": [],
    }


def get_buffer_frame_count(buffer):
    return len(buffer["timestamp"])


def flush_buffer(buffer, save_path):
    frame_count = get_buffer_frame_count(buffer)
    if frame_count <= 0:
        return

    timestamps = np.array(buffer["timestamp"], dtype=np.float64)
    arm_joints = np.stack(buffer["arm_joints"]).astype(np.float64)
    hand_joints = np.stack(buffer["hand_joints"]).astype(np.int32)
    wrist_rgb = np.stack(buffer["wrist_rgb"]).astype(np.uint8)
    front_rgb = np.stack(buffer["front_rgb"]).astype(np.uint8)
    wrist_depth = np.stack(buffer["wrist_depth"]).astype(np.uint16)
    front_depth = np.stack(buffer["front_depth"]).astype(np.uint16)

    with h5py.File(save_path, "a") as f:
        cur_len = f["observations/arm_joints"].shape[0]
        new_len = cur_len + frame_count
        hand_joint_dim = hand_joints.shape[1]

        f["timestamps"].resize((new_len,))
        f["observations/arm_joints"].resize((new_len, 6))
        f["observations/hand_joints"].resize((new_len, hand_joint_dim))
        f["observations/images/wrist_RGB"].resize((new_len, H, W, 3))
        f["observations/images/front_RGB"].resize((new_len, H, W, 3))
        f["observations/depth/wrist_depth"].resize((new_len, H, W))
        f["observations/depth/front_depth"].resize((new_len, H, W))

        f["timestamps"][cur_len:new_len] = timestamps
        f["observations/arm_joints"][cur_len:new_len] = arm_joints
        f["observations/hand_joints"][cur_len:new_len] = hand_joints
        f["observations/images/wrist_RGB"][cur_len:new_len] = wrist_rgb
        f["observations/images/front_RGB"][cur_len:new_len] = front_rgb
        f["observations/depth/wrist_depth"][cur_len:new_len] = wrist_depth
        f["observations/depth/front_depth"][cur_len:new_len] = front_depth


def finalize_episode(buffer, save_path, episode_idx):
    if save_path is None:
        return

    frame_count = get_buffer_frame_count(buffer)
    if frame_count > 0:
        hand_joint_dim = len(buffer["hand_joints"][0])
        create_hdf5_file(save_path, episode_idx, hand_joint_dim)
        flush_buffer(buffer, save_path)
        for values in buffer.values():
            values.clear()

    if frame_count <= 0:
        if os.path.exists(save_path):
            os.remove(save_path)
        print("当前 episode 未采集到数据，未生成 HDF5 文件。")
    else:
        print(f"当前 episode 采集完成，共 {frame_count} 帧，已保存到: {save_path}")


def request_exit(shared):
    with shared.lock:
        shared.exit_flag[0] = True


def safe_get_arm_joints(arm, fallback):
    try:
        return np.array(arm.get_actual_joints(), dtype=np.float64)
    except Exception as e:
        print(f"读取机械臂实际关节失败，回退到目标关节: {e}")
        return np.array(fallback, dtype=np.float64)


def safe_get_hand_joints(hand, fallback):
    try:
        state = hand.get_state()
        expected_dim = len(fallback)
        if state is None or len(state) != expected_dim:
            raise ValueError(f"返回状态长度异常: {state}")
        return np.array(state, dtype=np.int32)
    except Exception as e:
        print(f"读取灵巧手状态失败，回退到控制指令: {e}")
        return np.array(fallback, dtype=np.int32)


def control_capture_process(shm_name, record_event, current_episode):
    shared = SharedData(create=False, existing_shm_name=shm_name)
    hand_type = HAND_CONTROLLER_TYPE.strip().lower()
    if hand_type == "l10":
        hand_mapper = map_grip_to_l10
    elif hand_type == "inspire":
        hand_mapper = map_grip_to_inspire
    else:
        print(f"不支持的灵巧手类型: {HAND_CONTROLLER_TYPE}")
        request_exit(shared)
        return

    arm = RM65Controller(sim=False)
    if not arm.connect():
        print("机械臂连接失败，control_capture_process 退出")
        request_exit(shared)
        return

    try:
        hand = create_hand_controller(arm, hand_type=hand_type)
    except Exception as e:
        print(f"灵巧手初始化失败: {e}")
        arm.disconnect()
        request_exit(shared)
        return

    wrist_camera = RealSenseCamera(
        name="wrist_RGB",
        serial_number=CAMERAS["realsenseD305"]["serial_number"],
        fps=CAMERAS["realsenseD305"].get("fps", CAMERA_FPS),
        use_align=USE_RS_ALIGN,
    ) if ENABLE_WRIST_CAMERA else None
    front_camera = RealSenseCamera(
        name="front_RGB",
        serial_number=CAMERAS["realsenseD435"]["serial_number"],
        fps=CAMERAS["realsenseD435"].get("fps", CAMERA_FPS),
        use_align=USE_RS_ALIGN,
    ) if ENABLE_FRONT_CAMERA else None

    empty_rgb = np.zeros((H, W, 3), dtype=np.uint8)
    empty_depth = np.zeros((H, W), dtype=np.uint16)

    last_time = time.time()
    was_recording = False
    recording_episode_idx = None
    episode_buffer = create_episode_buffer()
    current_save_path = None
    cameras_started = False
    next_capture_time = 0.0
    captured_frames = 0
    hand_send_counter = 0

    def start_cameras():
        nonlocal cameras_started, next_capture_time
        if cameras_started:
            return
        if wrist_camera is not None:
            wrist_camera.start()
        if front_camera is not None:
            front_camera.start()
        cameras_started = True
        next_capture_time = time.time() + CAPTURE_DT

    def stop_cameras():
        nonlocal cameras_started
        if not cameras_started:
            return
        if wrist_camera is not None:
            wrist_camera.stop()
        if front_camera is not None:
            front_camera.stop()
        cameras_started = False

    try:
        while True:
            with shared.lock:
                target_joints = shared.joints.copy()
                target_grip = shared.grip[0]
                exit_flag = shared.exit_flag[0]

            if exit_flag:
                break

            arm.move_canfd(target_joints)
            hand_target = hand_mapper(target_grip)
            hand_send_counter += 1
            if hand_type != "inspire" or hand_send_counter % INSPIRE_SEND_INTERVAL == 0:
                hand.move_fingers(hand_target)

            is_recording = record_event.is_set()
            if is_recording and not was_recording:
                if not cameras_started:
                    try:
                        start_cameras()
                    except Exception as e:
                        print(f"相机启动失败，无法开始录制: {e}")
                        record_event.clear()
                        elapsed = time.time() - last_time
                        if elapsed < CONTROL_DT:
                            time.sleep(CONTROL_DT - elapsed)
                        last_time = time.time()
                        continue
                recording_episode_idx = current_episode.value
                current_save_path = build_save_path(recording_episode_idx)
                for values in episode_buffer.values():
                    values.clear()
                captured_frames = 0
                print(f"开始录制 episode {recording_episode_idx} -> {current_save_path}")
                was_recording = True
            elif not is_recording and was_recording:
                finalize_episode(episode_buffer, current_save_path, recording_episode_idx)
                was_recording = False
                recording_episode_idx = None
                current_save_path = None

            if not is_recording:
                elapsed = time.time() - last_time
                if elapsed < CONTROL_DT:
                    time.sleep(CONTROL_DT - elapsed)
                last_time = time.time()
                continue

            now = time.time()
            if now < next_capture_time:
                elapsed = now - last_time
                if elapsed < CONTROL_DT:
                    time.sleep(CONTROL_DT - elapsed)
                last_time = time.time()
                continue

            if wrist_camera is not None:
                try:
                    wrist_rgb, wrist_depth = wrist_camera.read()
                except Exception as e:
                    print(f"wrist_RGB 采集失败，本次采样跳过: {e}")
                    next_capture_time = time.time() + CAPTURE_DT
                    elapsed = time.time() - last_time
                    if elapsed < CONTROL_DT:
                        time.sleep(CONTROL_DT - elapsed)
                    last_time = time.time()
                    continue
            else:
                wrist_rgb = empty_rgb.copy()
                wrist_depth = empty_depth.copy()

            if front_camera is not None:
                try:
                    front_rgb, front_depth = front_camera.read()
                except Exception as e:
                    print(f"front_RGB 采集失败，本次采样跳过: {e}")
                    next_capture_time = time.time() + CAPTURE_DT
                    elapsed = time.time() - last_time
                    if elapsed < CONTROL_DT:
                        time.sleep(CONTROL_DT - elapsed)
                    last_time = time.time()
                    continue
            else:
                front_rgb = empty_rgb.copy()
                front_depth = empty_depth.copy()

            if SAVE_ACTUAL_STATES:
                arm_joints_to_save = safe_get_arm_joints(arm, target_joints)
                hand_joints_to_save = safe_get_hand_joints(hand, hand_target)
            else:
                arm_joints_to_save = np.array(target_joints, dtype=np.float64)
                hand_joints_to_save = np.array(hand_target, dtype=np.int32)
            record = {
                "timestamp": time.time(),
                "arm_joints": arm_joints_to_save,
                "hand_joints": hand_joints_to_save,
                "wrist_rgb": wrist_rgb,
                "front_rgb": front_rgb,
                "wrist_depth": wrist_depth,
                "front_depth": front_depth,
            }
            for key, value in record.items():
                episode_buffer[key].append(value)
            captured_frames += 1

            next_capture_time = time.time() + CAPTURE_DT
            elapsed = time.time() - last_time
            if elapsed < CONTROL_DT:
                time.sleep(CONTROL_DT - elapsed)
            last_time = time.time()
    finally:
        if was_recording:
            finalize_episode(episode_buffer, current_save_path, recording_episode_idx)
        stop_cameras()
        arm.disconnect()


def read_key_nonblocking(fd, timeout=0.1):
    readable, _, _ = select.select([fd], [], [], timeout)
    if readable:
        return sys.stdin.read(1).lower()
    return None


def main():
    start_episode_idx = get_next_episode_idx()
    shared = SharedData(create=True)
    shm_name = shared.get_shm_name()
    record_event = mp.Event()
    current_episode = mp.Value("i", start_episode_idx)

    p_joy = mp.Process(target=joystick_process, args=(shm_name,))
    p_ctrl = mp.Process(
        target=control_capture_process,
        args=(shm_name, record_event, current_episode),
    )

    p_joy.start()
    p_ctrl.start()
    cleaned_up = False

    def cleanup_and_exit():
        nonlocal cleaned_up
        if cleaned_up:
            return
        cleaned_up = True

        record_event.clear()
        request_exit(shared)

        p_joy.join(timeout=2)
        p_ctrl.join(timeout=2)
        for process in [p_joy, p_ctrl]:
            if process.is_alive():
                process.terminate()

        shared.close()
        shared.unlink()

    def signal_handler(*_):
        print("\n收到退出信号，准备停止遥操作并保存数据...")
        cleanup_and_exit()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print(
        f"程序已启动。保存目录: {DATA_SAVE_DIR}。"
        f"下一个 episode 编号: {start_episode_idx}。"
        "每个 episode 均按 R 开始采集，按 S 结束当前采集并保存。按 Ctrl+C 退出。"
    )

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    is_recording = False
    next_episode_idx = start_episode_idx

    try:
        tty.setcbreak(fd)
        while True:
            if not p_joy.is_alive() or not p_ctrl.is_alive():
                print("检测到子进程退出，主进程准备结束。")
                break

            key = read_key_nonblocking(fd, timeout=0.1)
            if key is None:
                continue

            if key == "r":
                if is_recording:
                    print("当前正在录制，R 键已忽略。")
                    continue
                with current_episode.get_lock():
                    current_episode.value = next_episode_idx
                record_event.set()
                is_recording = True
                print(f"开始录制 episode {next_episode_idx}")
                continue

            if key == "s":
                if is_recording:
                    record_event.clear()
                    is_recording = False
                    print(f"停止录制 episode {next_episode_idx}，正在保存...")
                    next_episode_idx += 1
                else:
                    print("当前未在录制中，请按 R 开始新的 episode。")
                continue
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        cleanup_and_exit()
        print("遥操作数据采集已结束。")


if __name__ == "__main__":
    main()
