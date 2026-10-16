# config.py
import os
# 机械臂配置
ARM_IP = "192.168.1.18"
ARM_PORT = 8080
ARM_WORK_FRAME = "RM65"
ARM_TOOL_FRAME = "Arm_Tip"
SIM_MODE = False

# 操纵杆配置
JOYSTICK_PORT = "/dev/ttyACM0"          # Windows下为COMx，Linux下可能为 /dev/ttyUSB0
JOYSTICK_BAUDRATE = 115200       # 根据你的SDK要求
JOYSTICK_HZ = 500                # 读取频率 (Hz)
JOYSTICK_DT = 1.0 / JOYSTICK_HZ

# 控制进程配置
CONTROL_HZ = 100                 # 机械臂+手控制频率
CONTROL_DT = 1.0 / CONTROL_HZ

# 相机配置
IMAGE_SIZE = [480, 640]          # [H, W]
CAMERA_FPS = 30
CAMERAS = {
    "realsenseD305": {
        "type": "RGBD",
        "serial_number": "260322279022",
        "fps": CAMERA_FPS,
    },
    "realsenseD435": {
        "type": "RGBD",
        "serial_number": "216322073710",
        "fps": CAMERA_FPS,
    },
}

# 灵巧手L10配置
# HAND_CONTROLLER_TYPE = "l10"  # 可选: "l10" / "inspire"
HAND_CONTROLLER_TYPE = "l10"

# L10 灵巧手配置
L10_CAN_INTERFACE = "can0"
L10_HAND_TYPE = "right"
L10_DEFAULT_SPEEDS = [250, 250, 250, 250, 250, 255, 255, 255, 255, 255]  # 拇指先动可在此设大拇指速度快

# Inspire 灵巧手配置
INSPIRE_HAND_DEFAULT_POSE = [0, 0, 1000, 1000, 1000, 0]
INSPIRE_HAND_INIT_BAUDRATE = 115200
INSPIRE_HAND_INIT_TIMEOUT = 2
INSPIRE_HAND_VOLTAGE = 3
INSPIRE_HAND_DEFAULT_SPEED_FORCE = [2, 232, 2, 232, 2, 232, 2, 232, 2, 232, 2, 232]

# 数据存储配置
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_DIR = os.path.dirname(PROJECT_DIR)
# DATA_SAVE_DIR = f"/mnt/1c714c0e-2ec6-4b54-b86f-8ab77eaea676/study/DexGraspVLA/data/Datas/{HAND_CONTROLLER_TYPE}"
DATA_SAVE_DIR = f"/mnt/1c714c0e-2ec6-4b54-b86f-8ab77eaea676/study/data/robot_data/Orange_in_bawl/{HAND_CONTROLLER_TYPE}"
TELEOP_EPISODE_PREFIX = "teleop_episode_"
EPISODE_IDX = 0                  # 无历史数据时的起始编号；如脚本支持扫描目录，可从已有最大编号后继续
QUEUE_MAXSIZE = 500              # 数据队列最大长度，防止内存爆满
HDF5_BUFFER_SIZE = 100           # 每积累多少条数据写入一次HDF5

# 角度转换参数（从你的测试代码迁移）
INITIAL_ANGLES = [0, 0, 132, 0, 0, 0]
COEFFICIENTS = [1, -1, -1, -1, -1, -1]
