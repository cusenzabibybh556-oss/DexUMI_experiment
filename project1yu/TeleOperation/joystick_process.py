import numpy as np
import time
from alicia_d_sdk import create_robot
from shared_memory import SharedData   # 导入自定义类
from teleop_utils import transform_joystick_angles
from config import JOYSTICK_PORT, JOYSTICK_DT

def joystick_process(shm_name: str):
    shared = SharedData(create=False, existing_shm_name=shm_name)
    # 连接操纵杆
    robot = create_robot(port=JOYSTICK_PORT)
    if not robot.is_connected():
        print("操纵杆连接失败")
        return

    last_grip_target = 1000

    while True:
        # 1. 读取关节角度（弧度）
        try:
            angles_rad = robot.get_robot_state(info_type="joint", timeout=0.01, cache=True)
        except Exception as e:
            print(f"操纵杆读取关节失败: {e}")
            time.sleep(0.01)
            continue

        if angles_rad is None:
            time.sleep(0.01)
            continue

        # 2. 弧度转角度并转换
        try:
            target_joints = transform_joystick_angles(angles_rad)
        except Exception as e:
            print(f"操纵杆角度转换失败: {e}")
            time.sleep(0.01)
            continue

        # 3. 读取夹爪力/开度（假设返回值 0~1000，1000=张开）
        try:
            grip_raw = robot.get_robot_state(info_type="gripper", timeout=0.01, cache=True)
        except Exception as e:
            print(f"操纵杆读取夹爪失败: {e}")
            grip_raw = None

        if grip_raw is None:
            grip_target = last_grip_target
        else:
            grip_target = int(grip_raw)
            last_grip_target = grip_target

        # 4. 加锁写入共享内存
        with shared.lock:
            shared.joints[:] = target_joints
            shared.grip[0] = grip_target
            if shared.exit_flag[0]:
                break

        # 5. 睡眠控制循环频率（例如 500Hz = 0.002秒）
        time.sleep(0.002)

    robot.disconnect()