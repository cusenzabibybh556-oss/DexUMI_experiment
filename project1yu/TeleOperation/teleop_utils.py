# utils.py
import numpy as np
import math
from config import INITIAL_ANGLES, COEFFICIENTS

def transform_joystick_angles(angles_rad):
    """
    将操纵杆读到的弧度值转换为机械臂目标关节角度（度）
    """
    angles_deg = [a * 180.0 / math.pi for a in angles_rad]
    calib = [angles_deg[i] - INITIAL_ANGLES[i] for i in range(6)]
    target = [calib[i] * COEFFICIENTS[i] for i in range(6)]
    return target

def map_grip_to_l10(grip_value):
    """
    将夹爪原始值（0~1000）线性映射到 L10 的10个手指角度（0~255）。
    - grip_value = 0   -> lower_pose
    - grip_value = 1000 -> initial_pose
    - 第2个值（Thumb_Yaw）使用开关式映射：
      grip_value != 1000 时直接到达目标位，grip_value == 1000 时回到初始位
    """
    lower_pose = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    initial_pose = [250, 250, 220, 220, 220, 220, 250, 250, 250, 250]

    grip_value = max(0, min(1000, int(grip_value)))
    ratio = grip_value / 1000.0

    target = []
    for idx, (low, high) in enumerate(zip(lower_pose, initial_pose)):
        if idx == 1:
            # target.append(high if grip_value == 1000 else low)
            target.append(0)
        elif idx == 9:
            target.append(200)
        else:
            target.append(int(round(low + (high - low) * ratio)))
    return target
    
def map_grip_to_inspire(grip_value):
    """
    将夹爪原始值（0~1000）映射到 Inspire 的 6 维手指角度（0~1000）。
    当前默认姿态参考 robotic_move.py 中的控制方式。
    """
    lower_pose = [0, 0, 0, 0, 0, 0]
    initial_pose = [1000, 1000, 1000, 1000, 1000, 0]

    grip_value = max(0, min(1000, int(grip_value)))
    ratio = grip_value / 1000.0

    target = []
    for low, high in zip(lower_pose, initial_pose):
        target.append(int(round(low + (high - low) * ratio)))
    return target

def joint_diff(target, current):
    """计算关节角度差（度）"""
    return np.array(target) - np.array(current)

def pose_follow(q_out, c_q):
    """角度限幅跟随算法"""
    tar_q = np.array(q_out)
    index = np.abs(joint_diff(q_out, c_q)) >= 10
    tar_q[index] = (np.array(c_q) + 9.5 * (joint_diff(q_out, c_q) / np.abs(joint_diff(q_out, c_q))))[index]
    return tar_q

def canfd_judge(target_angle, current_q):
    """判断角度差是否在透传允许范围内"""
    return np.max(joint_diff(target_angle, current_q)) < 10
