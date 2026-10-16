import math
import numpy as np
import torch
import pyrealsense2 as rs
import cv2
import json

from Robotic_Arm.rm_robot_interface import *

import numpy as np
import math

import time

def load_file(file_path):
    result = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip():  # 确保不是空行
                data = json.loads(line)  # 解析 JSON 格式
                point = data.get("point", [])  # 获取 "point" 数据
                if point:
                    result.append([x / 1000 for x in point])  # 将每个数除以1000
    return result


def test():


    arm_model = rm_robot_arm_model_e.RM_MODEL_RM_65_E  # RM_65机械臂
    force_type = rm_force_type_e.RM_MODEL_RM_B_E  # 标准版
    # 初始化算法的机械臂及末端型号
    algo_handle = Algo(arm_model, force_type)

    # 实例化RoboticArm类
    left_arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
    # 创建机械臂连接，打印连接id
    handle = left_arm.rm_create_robot_arm("192.168.1.18", 8080)
    print('handle id:', handle.id)
    # print(left_arm.rm_get_total_work_frame())
    print('Coordinate System:', left_arm.rm_change_work_frame("RM65"))

    # 灵巧手通讯初始化
    left_arm.rm_set_modbus_mode(1, 115200, 2)
    left_arm.rm_set_voltage(3)

    left_arm.rm_set_arm_run_mode(1)


    tr1 = load_file('traj/imitation01.txt')
    tr2 = load_file('traj/imitation02.txt')
    tr3 = load_file('traj/imitation03.txt')
    tr4 = load_file('traj/imitation04.txt')

    # 机械臂初始位置
    arm_state_ori = [0, -35, 95, 0, 88, 0]
    left_arm.rm_movej(arm_state_ori, 30, 0, 0, 1)

    # time.sleep(1)

    hand0 = [3, 232, 3, 232, 3, 232, 3, 232, 3, 232, 0, 0]
    hand1 = [3, 232, 3, 232, 1, 232, 1, 232, 0, 232, 0, 0]


    left_arm.rm_write_registers(rm_peripheral_read_write_params_t(1, 1486, 1, 6), hand0)

    time.sleep(2)

    for i in range(len(tr3)):
        left_arm.rm_movej_canfd(tr3[i], follow=True)
        time.sleep(0.005)

    # print(rm_peripheral_read_write_params_t(1, 1486, 2, 6))
    print(left_arm.rm_write_registers(rm_peripheral_read_write_params_t(1, 1486, 1, 6), hand1))

    time.sleep(0.25)
    for i in range(len(tr4)):
        left_arm.rm_movej_canfd(tr4[i], follow=True)
        time.sleep(0.005)

if __name__ == '__main__':
    test()