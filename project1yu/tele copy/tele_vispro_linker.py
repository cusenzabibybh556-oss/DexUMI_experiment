import os
import argparse
import numpy as np
import cv2
from avp_stream import VisionProStreamer
from pytransform3d import rotations

from loguru import logger

from Robotic_Arm.rm_robot_interface import *
from Robot_control.pos_process import *
from Robot_control.robotic_move_linker import *
from Robot_control.retargeting import *
from Robot_control.data_process import *
from Robot_control.linker_move import *
from constant import *
from hand_detector import *

import multiprocessing
from multiprocessing import Array, Process, shared_memory, Manager, Event, Semaphore

def main(episode_idx):
    robot_manager = Manager()
    # 共享字典 用于控制机械臂和灵巧手 (主进程更新字典后，子进程也能使用更新后的数据)
    robotic_data = robot_manager.dict()
    robotic_data['pose'] = None
    robotic_data['hand_row'] = None
    robotic_data['hand_qpos'] = None
    robotic_data['hand_write'] = None
    robotic_data['EXIT'] = False
    robotic_data['sim'] = False
    
    robot_name = RobotName.linkerl21

    """===============================机械臂控制子进程================================="""
    Process_robot = Process(target=Robotic_process_with_linker, args = (robotic_data, robot_name, episode_idx, 0, 'vr_right', 'visionpro'))
    Process_robot.daemon = True
    arm_start_time = time.time()
    Process_robot.start()

    """===============================灵巧手控制子进程================================="""
    if robot_name == RobotName.linkerl21:
        Process_hand = Process(target=Linker_Hand_Process, args=(robotic_data,))
        Process_hand.daemon = True
        Process_hand.start()

    """===============================从visionpro读取数据并进行转换================================="""   
     
    s = VisionProStreamer("10.126.230.168")     # 连接visionpro获取数据流
# 初始化GeoRT模型 (主进程)
    geort = GeoRTInference(
        model_path="./geo_retargeting/trained_geo_retargeting_model/linker_right/geort_model_final.pth",
        urdf_path="./geo_retargeting/robot/linker21_right/linkerhand_l21_right.urdf"
    )
    visionpro_hand_type = "Right"  # 假设我们只用右手

    # 实例化转换器 (用于 radians -> 0-255 )
    # 注意: 我们不需要连接硬件，所以 can_interface 可以随意填
    linker_converter = LinkerHandClient(hand_joint="L21", hand_type="right")
    # 从 geort 获取关节限制并传给转换器
    joint_limits = {finger: config["joint_limits"] for finger, config in geort.hand_config.items()}
    linker_converter.joint_limits = joint_limits

    last_right_mat = []
    try:
        while True:
            latest = s.latest
            # print(latest['right_fingers'][:,:3,3])      # 测试是否成功连接visionpro并接收到数据

            if latest is not None:
                right_wrist_mat=latest['right_wrist'][0]            # 手腕位姿矩阵（4*4）

                if right_wrist_mat is not None:    
                    right_wrist_pose= np.concatenate([right_wrist_mat[:3, -1] + np.array([0, 0, -0.8]),
                                        rotations.quaternion_from_matrix(right_wrist_mat[:3, :3])])   # 将手腕末端位姿矩阵转换为pose向量
                
                right_hand_mat=latest['right_fingers']     # 手指位姿
                robotic_data['pose'] = right_wrist_pose.copy()
                robotic_data['hand_row'] = right_hand_mat
                
                # ===== GeoRT 模型推理 (主进程执行) =====
                if right_hand_mat is not None:
                    # 坐标变换
                    fingertip_positions = extract_fingertip_positions(right_hand_mat, hand_type=visionpro_hand_type)
                    # 推理
                    robot_qpos, _ = geort.retarget(fingertip_positions)
                    # 转换: radians -> 0-255 ints
                    linker_values = linker_converter.radians_to_linker_values(robot_qpos)
                    # 存入共享内存 (直接存最终数值)
                    robotic_data['hand_qpos'] = linker_values
                robotic_data['hand_row'] = right_hand_mat

            if robotic_data['EXIT']:
                print('EXIT')
                break

    finally:
        Process_robot.join()
        if 'Process_hand' in locals() and Process_hand.is_alive():
             Process_hand.join()
        print('ok')
        exit(0)


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    main(0)

