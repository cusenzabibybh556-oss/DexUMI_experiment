import sys
sys.path.append("/home/user-iip/TeleOperation_DataCollection/Robot_control")
# from Robotic_Arm.rm_ctypes_wrap
from Robotic_Arm.rm_robot_interface import *
from constant import *

import numpy as np
import math

import h5py
import time
# import keyboard

from dex_retargeting.constants import RobotName, RetargetingType, HandType, get_default_config_path

from linker_hand_python_sdk.LinkerHand.linker_hand_api import LinkerHandApi

from data_process import *
from pos_process import *
from detect_process import *
from robotic_pathplan import *

'''=================================机械臂相关函数==========================='''
# 位姿跟随
def PoseFollow(q_out, c_q):
    tar_q = np.array(q_out)
    index = np.abs(JointDiff(q_out, c_q))>=10
    tar_q[index] = (np.array(c_q) + 9.5*(JointDiff(q_out, c_q)/np.abs(JointDiff(q_out, c_q))))[index]
    return tar_q

# 计算机械臂各关节角度差
def JointDiff(target_angle, current_q):
    a = np.array(target_angle)
    b = np.array(current_q)
    return  a-b

# 机械臂透传移动
def Roboric_move_CANFD(arm, algo_handle, arm_pose, euler = 0):
    current_q = arm.rm_get_joint_degree()
    params = rm_inverse_kinematics_params_t(current_q[1], arm_pose, euler)
    q_out = algo_handle.rm_algo_inverse_kinematics(params)
    if q_out[0]==0:
        if CANFD_judge(q_out[1], current_q[1]):
            arm.rm_movej_canfd(q_out[1], False)
            return True, np.array(q_out[1])
        else:
            tar_q = PoseFollow(q_out[1], current_q[1])
            arm.rm_movej_canfd(tar_q, False)
            return True, np.array(tar_q)
    else:
        print(f'Inverse_kinematics:Failed,q_out[0]={q_out[0]}')
        return False, 0

# 判断数据是否符合透传要求
def CANFD_judge(target_angle, current_q):
    return np.max(JointDiff(target_angle, current_q)) < 10

# 获取当前的机器人状态（臂+手）
def get_qpos(arm, hand_data):   
    '''
    arm.rm_get_joint_degree()  从机械臂获取机械臂的关节角度
    hand data    
    '''
    return np.concatenate([np.array(arm.rm_get_joint_degree()[1]),hand_data])

'''=================================机械臂执行进程（附加灵巧手）==========================='''
def Robotic_process_with_linker(robotic_data, robot_name, episode_idx, euler = 0, tool_frame = 'vr-right', work_frame = 'visionpro'):
    """----------------------摄像头初始化----------------------"""
     # 注册数据流，并对齐图像，将深度图与RGB图对齐
    align = rs.align(rs.stream.color)
    rs_config = rs.config()
    rs_config.enable_stream(rs.stream.depth, IMAGE_SIZE['width'], IMAGE_SIZE['height'], rs.format.z16, IMAGE_FPS)
    rs_config.enable_stream(rs.stream.color, IMAGE_SIZE['width'], IMAGE_SIZE['height'], rs.format.bgr8, IMAGE_FPS)

    pipeline1 = rs.pipeline()
    rs_config.enable_device(CAMERA_ID[0])
    # pipeline_profile1 = pipeline1.start(rs_config)
    pipeline1.start(rs_config)

    """----------------------机械臂初始化----------------------"""
    arm_model = rm_robot_arm_model_e.RM_MODEL_RM_65_E  # RM_65机械臂
    force_type = rm_force_type_e.RM_MODEL_RM_B_E  # 标准版
    # 初始化算法的机械臂及末端型号
    algo_handle = Algo(arm_model, force_type)
    # 实例化RoboticArm类
    right_arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
    # 创建机械臂连接，打印连接id
    handle = right_arm.rm_create_robot_arm("192.168.1.18", 8080)
    print('handle id:', handle.id)

    # 设置机械臂工作和工具坐标系
    print(f'Coordinate System_work_frame:{work_frame}', right_arm.rm_change_work_frame(work_frame))
    print(f'Coordinate System_tool_frame:{tool_frame}', right_arm.rm_change_tool_frame(tool_frame))

    # Sim or not
    if robotic_data['sim']:
        right_arm.rm_set_arm_run_mode(0)
    else:
        right_arm.rm_set_arm_run_mode(1)
    # 逆解模式
    algo_handle.rm_algo_set_redundant_parameter_traversal_mode(False)
    # 设置机械臂初始位置
    right_arm.rm_movej([-40,50,50,65,35,-10], 60, 0, 0, 1)

    # ------------数据输入和执行----------
    try:
        while True:
            frames1 = pipeline1.wait_for_frames()   # 读取到当前帧
            aligned_frames1 = align.process(frames1)
            # 将对其的RGB—D图取出来
            color_frame1 = aligned_frames1.get_color_frame()
            depth_frame1 = aligned_frames1.get_depth_frame()
            # 数组化数据便于处理
            color_image1 = np.asanyarray(color_frame1.get_data())
            depth_image1 = np.asanyarray(depth_frame1.get_data())

            # if robotic_data['pose'] is not None and robotic_data['hand'] is not None:
            if robotic_data['pose'] is not None:
                current_q = right_arm.rm_get_joint_degree()[1]   # 获取当前机械臂关节角度(这里耗时最久)
                params = rm_inverse_kinematics_params_t(current_q, robotic_data['pose'], euler)   # 因为逆解需要当前状态，所以current_q不可缺
                q_out = algo_handle.rm_algo_inverse_kinematics(params)  # 逆解得到机械臂关节角度  q_out[0]状态码；q_out[1] 关节角度

                # 对手臂关节角度进行透传控制
                if q_out[0]==0:
                    # print(f'Inverse_kinematics:Success  逆解成功')
                    if CANFD_judge(q_out[1],current_q):
                        tar_q = q_out[1]
                    else:
                        tar_q = PoseFollow(q_out[1],current_q)

                    right_arm.rm_movej_canfd(tar_q, False)
                else:
                    print(f'Inverse_kinematics:Failed  逆解失败，{robotic_data["pose"]}')

            else:
                print(f'Inverse_kinematics:Failed  逆解失败，{robotic_data["pose"]}')
                print(f"当前机械臂末端位姿：{right_arm.rm_get_joint_degree()[1]}")
            
            # 实时显示RGB
            cv2.imshow("detect result", color_image1)
            cv2.moveWindow("detect result", 250, 250)
            
            # 按q手动退出
            if cv2.waitKey(1) == ord("q"):
                robotic_data['EXIT'] = True
                break
                
    finally:
        right_arm.rm_delete_robot_arm()
        RoboticArm.rm_destroy()




    