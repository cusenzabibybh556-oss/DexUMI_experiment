import os
import argparse
import mmcv
import torch
import numpy as np

from mmdet.apis import inference_detector, init_detector
from robotic_arm_package.robotic_arm import *
from mmdet.registry import VISUALIZERS

from Robot_control.pos_process import *
from Robot_control.robotic_pathplan import *
from Robot_control.robotic_move import *
from Robot_control.Realsense import *
from Robot_control.detect_process import*
from Robot_control.com_drive import*

import mediapipe as mp
import cv2



def parse_args():
    parser = argparse.ArgumentParser(description='MMDetection robotic arm demo')
    # parser.add_argument('config', help='test config file path')
    # parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--grab_mode', type=int, default=2, help='Grab mode')
    # parser.add_argument('--file', type=str, default='joint_angles01.txt', help='Path to the txt file with joint angles and target names')
    parser.add_argument('--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument('--score-thr', type=float, default=0.7, help='bbox score threshold')
    parser.add_argument('--arm-ip', type=str, default='192.168.1.18', help='IP address of the robotic arm')
    # parser.add_argument('--arm-ip', type=str, default='192.168.33.80', help='IP address of the robotic arm')
    parser.add_argument('--com-id', type=str, default='/dev/ttyUSB0', help='ID number of serial port')
    parser.add_argument('--speed', type=int, default=15, help='Rotation speed of the robotic arm')
    parser.add_argument('--center-thr', type=int, default=10, help='Threshold for center alignment')
    parser.add_argument('--output-file', type=str, default='joint_angles.txt', help='File to save joint angles')

    args = parser.parse_args()
    return args


def main():
    # 参数
    args = parse_args()
    ##################################################################
    # RM65机械臂连接
    ########################### Begin ################################
    MyRobot = Myrobot_RM65(args.arm_ip)
    if MyRobot.robot.Arm_Socket_State() != 0:
        print("Failed to connect to robotic0.85128122 arm")
        return
    ###########################  End  ################################


    ##################################################################
    # Mediapipe配置
    ########################### Begin ################################
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_holistic = mp.solutions.holistic
    ###########################  End  ################################

    # 半自动抓取
    if args.grab_mode == 1 or args.grab_mode == 2:

        ##################################################################
        # 打开串口, 连接滑轨伺服，并设置基本参数(速度限幅、加减速时间等)
        ########################### Begin ################################
        My_Driver = IDS850(serial_open(args.com_id))
        My_Driver.velocity_lim(1500)
        My_Driver.Acce_dece_time_Set(100)
        My_Driver.driver_on()
        
        ###########################  End  ################################
        

        ##############################################################
        # 第一步:环扫&检测
        ########################## Begin #############################
        """不启用摄像头"""
        My_Driver.move_by_position(350000)
        MyRobot.detect_ready(option=1)
        time.sleep(0.05)
        ###########################  End  ############################
    
        while True:

        ##############################################################
        # 第二步:锁定抓取目标
        ########################### Begin ############################
            if My_Driver.position != 350000:
                My_Driver.move_by_position(350000)

            joint_angles_dict = read_joint_angles('joint_angles01.txt')
            joint_angles_dict_2 = read_joint_angles('joint_angles02.txt')
            # 输入待抓取目标位置
            target_name = input("Enter target name (or 'exit' to quit): ").strip()

            if target_name == 'None'or target_name == 'none':
                break

            elif target_name in joint_angles_dict:
                joint_angles = joint_angles_dict[target_name]
                joint_angles02 = joint_angles_dict_2[target_name]

                MyRobot.robot.Movej_Cmd(joint_angles, 30, trajectory_connect=0, block= True)
                time.sleep(0.2)
                MyRobot.robot.Set_Hand_Angle([1000,1000,1000,1000,1000,1000])
                MyRobot.robot.Movej_Cmd(joint_angles02, 50, trajectory_connect=0, block= True)
                time.sleep(0.2)
        ###########################  End  ############################


        ##############################################################
        # 第三步:计算目标位置并移动滑轨
        ########################### Begin ############################
            if target_name == 'orange':
                My_Driver.move_by_position(70000)

        ##########################  End  ############################
            elif target_name == 'medicine box':
                My_Driver.move_by_position(580000)

        ##############################################################
        # 第四步：抓取
        ########################### Begin ############################
            pipeline_1, config_1 = realsense_camara_enable(device_id=1)               # 手部检测摄像头激活
            align_1 = rs.align(rs.stream.color)
            profile_1 = pipeline_1.start(config_1)
            
                
            # 遥操作抓取
            default_grab_angle = 10
            MyRobot.robot.Movej_Cmd([default_grab_angle, 0, 0, 0, 0,0], v = 70, trajectory_connect = 0)
            remote_control(MyRobot, pipeline_1, align_1, mp_drawing, mp_drawing_styles, mp_holistic, grab_angle = default_grab_angle, cur_pose = None)
            MyRobot.robot.Movej_Cmd([90, 0, 0, 0, 0, 0], 30, trajectory_connect=0, block=True)
            My_Driver.move_by_position(0)
            MyRobot.put_target_semiauto2()
            pipeline_1.stop()
            cv2.destroyAllWindows()
        
        # pipeline.stop()
        # cv2.destroyAllWindows()
        MyRobot.robot.Movej_Cmd([0, 0, 0, 0, 0, 0], 30, trajectory_connect=0, block=False)
        My_Driver.move_by_position(0)

    # 纯遥操作
    elif args.grab_mode == 3:
        pipeline, config = realsense_camara_enable(device_id=1)               # 手部检测摄像头激活
        align = rs.align(rs.stream.color)
        profile = pipeline.start(config)
        grab_angle = 0
        MyRobot.robot.Movej_Cmd([grab_angle, 0, 0, 0, 0,0], v = 70, trajectory_connect = 0)
        remote_control(MyRobot, pipeline, align, mp_drawing, mp_drawing_styles, mp_holistic, grab_angle = grab_angle, cur_pose = None)
        
        pipeline.stop()
        cv2.destroyAllWindows()

    


if __name__ == '__main__':
    main()