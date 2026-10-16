import os
import argparse
import numpy as np
import cv2
from avp_stream import VisionProStreamer
import argparse 
from pytransform3d import rotations



from loguru import logger

from Robotic_Arm.rm_robot_interface import *
from Robot_control.pos_process import *
from Robot_control.robotic_move import *
from Robot_control.retargeting import *
from Robot_control.data_process import *
from constant import *
from hand_detector import *

import multiprocessing
from multiprocessing import Array, Process, shared_memory, Manager, Event, Semaphore



def main(episode_idx):
    robot_manager = Manager()
    # 共享字典 用于控制机械臂和灵巧手
    robotic_data = robot_manager.dict()
    robotic_data['pose'] = None
    robotic_data['hand'] = None
    robotic_data['hand_write'] = None
    robotic_data['EXIT'] = False
    robotic_data['sim'] = False
    
    # robot_name = RobotName.allegro
    robot_name = RobotName.inspire

    
    
    # 机械臂控制子进程----------------------------------------------------------------

    Process_robot = Process(target=Robotic_process_with_camera, args = (robotic_data, robot_name, episode_idx, 0, 'visionpro', 'visionpro'))
    Process_robot.daemon = True
    arm_start_time = time.time()
    Process_robot.start()
    # Process_robot.join()

    # s = VisionProStreamer("192.168.31.143")
    # s = VisionProStreamer("192.168.146.97")
    s = VisionProStreamer("10.15.241.16")
    try:
        while True:
            latest = s.latest
            if latest is not None:
                left_wrist_mat=latest['left_wrist'][0]
                
                
                if left_wrist_mat is not None:    
                    left_wrist_pose= np.concatenate([left_wrist_mat[:3, -1] + np.array([0, 0, -0.8]),
                                        rotations.quaternion_from_matrix(left_wrist_mat[:3, :3])])
                
                left_hand_mat=latest['left_fingers'][:,:3,3]

                angle=hand_angle_caculate(left_hand_mat,robot_name)
                # print(angle)

                robotic_data['pose'] = left_wrist_pose.copy()
                if robot_name == RobotName.inspire:
                    robotic_data['hand'] = HandPosePreprocess(angle).copy()
                    robotic_data['hand_write'] = HandPosePreprocess_inspire(angle).copy()
                elif robot_name == RobotName.allegro:
                    robotic_data['hand'] = angle.copy()
            if robotic_data['EXIT']:
                print('EXIT')
                break

    finally:
            Process_robot.join()
            print('ok')
            exit(0)


if __name__ == '__main__':
    main(99)
 



