import os
import sys
import argparse
import numpy as np
import cv2
from avp_stream import VisionProStreamer
import argparse
from pytransform3d import rotations
import time

from loguru import logger

import multiprocessing
from multiprocessing import Manager



def main(episode_idx):
    robot_manager = Manager()
    # 共享字典，用于控制退出信号
    robotic_data = robot_manager.dict()
    robotic_data['pose'] = None
    robotic_data['hand'] = None
    robotic_data['hand_write'] = None
    robotic_data['EXIT'] = False
    robotic_data['sim'] = False

    # 仅保留 VisionPro 数据读取和打印
    #s = VisionProStreamer("192.168.146.97")
    s = VisionProStreamer("10.39.46.16")
    # Diagnostic prints to help debug connection issues
    print("running with", sys.executable)
    try:
        # try to show common connection attributes if present
        info = {}
        for attr in ("connected", "is_connected", "status"):
            if hasattr(s, attr):
                info[attr] = getattr(s, attr)
        print("VisionProStreamer status:", info)
    except Exception as _e:
        print("VisionProStreamer introspect error:", _e)

    try:
        while True:
            latest = s.latest
            if latest is not None:
                left_wrist_mat = latest['left_wrist'][0]

                if left_wrist_mat is not None:
                    left_wrist_pose = np.concatenate([
                        left_wrist_mat[:3, -1] + np.array([0, 0, -0.8]),
                        rotations.quaternion_from_matrix(left_wrist_mat[:3, :3])
                    ])
                else:
                    left_wrist_pose = None

                left_hand_mat = latest['left_fingers'][:, :3, 3]

                # 打印 VisionPro 原始数据
                print("===== VisionPro data =====")
                print("keys:", list(latest.keys()))
                print("left_wrist_mat:", left_wrist_mat)
                print("left_wrist_pose:", left_wrist_pose)
                print("left_hand_mat shape:", left_hand_mat.shape)
                print("left_hand_mat:", left_hand_mat)
                print("==========================")

                robotic_data['pose'] = left_wrist_pose.copy() if left_wrist_pose is not None else None

            if robotic_data['EXIT']:
                print('EXIT')
                break

            time.sleep(0.05)

            if robotic_data['EXIT']:
                print('EXIT')
                break

            time.sleep(0.05)

    finally:
            print('ok')
            exit(0)


if __name__ == '__main__':
    main(99)
 



