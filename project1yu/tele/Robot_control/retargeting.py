import multiprocessing
from pathlib import Path
from queue import Empty
from typing import Optional

import sys
import cv2
import numpy as np
import sapien
import tyro
from loguru import logger
from sapien.asset import create_dome_envmap
from sapien.utils import Viewer

from dex_retargeting.constants import RobotName, RetargetingType, HandType, get_default_config_path
from dex_retargeting.retargeting_config import RetargetingConfig
from hand_detector import SingleHandDetector
import time
import socket

import pyrealsense2 as rs

from Robot_control import Realsense
from Robot_control.pos_process import *
from Robot_control.robotic_move import *


def qpos_norm(qpos, handtype):
    qpos_np = np.array(qpos)
    # handpos_temp = qpos_np.copy()
    if 'inspire' in handtype:
        norm_pos = []
        handpos_temp = np.zeros([12])

        if 'right' in handtype:
            handpos_temp[0:2] = qpos_np[4:6]
            handpos_temp[2:4] = qpos_np[6:8]
            handpos_temp[4:6] = qpos_np[2:4]
            handpos_temp[6:8] = qpos_np[0:2]
            handpos_temp[8:11] =  qpos_np[9:]
            handpos_temp[11] = qpos_np[8]

        for i in range(0,4):
            norm_pos.append(np.mean([handpos_temp[2*i]/1.47, (handpos_temp[2*i+1]+0.04545)/1.60545]))
        norm_pos.append(np.mean([handpos_temp[8]/0.6, handpos_temp[9]/0.8, handpos_temp[10]/0.4])*1.7)
        norm_pos.append(handpos_temp[11]/1.308)

        norm_pos = np.array(norm_pos)
        norm_pos = np.abs(norm_pos-1)

        return norm_pos
    
    elif 'allegro' in handtype:
        handpos_temp = np.zeros([16])
        if 'left' in handtype:
            handpos_temp[0:4] = qpos_np[12:]
            handpos_temp[4:8] = qpos_np[8:12]
            handpos_temp[8:12] = qpos_np[0:4]
            handpos_temp[12:] = qpos_np[4:8]


        return handpos_temp

def HandPosePreprocess_inspire(qpos):
    qpos = np.array(qpos)
    qpos[5] = 0.1
    qpos = (qpos*1000).astype(np.int16)
    return qpos

def HandPosePreprocess(qpos):
    qpos[qpos>1] = 1
    qpos[qpos<0] = 0

    hand_pose = convert_to_bytes(qpos)
    return  hand_pose

def convert_to_bytes(values):##最上面加这个函数
    # 转换系数
    scale_factors = 1000
    
    # 转换成16位整数
    integers = [int(val * scale_factors) for val in values]

    # 转换成高低字节
    result = []
    for num in integers:
        high = (num >> 8) & 0xFF
        low = num & 0xFF
        result.extend([high, low])
    return result

def qpos2handpos(qpos, handtype):
    norm_pos = qpos_norm(qpos, handtype)
    if 'allegro' in handtype:
        return norm_pos
    hand_pos = HandPosePreprocess(norm_pos)
    return hand_pos



# 通过重定向进行角度控制
def start_retargeting(robotic_data, robot_dir: str, config_path: str):

    RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    logger.info(f"Start retargeting with config {config_path}")
    retargeting = RetargetingConfig.load_from_file(config_path).build()

    hand_type = "Right" if "right" in config_path.lower() else "Left"
    # detector = SingleHandDetector(hand_type=hand_type, selfie=False)

    sapien.render.set_viewer_shader_dir("default")
    sapien.render.set_camera_shader_dir("default")

    config = RetargetingConfig.load_from_file(config_path)

    # Setup
    scene = sapien.Scene()
    render_mat = sapien.render.RenderMaterial()
    render_mat.base_color = [0.06, 0.08, 0.12, 1]
    render_mat.metallic = 0.0
    render_mat.roughness = 0.9
    render_mat.specular = 0.8
    scene.add_ground(-0.2, render_material=render_mat, render_half_size=[1000, 1000])

    # Lighting
    scene.add_directional_light(np.array([1, 1, -1]), np.array([3, 3, 3]))
    scene.add_point_light(np.array([2, 2, 2]), np.array([2, 2, 2]), shadow=False)
    scene.add_point_light(np.array([2, -2, 2]), np.array([2, 2, 2]), shadow=False)
    scene.set_environment_map(create_dome_envmap(sky_color=[0.2, 0.2, 0.2], ground_color=[0.2, 0.2, 0.2]))
    scene.add_area_light_for_ray_tracing(sapien.Pose([2, 1, 2], [0.707, 0, 0.707, 0]), np.array([1, 1, 1]), 5, 5)

    # Camera
    cam = scene.add_camera(name="Cheese!", width=600, height=600, fovy=1, near=0.1, far=10)
    cam.set_local_pose(sapien.Pose([0.50, 0, 0.0], [0, 0, 0, -1]))

    viewer = Viewer()
    viewer.set_scene(scene)
    viewer.control_window.show_origin_frame = False
    viewer.control_window.move_speed = 0.01
    viewer.control_window.toggle_camera_lines(False)
    viewer.set_camera_pose(cam.get_local_pose())

    # Load robot and set it to a good pose to take picture
    loader = scene.create_urdf_loader()
    filepath = Path(config.urdf_path)
    robot_name = filepath.stem
    loader.load_multiple_collisions_from_file = True
    if "ability" in robot_name:
        loader.scale = 1.5
    elif "dclaw" in robot_name:
        loader.scale = 1.25
    elif "allegro" in robot_name:
        loader.scale = 1.4
    elif "shadow" in robot_name:
        loader.scale = 0.9
    elif "bhand" in robot_name:
        loader.scale = 1.5
    elif "leap" in robot_name:
        loader.scale = 1.4
    elif "svh" in robot_name:
        loader.scale = 1.5

    if "glb" not in robot_name:
        filepath = str(filepath).replace(".urdf", "_glb.urdf")
    else:
        filepath = str(filepath)

    robot = loader.load(filepath)

    if "ability" in robot_name:
        robot.set_pose(sapien.Pose([0, 0, -0.15]))
    elif "shadow" in robot_name:
        robot.set_pose(sapien.Pose([0, 0, -0.2]))
    elif "dclaw" in robot_name:
        robot.set_pose(sapien.Pose([0, 0, -0.15]))
    elif "allegro" in robot_name:
        robot.set_pose(sapien.Pose([0, 0, -0.05]))
    elif "bhand" in robot_name:
        robot.set_pose(sapien.Pose([0, 0, -0.2]))
    elif "leap" in robot_name:
        robot.set_pose(sapien.Pose([0, 0, -0.15]))
    elif "svh" in robot_name:
        robot.set_pose(sapien.Pose([0, 0, -0.13]))

    # Different robot loader may have different orders for joints
    sapien_joint_names = [joint.get_name() for joint in robot.get_active_joints()]
    retargeting_joint_names = retargeting.joint_names
    retargeting_to_sapien = np.array([retargeting_joint_names.index(name) for name in sapien_joint_names]).astype(int)

    while True:

        joint_pos = robotic_data['joint_pos']
        # print('joint_pos is getted', joint_pos)
        if joint_pos is None:
            pass
            # logger.warning(f"{hand_type} hand is not detected.")
        else:
            # if robotic_data['angle']:
            #     # angle = hand_angle_caculate(joint_pos)
            #     # print('angle is get as',angle)
            #     robotic_data['hand'] = joint_pos.copy()

            # else:
            retargeting_type = retargeting.optimizer.retargeting_type
            indices = retargeting.optimizer.target_link_human_indices
            if retargeting_type == "POSITION":
                indices = indices
                ref_value = joint_pos[indices, :]
                print(1)
            else:
                origin_indices = indices[0, :]
                task_indices = indices[1, :]
                ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]

            qpos = retargeting.retarget(ref_value)
            handpos = qpos2handpos(qpos, robot_name)
            robotic_data['hand'] = handpos
                            
                

            time.sleep(0.025)
            if robotic_data['EXIT']:
                break

            robot.set_qpos(qpos[retargeting_to_sapien])

        for _ in range(2):
            viewer.render()


# 通过角度映射进行角度控制
def angle_mapping(robotic_data, robot_name: str):    

    if 'allegro' in robot_name:

        while True:
            angle = robotic_data['angle']
            if angle is None:
                pass
                # logger.warning(f"{hand_type} hand is not detected.")
            else:
                robotic_data['hand'] = handpos
            
            time.sleep(0.025)
            if robotic_data['EXIT']:
                break

    elif 'inspire'in robot_name:

        while True:
            angle = robotic_data['angle']
            # print('angle is getted', angle)
            if angle is None:
                pass
                # logger.warning(f"{hand_type} hand is not detected.")
            else:
                handpos = HandPosePreprocess_inspire(angle)
                robotic_data['hand'] = handpos
                print(handpos)
            time.sleep(0.025)
            if robotic_data['EXIT']:
                break