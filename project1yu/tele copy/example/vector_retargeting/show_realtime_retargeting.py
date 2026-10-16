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
from single_hand_detector import SingleHandDetector
import time
import socket

from Robotic_arm.Robot_control import Realsense
from Robotic_arm.Robot_control.pos_process import *
from Robotic_arm.Robot_control.robotic_move import *
import pyrealsense2 as rs

# def TestClient():
#     server_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#     server_client.connect(("localhost", 8888))
#     print(111)
#     info = "How are you?"
#     while True:
#         server_client.send(bytes(info, encoding='utf-8'))
#         print("send   : ", info)
#         recv_str = server_client.recv(1024)
#         if not recv_str:
#             break
#         print("receive: ", recv_str.decode("utf-8"))
#         time.sleep(0.5)
#
#     server_client.close()
#     print("server end, exit!")
#     exit()

def start_retargeting(queue: multiprocessing.Queue, queue_thumb: multiprocessing.Queue, robot_dir: str, config_path: str):

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

    server_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_client.connect(("localhost", 8888))

    while True:

        joint_pos = queue.get(timeout=5)
        thumb = queue_thumb.get(timeout=5)
        if (joint_pos is None) or (thumb is None):
            pass
            # logger.warning(f"{hand_type} hand is not detected.")
        else:
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
            # if thumb == 1:          # 大拇指张开
            #     qpos[4:8] = [0.5284, 0.3693, 0.8977, 0.4863]
            # else:
            #     qpos[4:8] = [0.7017, 0.2976, 0.9034, 0.8929]

            server_client.send(bytes(str(np.round(qpos,5)), encoding='utf-8'))
            recv_str = server_client.recv(1024)
            if not recv_str:
                break
            # print("receive: ", recv_str.decode("utf-8"))
            time.sleep(0.02)

            robot.set_qpos(qpos[retargeting_to_sapien])

        for _ in range(2):
            viewer.render()
    server_client.close()

def produce_frame(queue_joint: multiprocessing.Queue, queue_pose: multiprocessing.Queue,queue_thumb: multiprocessing.Queue, camera_path: Optional[str] = None):

    # if camera_path is None:
    pipeline, config = Realsense.realsense_camara_enable(device_id=1)  # 手部检测摄像头激活
    align = rs.align(rs.stream.color)
    profile = pipeline.start(config)

    hand_type = "Left"
    detector = SingleHandDetector(hand_type=hand_type, selfie=False)

    index = 0
    while True:
        index += 1
        try:
            frames = pipeline.wait_for_frames()
            time.sleep(1 / 30.0)
            bgr, depth_frame_aligned, _ = Realsense.Align_version(frames, align, show_pic=1)
            height, width, channel = bgr.shape
            # cv2.imshow('Detect Result', bgr)
            # bgr = queue.get(timeout=5)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        except Empty:
            logger.error(f"Fail to fetch image from camera in 5 secs. Please check your web camera device.")
            return

        _, joint_pos, keypoint_2d, _ = detector.detect(rgb)
        thumb = thumb_judge(keypoint_2d)
        queue_joint.put(joint_pos)
        queue_thumb.put(thumb)
        bgr = detector.draw_skeleton_on_image(bgr, keypoint_2d, style="default")

        if index == 10:
            if keypoint_2d is not None:
                pose = keypointProcess(height, width, keypoint_2d.landmark, depth_frame_aligned)
                queue_pose.put(pose)
            index = 0
        cv2.imshow("realtime_retargeting_demo", bgr)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

def robotic_arm_move(queue_pose: multiprocessing.Queue, camera_path: Optional[str] = None):
    MyRobot = Myrobot_RM65('192.168.1.18')
    if MyRobot.robot.Arm_Socket_State() != 0:
        print('ERROR:',MyRobot.robot.Arm_Socket_State(),"Failed to connect to robotic arm")
        return

    arm_pos_temp = None
    grab_angle = 0
    MyRobot.robot.Movej_Cmd([grab_angle, 0, 0, 0, 0, 0], v=70, trajectory_connect=0)
    while True:
        try:
            pose = queue_pose.get(timeout=15)
            print('pose is get as:', pose)
            # print('pose is get as:', pose)
            if arm_pos_temp is None:
                arm_pos_temp = pose
            elif isArmThreshold(pose, arm_pos_temp):
                pose_trans = pos_transform(pose, grab_angle)
                MyRobot.arm_move_bypose_CANFD(pose_trans)
                arm_pos_temp = pose
            else:
                arm_pos_temp = pose
        except Empty:
            logger.error(f"Fail to get poses to control RM65 arms.")
            return

def keypointProcess(height, width, keypoint, depth_frame_aligned):

    center_landmarks_x, center_landmarks_y = LM_mean(keypoint)
    cx, cy = int(center_landmarks_x * width), int(center_landmarks_y * height)
    if cx < 0 or cx >= width or cy < 0 or cy >= height:
        depth = None
        print('手掌超出摄像头范围')
    else:
        depth = depth_frame_aligned.get_distance(cx, cy)

    if depth != None:
        x, y, z = pix2position(cx, cy, depth, [640, 480])
        # print(x,y,z)

        # thumb rotation
        hand_vector_x = -normal_vertor_caculate(keypoint[5], keypoint[0], keypoint[9])
        hand_vector_z = hand_vector_z_caculate(keypoint[5], keypoint[17], keypoint[0])

        # wrist rotation
        hand_vector_x_robot = vector_normalize(pos_Transform2Robot(hand_vector_x))
        hand_vector_z_robot = vector_normalize(pos_Transform2Robot(hand_vector_z))

        rx, ry, rz = get_pose_rotate(-hand_vector_x_robot, hand_vector_z_robot)
        pose = [x, y, z, rx, ry, rz]
    return pose

def thumb_judge(keypoint):
    if keypoint is None:
        return None
    thumb_angle = angle_caculate(keypoint.landmark, 2, 3, 4)
    if thumb_angle >= 550:
        return 1
    else:
        return 0


def main(
    robot_name: RobotName, retargeting_type: RetargetingType, hand_type: HandType, camera_path: Optional[str] = None
):
    """
    Detects the human hand pose from a video and translates the human pose trajectory into a robot pose trajectory.

    Args:
        robot_name: The identifier for the robot. This should match one of the default supported robots.
        retargeting_type: The type of retargeting, each type corresponds to a different retargeting algorithm.
        hand_type: Specifies which hand is being tracked, either left or right.
            Please note that retargeting is specific to the same type of hand: a left robot hand can only be retargeted
            to another left robot hand, and the same applies for the right hand.
        camera_path: the device path to feed to opencv to open the web camera. It will use 0 by default.
    """
    config_path = get_default_config_path(robot_name, retargeting_type, hand_type)
    robot_dir = Path(__file__).absolute().parent.parent.parent / "assets" / "robots" / "hands"

    queue_joint = multiprocessing.Queue(maxsize=1000)
    queue_pose = multiprocessing.Queue(maxsize=1000)
    queue_thumb = multiprocessing.Queue(maxsize=1000)
    producer_process = multiprocessing.Process(target=produce_frame, args=(queue_joint, queue_pose, queue_thumb, camera_path))
    consumer_process = multiprocessing.Process(target=start_retargeting, args=(queue_joint, queue_thumb, str(robot_dir), str(config_path)))
    # robot_control = multiprocessing.Process(target=robotic_arm_move, args=(queue_pose, camera_path))

    producer_process.start()
    consumer_process.start()
    # robot_control.start()

    producer_process.join()
    consumer_process.join()
    # robot_control.join()
    time.sleep(5)

    print("done")


if __name__ == "__main__":

    tyro.cli(main)

