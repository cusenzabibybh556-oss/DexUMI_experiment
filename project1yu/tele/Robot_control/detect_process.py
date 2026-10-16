import os
import argparse
import cv2
# import mmcv
import torch

import time
import numpy as np
import math
import pyrealsense2 as rs
from Robotic_Arm.rm_robot_interface import *

# from mmdet.apis import inference_detector, init_detector

# from mmdet.registry import VISUALIZERS

from constant import *




def mmdetect_init():
    model_config = '/home/user-iip/code/TeleOperation/mmdetection/configs/faster-rcnn_5classes.py'
    checkpoint = '/home/user-iip/code/TeleOperation/mmdetection/checkpoint/epoch_8.pth'
    device = torch.device('cuda:0')

    model = init_detector(model_config, checkpoint, device=device)
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta
    return model, visualizer

def is_target_centered_x(bbox, img_shape, center_thr):
    img_center_x = img_shape[1] // 2
    bbox_center_x = (bbox[0] + bbox[2]) // 2
    dx = abs(bbox_center_x - img_center_x)
    return dx < center_thr

def is_target_centered_y(bbox, img_shape, center_thr):
    img_center_y = img_shape[0] // 2
    bbox_center_y = (bbox[1] + bbox[3]) // 2
    dy = abs(bbox_center_y - img_center_y)
    return dy < center_thr


def rotate_with_detection(robot, cap, speed=10, model=None, score_thr=0.7, visualizer=None, center_thr=10, output_file='./joint_angles.txt'):
    
    robot.rm_movej([60, -32, 76, 0, 111, 0], 70, 0, 0, block = 1)
    time.sleep(0.5)
    robot.rm_movej([-60, -32, 76, 0, 111, 0], speed, 0, 0, block = 0)
    with open(output_file, 'a') as f:
        while True:
            frames = cap.wait_frames()
            bgr_img_arm = cap.rgb_image(frames[1])
            result = inference_detector(model, bgr_img_arm)
            if hasattr(result, 'pred_instances') and result.pred_instances.bboxes.size(0) > 0:
                bboxes = result.pred_instances.bboxes.cpu()
                scores = result.pred_instances.scores.cpu()
                labels = result.pred_instances.labels.cpu()  # Get the labels

                for bbox, score, label in zip(bboxes, scores, labels):
                    if score > score_thr:
                        target_name = CLASS[label]  # Get the target label name
                        # print(f"Detected {target_name} with score {score} and bbox {bbox.numpy()}")

                        if is_target_centered_x(bbox.numpy(), bgr_img_arm.shape, center_thr):
                            print("Target detected and centered.")
                            current_joint_angles = robot.rm_get_joint_degree()
                            print("Current joint angles:", current_joint_angles[1])
                            print("Detected object:", target_name)
                            f.write(f"{current_joint_angles[1]} {target_name}\n")
                            f.flush()

            img_rgb = mmcv.imconvert(bgr_img_arm, 'bgr', 'rgb')
            visualizer.add_datasample(
                name='result',
                image=img_rgb,
                data_sample=result,
                draw_gt=False,
                pred_score_thr=score_thr,
                show=False)

            vis_img = visualizer.get_image()
            vis_img = mmcv.imconvert(vis_img, 'rgb', 'bgr')
            cv2.imshow('result', vis_img)

            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord('q') or ch == ord('Q'):
                break

        robot.rm_movej([0, -32, 76, 0, 111, 0], 70, 0, 0, block = 1)
    

def detect_target_centered_y(robot, model, cap, visualizer, target_name, score_thr = 0.7, center_thr_y=10):

    while True:
        frames = cap.wait_frames()
        bgr_img_arm = cap.rgb_image(frames[1])
        depth_frame_detect = cap.depth_frame(frames[1])
        
        result = inference_detector(model, bgr_img_arm)
        if hasattr(result, 'pred_instances') and result.pred_instances.bboxes.size(0) > 0:
            bboxes = result.pred_instances.bboxes.cpu()
            scores = result.pred_instances.scores.cpu()
            labels = result.pred_instances.labels.cpu()
            # print(labels[0])
            for bbox, score, label in zip(bboxes, scores, labels):
                if score > score_thr:
                    if CLASS[label] == target_name:
                        if is_target_centered_y(bbox.numpy(), bgr_img_arm.shape, center_thr_y):
                            print(robot.rm_set_arm_stop())
                            depth = depth_frame_detect.get_distance(320, int((bbox.numpy()[1] + bbox.numpy()[3]) // 2))
                            print("Target detected and centered in Y-axis.")
                            return depth

        img_rgb = mmcv.imconvert(bgr_img_arm, 'bgr', 'rgb')
        visualizer.add_datasample(
            name='result',
            image=img_rgb,
            data_sample=result,
            draw_gt=False,
            pred_score_thr=score_thr,
            show=False)

        vis_img = visualizer.get_image()
        vis_img = mmcv.imconvert(vis_img, 'rgb', 'bgr')
        cv2.imshow('result', vis_img)

        ch = cv2.waitKey(1)
        if ch == 27 or ch == ord('q') or ch == ord('Q'):
            break

    return False



def read_joint_angles(file_path):
    joint_angles_dict = {}
    with open(file_path, 'r') as f:
        for line in f:
            # 移除开头和结尾的方括号
            parts = line.strip().split('] ')
            joint_angles = list(map(float, parts[0].strip('[]').split(', ')))
            target_name = parts[1].strip()
            joint_angles_dict[target_name] = joint_angles
    return joint_angles_dict


def calcula_location(joint,end_point,depth,camera_length=0.107):
    joint = np.array(joint)
    angle=180-abs(joint[1]+joint[2]+joint[4])#计算关节角度

    y_length=camera_length*math.sin(math.radians(angle))
    x_length=camera_length*math.cos(math.radians(angle))
    z=depth*math.cos(math.radians(angle))+y_length#计算z轴距离手腕的高度

    xy_length=depth*math.sin(math.radians(angle))-x_length#计算xy平面的长度
    y=xy_length*math.sin(math.radians(joint[0]))#计算y轴距离手腕的长度
    x=xy_length*math.cos(math.radians(joint[0]))#计算x轴距离手腕的长度
   
    return [end_point[0]+x,end_point[1]+y,end_point[2]-z]