import os
import argparse
import cv2
import mmcv
import torch
import numpy as np
import pyrealsense2 as rs
from mmdet.apis import inference_detector, init_detector

from robotic_arm_package.robotic_arm import *
from mmdet.registry import VISUALIZERS

from Robot_control.pos_process import *
from Robot_control.robotic_move import *
from Robot_control.Realsense import *

# python detect_test.py /mnt/1c714c0e-2ec6-4b54-b86f-8ab77eaea676/study/code/ROS/mmdetection/configs/faster-rcnn_5classes.py /mnt/1c714c0e-2ec6-4b54-b86f-8ab77eaea676/study/code/ROS/mmdetection/checkpoint/epoch_8.pth

def parse_args():
    parser = argparse.ArgumentParser(description='MMDetection robotic arm demo')
    # parser.add_argument('config', help='test config file path')
    # parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument('--score-thr', type=float, default=0.7, help='bbox score threshold')
    parser.add_argument('--arm-ip', type=str, default='192.168.1.18', help='IP address of the robotic arm')
    parser.add_argument('--speed', type=int, default=20, help='Rotation speed of the robotic arm')
    parser.add_argument('--center-thr', type=int, default=20, help='Threshold for center alignment')
    parser.add_argument('--output-file', type=str, default='joint_angles.txt', help='File to save joint angles')
    args = parser.parse_args()
    return args

def is_target_centered(bbox, img_shape, center_thr):
    img_center_x = img_shape[1] // 2
    bbox_center_x = (bbox[0] + bbox[2]) // 2
    dx = abs(bbox_center_x - img_center_x)
    return dx < center_thr

def rotate_with_detection(MyRobot, degrees=360, speed=20, model=None, score_thr=0.7, pipeline=None, align=None, visualizer=None, center_thr=20, output_file='joint_angles.txt'):
    classes = ('apple', 'orange', 'medicine box', 'water', 'thermometer',)  # Define the classes tuple
    steps = 360
    step_angle = degrees / steps
    MyRobot.detect_ready(speed,1)
    with open(output_file, 'a') as f:
        for angle in range(0, degrees + 1, int(step_angle)):
            frames = pipeline.wait_for_frames()
            color_image_aligned, depth_image_aligned, depth_coloRobot_controlrmap_aligned = Align_version(frames, align)

            result = inference_detector(model, color_image_aligned)
            if hasattr(result, 'pred_instances') and result.pred_instances.bboxes.size(0) > 0:
                bboxes = result.pred_instances.bboxes.cpu()
                scores = result.pred_instances.scores.cpu()
                labels = result.pred_instances.labels.cpu()  # Get the labels

                for bbox, score, label in zip(bboxes, scores, labels):
                    if score > score_thr:
                        target_name = classes[label]  # Get the target label name
                        # print(f"Detected {target_name} with score {score} and bbox {bbox.numpy()}")

                        if is_target_centered(bbox.numpy(), color_image_aligned.shape, center_thr):
                            print("Target detected and centered.")
                            current_joint_angles = MyRobot.get_current_state('joint')
                            print("Current joint angles:", current_joint_angles)
                            print("Detected object:", target_name)
                            f.write(f"{current_joint_angles} {target_name}\n")
                            f.flush()

            img_rgb = mmcv.imconvert(color_image_aligned, 'bgr', 'rgb')
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

        MyRobot.robot.Movej_Cmd([0, 0, 0, 0, 0, 0], 70)

def main():
    args = parse_args()
    model_config = '/mnt/1c714c0e-2ec6-4b54-b86f-8ab77eaea676/study/code/ROS/mmdetection/configs/faster-rcnn_5classes.py'
    checkpoint = '/mnt/1c714c0e-2ec6-4b54-b86f-8ab77eaea676/study/code/ROS/mmdetection/checkpoint/epoch_8.pth'
    device = torch.device(args.device)
    model = init_detector(model_config, checkpoint, device=device)
    MyRobot = Myrobot_RM65(args.arm_ip)

    if MyRobot.robot.Arm_Socket_State() != 0:
        print("Failed to connect to robotic arm")
        return

    pipeline, config = realsense_camara_enable(0)            # 环扫：'048522073772'     骨骼点检测： '233622070776'
    align = rs.align(rs.stream.color)
    profile = pipeline.start(config)

    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta

    rotate_with_detection(MyRobot, degrees=360, speed=args.speed, 
                          model=model, score_thr=args.score_thr, 
                          pipeline=pipeline, align=align, visualizer=visualizer, 
                          center_thr=args.center_thr, output_file=args.output_file)

    pipeline.stop()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
