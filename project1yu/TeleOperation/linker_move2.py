# pyright: reportMissingImports=false
import os
import sys
import socket
import time
import argparse
import numpy as np
import torch
from avp_stream import VisionProStreamer
from loguru import logger
from constant import *
import pinocchio as pin
import multiprocessing
from multiprocessing import Process, Manager, Event

# 导入LinkerHand SDK
from linker_hand_python_sdk.LinkerHand.linker_hand_api import LinkerHandApi
from linker_hand_python_sdk.LinkerHand.utils.color_msg import ColorMsg

# 导入TransRetar模型相关模块

sys.path.append("/home/ub/snap/Projects/TransHandR")
from model.model_poseformer import PoseTransformer
from model.angle2real import create_hand_kinematics
# /home/ub/snap/Projects/TransHandR/checkpoint/models/20260308/model_final.pth
# ========================== TransRetar 配置参数 ========================== #
TRANSRETAR_CONFIG = {
    'model_path': "/home/ub/snap/Projects/TransHandR/checkpoint/models/20260310/model_final.pth",
    'urdf_file': "/home/ub/snap/Projects/TransHandR/dataset/robot/l21_right/linkerhand_l21_right.urdf",
    # 'scaling_factor': 1.0 / 0.061,
    # 'scaling_factor_rb': 1.0 / 0.064,
    'scaling_factor': 1.0 ,
    'scaling_factor_rb': 1.0 ,
    'receptive_field': 3,
    'num_joints': 25,
    'out_num_joint': 18,
    'embed_dim_ratio': 32,
    'spatial_depth': 6,
    'temporal_depth': 4,
    'spatial_mlp_ratio': 4.0,
    'temporal_mlp_ratio': 1.0,
    'num_heads': 8,
    'qkv_bias': True,
    'qk_scale': None,
    'drop_path_rate': 0.1,
}

# 关节角度限制 (弧度)
ANGLE_LIMIT_ROB = [
    [0.0, 0.0],           # hand_base_link
    [-0.18, 0.18],        # index_mcp_roll
    [0.0, 1.57],          # index_mcp_pitch
    [0.0, 1.57],          # index_pip
    [-0.18, 0.18],        # middle_mcp_roll
    [0.0, 1.57],          # middle_mcp_pitch
    [0.0, 1.57],          # middle_pip
    [-0.18, 0.18],        # ring_mcp_roll
    [0.0, 1.57],          # ring_mcp_pitch
    [0.0, 1.57],          # ring_pip
    [-0.18, 0.18],        # pinky_mcp_roll
    [0.0, 1.57],          # pinky_mcp_pitch
    [0.0, 1.57],          # pinky_pip
    [-0.6, 0.6],          # thumb_cmc_roll
    [0.0, 1.6],           # thumb_cmc_yaw
    [0.0, 1.0],           # thumb_cmc_pitch
    [0.0, 1.57],          # thumb_mcp
    [0.0, 1.57]           # thumb_ip
]

# 关节映射字典
JOINT_MAP = {
    0: 15, 1: 2, 2: 5, 3: 8, 4: 11, 5: 14, 
    6: 1, 7: 4, 8: 7, 9: 10, 10: 13,
    11: 0, 12: 0, 13: 0, 14: 0, 
    15: 16, 16: 0, 17: 0, 18: 0, 19: 0,
    20: 17, 21: 3, 22: 6, 23: 9, 24: 12
}

# 手部配置
HAND_CFG = {
    'joints_name': [
        'hand_base_link', 
        'index_mcp_roll', 'index_mcp_pitch', 'index_pip',
        'middle_mcp_roll', 'middle_mcp_pitch', 'middle_pip',
        'ring_mcp_roll', 'ring_mcp_pitch', 'ring_pip',
        'pinky_mcp_roll', 'pinky_mcp_pitch', 'pinky_pip',
        'thumb_cmc_roll', 'thumb_cmc_yaw', 'thumb_cmc_pitch', 'thumb_mcp', 'thumb_ip',
        'index_tip', 'middle_tip', 'ring_tip', 'pinky_tip', 'thumb_tip'
    ],
    'edges': [
        ['hand_base_link', 'index_mcp_roll'],
        ['index_mcp_roll', 'index_mcp_pitch'],
        ['index_mcp_pitch', 'index_pip'],
        ['hand_base_link', 'middle_mcp_roll'],
        ['middle_mcp_roll', 'middle_mcp_pitch'],
        ['middle_mcp_pitch', 'middle_pip'],
        ['hand_base_link', 'ring_mcp_roll'],
        ['ring_mcp_roll', 'ring_mcp_pitch'],
        ['ring_mcp_pitch', 'ring_pip'],
        ['hand_base_link', 'pinky_mcp_roll'],
        ['pinky_mcp_roll', 'pinky_mcp_pitch'],
        ['pinky_mcp_pitch', 'pinky_pip'],
        ['hand_base_link', 'thumb_cmc_roll'],
        ['thumb_cmc_roll', 'thumb_cmc_yaw'],
        ['thumb_cmc_yaw', 'thumb_cmc_pitch'],
        ['thumb_cmc_pitch', 'thumb_mcp'],
        ['thumb_mcp', 'thumb_ip'],
        ['index_pip', 'index_tip'],
        ['middle_pip', 'middle_tip'],
        ['ring_pip', 'ring_tip'],
        ['pinky_pip', 'pinky_tip'],
        ['thumb_ip', 'thumb_tip']
    ],
    'root_name': 'hand_base_link',
    'end_effectors': ['index_pip', 'middle_pip', 'ring_pip', 'pinky_pip', 'thumb_ip'],
    'elbows': ['index_mcp_pitch', 'middle_mcp_pitch', 'ring_mcp_pitch', 'pinky_mcp_pitch', 'thumb_mcp']
}


def clamp(value, min_val=0, max_val=255):
    """限制值在0-255范围内"""
    return max(min_val, min(max_val, int(value)))


def trans2realworld(angle):
    """
    将虚拟角度转换为真实角度(0-255),输入为弧度
    """
    angle_real = angle.copy() if isinstance(angle, list) else angle.tolist()
    
    # 归一化至0-255
    for i in range(len(angle_real)):
        low, high = ANGLE_LIMIT_ROB[i]
        norm_angle = (angle_real[i] - low) / (high - low) if high > low else 0.0
        angle_real[i] = int(norm_angle * 255)
    
    # 重排顺序并反转
    angle_mapped = [0] * 25
    for drive_idx, joint_idx in JOINT_MAP.items():
        angle_mapped[drive_idx] = 255 - angle_real[joint_idx]
    
    # 特定关节不需要反转
    for idx in [0, 6, 7, 8, 9]:
        angle_mapped[idx] = 255 - angle_mapped[idx]
    
    return [clamp(a) for a in angle_mapped]


def extract_fingertip_from_visionpro(right_hand_mat, hand_type="Right"):
    """
    从VisionPro数据中提取并处理手部关键点坐标
    返回: (25, 3) 的相对坐标数组
    """
    coordinates = []
    
    for i in range(len(right_hand_mat)):
        transform_matrix = right_hand_mat[i]
        # 坐标变换
        x = -transform_matrix[1][3]
        y = transform_matrix[2][3]
        z = -transform_matrix[0][3]
        coordinates.append([x, y, z])
    
    coordinates = np.array(coordinates)
    
    # 相对于手腕位置
    wrist_pos = coordinates[0]
    relative_coordinates = coordinates - wrist_pos
    
    return relative_coordinates


# ========================== 重定向进程 ========================== #
def TransRetar_retarget_Process(shared_data, stop_event):
    """
    TransRetar重定向进程：接收三帧人手数据，输出机器手关节角度
    """
    logger.info("TransRetar 重定向进程启动")
    
    config = TRANSRETAR_CONFIG
    
    try:
        # 初始化模型
        model = PoseTransformer(
            num_frame=config['receptive_field'],
            in_num_joints=config['num_joints'],
            in_chans=3,
            out_num_joint=config['out_num_joint'],
            out_chans=1,
            embed_dim_ratio=config['embed_dim_ratio'],
            spatial_depth=config['spatial_depth'],
            temporal_depth=config['temporal_depth'],
            spatial_mlp_ratio=config['spatial_mlp_ratio'],
            temporal_mlp_ratio=config['temporal_mlp_ratio'],
            num_heads=config['num_heads'],
            qkv_bias=config['qkv_bias'],
            qk_scale=config['qk_scale'],
            drop_path_rate=config['drop_path_rate'],
            angle_limit_rad=ANGLE_LIMIT_ROB
        )
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # 加载模型权重
        logger.info(f"Loading TransRetar model from: {config['model_path']}")
        checkpoint = torch.load(config['model_path'], map_location=device)
        model.load_state_dict(checkpoint['model_pos'], strict=False)
        model.eval()
        logger.info(f"TransRetar model loaded successfully on {device}")
        
        # 存储最近3帧数据
        recent_frames = []
        
        while not stop_event.is_set():
            # 获取原始手部数据
            hand_raw_data = shared_data.get('hand_row', None)
            
            if hand_raw_data is not None:
                try:
                    # 处理VisionPro数据
                    relative_coords = extract_fingertip_from_visionpro(hand_raw_data)
                    
                    # 添加到帧缓冲
                    recent_frames.append(relative_coords)
                    if len(recent_frames) > 3:
                        recent_frames.pop(0)
                    
                    # 当有3帧数据时进行推理
                    if len(recent_frames) == 3:
                        # 准备输入数据: (25, 3, 3) -> (1, 3, 25, 3)
                        stacked_frames = np.stack(recent_frames, axis=1)
                        reshaped_data = np.transpose(stacked_frames, (1, 0, 2))
                        reshaped_data = np.expand_dims(reshaped_data, axis=0)
                        reshaped_data *= config['scaling_factor']
                        
                        input_tensor = torch.from_numpy(reshaped_data.copy()).float().to(device)
                        
                        # 模型推理
                        with torch.no_grad():
                            output = model(input_tensor)
                        
                        # 提取结果
                        robot_angles = output.cpu().numpy()[0]
                        
                        # 转换为实际控制值
                        linker_values = trans2realworld(robot_angles)
                        
                        # 存入共享数据
                        shared_data['hand_qpos'] = linker_values
                        
                except Exception as e:
                    logger.error(f"TransRetar inference error: {e}")
            
            time.sleep(0.001)
    
    except Exception as e:
        logger.error(f"TransRetar 重定向进程出错: {e}")
    
    logger.info("TransRetar 重定向进程结束")


# ========================== 手部控制进程 ========================== #
def Linker_Hand_Process(shared_data, stop_event=None):
    """
    LinkerHand控制进程
    """
    logger.info("LinkerHand 控制进程启动")
    
    try:
        # 初始化手部API
        hand = LinkerHandApi(hand_joint="L21", hand_type="right", can="can0")
        hand.set_speed(speed=[60, 220, 220, 220, 220])
        hand.set_torque(torque=[200] * 5)
        
        logger.info("LinkerHand 初始化成功")
        
        # 设置初始位置
        initial_pose = [128] * 25
        hand.finger_move(pose=initial_pose)
        time.sleep(0.5)
        
        while True:
            # 检查退出条件
            if stop_event is not None and stop_event.is_set():
                break
            if shared_data.get('EXIT', False):
                break
            
            # 获取手部控制数据
            hand_qpos = shared_data.get('hand_qpos', None)
            
            if hand_qpos is not None:
                try:
                    # 执行手部控制
                    hand.finger_move(pose=hand_qpos)
                except Exception as e:
                    logger.error(f"Hand control error: {e}")
            
            time.sleep(0.005)
    
    except Exception as e:
        logger.error(f"LinkerHand 进程出错: {e}")
    
    logger.info("LinkerHand 控制进程结束")


# ========================== 集成进程（重定向+手部控制） ========================== #
def Linker_Hand_Process_With_TransRetar(shared_data, stop_event=None):
    """
    集成了TransRetar重定向和手部控制的进程
    适用于不想启动额外进程的情况
    """
    logger.info("TransRetar + LinkerHand 集成进程启动")
    
    config = TRANSRETAR_CONFIG
    
    try:
        # 初始化TransRetar模型
        model = PoseTransformer(
            num_frame=config['receptive_field'],
            in_num_joints=config['num_joints'],
            in_chans=3,
            out_num_joint=config['out_num_joint'],
            out_chans=1,
            embed_dim_ratio=config['embed_dim_ratio'],
            spatial_depth=config['spatial_depth'],
            temporal_depth=config['temporal_depth'],
            spatial_mlp_ratio=config['spatial_mlp_ratio'],
            temporal_mlp_ratio=config['temporal_mlp_ratio'],
            num_heads=config['num_heads'],
            qkv_bias=config['qkv_bias'],
            qk_scale=config['qk_scale'],
            drop_path_rate=config['drop_path_rate'],
            angle_limit_rad=ANGLE_LIMIT_ROB
        )
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        checkpoint = torch.load(config['model_path'], map_location=device)
        model.load_state_dict(checkpoint['model_pos'], strict=False)
        model.eval()
        logger.info(f"TransRetar model loaded on {device}")
        
        # 初始化手部API
        hand = LinkerHandApi(hand_joint="L21", hand_type="right", can="can0")
        hand.set_speed(speed=[60, 220, 220, 220, 220])
        hand.set_torque(torque=[200] * 5)
        logger.info("LinkerHand 初始化成功")
        
        # 设置初始位置
        initial_pose = [128] * 25
        hand.finger_move(pose=initial_pose)
        time.sleep(0.5)
        
        # 帧缓冲
        recent_frames = []
        
        while True:
            if stop_event is not None and stop_event.is_set():
                break
            if shared_data.get('EXIT', False):
                break
            
            hand_raw_data = shared_data.get('hand_row', None)
            
            if hand_raw_data is not None:
                try:
                    # 处理VisionPro数据
                    relative_coords = extract_fingertip_from_visionpro(hand_raw_data)
                    
                    recent_frames.append(relative_coords)
                    if len(recent_frames) > 3:
                        recent_frames.pop(0)
                    
                    if len(recent_frames) == 3:
                        # 准备数据
                        stacked_frames = np.stack(recent_frames, axis=1)
                        reshaped_data = np.transpose(stacked_frames, (1, 0, 2))
                        reshaped_data = np.expand_dims(reshaped_data, axis=0)
                        reshaped_data *= config['scaling_factor']
                        
                        input_tensor = torch.from_numpy(reshaped_data.copy()).float().to(device)
                        
                        # 推理
                        with torch.no_grad():
                            output = model(input_tensor)
                        
                        robot_angles = output.cpu().numpy()[0]
                        linker_values = trans2realworld(robot_angles)
                        
                        # 控制手部
                        hand.finger_move(pose=linker_values)
                        
                        # 同时更新共享数据（供其他进程使用）
                        shared_data['hand_qpos'] = linker_values
                        
                except Exception as e:
                    logger.error(f"Error: {e}")
            
            time.sleep(0.005)
    
    except Exception as e:
        logger.error(f"集成进程出错: {e}")
    
    logger.info("TransRetar + LinkerHand 集成进程结束")


# ========================== 便捷启动函数 ========================== #
def start_transretar_hand_control(shared_data, use_separate_processes=True):
    """
    启动TransRetar手部控制
    
    Args:
        shared_data: 共享数据字典
        use_separate_processes: 是否使用分离的重定向和控制进程
    
    Returns:
        启动的进程列表
    """
    processes = []
    stop_event = Event()
    
    if use_separate_processes:
        # 分离模式：重定向进程 + 控制进程
        #================== 启动重定向进程 ================== 
        retarget_proc = Process(
            target=TransRetar_retarget_Process,
            args=(shared_data, stop_event)
        )
        retarget_proc.daemon = True
        retarget_proc.start()
        processes.append(retarget_proc)
        
        #==================  启动手控制进程 ================== 
        hand_proc = Process(
            target=Linker_Hand_Process,
            args=(shared_data, stop_event)
        )
        hand_proc.daemon = True
        hand_proc.start()
        processes.append(hand_proc)
    else:
        # 集成模式：单进程处理
        integrated_proc = Process(
            target=Linker_Hand_Process_With_TransRetar,
            args=(shared_data, stop_event)
        )
        integrated_proc.daemon = True
        integrated_proc.start()
        processes.append(integrated_proc)
    
    return processes, stop_event