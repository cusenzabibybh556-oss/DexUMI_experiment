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
# 导入GeoRT模型相关模块
# from geo_retargeting.geo_retargeting_model import GeoRTModel
import pinocchio as pin

# 导入LinkerHand SDK
from linker_hand_python_sdk.LinkerHand.linker_hand_api import LinkerHandApi

NO_INVERT_LINKER_INDICES = [0]
# ========================== 关节值翻转 ========================== #
def invert_linker_pose(pose):
    """
    仅对 LINKER_VALID_JOINT_INDICES 中的关节值做反向(255 - v)
    保留其它索引不变。返回新列表（不修改原列表）。
    """
    if pose is None:
        return pose
    inv = list(pose)  # 复制一份
    for idx in LINKER_VALID_JOINT_INDICES:
        if 0 <= idx < len(inv):
            if idx in NO_INVERT_LINKER_INDICES:
                continue
            v = inv[idx]
            if isinstance(v, int):
                # 若无效值保持不变
                if v == LINKER_INVALID_VALUE:
                    continue
                inv[idx] = 255 - v
    return inv

# ========================== 低通滤波器 ========================== #
class LPFilter:
    """低通滤波器，用于平滑关节角度输出"""
    def __init__(self, alpha):
        self.alpha = alpha
        self.y = None
        self.is_init = False

    def next(self, x):
        if not self.is_init:
            self.y = x
            self.is_init = True
            return self.y.copy()
        self.y = self.y + self.alpha * (x - self.y)
        return self.y.copy()

    def reset(self):
        self.y = None
        self.is_init = False

# ========================== GeoRT推理器 ========================== #
class GeoRTInference:
    """
    GeoRT模型推理器
    加载训练好的模型，将人手指尖坐标转换为机器人关节角度
    """
    def __init__(self, 
                    model_path: str, 
                    urdf_path: str,
                    device: str = "cuda" if torch.cuda.is_available() else "cpu"):
            
            logger.info("--- 初始化 GeoRT 推理器 ---")
            self.device = torch.device(device)
            logger.info(f"使用设备: {self.device}")

            # 1. 加载机器人配置
            self.hand_config = ROBOT_CONFIG["hand_config"]
            self.finger_names = list(self.hand_config.keys())  # ["thumb", "index", "middle", "ring", "pinky"]
            
            # 2. 从URDF加载关节限制
            logger.info(f"从URDF加载关节限制: {urdf_path}")
            pin_model = pin.buildModelFromUrdf(urdf_path)
            for finger, config in self.hand_config.items():
                limits = []
                for joint_name in config["joint_names"]:
                    joint_id = pin_model.getJointId(joint_name)
                    joint = pin_model.joints[joint_id]
                    limit = (
                        pin_model.lowerPositionLimit[joint.idx_q: joint.idx_q + joint.nq],
                        pin_model.upperPositionLimit[joint.idx_q: joint.idx_q + joint.nq]
                    )
                    limits.append(np.array(limit).T)
                config["joint_limits"] = np.vstack(limits)
            
            # 3. 实例化模型
            logger.info("创建 GeoRT 模型...")
            self.model = GeoRTModel(self.hand_config)
            
            # 4. 加载训练好的权重
            logger.info(f"加载模型权重: {model_path}")
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            
            # 5. 设置为评估模式
            self.model.to(self.device)
            self.model.eval()
            
            # 6. 初始化低通滤波器
            self.filter = LPFilter(alpha=1.0)
            
            logger.info("--- GeoRT 推理器初始化完成 ---")

    def retarget(self, human_fingertips: dict) -> tuple:
            """
            执行一次重定向推理
            
            Args:
                human_fingertips: 字典，key为手指名称，value为3D指尖坐标 (相对于手腕坐标系)
            
            Returns:
                (robot_qpos, inference_time_ms): 17维关节角度数组和推理时长(毫秒)
            """
            t_start = time.time()
            
            # 准备输入数据
            t_prep_start = time.time()
            input_dict = {}
            for finger_name in self.finger_names:
                if finger_name in human_fingertips:
                    fingertip_pos = human_fingertips[finger_name]
                    fingertip_tensor = torch.from_numpy(fingertip_pos).float().unsqueeze(0)
                    input_dict[finger_name] = fingertip_tensor.to(self.device)
            t_prep = (time.time() - t_prep_start) * 1000
            
            # 模型推理
            t_model_start = time.time()
            with torch.no_grad():
                # 模型输出 [-1, 1] 的标准化值
                normalized_output = self.model(input_dict)
                # 映射到真实关节角度范围
                real_output = self.model.rescale_to_real_limits(normalized_output)
            t_model = (time.time() - t_model_start) * 1000
            
            # 拼接各手指的关节角度
            t_post_start = time.time()
            # 顺序: thumb(5) + index(5) + middle(5) + ring(5) + pinky(5) = 25个关节
            qpos_tensors = [real_output[finger] for finger in self.finger_names]
            full_qpos_tensor = torch.cat(qpos_tensors, dim=1)
            robot_qpos = full_qpos_tensor.squeeze(0).cpu().numpy()
            
            # 应用低通滤波器平滑输出
            robot_qpos = self.filter.next(robot_qpos)
            t_post = (time.time() - t_post_start) * 1000
            
            t_total = (time.time() - t_start) * 1000
            
            # 返回结果和时间信息
            timing_info = {
                "prep_ms": t_prep,
                "model_ms": t_model,
                "post_ms": t_post,
                "total_ms": t_total
            }
            
            return robot_qpos, timing_info

    def reset_filter(self):
        """重置滤波器状态"""
        self.filter.reset()


# ========================== LinkerHand通信模块 ========================== #
class LinkerHandClient:
    """LinkerHand灵巧手控制客户端"""
    
    def __init__(self, 
                 hand_joint: str = "L21", 
                 hand_type: str = "right", 
                 can_interface: str = "can0",
                 speed: int = 120,
                 torque: int = 200):
        self.hand_joint = hand_joint
        self.hand_type = hand_type
        self.can_interface = can_interface
        self.speed = speed
        self.torque = torque
        self.hand = None
        self.connected = False
        
        # LinkerHand L21 关节角度范围 (弧度转换到0-255)
        # 需要根据实际URDF中的关节限制来设置
        self.joint_limits = None  # 将在connect时从配置中获取
    
        self.total_joints = 25
        self.valid_joints = 17
    
    def connect(self, joint_limits: dict = None) -> bool:
        """连接到LinkerHand"""
        try:
            # logger.info(f"初始化 LinkerHand API (型号: {self.hand_joint}, 类型: {self.hand_type}, CAN: {self.can_interface})...")
            self.hand = LinkerHandApi(
                hand_joint=self.hand_joint, 
                hand_type=self.hand_type, 
                can=self.can_interface
            )
            
            # 设置速度和力矩
            # logger.info(f"设置速度: {self.speed}, 力矩: {self.torque}")
            self.hand.set_speed(speed=[self.speed] * 5)
            self.hand.set_torque(torque=[self.torque] * 5)
            
            # 保存关节限制用于后续转换
            self.joint_limits = joint_limits
            
            self.connected = True
            logger.info("LinkerHand 连接成功")
            return True
            
        except Exception as e:
            logger.error(f"LinkerHand 连接失败: {e}")
            self.connected = False
            return False
    
    def radians_to_linker_values(self, joint_angles: np.ndarray) -> list:
        """
        将弧度制关节角度转换为LinkerHand的控制值 (0-255)
        
        LinkerHand L21 有25个关节位置, 但只有17个是有效的:
        - 有效关节 (0-10, 15, 20-24): 根据URDF关节限制进行线性映射
        - 无效关节 (11-14, 16-19): 固定为255
        
        Args:
            joint_angles: 17维弧度制关节角度数组 (模型输出)
        
        Returns:
            25个整数值的列表 (0-255)
        """
        # 初始化25维输出，无效关节设为255
        linker_values = [LINKER_INVALID_VALUE] * self.total_joints
        
        if self.joint_limits is None:
            # 如果没有关节限制信息，使用默认映射
            logger.warning("未设置关节限制，使用默认映射")
            SPECIAL_RANGE_L0, SPECIAL_RANGE_HI = 64, 192
            SPECIAL_INDEX = {6, 7, 8, 9}
            for model_idx, linker_idx in MODEL_TO_LINKER_MAPPING.items():
                if model_idx < len(joint_angles):
                    angle = joint_angles[model_idx]
                    # 假设关节范围是 [-1, 1] -> [0, 255]
                    normalized = (angle + 1.0) / 2.0
                    if linker_idx in SPECIAL_INDEX:
                        value = int(SPECIAL_RANGE_L0 + normalized * (SPECIAL_RANGE_HI - SPECIAL_RANGE_L0))
                    else:
                        value = int(normalized * 255)
                    value = max(0, min(255, value))
                    linker_values[linker_idx] = value
            return linker_values
        
        # 使用实际关节限制进行映射
        # 按手指顺序遍历，计算模型输出索引
        model_idx = 0
        finger_order = ["thumb", "index", "middle", "ring", "pinky"]
        
        for finger in finger_order:
            limits = self.joint_limits[finger]  # shape: (num_joints, 2)
            num_joints = limits.shape[0]
            
            for j in range(num_joints):
                if model_idx in MODEL_TO_LINKER_MAPPING:
                    linker_idx = MODEL_TO_LINKER_MAPPING[model_idx]
                    angle = joint_angles[model_idx]
                    lower = limits[j, 0]
                    upper = limits[j, 1]
                    
                    # 线性映射: [lower, upper] -> [0, 255]
                    if upper > lower:
                        normalized = (angle - lower) / (upper - lower)
                    else:
                        normalized = 0.5
                    
                    value = int(normalized * 255)
                    value = max(0, min(255, value))
                    linker_values[linker_idx] = value
                
                model_idx += 1
        
        return linker_values
    
    def send_joint_angles(self, joint_angles: np.ndarray) -> str:
        """
        发送关节角度到LinkerHand灵巧手
        
        Args:
            joint_angles: 25维关节角度数组 (弧度制)
        
        Returns:
            状态字符串
        """
        if not self.connected or self.hand is None:
            return "未连接到LinkerHand"
        
        try:
            # 将弧度制角度转换为LinkerHand的0-255值
            linker_values = self.radians_to_linker_values(joint_angles)
            
            # 发送到LinkerHand
            self.hand.finger_move(pose=invert_linker_pose(linker_values))
            
            return "OK"
            
        except Exception as e:
            logger.error(f"发送关节角度时发生错误: {e}")
            return f"发送失败: {e}"
    
    def send_linker_values(self, linker_values: list) -> str:
        """
        直接发送0-255的LinkerHand控制值
        """
        if not self.connected or self.hand is None:
            return "未连接到LinkerHand"
        try:
            self.hand.finger_move(pose=invert_linker_pose(linker_values))
            return "OK"
        except Exception as e:
            logger.error(f"发送控制值时发生错误: {e}")
            return f"发送失败: {e}"

    def send_home_pose(self):
        """发送归零/初始姿态"""
        if self.connected and self.hand is not None:
            try:
                # 构建初始姿态：有效关节为128，无效关节为255
                home_pose = [LINKER_INVALID_VALUE] * self.total_joints
                for linker_idx in LINKER_VALID_JOINT_INDICES:
                    home_pose[linker_idx] = 128  # 中间位置
                
                self.hand.finger_move(pose=home_pose)
                logger.info("已发送归零姿态")
            except Exception as e:
                logger.error(f"发送归零姿态失败: {e}")
    
    def disconnect(self):
        """断开连接"""
        if self.hand is not None:
            # LinkerHand SDK可能需要特定的关闭方法
            # 如果SDK有close方法，在这里调用
            self.connected = False
            logger.info("已断开与 LinkerHand 的连接")



# ========================== VisionPro数据处理 ========================== #

def estimate_frame_from_hand_points(keypoint_3d_array: np.ndarray) -> np.ndarray:
    '''
    使用手腕、食指根部、中指根部三个点建立手掌平面，计算局部坐标系的旋转矩阵
    （与训练数据采集时的 collect_hworkspace_visionpro.py 保持一致）
    '''
    # 提取手腕(0)、食指根部(6)、中指根部(16) 三个点
    points = keypoint_3d_array[[0, 6, 16], :]
    
    x_vector = points[0] - points[2]  # 从手腕到中指根的向量，作为x轴方向
    points = points - np.mean(points, axis=0, keepdims=True)  # 中心化
    
    u, s, v = np.linalg.svd(points)  # SVD分解
    normal = v[2, :]  # 提取手掌平面的法向量
    
    x = x_vector - np.sum(x_vector * normal) * normal  # 将x_vector投影到手掌平面
    x = x / np.linalg.norm(x)  # 归一化x向量
    z = np.cross(x, normal)  # x和normal的叉积，得到第三个方向
    
    if np.sum(z * (points[1] - points[2])) < 0:
        normal *= -1
        z *= -1
    
    return np.stack([x, normal, z], axis=1)

def extract_fingertip_positions(fingers_mats: np.ndarray, hand_type: str = "Right") -> dict:
    """
    从VisionPro数据中提取手指指尖位置（使用与训练时相同的坐标变换）
    
    Args:
        fingers_mats: VisionPro返回的手指关节矩阵 (25, 4, 4)
        hand_type: "Left" 或 "Right"
    
    Returns:
        字典，包含各手指指尖在 MANO 坐标系下的 3D 坐标
    """
    # 1. 从变换矩阵中提取所有关键点的3D位置
    keypoint_3d_array = np.zeros((25, 3))
    for i in range(25):
        keypoint_3d_array[i] = fingers_mats[i][:3, 3]
    
    # 2. 中心化: 将所有点移动到以手腕为原点的坐标系
    wrist_pos = keypoint_3d_array[VISIONPRO_WRIST_IDX]
    centered_keypoints = keypoint_3d_array - wrist_pos
    
    # 3. 定向: 使用手腕、食指根部、中指根部三个点估算手腕的局部坐标系(旋转矩阵)
    wrist_rotation = estimate_frame_from_hand_points(centered_keypoints)
    
    # 4. 选择对应的变换矩阵
    operator2mano = OPERATOR2MANO_LEFT if hand_type == "Left" else OPERATOR2MANO_RIGHT
    
    # 5. 对齐: 将所有点应用旋转，变换到最终的 MANO 坐标系
    joint_pos_mano = centered_keypoints @ wrist_rotation @ operator2mano
    
    # 6. 提取指尖位置
    fingertip_positions = {}
    for finger_name, idx in VISIONPRO_FINGERTIP_INDICES.items():
        fingertip_positions[finger_name] = joint_pos_mano[idx].astype(np.float32)
    
    return fingertip_positions

def Linker_Hand_Process(robotic_data, 
                        visionpro_ip="10.126.230.168",
                        hand_joint="L21",
                        hand_type="right",
                        can_interface="can0",
                        speed=120,
                        torque=200,
                        control_rate=90.0):
    
    # logger.info("=" * 60)
    logger.info("LinkerHand Process Started")
    # logger.info("=" * 60)
    
    # 2. 连接LinkerHand
    logger.info(f"\n连接 LinkerHand...")
    linker_hand = LinkerHandClient(
        hand_joint=hand_joint,
        hand_type=hand_type,
        can_interface=can_interface,
        speed=speed,
        torque=torque
    )
    
    # 因为主进程已经处理了模型到0-255数值的转换，子进程不需要加载URDF或知道joint_limits
    # 直接连接即可
    if not linker_hand.connect(joint_limits=None):
        logger.warning("LinkerHand 连接失败，将无法发送控制指令")
        linker_hand = None
    
    # logger.info("\n" + "=" * 60)
    logger.info("灵巧手控制进程就绪 (等待主进程 0-255 数据)...")
    # logger.info("=" * 60 + "\n")

    # 控制循环
    control_interval = 1.0 / control_rate
    
    try:
        while not robotic_data['EXIT']:
            t0_loop = time.time()
            
            # ===== 1. 获取 0-255 控制值 (从共享内存) =====
            if 'hand_qpos' in robotic_data.keys() and robotic_data['hand_qpos'] is not None:
                linker_values = robotic_data['hand_qpos']
                
                # ===== 2. 发送到LinkerHand =====
                if linker_hand and linker_hand.connected:
                    linker_hand.send_linker_values(linker_values)
                
                # ===== 3. 循环频率控制 =====
                elapsed = time.time() - t0_loop
                sleep_time = control_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
            else:
                time.sleep(0.002)
                pass
                
    except Exception as e:
        logger.error(f"LinkerHand Process 运行时错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理资源
        if linker_hand:
            logger.info("发送归零姿态并断开连接...")
            linker_hand.send_home_pose()
            time.sleep(0.5)
            linker_hand.disconnect()
        logger.info("LinkerHand Process 退出")
