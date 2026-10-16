# robot_interface.py
import sys, os, time
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sdk_root = os.path.join(base_path, "linkerhand-python-sdk")
this_dir = os.path.dirname(__file__)
linkerhand_internal = os.path.join(sdk_root, "LinkerHand")

# 保证本地 TeleOperation 目录优先，这样 import config 会找到 d:\alicia\workplace\TeleOperation\config.py
if this_dir not in sys.path:
    sys.path.insert(0, this_dir)
# LinkerHand SDK 内部的 utils、core 等目录需要被发现
if linkerhand_internal not in sys.path:
    sys.path.insert(1, linkerhand_internal)
# LinkerHand 包根目录也需要在路径中，以便导入 LinkerHand 包
if sdk_root not in sys.path:
    sys.path.insert(2, sdk_root)
# Alicia-D-SDK 也加入路径
sdk_root_arm = os.path.join(base_path, "Alicia-D-SDK")
if sdk_root_arm not in sys.path:
    sys.path.append(sdk_root_arm)

# 现在可以随意双向导入
from linkerhand_python_sdk.LinkerHand.linker_hand_api import LinkerHandApi  # 灵巧手
from Robotic_Arm.rm_robot_interface import *
# from constant import *
from config import (
    ARM_IP,
    ARM_PORT,
    ARM_WORK_FRAME,
    ARM_TOOL_FRAME,
    HAND_CONTROLLER_TYPE,
    INSPIRE_HAND_DEFAULT_POSE,
    INSPIRE_HAND_DEFAULT_SPEED_FORCE,
    INSPIRE_HAND_INIT_BAUDRATE,
    INSPIRE_HAND_INIT_TIMEOUT,
    INSPIRE_HAND_VOLTAGE,
    L10_CAN_INTERFACE,
    L10_HAND_TYPE,
    L10_DEFAULT_SPEEDS,
)
import numpy as np
import time

class RM65Controller:
    def __init__(self, sim=False):
        self.sim = sim
        self.arm = None
        self.algo_handle = None
        
    def connect(self):
        arm_model = rm_robot_arm_model_e.RM_MODEL_RM_65_E
        force_type = rm_force_type_e.RM_MODEL_RM_B_E
        self.algo_handle = Algo(arm_model, force_type)
        
        self.arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
        handle = self.arm.rm_create_robot_arm(ARM_IP, ARM_PORT)
        handle_id = getattr(handle, 'id', None)
        print(f"机械臂连接 handle id: {handle_id}")
        if handle_id is None or handle_id < 0:
            print("机械臂连接失败：rm_create_robot_arm 返回无效 handle")
            return False

        self.arm.rm_change_work_frame(ARM_WORK_FRAME)
        self.arm.rm_change_tool_frame(ARM_TOOL_FRAME)

        run_mode = 0 if self.sim else 1
        self.arm.rm_set_arm_run_mode(run_mode)
        self.algo_handle.rm_algo_set_redundant_parameter_traversal_mode(True)
        return True
    
    def move_canfd(self, target_joints_deg):
        """
        直接发送关节角度目标，CANFD 透传模式
        返回 (success, actual_qpos)  success 表示发送成功
        """
        try:
            # 直接调用底层 CANFD 移动，不阻塞
            self.arm.rm_movej_canfd(target_joints_deg.tolist(), False)
            # 获取实际位置（也可以不获取，节省时间）
            # actual = self.arm.rm_get_joint_degree()[1]
            # return True, np.array(actual)
            return True, np.array(target_joints_deg)  # 直接返回目标作为实际，简化流程
        except Exception as e:
            print(f"CANFD移动失败: {e}")
            return False, np.array(self.get_actual_joints())
    
    def get_actual_joints(self):
        """获取机械臂当前实际关节角（度）"""
        return self.arm.rm_get_joint_degree()[1]
    
    def disconnect(self):
        if self.arm:
            self.arm.rm_delete_robot_arm()
            RoboticArm.rm_destroy()

class L10HandController:
    joint_dim = 10

    def __init__(self, can_interface=None, hand_type=None):
        if can_interface is None:
            can_interface = L10_CAN_INTERFACE
        if hand_type is None:
            hand_type = L10_HAND_TYPE
        self.hand = LinkerHandApi(hand_joint="L10", hand_type=hand_type, can=can_interface)
        self.set_speed(L10_DEFAULT_SPEEDS)
    
    def set_speed(self, speeds):
        self.hand.set_speed(speeds)
    
    def move_fingers(self, target_angles_10):
        """
        target_angles_10: list of 10 ints, 0~255
        一次性发送所有手指目标
        """
        self.hand.finger_move(target_angles_10)

    def get_state(self):
        """获取灵巧手当前10维关节状态"""
        return self.hand.get_state()
    
    def move_thumb_first(self, target_angles_10, delay_s=0.03):
        """
        拇指先动版本：假设前两个索引为大拇指关节
        """
        # 先发大拇指
        thumb_cmd = target_angles_10[:2] + [0]*8
        self.hand.finger_move(thumb_cmd)
        time.sleep(delay_s)
        # 再发全部
        self.hand.finger_move(target_angles_10)


class InspireHandController:
    joint_dim = 6

    def __init__(self, arm_controller, default_pose=None):
        if arm_controller is None or arm_controller.arm is None:
            raise ValueError("Inspire 灵巧手需要已连接的机械臂控制器")

        self.arm_controller = arm_controller
        self.arm = arm_controller.arm
        self.last_command = list(default_pose or INSPIRE_HAND_DEFAULT_POSE)
        self._init_hand()

    def _init_hand(self):
        self.arm.rm_set_modbus_mode(1, INSPIRE_HAND_INIT_BAUDRATE, INSPIRE_HAND_INIT_TIMEOUT)
        self.arm.rm_set_voltage(INSPIRE_HAND_VOLTAGE, True)
        self.arm.rm_write_registers(
            rm_peripheral_read_write_params_t(1, 1522, 1, 6),
            INSPIRE_HAND_DEFAULT_SPEED_FORCE,
        )
        self.move_fingers(self.last_command)

    def move_fingers(self, target_angles_6):
        command = [int(x) for x in target_angles_6]
        if len(command) != self.joint_dim:
            raise ValueError(f"Inspire 灵巧手目标维度应为 {self.joint_dim}，实际为 {len(command)}")
        self.arm.rm_set_hand_angle(command, block=False, timeout=1)
        self.last_command = command

    def get_state(self):
        # 当前工程中未封装 Inspire 实际状态读取，默认回退保存最近一次下发目标
        return self.last_command


def create_hand_controller(arm_controller, hand_type=None):
    controller_type = (hand_type or HAND_CONTROLLER_TYPE).strip().lower()
    if controller_type == "l10":
        return L10HandController()
    if controller_type == "inspire":
        return InspireHandController(arm_controller)
    raise ValueError(f"不支持的灵巧手类型: {controller_type}")
