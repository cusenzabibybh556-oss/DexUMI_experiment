# control_process.py
import time
import multiprocessing as mp
from shared_memory import SharedData
from robot_interface import RM65Controller, L10HandController
from teleop_utils import map_grip_to_l10
from config import CONTROL_DT

def control_process(shm_name, data_queue):
    shared = SharedData(create=False, existing_shm_name=shm_name)
    arm = RM65Controller(sim=False)
    if not arm.connect():
        print("机械臂连接失败，control_process 退出")
        return
    hand = L10HandController()
    
    last_time = time.time()
    while True:
        with shared.lock:
            target_joints = shared.joints.copy()
            target_grip = shared.grip[0]
            exit_flag = shared.exit_flag[0]
        if exit_flag:
            break
        
        # 直接控制机械臂（关节角度）
        success, actual_joints = arm.move_canfd(target_joints)
        
        # 控制灵巧手
        hand_target = map_grip_to_l10(target_grip)
        hand.move_fingers(hand_target)
        
        # 记录数据
        timestamp = time.time()
        data_queue.put((timestamp, target_joints, target_grip, actual_joints, target_grip))  # 实际夹爪暂用目标代替
        
        # 频率控制
        elapsed = time.time() - last_time
        if elapsed < CONTROL_DT:
            time.sleep(CONTROL_DT - elapsed)
        last_time = time.time()
    
    arm.disconnect()