import sys
sys.path.append("//home/ub/TeleOperation_DataCollection/Robot_control")
# from Robotic_Arm.rm_ctypes_wrap
from Robotic_Arm.rm_robot_interface import *
from constant import *

import numpy as np
import math

import h5py
import socket
import time
# import keyboard

from dex_retargeting.constants import RobotName, RetargetingType, HandType, get_default_config_path

from data_process import *
from pos_process import *
from detect_process import *
from robotic_pathplan import *

       
def robotic_arm_connect(robotic_data, ip = "192.168.1.18", port = 8080):
    arm_model = rm_robot_arm_model_e.RM_MODEL_RM_65_E  # RM_65机械臂
    force_type = rm_force_type_e.RM_MODEL_RM_B_E  # 标准版
    # 初始化算法的机械臂及末端型号
    algo_handle = Algo(arm_model, force_type)
    print('ok')
    # 实例化RoboticArm类
    left_arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
    # 创建机械臂连接，打印连接id
    handle = left_arm.rm_create_robot_arm(ip, port)
    print('handle id:', handle.id)
    # print(left_arm.rm_get_total_work_frame())
    print('Coordinate System:', left_arm.rm_change_work_frame("RM65"))

    # Sim or not
    if robotic_data['sim']:
        left_arm.rm_set_arm_run_mode(0)
    else:
        left_arm.rm_set_arm_run_mode(1)

    algo_handle.rm_algo_set_redundant_parameter_traversal_mode(True)

    return left_arm, algo_handle

# 位姿跟随
def PoseFollow(q_out, c_q):
    tar_q = np.array(q_out)
    index = np.abs(JointDiff(q_out, c_q))>=10
    tar_q[index] = (np.array(c_q) + 9.5*(JointDiff(q_out, c_q)/np.abs(JointDiff(q_out, c_q))))[index]
    return tar_q

# 计算机械臂各关节角度差
def JointDiff(target_angle, current_q):
    a = np.array(target_angle)
    b = np.array(current_q)
    return  a-b

# 机械臂透传移动
def Roboric_move_CANFD(arm, algo_handle, arm_pose, euler = 0):
    current_q = arm.rm_get_joint_degree()
    params = rm_inverse_kinematics_params_t(current_q[1], arm_pose, euler)
    q_out = algo_handle.rm_algo_inverse_kinematics(params)
    if q_out[0]==0:
        if CANFD_judge(q_out[1], current_q[1]):
            arm.rm_movej_canfd(q_out[1], False)
            return True, np.array(q_out[1])
        else:
            tar_q = PoseFollow(q_out[1], current_q[1])
            arm.rm_movej_canfd(tar_q, False)
            return True, np.array(tar_q)
    else:
        print(f'Inverse_kinematics:Failed,q_out[0]={q_out[0]}')
        return False, 0

# 判断数据是否符合透传要求
def CANFD_judge(target_angle, current_q):
    return np.max(JointDiff(target_angle, current_q)) < 10

def pos_transform(pose, grab_angle):
    # print(f'检测到的pose:{pose}')
    pose = np.array(pose)
    angle = grab_angle/180*math.pi
    # print(f'angle{angle}')
    x1 = math.cos(-angle)*pose[0] + math.sin(-angle)*pose[1]
    y1 = -math.sin(-angle)*pose[0] + math.cos(-angle)*pose[1]
    pose[0] = x1
    pose[1] = y1
    pose[5] = pose[5] + angle
    # print(f'变换后的pose:{pose}')
    return pose


# 机械臂控制进程
def Robotic_process(robotic_data, robot_name, euler = 0, tool_frame = 'Arm_Tip', work_frame = 'RM65',dataset_dir=None):
    
    tim = {}
    # time_delay['COUNTER'] = 0
    tim['START'] = []
    tim['END_CAMERA_CAP'] = []
    tim['END_HAND_DETECT'] = []

    tim['START_ARM_MOVE'] = []
    tim['END_ARM_MOVE'] = []

    
    tim['START_HAND_MOVE'] = []
    tim['END_HAND_MOVE'] = []

    """共享字典 用于分别保留轨迹"""
    tra = {}
    tra['POS_X'] = []
    tra['POS_Y'] = []
    tra['POS_Z'] = []

    tra['ARM_X'] = []
    tra['ARM_Y'] = []
    tra['ARM_Z'] = []

    start_time = time.time()
    arm_model = rm_robot_arm_model_e.RM_MODEL_RM_65_E  # RM_65机械臂
    force_type = rm_force_type_e.RM_MODEL_RM_B_E  # 标准版
    # 初始化算法的机械臂及末端型号
    algo_handle = Algo(arm_model, force_type)

    # 实例化RoboticArm类
    left_arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
    # 创建机械臂连接，打印连接id
    handle = left_arm.rm_create_robot_arm("192.168.1.18", 8080)
    print('handle id:', handle.id)
    # print(left_arm.rm_get_total_work_frame())
    print('Coordinate System:', left_arm.rm_change_work_frame(work_frame))
    print('Coordinate System:', left_arm.rm_change_tool_frame(tool_frame))

    # 灵巧手通讯初始化
    if robot_name == RobotName.inspire:
        left_arm.rm_set_modbus_mode(1, 115200, 2)
        left_arm.rm_set_voltage(3)
        # 设置灵巧手速度和力阈值
        left_arm.rm_write_registers(rm_peripheral_read_write_params_t(1, 1522, 1, 6), [2, 232, 2, 232, 2, 232, 2, 232, 2, 232, 2, 232])

    elif robot_name == RobotName.allegro:
        server_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_client.connect(("localhost", 8888))


    # left_arm.rm_set_hand_speed(800)
    # left_arm.rm_set_hand_force(600)

    # Sim or not
    if robotic_data['sim']:
        left_arm.rm_set_arm_run_mode(0)
    else:
        left_arm.rm_set_arm_run_mode(1)

    algo_handle.rm_algo_set_redundant_parameter_traversal_mode(True)
    
    # 机械臂初始位置
    # left_arm.rm_movej([0, 0, 73, 0, 80, 0], 70, 0, 0, 1)


    index = 0
    try:
        while True:
            
            index += 1
            if robotic_data['pose'] is not None:
                # print('pose is get as:', robotic_data['pose'])
                tim['START_ARM_MOVE'].append(time.time()- start_time)
                control_result = Roboric_move_CANFD(left_arm, algo_handle, robotic_data['pose'], euler)
                if not control_result:
                    tim['END_ARM_MOVE'].append(time.time()- start_time)
                    
            else:
                print('wait for Left Arm control signal')

            if robotic_data['hand'] is not None:
                # print('hand_data is get as', robotic_data['hand'])

                tim['START_HAND_MOVE'].append(time.time()- start_time)


                if robot_name == RobotName.inspire and index%3==0:
                    # hand_result = left_arm.rm_write_registers(rm_peripheral_read_write_params_t(1, 1486, 1, 6),robotic_data['hand'])
                    hand_result = left_arm.rm_set_hand_angle(robotic_data['hand'])
                    index = 0
                    if not hand_result:
                        tim['END_HAND_MOVE'].append(time.time()- start_time)

                elif robot_name == RobotName.allegro:
                    # print('hand_date is getted',robotic_data['hand'])
                    server_client.send(bytes(str(np.round(robotic_data['hand'],4))[2:-1], encoding='utf-8'))
                    recv_str = server_client.recv(1024)
                    if not recv_str:
                        break
                    # print("receive: ", recv_str.decode("utf-8"))
                    # time_delay['END_HAND_MOVE'].append(time.time())
                    # time.sleep(1)

            else:
                print('wait for Left Hand control signal')

            time.sleep(0.03)
            pose = left_arm.rm_get_current_arm_state()[1]['pose']

            tra['ARM_X'].append(pose[0])
            tra['ARM_Y'].append(pose[1])
            tra['ARM_Z'].append(pose[2])

            if robotic_data['EXIT']:

                print(tra)
                print(tim)
                excel_path = './datas'

                save_dict_to_excel(tim, os.path.join(excel_path, 'tim.xlsx'))
                save_dict_to_excel(tra, os.path.join(excel_path, 'tra.xlsx'))
                break

    finally:
        if robot_name == RobotName.allegro:
            server_client.close()
        left_arm.rm_delete_robot_arm()
        RoboticArm.rm_destory()


def get_qpos(arm, hand_data):
    return np.concatenate([np.array(arm.rm_get_joint_degree()[1]),hand_data])

def get_qvel(last_qpos, cur_qpos,durtime):
#     cur_qpos = np.array(cur_qpos)
#     last_qpos = np.array(last_qpos)
    return (cur_qpos-last_qpos)/np.array(durtime)

def Robotic_process_with_camera(robotic_data, robot_name, episode_idx, euler = 0, tool_frame = 'Arm_Tip', work_frame = 'RM65'):
    """================================数据保存设置================================"""
    Task_id = 1
    task_name = TASK_NAME[Task_id]  # 任务名
    dataset_dir = '/home/ub/TeleOperation_DataCollection/data/' + task_name  # 保存路径
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir, exist_ok=True)
    
    # 记录数据
    #-----------------------------------------------------------------------------
    data_dict = {'/observations/qpos': [], '/observations/qvel': [], '/action': [],
                 '/observations/images/front_RGB': [], '/observations/images/front_depth': []}
    """=========================================================================="""
    

    """===============================摄像头初始化================================="""
    # 注册数据流，并对齐图像
    align = rs.align(rs.stream.color)
    rs_config = rs.config()
    rs_config.enable_stream(rs.stream.depth, IMAGE_SIZE['width'], IMAGE_SIZE['height'], rs.format.z16, IMAGE_FPS)
    rs_config.enable_stream(rs.stream.color, IMAGE_SIZE['width'], IMAGE_SIZE['height'], rs.format.bgr8, IMAGE_FPS)

    pipeline1 = rs.pipeline()
    rs_config.enable_device(CAMERA_ID[0])
    # pipeline_profile1 = pipeline1.start(rs_config)
    pipeline1.start(rs_config)
    """=========================================================================="""


    """===============================机械臂初始化================================="""
    arm_model = rm_robot_arm_model_e.RM_MODEL_RM_65_E  # RM_65机械臂
    force_type = rm_force_type_e.RM_MODEL_RM_B_E  # 标准版
    # 初始化算法的机械臂及末端型号
    algo_handle = Algo(arm_model, force_type)

    # 实例化RoboticArm类
    left_arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
    # 创建机械臂连接，打印连接id
    handle = left_arm.rm_create_robot_arm("192.168.1.18", 8080)
    print('handle id:', handle.id)
    # 设置机械臂工作和工具坐标系
    print('Coordinate System:', left_arm.rm_change_work_frame(work_frame))
    print('Coordinate System:', left_arm.rm_change_tool_frame(tool_frame))

    # 灵巧手通讯初始化
    if robot_name == RobotName.inspire:
        # left_arm.rm_set_hand_speed(800)
        # left_arm.rm_set_hand_force(600)
        left_arm.rm_set_modbus_mode(1, 115200, 2)
        left_arm.rm_set_voltage(3, True)
        # 设置灵巧手速度和力阈值
        left_arm.rm_write_registers(rm_peripheral_read_write_params_t(1, 1522, 1, 6), [2, 232, 2, 232, 2, 232, 2, 232, 2, 232, 2, 232])

    elif robot_name == RobotName.allegro:
        server_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_client.connect(("localhost", 8888))

    # Sim or not
    if robotic_data['sim']:
        left_arm.rm_set_arm_run_mode(0)
    else:
        left_arm.rm_set_arm_run_mode(1)

    algo_handle.rm_algo_set_redundant_parameter_traversal_mode(False)
    """==========================================================================="""
    
    # 机械臂初始位置
    left_arm.rm_movej([0, 0, 120, 0, -30, -80], 60, 0, 0, 1)
    left_arm.rm_set_hand_angle([0,0,1000,1000,1000,0], block=False, timeout=1)

    index = 0
    count = 0
    start_time = np.array(time.time())
    hand_data = [1000,1000,1000,1000,800,0]
    last_qpos = get_qpos(left_arm, hand_data)
    try:
        while True:
            count += 1
            # 摄像头数据处理
            frames1 = pipeline1.wait_for_frames()
            # 将进来的RGB-D数据对齐
            aligned_frames1 = align.process(frames1)
            # 将对其的RGB—D图取出来
            color_frame1 = aligned_frames1.get_color_frame()
            depth_frame1 = aligned_frames1.get_depth_frame()
            # 数组化数据便于处理
            color_image1 = np.asanyarray(color_frame1.get_data())
            depth_image1 = np.asanyarray(depth_frame1.get_data())

            depth_colormap1 = cv2.applyColorMap(cv2.convertScaleAbs(depth_image1, alpha=0.03), cv2.COLORMAP_JET)
            #### 机械臂执行动作 ######
            end_time = start_time.copy()
            star_time = np.array(time.time())
            cur_qpos = get_qpos(left_arm, hand_data)
            cur_vel = get_qvel(last_qpos, cur_qpos, end_time - star_time)
            
            
            if robotic_data['pose'] is not None and robotic_data['hand'] is not None:
                # print('pose is get as:', robotic_data['pose'])
                current_q = left_arm.rm_get_joint_degree()[1]
                params = rm_inverse_kinematics_params_t(current_q, robotic_data['pose'], euler)
                q_out = algo_handle.rm_algo_inverse_kinematics(params)
                # control_result, qpos = Roboric_move_CANFD(left_arm, algo_handle, robotic_data['pose'], euler)
                if q_out[0]==0:
                    if CANFD_judge(q_out[1], current_q):
                        tar_q = q_out[1]
                        
                    else:
                        tar_q = PoseFollow(q_out[1], current_q)

                    left_arm.rm_movej_canfd(tar_q, False)

                    if robot_name == RobotName.inspire and count%3 == 0:
                        hand_result = left_arm.rm_write_registers(rm_peripheral_read_write_params_t(1, 1486, 1, 6),robotic_data['hand'])
                        if hand_result == 0:
                            hand_data = robotic_data['hand_write']

                    elif robot_name == RobotName.allegro:
                        # print('hand_date is getted',robotic_data['hand'])
                        server_client.send(bytes(str(np.round(robotic_data['hand'],4))[2:-1], encoding='utf-8'))
                        recv_str = server_client.recv(1024)
                        if not recv_str:
                            break


                else:
                    print('Inverse_kinematics:Failed')
                action = get_qpos(left_arm, hand_data)

            time.sleep(0.035)

            #######################
            index += 1
            data_dict['/observations/qpos'].append(cur_qpos.copy())
            data_dict['/observations/qvel'].append(cur_vel.copy())
            data_dict['/action'].append(action.copy())
            data_dict[f'/observations/images/front_RGB'].append(color_image1.copy())
            data_dict[f'/observations/images/front_depth'].append(depth_image1.copy())
            last_qpos = cur_qpos.copy()

            cv2.imshow("detect result", color_image1)
            cv2.moveWindow("detect result", 250, 250)


            if cv2.waitKey(1) == ord("q"):
                robotic_data['EXIT'] = True
                break


    finally:
        if robot_name == RobotName.allegro:
            server_client.close()
        
        """===========================================================HDF5 保存================================================================"""
        max_timesteps = index
        t0 = time.time()
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}')
        with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
            root.attrs['sim'] = True
            obs = root.create_group('observations')
            image = obs.create_group('images')

            front_RGB = image.create_dataset('front_RGB', (max_timesteps, IMAGE_SIZE['height'], IMAGE_SIZE['width'], 3), dtype='uint8',
                                            chunks=(1, IMAGE_SIZE['height'], IMAGE_SIZE['width'], 3), )
            front_depth = image.create_dataset('front_depth', (max_timesteps, IMAGE_SIZE['height'], IMAGE_SIZE['width']), dtype='uint16',
                                            chunks=(1, IMAGE_SIZE['height'], IMAGE_SIZE['width']), )
            # compression='gzip',compression_opts=2,)
            # compression=32001, compression_opts=(0, 0, 0, 0, 9, 1, 1), shuffle=False)
            qpos = obs.create_dataset('qpos', (max_timesteps, 12))
            qvel = obs.create_dataset('qvel', (max_timesteps, 12))
            action = root.create_dataset('action', (max_timesteps, 12))

            for name, array in data_dict.items():
                root[name][...] = array
        print(f'Saving: {time.time() - t0:.1f} secs\n')

        print(f'Saved to {dataset_dir}')
        """===================================================================================================================================="""

        left_arm.rm_delete_robot_arm()
        RoboticArm.rm_destory()

if __name__ == '__main__':

    print(0)