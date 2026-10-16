import sys
sys.path.append("/home/ub/TeleOperation_DataCollection/Robot_control")
# from Robotic_Arm.rm_ctypes_wrap
from Robotic_Arm.rm_robot_interface import *
from constant import *

import numpy as np
import math
import signal

import h5py
import time
# import keyboard

from dex_retargeting.constants import RobotName, RetargetingType, HandType, get_default_config_path

from linker_hand_python_sdk.LinkerHand.linker_hand_api import LinkerHandApi

from data_process import *
from pos_process import *
from detect_process import *
from robotic_pathplan import *

'''=================================机械臂相关函数==========================='''
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

# 获取当前的机器人状态（臂+手）
def get_qpos(arm, hand_data):   
    '''
    arm.rm_get_joint_degree()  从机械臂获取机械臂的关节角度
    hand data    
    '''
    return np.concatenate([np.array(arm.rm_get_joint_degree()[1]),hand_data])

'''=================================机械臂执行进程（附加灵巧手）==========================='''
def Robotic_process_with_linker(robotic_data, robot_name, episode_idx, euler = 0, tool_frame = 'vr_right', work_frame = 'visionpro'):
    """----------------------数据记录初始化----------------------"""
    Task_id = 5
    task_name = TASK_NAME.get(Task_id, 'defualt_task')
    dataset_dir = '/home/ub/TeleOperation_DataCollection/data/' + task_name
    # dataset_dir = '/media/ub/My Book/04-Imitation Learning/自采抓取数据/' + task_name
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir, exist_ok=True)

    data_dict = {
        '/observations/qpos': [],
        '/observations/qvel': [],
        '/action': [],
        '/observations/images/front_RGB': [],
        '/observations/images/front_depth': [],
        '/observations/images/wrist_RGB': [],
        '/observations/images/wrist_depth': [],
        '/observations/hand_qpos': []  # 灵巧手关节角度
    }
    start_time = time.time()
    frame_count = 0
    """----------------------摄像头初始化----------------------"""
     # 注册数据流，并对齐图像，将深度图与RGB图对齐
    align = rs.align(rs.stream.color)
    rs_config = rs.config()
    rs_config.enable_stream(rs.stream.depth, IMAGE_SIZE['width'], IMAGE_SIZE['height'], rs.format.z16, IMAGE_FPS)
    rs_config.enable_stream(rs.stream.color, IMAGE_SIZE['width'], IMAGE_SIZE['height'], rs.format.bgr8, IMAGE_FPS)

    # ---------------------- front摄像头初始化 ----------------------
    pipeline1 = rs.pipeline()
    rs_config.enable_device(CAMERA_ID[0])
    # pipeline_profile1 = pipeline1.start(rs_config)
    pipeline1.start(rs_config)
    print(f"front摄像头初始化成功: {CAMERA_ID[0]}")
    # ---------------------- Wrist摄像头初始化 ----------------------
    pipeline2 = rs.pipeline()
    rs_config2 = rs.config()
    rs_config2.enable_stream(rs.stream.depth, IMAGE_SIZE['width'], IMAGE_SIZE['height'], rs.format.z16, IMAGE_FPS)
    rs_config2.enable_stream(rs.stream.color, IMAGE_SIZE['width'], IMAGE_SIZE['height'], rs.format.bgr8, IMAGE_FPS)
    rs_config2.enable_device(CAMERA_ID[1])
    pipeline2.start(rs_config2)
    print(f"Wrist摄像头初始化成功: {CAMERA_ID[1]}")



    """----------------------机械臂初始化----------------------"""
    arm_model = rm_robot_arm_model_e.RM_MODEL_RM_65_E  # RM_65机械臂
    force_type = rm_force_type_e.RM_MODEL_RM_B_E  # 标准版
    # 初始化算法的机械臂及末端型号
    algo_handle = Algo(arm_model, force_type)
    # 实例化RoboticArm类
    right_arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
    # 创建机械臂连接，打印连接id
    handle = right_arm.rm_create_robot_arm("192.168.1.18", 8080)
    print('handle id:', handle.id)

    # 设置机械臂工作和工具坐标系
    print(f'Coordinate System_work_frame:{work_frame}', right_arm.rm_change_work_frame(work_frame))
    print(f'Coordinate System_tool_frame:{tool_frame}', right_arm.rm_change_tool_frame(tool_frame))

    # Sim or not
    if robotic_data['sim']:
        right_arm.rm_set_arm_run_mode(0)
    else:
        right_arm.rm_set_arm_run_mode(1)
    # 逆解模式
    algo_handle.rm_algo_set_redundant_parameter_traversal_mode(False)
    # 设置机械臂初始位置
    right_arm.rm_movej([-10,25,70,70,27,-20], 60, 0, 0, 1)
    # ------------设置手初始动作---------（未定义）
    hand_pose = [128] * 25

    # ------------数据输入和执行----------

    hand_data = hand_pose
    last_qpos = get_qpos(right_arm, hand_data)   # 读取当前臂+手的完整状态，每次循环用cur_qpos更新
    try:
        while True:
            frames1 = pipeline1.wait_for_frames()   # 读取到front当前帧
            aligned_frames1 = align.process(frames1)
            # 将对其的RGB—D图取出来
            color_frame1 = aligned_frames1.get_color_frame()
            depth_frame1 = aligned_frames1.get_depth_frame()
            # 数组化数据便于处理
            color_image1 = np.asanyarray(color_frame1.get_data())
            depth_image1 = np.asanyarray(depth_frame1.get_data())

            # ---------------------- Wrist摄像头图像读取 ----------------------
            frames2 = pipeline2.wait_for_frames()   # 读取wrist摄像头当前帧
            aligned_frames2 = align.process(frames2)
            color_frame2 = aligned_frames2.get_color_frame()
            depth_frame2 = aligned_frames2.get_depth_frame()
            color_image2 = np.asanyarray(color_frame2.get_data())
            depth_image2 = np.asanyarray(depth_frame2.get_data())

            cur_qpos = get_qpos(right_arm, hand_data)

            # if robotic_data['pose'] is not None and robotic_data['hand'] is not None:
            if robotic_data['pose'] is not None:
                current_q = right_arm.rm_get_joint_degree()[1]   # 获取当前机械臂关节角度(这里耗时最久)
                params = rm_inverse_kinematics_params_t(current_q, robotic_data['pose'], euler)   # 因为逆解需要当前状态，所以current_q不可缺
                q_out = algo_handle.rm_algo_inverse_kinematics(params)  # 逆解得到机械臂关节角度  q_out[0]状态码；q_out[1] 关节角度

                # 对手臂关节角度进行透传控制
                if q_out[0]==0:
                    # print(f'Inverse_kinematics:Success  逆解成功')
                    if CANFD_judge(q_out[1],current_q):
                        tar_q = q_out[1]
                    else:
                        tar_q = PoseFollow(q_out[1],current_q)

                    right_arm.rm_movej_canfd(tar_q, False)
                else:
                    print(f'Inverse_kinematics:Failed  逆解失败，{robotic_data["pose"]}')



            else:
                print(f'Inverse_kinematics:Failed  逆解失败，{robotic_data["pose"]}')
                print(f"当前机械臂末端位姿：{right_arm.rm_get_joint_degree()[1]}")
            
            # ---------------------- 数据记录 ----------------------
            cur_time = time.time()
            cur_qpos = get_qpos(right_arm, hand_data)
            cur_qvel = (cur_qpos - last_qpos) / max(cur_time - start_time, 0.001) if frame_count > 0 else np.zeros_like(cur_qpos)

            data_dict['/observations/qpos'].append(cur_qpos.copy())
            data_dict['/observations/qvel'].append(cur_qvel.copy())
            data_dict['/action'].append(cur_qpos.copy())
            data_dict['/observations/images/front_RGB'].append(color_image1.copy())
            data_dict['/observations/images/front_depth'].append(depth_image1.copy())
            data_dict['/observations/images/wrist_RGB'].append(color_image2.copy())      
            data_dict['/observations/images/wrist_depth'].append(depth_image2.copy())    
       
            # 记录灵巧手数据
            hand_qpos = robotic_data.get('hand_qpos', None)
            if hand_qpos is not None:
                data_dict['/observations/hand_qpos'].append(np.array(hand_qpos).copy())
            
            last_qpos = cur_qpos.copy()
            start_time = cur_time
            frame_count += 1
            
            # 实时显示RGB
            cv2.imshow("detect result", color_image1)
            cv2.moveWindow("detect result", 250, 250)
            cv2.imshow("wrist camera", color_image2)       
            cv2.moveWindow("wrist camera", 900, 250)        
            
            # 按q手动退出
            if cv2.waitKey(1) == ord("q"):
                robotic_data['EXIT'] = True
                break
                
    finally:
                 # 先停止摄像头
        try:
            pipeline1.stop()
            pipeline2.stop()
            print("摄像头已关闭")
        except:
            pass
        
        # 关闭 OpenCV 窗口
        cv2.destroyAllWindows()
    # ---------------------- 保存数据到HDF5 ----------------------
        if frame_count > 0:
            print(f"保存 {frame_count} 帧数据...")
            t0 = time.time()
            dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
            
            # 保存数据时暂时屏蔽信号，防止写入被中断
            original_sigterm = signal.signal(signal.SIGTERM, signal.SIG_IGN)
            original_sigint = signal.signal(signal.SIGINT, signal.SIG_IGN)
            
            try:
                with h5py.File(dataset_path, 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
                    root.attrs['sim'] = robotic_data.get('sim', False)
                    root.attrs['num_timesteps'] = frame_count
                    
                    obs = root.create_group('observations')
                    image = obs.create_group('images')
                    
                    # 保存图像
                    image.create_dataset('front_RGB', data=np.array(data_dict['/observations/images/front_RGB']), dtype='uint8')
                    image.create_dataset('front_depth', data=np.array(data_dict['/observations/images/front_depth']), dtype='uint16')
                    image.create_dataset('wrist_RGB', data=np.array(data_dict['/observations/images/wrist_RGB']), dtype='uint8')      # 新增
                    image.create_dataset('wrist_depth', data=np.array(data_dict['/observations/images/wrist_depth']), dtype='uint16')  # 新增
                    
                    
                    # 保存状态
                    obs.create_dataset('qpos', data=np.array(data_dict['/observations/qpos']))
                    obs.create_dataset('qvel', data=np.array(data_dict['/observations/qvel']))
                    root.create_dataset('action', data=np.array(data_dict['/action']))
                    
                    # 保存灵巧手数据
                    if len(data_dict['/observations/hand_qpos']) > 0:
                        obs.create_dataset('hand_qpos', data=np.array(data_dict['/observations/hand_qpos']))
                
                print(f'数据保存完成: {dataset_path}, 耗时: {time.time() - t0:.1f}s')
            finally:
                # 恢复信号处理器
                signal.signal(signal.SIGTERM, original_sigterm)
                signal.signal(signal.SIGINT, original_sigint)

        # 断开机械臂
        right_arm.rm_delete_robot_arm()
        RoboticArm.rm_destroy()




    