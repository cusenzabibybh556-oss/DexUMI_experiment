import os
import numpy as np
import time
from avp_stream import VisionProStreamer
from pytransform3d import rotations

from Robotic_Arm.rm_robot_interface import *
from constant import *

import multiprocessing
from multiprocessing import Manager, Process


def Robotic_arm_control_process(robotic_data):
    """机械臂控制进程 - 只控制机械臂位置，不涉及灵巧手"""
    
    # 机械臂初始化
    arm_model = rm_robot_arm_model_e.RM_MODEL_RM_65_E  # RM_65机械臂
    force_type = rm_force_type_e.RM_MODEL_RM_B_E  # 标准版
    
    # 初始化算法的机械臂及末端型号
    algo_handle = Algo(arm_model, force_type)
    
    # 实例化RoboticArm类
    left_arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
    
    # 创建机械臂连接
    handle = left_arm.rm_create_robot_arm("192.168.1.18", 8080)
    print('机械臂连接成功，handle id:', handle.id)
    
    # 设置机械臂工作和工具坐标系
    print('设置工作坐标系:', left_arm.rm_change_work_frame("RM65"))
    print('设置工具坐标系:', left_arm.rm_change_tool_frame("Arm_Tip"))
    
    # 设置运行模式（实际运行模式）
    if robotic_data['sim']:
        left_arm.rm_set_arm_run_mode(0)  # 仿真模式
    else:
        left_arm.rm_set_arm_run_mode(1)  # 实际运行模式
    
    algo_handle.rm_algo_set_redundant_parameter_traversal_mode(True)
    
    # 移动到初始位置
    print("移动到初始位置...")
    left_arm.rm_movej([0, 0, 120, 0, -30, -80], 60, 0, 0, 1)
    print("机械臂初始化完成")
    
    try:
        while True:
            if robotic_data['pose'] is not None:
                # 获取当前关节角度
                current_q = left_arm.rm_get_joint_degree()[1]
                
                # 逆运动学求解
                params = rm_inverse_kinematics_params_t(current_q, robotic_data['pose'], 0)
                q_out = algo_handle.rm_algo_inverse_kinematics(params)
                
                if q_out[0] == 0:  # 逆运动学求解成功
                    if CANFD_judge(q_out[1], current_q):
                        # 角度变化在安全范围内，直接移动
                        left_arm.rm_movej_canfd(q_out[1], False)
                    else:
                        # 角度变化过大，进行平滑处理
                        tar_q = PoseFollow(q_out[1], current_q)
                        left_arm.rm_movej_canfd(tar_q, False)
                else:
                    print(f'逆运动学求解失败，错误代码: {q_out[0]}')
            else:
                print('等待机械臂控制信号...')
            
            # 检查退出信号
            if robotic_data['EXIT']:
                print('收到退出信号')
                break
                
            time.sleep(0.03)  # 控制频率
            
    except Exception as e:
        print(f"机械臂控制进程发生错误: {e}")
    finally:
        left_arm.rm_delete_robot_arm()
        RoboticArm.rm_destory()
        print("机械臂资源已释放")


def PoseFollow(q_out, c_q):
    """位姿跟随 - 限制单次关节角度变化幅度"""
    tar_q = np.array(q_out)
    index = np.abs(JointDiff(q_out, c_q)) >= 10
    tar_q[index] = (np.array(c_q) + 9.5*(JointDiff(q_out, c_q)/np.abs(JointDiff(q_out, c_q))))[index]
    return tar_q


def JointDiff(target_angle, current_q):
    """计算机械臂各关节角度差"""
    a = np.array(target_angle)
    b = np.array(current_q)
    return a - b


def CANFD_judge(target_angle, current_q):
    """判断数据是否符合透传要求"""
    return np.max(np.abs(JointDiff(target_angle, current_q))) < 10


def main():
    """主函数"""
    print("启动VisionPro机械臂遥操作系统...")
    
    # 创建进程间通信管理器
    robot_manager = Manager()
    
    # 共享字典用于进程间数据传递
    robotic_data = robot_manager.dict()
    robotic_data['pose'] = None
    robotic_data['EXIT'] = False
    robotic_data['sim'] = False  # 设置为True开启仿真模式
    
    # 启动机械臂控制子进程
    print("启动机械臂控制进程...")
    Process_robot = Process(target=Robotic_arm_control_process, args=(robotic_data,))
    Process_robot.daemon = True
    Process_robot.start()
    
    # 初始化VisionPro数据流
    print("连接VisionPro...")
    # s = VisionProStreamer("192.168.31.143")  # 备用IP
    s = VisionProStreamer("192.168.146.97")
    print("VisionPro连接成功")
    
    try:
        print("开始遥操作，按Ctrl+C退出...")
        while True:
            # 获取VisionPro最新数据
            latest = s.latest
            
            if latest is not None:
                # 提取左手腕变换矩阵
                left_wrist_mat = latest['left_wrist'][0]
                
                if left_wrist_mat is not None:
                    # 提取位置信息并添加偏移
                    position = left_wrist_mat[:3, -1] + np.array([0, 0, -0.8])
                    
                    # 提取旋转矩阵并转换为四元数
                    rotation_quat = rotations.quaternion_from_matrix(left_wrist_mat[:3, :3])
                    
                    # 组合成7维位姿向量：[x, y, z, qx, qy, qz, qw]
                    left_wrist_pose = np.concatenate([position, rotation_quat])
                    
                    # 更新机械臂目标位姿
                    robotic_data['pose'] = left_wrist_pose.copy()
                    
                    # 打印当前位姿信息（可选）
                    # print(f"位置: {position}, 四元数: {rotation_quat}")
            
            # 检查退出条件
            if robotic_data['EXIT']:
                print('系统退出')
                break
                
            time.sleep(0.01)  # 主循环频率
            
    except KeyboardInterrupt:
        print("\n用户中断，正在退出...")
        robotic_data['EXIT'] = True
    except Exception as e:
        print(f"主进程发生错误: {e}")
        robotic_data['EXIT'] = True
    finally:
        # 等待子进程结束
        Process_robot.join(timeout=5)
        if Process_robot.is_alive():
            Process_robot.terminate()
        print('系统已安全退出')


if __name__ == '__main__':
    main()
