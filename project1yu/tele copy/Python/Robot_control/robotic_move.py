import sys
sys.path.append("/mnt/1c714c0e-2ec6-4b54-b86f-8ab77eaea676/study/code/ROS/mmdetection/Python/Robot_control")
from robotic_arm_package.robotic_arm import * 

import numpy as np
import math

import mediapipe
import cv2
from Realsense import Align_version
from pos_process import *
from robotic_pathplan import *

def MCallback(data):
    print("MCallback MCallback MCallback")
    # 判断接口类型
    if data.codeKey == MOVEJ_CANFD_CB: # 角度透传
        print("透传结果:", data.errCisArmThresholdrCode)
        print("当前角度:", data.joint[0], data.joint[1], data.joint[2], data.joint[3], data.joint[4], data.joint[5])
        print("当前位姿:", data.pose.position.x, data.pose.position.y, data.pose.position.z, data.pose.euler.rx, data.pose.euler.ry, data.pose.euler.rz)
    elif data.codeKey == FORCE_POSITION_MOVE_CB: # 力位混合透传
        print("透传结果:", data.errCode)
        print("当前力度：", data.nforce)

class Myrobot_RM65():
    """用于获取机械臂当前动作状态以选择延展方式"""
    def __init__(self,ip, v=70, target_angle = 0):
        
        self.type = RM65                        # 机器人型号

        callback = CANFD_Callback(MCallback)
        self.robot = Arm(self.type, ip, callback)

        self.state = self.robot_state_check()
        # self.block = block                      # 机械臂运动阻塞 True or False
        self.v = v                              # 机械臂运动速度
        self.current_target_angle = target_angle

        # 控制模块

    def robot_state_check(self):
        res, joint, pose, arm_err, sys_err = self.robot.Get_Current_Arm_State()
        joint = np.array(joint)
        if np.all(np.abs(joint)<=0.003):
            state = 'Zero_Pos'
        else:
            state = 'Doing'
        return state

    def reset(self,hand,arm):
        """设置机械臂及灵巧手复位
            hand: 逻辑值 0 或 1, 1表示手掌执行复位
            arm: 逻辑值 0 或 1, 1表示手臂执行复位
        """
        if hand:
            angle0 = [0, 0, 0, 0, 1000, 1000]
            angle1 = [1000, 1000, 1000, 1000, 0, 1000]
            self.robot.Set_Hand_Angle(angle0,True)
            time.sleep(0.5)
            self.robot.Set_Hand_Angle(angle1,True)
        if arm:
            zero = [0, 0, 0, 0, 0, 0]
            self.robot.Movej_Cmd(zero, 20)
        time.sleep(0.5)

    def hand_angle_set(self,angle):
        """控制灵巧手"""
        self.robot.Set_Hand_Angle(angle,True)
        time.sleep(0.05)

    # Arm move
    def arm_move_bypose(self, pose, TC):
        pose_np = np.array(pose)
        # print(pose_np)
        if is_pos_lim(pose_np[0:3]):
            print("目标位置超出抓取范围")
        else:
            # 关节移动
            # self.robot.Movej_P_Cmd(pose, self.v, block=False)
            self.robot.Movej_P_Cmd(pose, self.v, trajectory_connect = TC, block=True)
            time.sleep(0.02)

    def arm_move_bypose_CANFD(self, pose, d_t = 0.005):
        "高跟随位姿透传"
        start_angle = self.get_current_state('joint')
        res, angle = Arm.Algo_Inverse_Kinematics(start_angle, pose, 1)
        if res == 0:        # 逆运算无误
            end_angle = angle[:-1]
            max_dif = np.round(get_joint_max_dif(start_angle, end_angle))
            end_time = max_dif/50
            qq, vv, aa, t = Robot_Path_Planning(start_angle, end_angle, end_time, d_t)
            # Planning_plot(qq, vv, aa, t, 6)
            # print(max(aa))
            joint = np.array(qq).T
            step = joint.shape[0]
            for i in range(step):
                # print(f'最大角度差{max_dif}, 透传步数:{i+1}/{step}, 当前角度:{joint[i]}')
                self.robot.Movej_CANFD(list(joint[i]), False)
                time.sleep(d_t)
        

    def get_current_state(self, option):
        res, joints, poses, arm_err, sys_err = self.robot.Get_Current_Arm_State()
        if res == 0:  # 成功获取状态
            if option == 'joint':
                joints = [round(joint, 2) for joint in joints]
                return joints
            elif option == 'pose':
                poses = [round(pose,3) for pose in poses]
                return poses
        else:
            print("Failed to get current arm state")
            return False  # 返回默认值

    # 环扫检测
    def detect_ready(self, speed = 15, option = 0):
        
        self.robot.Set_Hand_Angle([1000,1000,1000,1000,0,1000])
        time.sleep(0.02)
        if option == 0:
            self.robot.Movej_Cmd([-90, 0, -90, 0, -90, 0], 60, trajectory_connect=0)
            time.sleep(0.02)
            self.robot.Movej_Cmd([90, 0, -90, 0, -90, 0], speed, trajectory_connect=0, block=False)
        
        elif option == 1:
            self.robot.Movej_Cmd([-90, 20, -40, 0, -95, 0], 60, trajectory_connect=0)
            time.sleep(0.02)
            self.robot.Movej_Cmd([90, 20, -40, 0, -95, 0], speed, trajectory_connect=0, block=False)

        elif option == 2:
            self.robot.Movej_Cmd([-90, 0, 0, 0, -110, 0], 60, trajectory_connect=0)
            time.sleep(0.02)
            self.robot.Movej_Cmd([90, 0, 0, 0, -110, 0], speed, trajectory_connect=0, block=False)


    def robot_rotate(self,angle):
        if self.state == 'grab_style_1':
            self.robot.Movej_Cmd()
    # 遥操作抓取
    # def Arm_move_designed_2(self, pos):
    #     distance_horiz = math.sqrt(sum(pos[0:2]**2))
    #     distance = sum(pos**2)
    #     angle = get_move_angle(pos)

    # 半自动化抓取
    def Arm_move_semiauto(self, pos, target_name):
        """控制机械臂末端移动到指定位置, pos = [x, y, z], 为归一化的坐标。
            x、y 的范围: [-1,1]
            z 的范围: [0,1]
        """
        angle = get_move_angle(pos)
        grab_angle = self.step1_angle_move(angle)
        if target_name == 'water':
            self.step2_Arm_extension(grab_angle, option=1)
        else:
            self.step2_Arm_extension(grab_angle, option=0)
        self.step3_Slow_approach(grab_angle, pos, peroid = 0.02, step = 50)

        return angle


    # 半自动化抓取规划
    def step1_angle_move(self, angle, hand_rotate = 0):
        """机械臂移动第一步: 根据目标位置, 调整抓取角度"""
        grab_angle = angle
        joint_step1 = [grab_angle, 0, 0, 0, 0, hand_rotate]
        self.robot.Movej_Cmd(joint_step1, 50, trajectory_connect=0)
        time.sleep(0.01)
        return grab_angle

    def step2_Arm_extension(self, grab_angle, option):
        """机械臂移动第二步: 根据目标距离, 选择机械臂延展方式与手臂抓握方式"""
        self.robot.Set_Hand_Angle([1000,1000,1000,1000,1000,0])
        if option == 0:
            self.robot.Movej_Cmd([grab_angle, 20, -100, 0, -50, 0], 50, trajectory_connect=0)

        elif option == 1:
            self.robot.Movej_Cmd([grab_angle-90, -10, -20, 60, -95, -80], 50, trajectory_connect=0)
            self.robot.Movej_Cmd([grab_angle-90, -20, -110, 60, -95, -80], 50, trajectory_connect=0)
            self.robot.Movej_Cmd([grab_angle-90, -60, -110, 60, -95, -80], 50, trajectory_connect=0)
        time.sleep(0.01)

    def step3_Slow_approach(self,grab_angle, target_pos, peroid = 0.02, step = 50):
        """缓速低跟随靠近目标位置, 输入target_pos为目标真实世界位置"""
        # 周期 20ms，低跟随位姿透传
        # bias_pos = pos_bias_calculate(grab_angle, target_pos)
        pose = self.get_current_state('pose')
        pose = np.array(pose)
        pose[0:3] = target_pos
        # # bias_pos[2] += 0True.24         # 以关节2为参照
        # pose_dif = np.array(bias_pos) - pose[0:3]
        # d_step = pose_dif/step
        # # print(d_step)
        self.arm_move_bypose_CANFD(pose)

    def put_target_semiauto1(self):
        joint = self.get_current_state('joint')
        joint[0]  = 90
        self.robot.Movej_Cmd(joint, v=50, trajectory_connect=0)
        self.robot.Movej_Cmd([90,-40,-50,90,-50,0], v=50, trajectory_connect=0)
        time.sleep(1.5)
        self.robot.Set_Hand_Angle([1000,1000,1000,1000,1000,0])

    def put_target_semiauto2(self):
        self.robot.Movej_Cmd([90, 0, 0, 0, 0, 0], 80, trajectory_connect=0, block=True)
        # MyRobot.arm_move_bypose_CANFD([0.076, 0.365, 0.406, -1.623, -0.048, 0.534])
        self.arm_move_bypose_CANFD([-0.031, 0.174, 0.465, -1.591, -0.12, 1.586])
        time.sleep(1)
        self.robot.Set_Hand_Angle([1000,1000,1000,1000,1000,0])
        time.sleep(1)
        self.robot.Movej_Cmd([0, 0, 0, 0, 0, 0], v = 30, trajectory_connect = 0)
        time.sleep(0.5)

    def put_target_semiauto3(self,count):
        self.arm_move_bypose_CANFD([-0.187, 0.305, 0.375, 3.075, -0.529, 2.392])
        time.sleep(1.5)
        if count == 1:
            self.arm_move_bypose_CANFD([-0.267, 0.325, -0.069, -3.039, -0.777, 2.126])
            time.sleep(0.02)
            self.robot.Set_Hand_Angle([1000,1000,1000,1000,1000,0],block=True)
            time.sleep(0.02)
            self.arm_move_bypose_CANFD([-0.259, 0.375, -0.015, -2.861, -0.287, 1.502])
            time.sleep(0.02)
            self.arm_move_bypose_CANFD([-0.261, 0.331, -0.022, -2.871, -0.021, 1.336])
            time.sleep(0.02)
        elif count == 2:
            self.arm_move_bypose_CANFD([-0.192, 0.361, -0.061, -2.957, -0.558, 1.762])
            time.sleep(0.02)
            self.robot.Set_Hand_Angle([1000,1000,1000,1000,1000,0],block=True)
            self.arm_move_bypose_CANFD([-0.212, 0.37, -0.009, -2.951, -0.232, 1.425])
            time.sleep(0.02)
            self.arm_move_bypose_CANFD([-0.218, 0.331, -0.015, -3.018, -0.01, 1.386])
            time.sleep(0.02)
        elif count == 3:
            self.arm_move_bypose_CANFD([-0.269, 0.372, -0.034, 3.046, -0.773, 2.302])
            time.sleep(0.02)
            self.robot.Set_Hand_Angle([1000,1000,1000,1000,1000,0],block=True)
        elif count == 4:
            self.arm_move_bypose_CANFD([-0.193, 0.368, -0.034, 3.064, -0.634, 2.149])
            time.sleep(0.02)
            self.robot.Set_Hand_Angle([1000,1000,1000,1000,1000,0],block=True)
        elif count == 5:
            self.arm_move_bypose_CANFD([-0.169, 0.318, 0.161, -1.543, 0.005, 0.482])
            time.sleep(0.02)
            self.arm_move_bypose_CANFD([-0.19, 0.361, -0.106, -1.653, -0.103, 0.474])
            time.sleep(0.02)
            self.robot.Set_Hand_Angle([1000,1000,1000,1000,1000,0],block=True)

        time.sleep(0.5)
        self.arm_move_bypose_CANFD([-0.187, 0.305, 0.375, 3.075, -0.529, 2.392])
        time.sleep(0.5)

def demo7(robot):
    #   初始位置
    joint = [0, 0, -90, 0, -90, 0]
    ret = robot.Movej_Cmd(joint, 20)
    print("Movej_Cmd ret:" + str(ret))
    time.sleep(1)

    #   正解
    compute_pose = Arm.Algo_Forward_Kinematics(joint)

    print(f'正解结果：{compute_pose}')

    #   逆解
    compute_pose[0] += 0.
    res = Arm.Algo_Inverse_Kinematics(joint, compute_pose, 1)
    print(f'逆解：{res}')



def is_pos_lim(pos):
    """用于判断目标位置是否在机械臂可达范围内, 超出范围返回1, 否则返回0"""
    rela_pos = pos
    rela_pos[2] -= 0.2405
    # print(get_distance(rela_pos))
    if get_distance(rela_pos) > 0.6:
        # 短距离抓取
        return True
    else:
        return False

def get_distance(pos):
    return math.sqrt(sum(pos**2))

def get_move_angle(pos):
    """用于计算目标位置相对于机械臂的旋转角度"""
    rad = math.atan2(pos[1],pos[0])
    angle = rad/math.pi*180
    return angle

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

def pos_bias_calculate(grab_angle, pos):
    dis = math.sqrt(sum(pos[0:2]**2))
    # print(dis_bias)
    grab_angle_rad = grab_angle/180*math.pi
    pos_bias = [dis*math.cos(grab_angle_rad), dis*math.sin(grab_angle), pos[2]]
    # print(pos_bias)
    return pos_bias

def remote_control(MyRobot, pipeline_1, align_1, mp_drawing, mp_drawing_styles, mp_holistic, grab_angle, cur_pose, fine_grab = False, weight = 0.005):
    pTime = 0
    cTime = 0
    

    # 手指和手臂信息寄存
    finger_angle_temp = None
    hand_move = False
    arm_pos_temp = None
    count = 0
    sample_rate = 3
    # # 机械臂状态寄存
    # start_flag = False
    # end_flag = False
    with mp_holistic.Holistic(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as holistic:
        while True:
            frames_1 = pipeline_1.wait_for_frames()

            color_image_aligned_1,depth_frame_aligned,depth_colormap_aligned=Align_version(frames_1,align_1,show_pic=1)
            # cv2.imshow('image',color_image_aligned)
            color_image_aligned_1 = cv2.cvtColor(color_image_aligned_1, cv2.COLOR_BGR2RGB)
            Mediapipe_result = holistic.process(color_image_aligned_1)
            color_image_aligned_1 = cv2.cvtColor(color_image_aligned_1, cv2.COLOR_RGB2BGR)

            height, width, channel = color_image_aligned_1.shape

            mp_drawing.draw_landmarks(
                color_image_aligned_1,
                Mediapipe_result.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
            
            if Mediapipe_result.right_hand_landmarks:
                right_landmarks = Mediapipe_result.right_hand_landmarks.landmark
                count += 1
                if count % sample_rate == 0:
                    count = 0
                    center_landmarks_x, center_landmarks_y = LM_mean(right_landmarks)
                    cx, cy = int(center_landmarks_x *width), int(center_landmarks_y *height)
                    if cx<0 or cx>= width or cy<0 or cy>=height:
                        depth = None
                        print('手掌超出摄像头范围')
                    else:
                        depth = depth_frame_aligned.get_distance(cx,cy)

                    if depth != None:
                        x,y,z = pix2position(cx,cy,depth, [640, 480])
                        # print(x,y,z)

                        # finger angle
                        angle_right_finger = []

                        angle_right_finger.append(angle_caculate(right_landmarks, 17, 18, 19))  # 1号手指(小拇指)
                        angle_right_finger.append(angle_caculate(right_landmarks, 13, 14, 15))  # 2号手指(无名指)
                        angle_right_finger.append(angle_caculate(right_landmarks, 9, 10, 11))   # 3号手指(中指)
                        angle_right_finger.append(angle_caculate(right_landmarks, 5, 6, 7))     # 4号手指(食指)
                        angle_right_finger.append(angle_caculate(right_landmarks, 2, 3, 4))     # 5号手指(大拇指)
                        angle_right_finger.append(int(0))                                       # 大拇指侧向开合
                        # thumb rotation
                        hand_vector_x = normal_vertor_caculate(right_landmarks[5], right_landmarks[0], right_landmarks[9])
                        hand_vector_z = hand_vector_z_caculate(right_landmarks[5],right_landmarks[17],right_landmarks[0])
                        
                        
                        if finger_angle_temp == None:
                            finger_angle_temp = angle_right_finger
                        elif isFingerThreshold(angle_right_finger,finger_angle_temp):
                            if np.mean(angle_right_finger[0])>500:
                                MyRobot.hand_angle_set([1000,1000,1000,1000,1000,0])
                            else:
                                MyRobot.hand_angle_set([0,0,0,0,0,0])
                            # MyRobot.hand_angle_set(angle_right_finger)
                            finger_angle_temp = angle_right_finger
                            hand_move = True
                        else:
                            finger_angle_temp = angle_right_finger
                            hand_move = False

                        # wrist rotation
                        hand_vector_x_robot = vector_normalize(pos_Transform2Robot(hand_vector_x))
                        hand_vector_z_robot = vector_normalize(pos_Transform2Robot(hand_vector_z))

                        # hand pose
                        
                        if hand_vector_x_robot.all() != None and hand_vector_z_robot.all() != None and hand_move == False:
                            rx, ry ,rz = get_pose_rotate(-hand_vector_x_robot, hand_vector_z_robot)
                            pose = [x,y,z,rx,ry,rz]
                            
                            if arm_pos_temp == None:
                                arm_pos_temp = pose
                            elif isArmThreshold(pose,arm_pos_temp):
                                # print('-----------------------------------------------------------')
                                # print('机械臂开始移动:',pose)
                                pose_trans = pos_transform(pose, grab_angle)
                                if fine_grab == True:
                                    cur_pose = np.array(cur_pose)
                                    # target_pos = cur_pose + weight*(pose_trans - cur_pose)
                                    target_pos = cur_pose
                                    target_pos[0:3] = cur_pose[0:3] + weight*(pose_trans[0:3]-cur_pose[0:3])
                                    MyRobot.arm_move_bypose_CANFD(list(target_pos))

                                    # MyRobot.arm_move_bypose(list(pose_trans))
                                elif fine_grab == False:
                                    # print('机械臂开始移动:',pose_trans)
                                    MyRobot.arm_move_bypose_CANFD(pose_trans)
                                    # MyRobot.arm_move_bypose(pose_trans, TC =1)
                                arm_pos_temp = pose
                            else:
                                arm_pos_temp = pose
                                

            cTime = time.time()
            fps = 1/(cTime-pTime)
            pTime = cTime
            # print(color_image_aligned)
            cv2.putText(color_image_aligned_1,str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
            cv2.imshow('Hand poseture detection',color_image_aligned_1)
            cv2.moveWindow("Hand poseture detection", 50, 50)
            cv2.imwrite(os.path.join('test.png'),color_image_aligned_1)
            cv2.waitKey(1)
            
            if cv2.waitKey(5) & 0xFF == 27:
                break


if __name__ == '__main__':

    # main
    ip = "192.168.1.18"

    MyRobot = Myrobot_RM65(ip)
    
    # 查询链接状态:成功返回0finger_angle1 = [0, 0, 0, 0, 1000, 500]
    print(MyRobot.robot.Arm_Socket_State())
    pose =MyRobot.get_current_state('pose')
    print(pose)

#   demo1
#     pos = np.array([0, 0.35, 0.19])    # 目标真实位置, 单位: cm
#     # 水平距离:[0.2:0.5]    z:[0.18:0.4]
# #   demo2
#     pos = np.array([0, 35, 0.08])
#     MyRobot.Arm_move_designed(pos)

    # demo7(MyRobot.robot)
    