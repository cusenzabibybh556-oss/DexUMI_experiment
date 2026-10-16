import mediapipe as mp
import cv2
import numpy as np

from Robot_control.pos_process import*
from Robot_control.robotic_move import *
from Robot_control.Realsense import *



# 基本参数设定

ip = '192.168.1.18'         # 机械臂若采用无限网口连接，ip为 192.168.33.xx, 以太网连接，ip为 192.168.1.xx

sampling_rate = 2         # Mediapipe输出坐标采样间隔
count = 0                   # 计数器[hand,pose]
img_size = [640, 480]           # 图像大小

# RM65机械臂连接

# MyRobot = Myrobot(ip)

# Mediapipe配置
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

# 创建管道并配置深度与RGB流
pipeline, config = realsense_camara_enable(img_size)
align = rs.align(rs.stream.color)
profile = pipeline.start(config)


# 手指和手臂信息寄存
finger_angle_temp = None
hand_move = False
arm_pos_temp = None


# 用于计算帧数的计时器
pTime = 0
cTime = 0

with mp_holistic.Holistic(
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as holistic:
    while True:

        frames = pipeline.wait_for_frames()

        color_image_aligned,depth_frame_aligned,depth_colormap_aligned=Align_version(frames,align,show_pic=1)

        color_image_aligned = cv2.cvtColor(color_image_aligned, cv2.COLOR_BGR2RGB)
        Mediapipe_result = holistic.process(color_image_aligned)

        color_image_aligned = cv2.cvtColor(color_image_aligned, cv2.COLOR_RGB2BGR)

        height, width, channel = color_image_aligned.shape

        mp_drawing.draw_landmarks(
            color_image_aligned,
            Mediapipe_result.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
        
        if Mediapipe_result.right_hand_landmarks:
            right_landmarks = Mediapipe_result.right_hand_landmarks.landmark

            count += 1
            if count % sampling_rate == 0:
                count = 0
                
                center_landmarks_x, center_landmarks_y = LM_mean(right_landmarks)
                cx, cy = int(center_landmarks_x *width), int(center_landmarks_y *height)
                if cx<0 or cx>= width or cy<0 or cy>=height:
                    print('手掌超出摄像头范围')
                else:
                    depth = depth_frame_aligned.get_distance(cx,cy)

                x,y,z = pix2position(cx,cy,depth, img_size)
                # print(x,y,z)

                # finger angle
                angle_right_finger = []

                angle_right_finger.append(angle_caculate(right_landmarks, 17, 18, 19))  # 1号手指(小拇指)
                angle_right_finger.append(angle_caculate(right_landmarks, 13, 14, 15))  # 2号手指(无名指)
                angle_right_finger.append(angle_caculate(right_landmarks, 9, 10, 11))   # 3号手指(中指)
                angle_right_finger.append(angle_caculate(right_landmarks, 5, 6, 7))     # 4号手指(食指)
                angle_right_finger.append(angle_caculate(right_landmarks, 2, 3, 4))     # 5号手指(大拇指)

                # thumb rotation
                hand_vector_x = normal_vertor_caculate(right_landmarks[5], right_landmarks[0], right_landmarks[9])
                hand_vector_z = hand_vector_z_caculate(right_landmarks[5],right_landmarks[17],right_landmarks[0])
                thumb_normal_vector = -normal_vertor_caculate(right_landmarks[5], right_landmarks[0], right_landmarks[1])
                thumb_rotation = np.arccos(np.dot(hand_vector_x, thumb_normal_vector) / (
                        np.linalg.norm(hand_vector_x) * np.linalg.norm(thumb_normal_vector)))
                if thumb_rotation > np.pi / 2:
                    thumb_rotation = np.pi / 2
                elif thumb_rotation < 0:
                    thumb_rotation = 0
                thumb_rotation = thumb_rotation / (np.pi / 2) * 1000
                thumb_rotation = (thumb_rotation -300)*2.8
                thumb_rotation = -thumb_rotation+800
                if thumb_rotation >= 1000:
                    thumb_rotation = 1000
                elif thumb_rotation <=0:
                    thumb_rotation = 0
                
                angle_right_finger.append(int(thumb_rotation))
                
                if finger_angle_temp == None:
                    finger_angle_temp = angle_right_finger
                elif isFingerThreshold(angle_right_finger,finger_angle_temp):
                    print('-----------------------------------------------------------')
                    print('手指开始移动:',angle_right_finger)
                    
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
                    rx, ry ,rz = get_pose_rotate(hand_vector_x_robot, hand_vector_z_robot)
                    pose = [x,y,z,rx,ry,rz]
                    if arm_pos_temp == None:
                        arm_pos_temp = pose
                    elif isArmThreshold(pose,arm_pos_temp):
                        print('-----------------------------------------------------------')
                        print('机械臂开始移动:',pose)
                        # MyRobot.arm_move_bypose_CANFD(pose)
                        arm_pos_temp = pose
                    else:
                        arm_pos_temp = pose
                    
        
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        # images_aligned = np.hstack((color_image_aligned, depth_colormap_aligned))
        # cv2.putText(images_aligned,str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
        # cv2.imshow('MediaPipe Holistic', images_aligned)

        cv2.putText(color_image_aligned,str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
        cv2.imshow('MediaPipe Holistic', color_image_aligned)


        if cv2.waitKey(5) & 0xFF == 27:
            break

# cap.release()
pipeline.stop()
cv2.destroyAllWindows()
