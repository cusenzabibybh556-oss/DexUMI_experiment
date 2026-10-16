import numpy as np

import math


from dex_retargeting.constants import RobotName, RetargetingType, HandType, get_default_config_path

def keypoint2armpose_semiauto(height, width, keypoint, depth_frame_aligned):

    center_landmarks_x, center_landmarks_y = LM_mean(keypoint)
    cx, cy = int(center_landmarks_x * width), int(center_landmarks_y * height)
    if cx < 0 or cx >= width or cy < 0 or cy >= height:
        depth = None
        print('手掌超出摄像头范围')
        return None
    else:
        depth = depth_frame_aligned.get_distance(cx, cy)

    if depth != None:
        x, y, z = pix2position(cx, cy, depth, [640, 480], semiauto=True)
        print(x,y,z)

        pose = [x, y, z]
    return pose

def keypoint2armpose(height, width, keypoint, depth_frame_aligned):

    center_landmarks_x, center_landmarks_y = LM_mean(keypoint)
    cx, cy = int(center_landmarks_x * width), int(center_landmarks_y * height)
    if cx < 0 or cx >= width or cy < 0 or cy >= height:
        depth = None
        print('手掌超出摄像头范围')
        return None
    else:
        depth = depth_frame_aligned.get_distance(cx, cy)

    if depth != None:
        x, y, z = pix2position(cx, cy, depth, [640, 480])
        # print(x,y,z)

        # thumb rotation
        hand_vector_x = -normal_vertor_caculate(keypoint[5], keypoint[0], keypoint[9])
        hand_vector_z = hand_vector_z_caculate(keypoint[5], keypoint[17], keypoint[0])

        # wrist rotation
        hand_vector_x_robot = vector_normalize(pos_Transform2Robot(hand_vector_x))
        hand_vector_z_robot = vector_normalize(pos_Transform2Robot(hand_vector_z))

        rx, ry, rz = get_pose_rotate(-hand_vector_x_robot, hand_vector_z_robot)
        pose = [x, y, z, rx, ry, rz]

    return pose

def thumb_judge(keypoint):
    if keypoint is None:
        return None
    thumb_angle = angle_caculate(keypoint.landmark, 2, 3, 4, 'inspire')
    if thumb_angle >= 550:
        return 1
    else:
        return 0

def pos_ViP2RM(pose):
    pose[0] = (pose[0] + 0.6) * 1.75
    pose[1] = (pose[1] -0.1) *1.2
    pose[2] = pose[2] - 1.3
    return pose

def pos_Transform2Robot(vector):
    """"实现摄像头估计坐标到机械臂工作坐标系的坐标转换"""
    arm_x = -vector[2]
    arm_y = vector[0]
    arm_z = -vector[1]

    return np.array([arm_x, arm_y, arm_z])

def pix2position(px,py,depth,image_size,depth_fix = 0.5, semiauto = False):
    y = (px-image_size[0]/2)/image_size[0]*2
    if abs(y) > 0.58:
        y = abs(y)/y*0.58
    if semiauto:
        z = -(py-image_size[1]/2)/image_size[1]*2
        if abs(z) > 0.15:
            z = abs(z)/z*0.15
    else:
        z = (image_size[1]-py-20)/image_size[1]*0.9
        if z> 0.78:
            z = 0.78
        elif z<0:
            z = 0
    if depth<=depth_fix:
        x = 0.58
    else:
        x = 0.58-(depth-depth_fix)
    return x,y,z

def normal_vertor_caculate(a, b, c):
    vector_1 = np.array([a.x - b.x, a.y - b.y, a.z - b.z])
    vector_2 = np.array([c.x - a.x, c.y - a.y, c.z - a.z])
    return np.cross(vector_1, vector_2)

def hand_vector_z_caculate(a,b,c):
    vector_1 = np.array([a.x + b.x, a.y + b.y, a.z + b.z])/2
    vector_2 = np.array([c.x, c.y, c.z])
    hand_vector_z = vector_1 - vector_2
    return hand_vector_z

def hand_angle_caculate(left_hand_mat, robot_name):

    if left_hand_mat is None:
        return None
    if robot_name == RobotName.inspire:
        hand_angle = np.zeros(6)
        hand_angle[0] = angle_caculate(left_hand_mat, 21, 22, 23,'inspire')
        # hand_angle[0] = 0
        # angle_caculate(left_hand_mat, 17, 18, 19)
        hand_angle[1] = angle_caculate(left_hand_mat, 16, 17, 18,'inspire')
        # hand_angle[1] = 0

        # angle_caculate(left_hand_mat, 13, 14, 15)
        # hand_angle[2] = angle_caculate(left_hand_mat, 9, 10, 11,'inspire')
        
        # hand_angle[3] = angle_caculate(left_hand_mat, 5, 6, 7,'inspire')      # mediapipe
        # hand_angle[4] = np.mean([hand_angle[2],hand_angle[3]])
        hand_angle[2] = angle_caculate(left_hand_mat, 11, 12, 13,'inspire')
        hand_angle[3] = angle_caculate(left_hand_mat, 6, 7, 8,'inspire')
        hand_angle[4] = np.mean([hand_angle[2],hand_angle[3]])
        hand_angle[5] = 0

        return hand_angle.astype(np.int16)
        
    # hand_angle[5] = 0

    elif robot_name == RobotName.allegro:
        hand_angle = np.zeros(16)
        # hand_angle[1] = angle_caculate(left_hand_mat, 0, 5, 6, 'allegro')
        # hand_angle[2] = angle_caculate(left_hand_mat, 5, 6, 7,'allegro')
        # hand_angle[3] = angle_caculate(left_hand_mat, 6, 7, 8,'allegro')

        # hand_angle[5] = angle_caculate(left_hand_mat, 0, 9, 10,'allegro')
        # hand_angle[6] = angle_caculate(left_hand_mat, 9, 10, 11,'allegro')
        # hand_angle[7] = angle_caculate(left_hand_mat, 10, 11, 12,'allegro')

        # hand_angle[9] = angle_caculate(left_hand_mat, 0, 13, 14,'allegro')
        # hand_angle[10] = angle_caculate(left_hand_mat, 13, 14, 15,'allegro')
        # hand_angle[11] = angle_caculate(left_hand_mat, 14, 15, 16,'allegro')
        
        # hand_angle[12] = thumb_swing_1(left_hand_mat)
        # hand_angle[13] = thumb_swing_3(left_hand_mat)
        # hand_angle[14] = angle_caculate(left_hand_mat, 1, 2, 3,'allegro')
        # hand_angle[15] = angle_caculate(left_hand_mat, 2, 3, 4,'allegro')
        hand_angle[1] = angle_caculate(left_hand_mat, 5, 6, 7, 'allegro')
        hand_angle[2] = angle_caculate(left_hand_mat, 6, 7, 8, 'allegro')
        hand_angle[3] = angle_caculate(left_hand_mat, 7, 8, 9,'allegro')

        hand_angle[5] = angle_caculate(left_hand_mat, 10, 11, 12,'allegro')
        hand_angle[6] = angle_caculate(left_hand_mat, 11, 12, 13,'allegro')
        hand_angle[7] = angle_caculate(left_hand_mat, 12, 13, 14,'allegro')

        hand_angle[9] = angle_caculate(left_hand_mat, 15, 16, 17,'allegro')
        hand_angle[10] = angle_caculate(left_hand_mat, 16, 17, 18,'allegro')
        hand_angle[11] = angle_caculate(left_hand_mat, 17, 18, 19,'allegro')
        
        hand_angle[12] = thumb_swing_1(left_hand_mat)
        hand_angle[13] = thumb_swing_3(left_hand_mat)
        hand_angle[14] = angle_caculate(left_hand_mat, 1, 2, 3,'allegro')
        hand_angle[15] = angle_caculate(left_hand_mat, 2, 3, 4,'allegro')


        return hand_angle


"""--------------------only for allegro hand--------------------"""
def thumb_swing_1(points):

    #手掌法向量
    # v1 = landmark_p2v(points[9], points[0])
    # v2 = landmark_p2v(points[17], points[0])
    v1 = landmark_p2v(points[10], points[0])
    v2 = landmark_p2v(points[20], points[0])
    n = np.cross(v2, v1)
    n_normalized = n / np.linalg.norm(n)

    # 计算大拇指方向向量
    t = landmark_p2v(points[2], points[0])

    # 计算t与平面夹角
    cos_theta = np.dot(t, n_normalized) / np.linalg.norm(t)
    theta = np.arccos(cos_theta)
    theta=np.pi/2-theta

    # 转换为角度（可选）
    # theta_degrees = np.degrees(theta)

    return -min(0.5,theta)/0.5*np.pi/2


def thumb_swing_2(points):

    a = landmark_p2v(points[2], points[0])
    b = landmark_p2v(points[4], points[2])
    # 计算点积
    dot_product = np.dot(a, b)
    # 计算叉积
    cross_product = np.cross(a, b)
    # 计算模
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    cross_norm = np.linalg.norm(cross_product)
    # 计算余弦值和正弦值
    cos_theta = dot_product / (a_norm * b_norm)
    sin_theta = cross_norm / (a_norm * b_norm)
    # 确定夹角的正负
    sign = np.sign(cross_product[1])  # 根据右手定则对左手坐标系进行分析，y朝下为正
    # 计算夹角
    theta = np.arctan2(sin_theta * sign, cos_theta)
    # print('theta:',theta)
    return -theta


def thumb_swing_3(points):
    distance=[]
    # for i in range(8,17,4):
    for i in range(9,20,5):
        distance.append(np.linalg.norm(landmark_p2v(points[i], points[4])))

    index=np.argmin(distance)
    if index==0:
        return 0
    elif index==1:
        return np.pi/8
    else:
        return np.pi/4

"""-------------------------------------------------------------------"""

def landmark_p2v(a, b):
    # return np.array([a.x-b.x,a.y-b.y,a.z-b.z])
    return np.array([a[0] - b[0], a[1] - b[1], a[2] - b[2]])


def angle_caculate(landmarks, i, j, k, hand):
    # vector_1 = np.array(
    #     [landmarks[i].x - landmarks[j].x, landmarks[i].y - landmarks[j].y,
    #      landmarks[i].z - landmarks[j].z])
    # vector_2 = np.array(
    #     [landmarks[k].x - landmarks[j].x, landmarks[k].y - landmarks[j].y,
    #      landmarks[k].z - landmarks[j].z])
    
    vector_1 = np.array(
        [landmarks[i][0] - landmarks[j][0], landmarks[i][1] - landmarks[j][1],
         landmarks[i][2] - landmarks[j][2]])
    vector_2 = np.array(
        [landmarks[k][0] - landmarks[j][0], landmarks[k][1] - landmarks[j][1],
         landmarks[k][2] - landmarks[j][2]])

    if hand =='inspire':
        hand_angle = np.arccos(np.dot(vector_1, vector_2) / (np.linalg.norm(vector_1) * np.linalg.norm(vector_2)))*0.92

        if hand_angle < np.pi/1.5:
            hand_angle = np.pi/2
        elif hand_angle > np.pi/1.5:
            hand_angle = np.pi
    
        return (hand_angle - np.pi / 2) / (np.pi / 2)
    
    elif hand == 'allegro':
        
        hand_angle = np.arccos(np.dot(vector_1, vector_2) / (np.linalg.norm(vector_1) * np.linalg.norm(vector_2)))
        # print(vector_1,vector_2)
        if hand_angle < np.pi / 2:
            hand_angle = np.pi / 2
        elif hand_angle > np.pi:
            hand_angle = np.pi
        return (np.pi-hand_angle)


def calOrientationFrom2Vectors(Vs1, Vs2, Vn1, Vn2):
        '''
        功能：根据两个不共线的矢量分别在两个坐标系下的矢量坐标求解这两个坐标系之间的旋转矩阵；
        输入:两个矢量在坐标系s下的坐标:Vs1(xs1,ys1,zs1),Vs2(xs2,ys2,zs2);
            在坐标系n下的坐标:Vn1(xn1,yn1,zn1),Vn2(xn2,yn2,zn2);
        输出:坐标系n到坐标系s的旋转矩阵Rns, 输出格式为矩阵;
        '''
        # frame s
        a = Vs1
        b = np.cross(Vs1, Vs2) / np.linalg.norm(np.cross(Vs1, Vs2))
        c = np.cross(a, b)
        # 参考坐标系frame d
        Rsd = np.array([a, b, c])

        # frame n
        A = Vn1
        B = np.cross(Vn1, Vn2) / np.linalg.norm(np.cross(Vn1, Vn2))
        C = np.cross(A, B)
        Rnd = np.array([A, B, C])
        Rns = np.dot(np.linalg.inv(Rsd), Rnd)

        return Rns

def rotateMatrixToEulerAngles(Rotate_matrix):
    """
    旋转矩阵到欧拉角，弧度制
    """
    theta_z = np.arctan2(Rotate_matrix[1, 0], Rotate_matrix[0, 0])
    theta_y = np.arctan2(-1 * Rotate_matrix[2, 0], np.sqrt(Rotate_matrix[2, 1] * Rotate_matrix[2, 1] + Rotate_matrix[2, 2] * Rotate_matrix[2, 2]))
    theta_x = np.arctan2(Rotate_matrix[2, 1], Rotate_matrix[2, 2])
    # print(f"Euler angles:\ntheta_x: {theta_x}\ntheta_y: {theta_y}\ntheta_z: {theta_z}")
    return theta_x, theta_y, theta_z

def LM_mean(hand_landmarks):
    x = (hand_landmarks[0].x +hand_landmarks[5].x + hand_landmarks[17].x) /3
    y = (hand_landmarks[0].y +hand_landmarks[5].y + hand_landmarks[17].y) /3
    return x, y
        

def get_pose_rotate(Vs1,Vs2):
    """
    获取到位姿信息rx,ry,rz
    """
    Vn1 = [1,0,0]
    Vn2 = [0,0,1]
    matrix = calOrientationFrom2Vectors(Vs1, Vs2, Vn1, Vn2)
    rotate_matrix = np.array(matrix)
    rx,ry,rz = rotateMatrixToEulerAngles(rotate_matrix)
    return rx, ry, rz

def vector_normalize(vector):
    value = math.sqrt(sum(vector[0:3]**2))
    norm_vector = vector/value
    return norm_vector


def vector_limitation(vector):
    vector = np.array(vector)
    vector_value = math.sqrt(sum(vector**2))
    if vector_value >= 0.6:
        vector_return = vector/vector_value*0.6
    else: 
        vector_return = vector
    return vector_return[0],vector_return[1],vector_return[2]




# Threshold Judge
def isFingerThreshold(finger_angle, finger_temp, threshold = 300):
    finger_angle = np.array(finger_angle)
    finger_temp = np.array(finger_temp)
    dif_finger = sum(abs(finger_angle - finger_temp))
    if dif_finger>=threshold:
        return True
    else:
        return False
    
def isArmThreshold(arm_pose, arm_temp, lower_threshold = 0.05):
    arm_pose = np.array(arm_pose)
    arm_temp = np.array(arm_temp)
    dif_arm = 5*sum(abs(arm_pose[0:3] - arm_temp[0:3])) + 0.6*sum(abs(arm_pose[3:]-arm_temp[3:]))
    if dif_arm >= lower_threshold:
        return True
    else:
        return False