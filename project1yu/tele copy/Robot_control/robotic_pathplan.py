# encoding: utf-8

import numpy as np
from scipy.interpolate import CubicHermiteSpline
import matplotlib as mpl
import matplotlib.pyplot as plt
import math

# import time

# from robotic_move import*

def Robot_Path_Planning(start_angle, end_angle, end_time, d_t = 0.005):
        
    # 我用的角度，如果需要弧度可以放开下面的两行代码
    # start_angle = np.deg2rad(start_angle1)
    # end_angle = np.deg2rad(end_angle1)

    # 设定起始时间
    start_time = 0
    # t_time = np.array([0, 0, 0, 0, 0, 0])

    # 给定关节始末角速度与角加速度
    start_vel = np.array([0, 0, 0, 0, 0, 0])
    end_vel = np.array([0, 0, 0, 0, 0, 0])

    start_acc = np.array([0, 0, 0, 0, 0, 0])
    end_acc = np.array([0, 0, 0, 0, 0, 0])

    # 初始化保存每个关节角度、速度和加速度随时间变化的列表
    qq = []     # 关节角度
    vv = []     # 关节速度
    aa = []     # 关节加速度

    # 计算a0-a5，并带入多项式中求解曲线
    for i in range(len(start_angle)):
        # i = 1
        # print(len(start_angle))
        #初始状态
        t=[start_time]
        q=[start_angle[i]]
        v=[start_vel[i]]
        a=[start_acc[i]]

        a0 = start_angle[i]
        a1 = start_vel[i]
        a2 = 0.5*start_acc[i]
        # print(30*start_angle[i]-30*start_angle[0])
        a3 = (20*(end_angle[i]-start_angle[i])-(8*end_vel[i]+12*start_vel[i])*end_time+(start_acc[i]-3*end_acc[i])*math.pow(end_time, 2))/(2 * math.pow(end_time, 3))
        a4 = (-30*(end_angle[i]-start_angle[i])+(14*end_vel[i]+16*start_vel[i])*end_time-(2*start_acc[i]-3*end_acc[i])*math.pow(end_time, 2))/(2 * math.pow(end_time, 4))
        a5 = (12*(end_angle[i]-start_angle[i])-(6*end_vel[i]+6*start_vel[i])*end_time+(start_acc[i]-end_acc[i])*math.pow(end_time, 2))/(2 * math.pow(end_time, 5))
        # print(a0,a1,a2,a3,a4,a5)
        ti = np.arange(start_time, end_time, d_t)
        
        # 求解出角度，角速度，角加速度随某个时间区间随时间变换的曲线
        qi = a0 + a1 * ti + a2 * np.power(ti, 2) + a3 * np.power(ti, 3) + a4 * np.power(ti, 4) + a5 * np.power(ti, 5)
        vi = a1 + 2 * a2 * ti  + 3 * a3 * np.power(ti , 2) + 4 * a4 * np.power(ti, 3) + 5 * a5 * np.power(ti, 4)
        ai = 2 * a2 + 6 * a3 * ti + 12 * a4 * np.power(ti, 2) + 20 * a5 * np.power(ti, 3)

        # # 将矩阵转换为List，否则进行数据整合会报错
        ti = ti.tolist()  
        qi = qi.tolist()  
        vi = vi.tolist()  
        ai = ai.tolist()  
        # print(ai)

        #进行数据整合，用来绘制函数图像
        t = t + ti[1:] 
        q = q + qi[1:] 
        v = v + vi[1:] 
        a = a + ai[1:] 
        # print(type(q))

        qq.append(q)
        vv.append(v)
        aa.append(a)
    return qq, vv, aa, t

def Planning_plot(qq, vv, aa, t, joint_num):
    # 绘制图像
    plt.figure()
    for j in range(joint_num):

        plt.subplot(3,2,j+1)

        plt.plot(t, qq[j], linestyle=":", color=(1, 0, 0, 1.0), label="angle curve")  # 角度变化曲线
        plt.plot(t, vv[j], linestyle="-", color=(1, 0.5, 1, 1.0), label="speed curve")  # 角速度变化曲线
        plt.plot(t, aa[j], linestyle="-.", color=(0, 0, 0, 1.0), label="acceleration curve")  # 角加速度变化曲线
        plt.title("Angle {}".format(j+1))
        plt.xlabel("Time (s)", x=1)

        # plt.rcParams['font.size'] = 15   #设置字体大小
        plt.suptitle("5th Polynomial Interpolation (6-DOF-ROBOT)")
        plt.grid()  # 显示网格
        plt.legend()  # 显示图例

    plt.tight_layout()  # 调整子图布局
    plt.show()

def get_joint_max_dif(start_angle,end_angle):
        joint_dif = np.array(end_angle) - np.array(start_angle)
        return max(abs(joint_dif))

def end_time_test():
    with open('output-65.txt', 'r') as f:
        lines = f.readlines()

    # 转换为浮点数列表
    point = []
    for line in lines:
        nums = line.strip().split(',')
        point.append([float(num) for num in nums])
    ti = np.arange(0,point.__len__())
    point = np.array(point).T
    print(point)
    ti = ti*0.005
    print(ti)
    # 给定关节起始位置 
    start_angle = np.array([15, 45, 56, 34, 66, 90])
    end_angle = np.array([75, 45, 45, 56, 43, 70])
    end_time = 2
    get_joint_max_dif(start_angle,end_angle)
    qq, vv, aa, t = Robot_Path_Planning(start_angle, end_angle, end_time)
    # Planning_plot(point)
    for j in range(6):
        plt.subplot(3,2,j+1)
        plt.plot(ti,point[j], linestyle=":", color=(1, 0, 0, 1.0), label="angle curve")  # 角度变化曲线
        # plt.title("5th Polynomial Interpolation (6-DOF-ROBOT)")
        plt.grid()  # 显示网格
        plt.legend()  # 显示图例

    plt.tight_layout()  # 调整子图布局
    plt.show()


def main():
    # ptime = time.time()
    # 给定关节起始位置 
    start_angle = np.array([35, 45, 60, -60, 20, 90])
    end_angle = np.array([75, 45, 45, 30, 50, -20])
    max_dif = np.round(get_joint_max_dif(start_angle,end_angle))
    end_time = max_dif/50
    # print(end_time)
    qq, vv, aa, t = Robot_Path_Planning(start_angle, end_angle, end_time)
    # angle = np.array(qq).T
    # for i in range(angle.__len__()):
    #     print(angle[i])
    Planning_plot(qq, vv, aa, t, 6)
    # ctime = time.time()
    # print(ctime - ptime)


if __name__ == '__main__':
    # end_time_test()
    main()