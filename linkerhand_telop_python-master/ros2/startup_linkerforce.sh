#!/bin/bash
# LinkerForce 手套遥操作启动脚本

# 设置串口权限
sudo chmod 666 /dev/ttyUSB0
sudo chmod 666 /dev/ttyUSB1

# source ROS 工作空间(请输入正确的目录)
source /home/linker-brunt/project/linkerhand_telop_sdk/v2/ros2/install/setup.bash

# 启动程序（强制标定）
# ros2 run linkerhand_retarget handretarget --ros-args -p calibration:=True

# 无标定启动
ros2 run linkerhand_retarget handretarget 