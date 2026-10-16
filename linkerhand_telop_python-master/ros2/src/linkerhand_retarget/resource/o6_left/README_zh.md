# O6 左手 MuJoCo 仿真

该仿真订阅 `/cb_left_hand_control_cmd`，显示外骨骼重定向产生的灵心巧手 O6
左手姿态。仿真进程不导入 LinkerHand SDK、不访问 CAN，也不会向真机发送指令。

模型使用仓库自带的 `linkerhand_O6_left.urdf` 和 O6 网格，包含 6 个主动关节与
5 个机械随动关节。六路指令按照以下顺序映射：

```text
[拇指弯曲, 拇指侧摆, 食指弯曲, 中指弯曲, 无名指弯曲, 小指弯曲]
```

触觉传感器是 O6 的选配硬件。当前仿真负责验证运动学、关节限位和碰撞，不生成
虚假的触觉数据；真机触觉应从 O6 SDK 的力与矩阵触觉接口读取。

## 安装

```bash
python3 -m pip install --user 'mujoco>=3.2,<4'

cd /home/zmy/Projects/Dexumi/linkerhand_telop_python-master/ros2
source /opt/ros/humble/setup.zsh
colcon build --symlink-install
source install/setup.zsh
```

## 运行

左手外骨骼手套已连接时，可用联合 launch 启动完整链路：

```text
LinkerForce 左手套 -> /cb_left_hand_control_cmd -> O6 左手 MuJoCo
```

```bash
ros2 launch linkerhand_retarget o6_left_glove_sim.launch.py
```

该 launch 只读取手套串口并启动仿真，不启动 `o6_bridge`、不访问
`can0`、不会控制真手。如果串口路径不同，可覆盖参数：

```bash
ros2 launch linkerhand_retarget o6_left_glove_sim.launch.py \
  port:=/dev/ttyUSB0 baudrate:=2000000
```

O6 指令值中 `255` 接近张开，`0` 接近完全弯曲。要限制第 1 通道的
拇指最大弯曲程度，调高 `thumb_flex_min`：

```bash
ros2 launch linkerhand_retarget o6_left_glove_sim.launch.py \
  thumb_flex_min:=120
```

`0` 保留全部弯曲范围，数值越大，允许的最大弯曲越小。建议从
`80`、`100`、`120` 逐步在仿真中试验。

六通道的仿真和真机限位分开保存在：

```text
config/o6_command_limits.yaml
```

`o6_left_sim_node` 是仿真限位，`o6_bridge_node` 是真机限位。真机启动时
显式加载该文件：

```bash
ros2 run linkerhand_retarget o6_bridge --ros-args \
  --params-file install/linkerhand_retarget/share/linkerhand_retarget/config/o6_command_limits.yaml
```

首次使用左手手套或映射范围不正确时，运行交互式三姿态标定：

```bash
ros2 launch linkerhand_retarget o6_left_glove_sim.launch.py calibration:=true
```

只启动 O6 左手仿真节点：

```bash
ros2 launch linkerhand_retarget o6_left_sim.launch.py
```

默认订阅左手重定向结果。只有右手套时，可以先让右手数据驱动左手虚拟模型：

```bash
ros2 launch linkerhand_retarget o6_left_sim.launch.py \
  topic:=/cb_right_hand_control_cmd
```

窗口左上角显示话题状态、消息数、输入限幅次数和非掌部碰撞数，左下角显示六路
原始指令和关节目标角。按 `R` 恢复 O6 左手张开姿态。

仿真关节状态发布到：

```bash
ros2 topic echo /o6_left_sim/joint_states --once
```

在仿真中确认通道方向、张开/握拳姿态和运动范围之后，再单独启动真机桥接。
