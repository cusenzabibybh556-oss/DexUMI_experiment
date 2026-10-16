# O6 左手外骨骼遥操作手册

## TransHandR 统一采集

统一平台会自动管理本页的 ROS 进程。推荐入口：

```bash
conda activate versionprodemo
cd /home/zmy/Projects/Dexumi/TransHandR0725/TransHandR
python3 scripts/run_teleoperation_platform.py --mode hardware
```

网页地址为 `http://127.0.0.1:8080/overview`。选择 `LinkerForce 外骨骼`、`O6`、`左手` 后启动系统。

| 数据 | Topic | 频率 |
|---|---|---:|
| 外骨骼原始关节 | `/operator/linkerforce/left/joint_states` | 跟随手套 |
| 实际下发 Action | `/robot/o6/left/action_applied` | 约 50 Hz |
| O6 State | `/robot/o6/left/joint_states` | 约 30 Hz |
| O6 矩阵触觉 | `/robot/o6/left/tactile_matrix` | 默认 10 Hz |

`o6_bridge` 是 O6 SDK 与 `can0` 的唯一所有者，同时向 `/cb_left_hand_matrix_touch` 发布五指触觉，供外骨骼力反馈使用。B1 触觉版还在同一个 `/robot/o6/left/tactile_matrix` 消息中提供 `palm_matrix`（14×15）；采集器将掌心事件独立写入 `dexterous_hand/palm_stream`，网页显示掌心二维热图和模型掌部热图。没有掌心板时该流为空。物理行列方向需要单点按压校正。

本文档说明以下链路的启动、标定与排障：

```text
LinkerForce 左手套
  -> /cb_left_hand_control_cmd
  -> O6 左手 MuJoCo 仿真或真机
```

O6 指令顺序为拇指弯曲、拇指侧摆、食指、中指、无名指、小指。
`255` 接近张开，`0` 接近完全弯曲。

## 1. 编译

```bash
cd /home/zmy/Projects/Dexumi/linkerhand_telop_python-master/ros2
source /opt/ros/humble/setup.zsh
python3 -m pip install --user 'mujoco>=3.2,<4'
colcon build --symlink-install --packages-select linkerhand_retarget
source install/setup.zsh
```

## 2. 左手套

检查串口：

```bash
ls -l /dev/serial/by-id/
```

启动重定向：

```bash
ros2 run linkerhand_retarget handretarget --ros-args \
  -p ports:="['/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0']" \
  -p baudrate:=2000000 \
  -p auto_scan:=True
```

新终端检查输出：

```bash
source /opt/ros/humble/setup.zsh
source /home/zmy/Projects/Dexumi/linkerhand_telop_python-master/ros2/install/setup.zsh
ros2 topic hz /cb_left_hand_control_cmd
ros2 topic echo /cb_left_hand_control_cmd
```

活动手套时，六个 `position` 必须随之变化。数值固定时不要启动真机。

### 重新标定

停止普通重定向节点后运行：

```bash
ros2 run linkerhand_retarget handretarget --ros-args \
  -p ports:="['/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0']" \
  -p baudrate:=2000000 \
  -p auto_scan:=True \
  -p calibration:=True
```

按提示依次保持五指张开、O 型手势和完全握拳。

## 3. MuJoCo 仿真

手套和仿真联合启动：

```bash
ros2 launch linkerhand_retarget o6_left_glove_sim.launch.py
```

只启动仿真：

```bash
ros2 launch linkerhand_retarget o6_left_sim.launch.py
```

仿真节点不访问 `can0`，不会控制真手。关节状态可通过以下命令检查：

```bash
ros2 topic echo /o6_left_sim/joint_states --once
```

## 4. 六通道限位

编辑：

```text
src/linkerhand_retarget/config/o6_command_limits.yaml
```

```yaml
o6_left_sim_node:
  ros__parameters:
    command_min: [100.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    command_max: [255.0, 255.0, 255.0, 255.0, 255.0, 255.0]

o6_bridge_node:
  ros__parameters:
    command_min: [100.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    command_max: [255.0, 255.0, 255.0, 255.0, 255.0, 255.0]
```

两个节点的限位相互独立。提高 `command_min` 可减小对应通道的最大
弯曲程度。修改后重新编译。

## 5. O6 真机

### 启动 CAN

```bash
cd /home/zmy/Projects/Dexumi/TransHandR0725/TransHandR
zsh scripts/start_can.sh
ip -br link show type can
```

`can0` 应显示为 `UP`。

### 干跑

保持 `handretarget` 运行，在新终端执行：

```bash
source /opt/ros/humble/setup.zsh
source /home/zmy/Projects/Dexumi/linkerhand_telop_python-master/ros2/install/setup.zsh

ros2 run linkerhand_retarget o6_bridge --ros-args \
  --params-file /home/zmy/Projects/Dexumi/linkerhand_telop_python-master/ros2/install/linkerhand_retarget/share/linkerhand_retarget/config/o6_command_limits.yaml \
  -p topic:=/cb_left_hand_control_cmd \
  -p hand_type:=left \
  -p can:=can0 \
  -p speed:="[180,180,180,180,180,180]" \
  -p torque:="[40,40,40,40,40,40]" \
  -p min_send_interval:=0.02 \
  -p dry_run:=True
```

确认 `dry_run pose` 随手套变化后再控制真机。

### 真机运行

```bash
ros2 run linkerhand_retarget o6_bridge --ros-args \
  --params-file /home/zmy/Projects/Dexumi/linkerhand_telop_python-master/ros2/install/linkerhand_retarget/share/linkerhand_retarget/config/o6_command_limits.yaml \
  -p topic:=/cb_left_hand_control_cmd \
  -p hand_type:=left \
  -p can:=can0 \
  -p speed:="[180,180,180,180,180,180]" \
  -p torque:="[40,40,40,40,40,40]" \
  -p min_send_interval:=0.02
```

`180` 是 bridge 默认速度。力矩从 `40` 起步，不足时逐步增加。

## 6. 排障和停止

```bash
ros2 node list
ros2 topic info /cb_left_hand_control_cmd -v
ip -details -statistics link show can0
pgrep -af 'handretarget|o6_bridge'
```

真机运行时，左手话题应有一个发布者和一个订阅者。

- 参数文件打不开：使用本文的绝对路径。
- `can0` 不存在：重新运行 `scripts/start_can.sh`。
- 真手只到达一个姿态：检查手套话题是否变化，必要时拔插手套并重新标定。
- 话题没有订阅者：`o6_bridge` 未运行。

停止时先结束 `o6_bridge`，再结束 `handretarget`。需要时关闭 CAN：

```bash
sudo ip link set can0 down
```
