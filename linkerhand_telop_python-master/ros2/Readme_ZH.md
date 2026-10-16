# LinkerHand ROS2 遥操作

本目录提供 LinkerForce 外骨骼手套到 LinkerHand 灵巧手的 ROS2 重定向程序。

## O6 左手

O6 左手的编译、标定、MuJoCo 仿真、通道限位、CAN 和真机操作请参阅：

[O6 左手外骨骼遥操作手册](docs/o6_left_teleoperation.md)

下面保留厂商通用 SDK 说明。

---

# 1. LinkerFFG手套

## 1.1 产品介绍
本产品的具体介绍参考，内含标定示例说明
附件1、Linker FFG(FFG01)产品说明手册

## 1.2 软件使用
本产品提供了Ubuntu下的ROS1和ROS2 SDK包，配合Linkerhand的ROS1 SDK，ROS2 SDK实现遥操作
具体使用请跳转到第四章节 遥操重定向系统

## 1.3 通讯协议
本产品的具体介绍参考附件2、Linker FFG(FFG01) 通讯协议

---

# 2. 遥操SDK系统

## 2.1 系统介绍
本系统重定向程序是基于ROS1/ROS2平台开发的，由一个节点和四个publisher的话题以及两个subscription的话题组成
Linker FFG手套通过串口协议进行数据推送到重定向程序

### 2.1.1 下载地址
https://github.com/linker-bot/linkerhand-ros-teleo
请联系客服获取最新SDK

### 2.1.2 安装遥操重定向程序
遥操重定向程序以src的形式提供，因此需要用户自行编译，才可以在多个不同的平台下使用，遥操重定向程序提供了ROS2版本,ROS1版尚在开发中

将src复制到工作空间中，然后进行编译
```bash
colcon build --symlink-install
```

过程中可能会存在权限不足的问题导致无法运行，使用以下代码
```bash
chmod 777 -R *
```

### 2.1.3 程序配置
该目录输入ls命令
```bash
cd src/linkerhand_retarget/linkerhand_retarget/config
ls -1
```
输入以上命令后正常应该进入config的目录，ls -1后即可看到以下清单
```
base_config.yml
body_custom_pose.yml
body_unity_pose.yml
hand_config.yml
human_hand_info.yml
linker_hand_info.yml
model_config.yml
retarget_config.yml
speed_config.yml
```

其中我们要唯一修改的配置文件是base_config.yml，其他配置文件均不可修改

以下是针对base_config.yml的配置内容展开进行介绍：

**1. 配置机械手类型**
```yaml
robotname_r: l10
robotname_l: l10
```
- LinkerHand_L10系列的手要配置成:l10
- LinkerHand_L20系列的手要配置成:l20
- LinkerHand_O7系列的手要配置成:o7
- LinkerHand_T25系列的手要配置成:t25
- LinkerHand_L25系列的手要配置成:l25
- LinkerHand_L6系列的手要配置成:l6
- LinkerHand_O6系列的手要配置成:o6

注意事项：以上均为小写，大写不会被正确识别

**2. 配置数据打印**
```yaml
debug:
  joint_pub_debug: false
  joint_motor_debug_r: true
  joint_motor_debug_l: false
```
- joint_pub_debug:设置为true可打印Topic Pub后的内容，刷新率较慢，可左右手同时显示
- joint_motor_debug_r:设置为true可快速刷新右手的输出数据，范围在255-0，代表张开到聚拢的变化幅度
- joint_motor_debug_l:设置为true可快速刷新左手的输出数据，范围在255-0，代表张开到聚拢的变化幅度

**3. 配置数据源**
```yaml
motion_type: linkerforce
```
- motion_type:使用LinkerFFG手套要设置为linkerforce，motion_device无需设置

### 2.1.4 程序启动(ROS2版)
之后执行以下代码确认本设备的USB清单，然后分别插入2个力反馈手套，再次执行命令，确认刷新出来的USB设备序号，以/dev/ttyUSB0和/dev/ttyUSB1为例
```bash
ls /dev/ttyUSB*
sudo chmod 666 /dev/ttyUSB0
sudo chmod 666 /dev/ttyUSB1
```

然后执行
```bash
# 启动分为两种，
# 一种带标定。第一次没标定过必须执行
ros2 run linkerhand_retarget handretarget --ros-args -p calibration := True
# 一种正常启动，默认不执行标定
ros2 run linkerhand_retarget handretarget 
```

**注意事项：**
为了客户启动方便，提供了一个启动脚本，在src同级目录下，文件名为startup_linkerforce.sh

```bash
#!/bin/bash
# LinkerForce 手套遥操作启动脚本

# 设置串口权限
sudo chmod 666 /dev/ttyUSB0
sudo chmod 666 /dev/ttyUSB1

# source ROS 工作空间(请输入正确的目录)
source ~/project/linkerhand_telop_ws/ros2/v2.8.7/install/setup.bash

# 启动程序（强制标定）
ros2 run linkerhand_retarget handretarget --ros-args -p calibration:=True

# 无标定启动
# ros2 run linkerhand_retarget handretarget 
```

编辑完成后即可这样启动
```bash
./startup_linkerforce.sh
# 如果遇到Tab无法识别的情况，请赋予权限
sudo chmod a+x ./startup_linkerforce.sh
```

### 2.1.5 程序启动(ROS1版)
之后执行以下代码确认本设备的USB清单，然后分别插入2个力反馈手套，再次执行命令，确认刷新出来的USB设备序号，以/dev/ttyUSB0和/dev/ttyUSB1为例
```bash
ls /dev/ttyUSB*
sudo chmod 666 /dev/ttyUSB0
sudo chmod 666 /dev/ttyUSB1
```

然后执行
```bash
# 启动分为两种，
# 一种带标定。第一次没标定过必须执行
rosrun ros_linkerhand_retarget handretarget.py _calibrate:=true
# 一种正常启动，默认不执行标定
rosrun ros_linkerhand_retarget handretarget.py
```

**注意事项：**
为了客户启动方便，提供了一个启动脚本，在src同级目录下，文件名为startup_linkerforce.sh

```bash
#!/bin/bash
# LinkerForce 手套遥操作启动脚本

# 设置串口权限
sudo chmod 666 /dev/ttyUSB0
sudo chmod 666 /dev/ttyUSB1

# source ROS 工作空间(请输入正确的目录)
source ~/linkerhand_telop/ros1/v2.8.6/install/setup.bash

# 启动程序（强制标定）
rosrun ros_linkerhand_retarget handretarget.py _calibrate:=true

# 无标定启动
# rosrun ros_linkerhand_retarget handretarget.py
```

编辑完成后即可这样启动
```bash
./startup_linkerforce.sh
# 如果遇到Tab无法识别的情况，请赋予权限
sudo chmod a+x ./startup_linkerforce.sh
```

## 2.2 Topic话题说明
SDK的工作原理是将上位机传输过来的数据经过清洗和整理后，统一通过Topic的方式进行数据分发，因此Topic是数据中转站，也是判断遥操工作是否正常的重要依据

- ROS1命令：`rostopic list`
- ROS2命令：`ros2 topic list`

**1. ROS2命令：**
```bash
ros2 topic echo /cb_left_hand_control_cmd
# 或
ros2 topic echo /cb_right_hand_control_cmd
```

### 数据说明

#### L25机械手（21个自由度）
| 关节 | 手指 | 动作 | 默认值 |
|------|------|------|--------|
| joint1 | 拇指 | 根部弯曲 | 255 |
| joint2 | 食指 | 根部弯曲 | 255 |
| joint3 | 中指 | 根部弯曲 | 255 |
| joint4 | 无名指 | 根部弯曲 | 255 |
| joint5 | 小指 | 根部弯曲 | 255 |
| joint6 | 拇指 | 侧摆 | 255 |
| joint7 | 食指 | 侧摆 | 128 |
| joint8 | 中指 | 侧摆 | 128 |
| joint9 | 无名指 | 侧摆 | 128 |
| joint10 | 小指 | 侧摆 | 128 |
| joint11 | 拇指 | 旋转 | 255 |
| joint12 | 预留 | - | - |
| joint13 | 预留 | - | - |
| joint14 | 预留 | - | - |
| joint15 | 预留 | - | - |
| joint16 | 拇指 | 第二关节弯曲 | 255 |
| joint17 | 食指 | 第二关节弯曲 | 255 |
| joint18 | 中指 | 第二关节弯曲 | 255 |
| joint19 | 无名指 | 第二关节弯曲 | 255 |
| joint20 | 小指 | 第二关节弯曲 | 255 |
| joint21 | 拇指 | 末端弯曲 | 255 |
| joint22 | 食指 | 末端弯曲 | 255 |
| joint23 | 中指 | 末端弯曲 | 255 |
| joint24 | 无名指 | 末端弯曲 | 255 |
| joint25 | 小指 | 末端弯曲 | 255 |

#### L20机械手（16个自由度）
| 关节 | 手指 | 动作 | 默认值 |
|------|------|------|--------|
| joint1 | 拇指 | 根部弯曲 | 255 |
| joint2 | 食指 | 根部弯曲 | 255 |
| joint3 | 中指 | 根部弯曲 | 255 |
| joint4 | 无名指 | 根部弯曲 | 255 |
| joint5 | 小指 | 根部弯曲 | 255 |
| joint6 | 拇指 | 侧摆 | 255 |
| joint7 | 食指 | 侧摆 | 128 |
| joint8 | 中指 | 侧摆 | 128 |
| joint9 | 无名指 | 侧摆 | 128 |
| joint10 | 小指 | 侧摆 | 128 |
| joint11 | 拇指 | 旋转 | 255 |
| joint12 | 预留 | - | - |
| joint13 | 预留 | - | - |
| joint14 | 预留 | - | - |
| joint15 | 预留 | - | - |
| joint16 | 拇指 | 末端弯曲 | 255 |
| joint17 | 食指 | 末端弯曲 | 255 |
| joint18 | 中指 | 末端弯曲 | 255 |
| joint19 | 无名指 | 末端弯曲 | 255 |
| joint20 | 小指 | 末端弯曲 | 255 |

#### L10机械手（10个自由度）
| 关节 | 手指 | 动作 | 默认值 |
|------|------|------|--------|
| joint1 | 拇指 | 根部弯曲 | 255 |
| joint2 | 拇指 | 侧摆 | 128 |
| joint3 | 食指 | 根部弯曲 | 255 |
| joint4 | 中指 | 根部弯曲 | 255 |
| joint5 | 无名指 | 根部弯曲 | 255 |
| joint6 | 小指 | 根部弯曲 | 255 |
| joint7 | 食指 | 侧摆 | 128 |
| joint8 | 无名指 | 侧摆 | 128 |
| joint9 | 小指 | 侧摆 | 128 |
| joint10 | 拇指 | 旋转 | 255 |

#### O7机械手（7个自由度）
| 关节 | 手指 | 动作 | 默认值 |
|------|------|------|--------|
| joint1 | 拇指 | 根部弯曲 | 255 |
| joint2 | 拇指 | 侧摆 | 128 |
| joint3 | 食指 | 根部弯曲 | 255 |
| joint4 | 中指 | 根部弯曲 | 255 |
| joint5 | 无名指 | 根部弯曲 | 255 |
| joint6 | 小指 | 根部弯曲 | 255 |
| joint7 | 拇指 | 旋转 | 255 |

#### O6/L6机械手（7个自由度）
| 关节 | 手指 | 动作 | 默认值 |
|------|------|------|--------|
| joint1 | 拇指 | 根部弯曲 | 255 |
| joint2 | 拇指 | 侧摆 | 128 |
| joint3 | 食指 | 根部弯曲 | 255 |
| joint4 | 中指 | 根部弯曲 | 255 |
| joint5 | 无名指 | 根部弯曲 | 255 |
| joint6 | 小指 | 根部弯曲 | 255 |

## 2.3 故障自检排查

### 2.3.1 网络故障
遥操和控制设备都是通过网络连接的，网络异常会导致彼此设备工作不正常，首先要检查网线的连接是否正确，保证双方的ip相互ping通以及防火墙要确保对应的端口是开放的

### 2.3.2 话题异常
在网络正常的前提下，可以在其他网络的设备上通过命令确认话题是否存在
具体细节参考 2.2 Topic话题说明

## 2.4 映射参数的设置

### 2.4.1 原理
映射系统有3种模式

| 映射模式 | 对应英文 | 对应中文 | 适用场景 |
|----------|----------|----------|----------|
| 三态映射 | origin-opose-fist | 张手-O手势-握拳 | 希望能兼顾机械手最大行程，同时能满足对指的，可以满足绝大部分映射要求 |
| 双端映射 | origin-fist | 张手-握拳 | 最大限度覆盖机械手的全部行程 |
| 对指映射 | origin-opose- | 张手-O手势 | 以对指为主要目标，以抓取为目标，提高抓取的颗粒度 |

### 2.4.2 配置文件
该映射系统配置对机械手进行参数绑定，映射配置目录在
```
~/src/linkerhand_retarget/linkerhand_retarget/motion/linkerforce/config
```

目前提供的映射参数包如下
```
g20_config.py
l10_config.py
l20_config.py
l6_config.py
o6_config.py
o7_config.py
```

### 2.4.3 配置介绍
以下以l6_config.py为例

#### 2.4.3.1 手指配置常量FINGER_CONFIGS

##### 2.4.3.1.1 公共常量表
（内容省略）

##### 2.4.3.1.2 reverse_motion属性
反转信号：默认为False

##### 2.4.3.1.3 dynamic_weight属性
动态权重：用于区间细节表达，默认为None，本版本不启用

##### 2.4.3.1.4 extended_mapping属性
扩展映射配置
```python
'extended_mapping': {    # 新增扩展映射配置
    'enabled': True,
    'scale_factor': 1.2,  # 缩放因子，默认1.0
    'extended_exp_factor': 80  # 新增：延伸指数参数
}
```
- enabled：仅在对指映射启用，False为不启用
- scale_factor：缩放因子，用于快速锁定在opose的姿态位
- extended_exp_factor：延伸参数，用于opose之后的延续动作，越小表示很难达到机械手的机械位置，利于更多的表达抓取细节，数值越大越容易快速达到驱动关节极限位置

#### 2.4.3.2 映射顺序MAPPING_ORDER
以L6为例，表示处理的关节顺序优先级
```python
MAPPING_ORDER = [
    'thumb_abduction', 'thumb_flexion', 
    'index', 'middle', 'ring', 'pinky'
]
```

#### 2.4.3.3 状态配置STATE_CONFIG（未启用）

#### 2.4.3.4 机械手OPOSE校准位：ROBOT_OPOSE_LEFT
以下以L6的配置为例，以下角度对应左手机械手的OPOSE位置，注意以下顺序对应URDF的序列
```python
ROBOT_OPOSE_LEFT = [
    1.4, 0.54, 0.0, 0.38, 0.0, 0.38, 0.0, 0.38, 0.0, 0.38, 0.0
]
```

#### 2.4.3.5 机械手OPOSE校准位：ROBOT_OPOSE_RIGHT
以下以L6的配置为例，以下角度对应右手机械手的OPOSE位置，注意以下顺序对应URDF的序列
```python
ROBOT_OPOSE_RIGHT = [
    1.4, 0.54, 0.0, 0.38, 0.0, 0.38, 0.0, 0.38, 0.0, 0.38, 0.0
]
```

### 2.4.4 机械手校准
在2.4.3.4和2.4.3.5章节已经分别介绍了左右机械手OPOSE校准位，用于处理当前机械手位置和URDF之间的差异，第一次使用的时候，需要在以下代码进行改写

路径：`./src/linkerhand_retarget/linkerhand_retarget/motion/linkerforce/hand`

以L6机械手为例，在打开linkerforce_l6.py之后，在约计149行
```python
if self.calibrationoriginal is not None 
and self.calibrationfistpose is not None 
and self.calibrationopose is not None:
    # for i in range(20):
    #     self.multi_state_mapper.debug_value[i] = joint_arc[i]
    # arc_value = self.multi_state_mapper.map_glove_to_robot(joint_arc)   # 注释掉
    arc_value = ROBOT_OPOSE_RIGHT          # 启用校准
    qpos[17] = self.g_jointpositions_arc[1] = arc_value[0]
    qpos[20] = self.g_jointpositions_arc[0] = arc_value[1]
    qpos[1] = self.g_jointpositions_arc[2] = arc_value[3]
    qpos[9] = self.g_jointpositions_arc[3] = arc_value[5]
    qpos[13] = self.g_jointpositions_arc[4] = arc_value[7]
    qpos[5] = self.g_jointpositions_arc[5] = arc_value[9]
```

改写成如上图的示例，即可启用右机械手的校准，启动后就会让机械手按照映射的角度固定在当前角度

左手对应的在360行
```python
if self.calibrationoriginal is not None and self.calibrationfistpose is not None and self.calibrationopose is not None:
    # for i in range(20):
    #     self.multi_state_mapper.debug_value[i] = joint_arc[i]
    # arc_value = self.multi_state_mapper.map_glove_to_robot(joint_arc) # 注释掉
    # 启用校准
    arc_value = ROBOT_OPOSE_LEFT
    qpos[17] = self.g_jointpositions_arc[1] = arc_value[0]
    qpos[20] = self.g_jointpositions_arc[0] = arc_value[1]
    qpos[1] = self.g_jointpositions_arc[2] = arc_value[3]
    qpos[9] = self.g_jointpositions_arc[3] = arc_value[5]
    qpos[13] = self.g_jointpositions_arc[4] = arc_value[7]
    qpos[5] = self.g_jointpositions_arc[5] = arc_value[9]
```

改写成如上图的示例，即可启用右机械手的校准，启动后就会让机械手按照映射的角度固定在当前角度

当左右两手都达到期望的对指位置后，就可以恢复原状，按正常顺序使用遥操系统
