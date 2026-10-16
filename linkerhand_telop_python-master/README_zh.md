# LinkerHand ROS 遥操作 SDK

ROS/ROS2 遥操作与运动重定向 SDK。

## 支持设备

### 数据手套与机械手型号

| 数据手套 | O6 | L7 | L10 | G20 | R20 | L25 | 模块 | 文档 |
|----------|----|----|-----|-----|-----|-----|------|------|
| LinkerFFG | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | `motion/linkerforce` | [点击此处](ros2/src/linkerhand_retarget/linkerhand_retarget/motion/linkerforce/README_zh.md) |
| VTR-DYN | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | `motion/vtrdyn` | [点击此处](ros2/src/linkerhand_retarget/linkerhand_retarget/motion/vtrdyn/README_zh.md) |
| LinkerTG | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | `motion/udexreal` | [点击此处](ros2/src/linkerhand_retarget/linkerhand_retarget/motion/udexreal/README_zh.md) |
| LinkerMCG | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | `motion/linkermcg` | [点击此处](ros2/src/linkerhand_retarget/linkerhand_retarget/motion/linkermcg/README_zh.md) |
| LinkerEG | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | `motion/linkereg` | [点击此处](ros2/src/linkerhand_retarget/linkerhand_retarget/motion/linkereg/README_zh.md) |

### 数据手套 ROS 版本支持

| 数据手套 | ROS1 Noetic 文档 | ROS2 文档 |
|----------|------------------|-----------|
| LinkerFFG | [点击此处](ros1/src/motion/linkerforce/README_zh.md) | [点击此处](ros2/src/linkerhand_retarget/linkerhand_retarget/motion/linkerforce/README_zh.md) |
| VTR-DYN | - | [点击此处](ros2/src/linkerhand_retarget/linkerhand_retarget/motion/vtrdyn/README_zh.md) |
| LinkerTG | - | [点击此处](ros2/src/linkerhand_retarget/linkerhand_retarget/motion/udexreal/README_zh.md) |
| LinkerMCG | - | [点击此处](ros2/src/linkerhand_retarget/linkerhand_retarget/motion/linkermcg/README_zh.md) |
| LinkerEG | - | [点击此处](ros2/src/linkerhand_retarget/linkerhand_retarget/motion/linkereg/README_zh.md) |

## ROS 版本兼容

| ROS版本 | Python | x86_64 | ARM64 | Windows |
|---------|--------|--------|-------|---------|
| Noetic (ROS1) | 3.8 | ✅ | ✅ | - |
| Foxy (ROS2) | 3.8 | ✅ | ✅ | - |
| Humble (ROS2) | 3.10 | ✅ | ✅ | - |
| Jazzy (ROS2) | 3.12 | ✅ | ✅ | ✅ |

## 目录结构

```
linkerhand_telop_sdk/
├── ros1/src/              # ROS1 Noetic 包
│   ├── linkerhand/        # 核心库
│   ├── motion/            # 设备驱动
│   ├── config/           # 配置文件
│   ├── scripts/          # 入口脚本
│   └── assets/           # 机器人模型资源
│
├── ros2/src/              # ROS2 包
│   └── linkerhand_retarget/
│       └── linkerhand_retarget/
│           ├── handretarget.py  # 入口脚本
│           ├── linkerhand/      # 核心库
│           ├── motion/          # 设备驱动
│           ├── config/         # 配置文件
│           ├── launch/        # Launch 文件
│           ├── assets/         # 机器人模型资源
│           ├── resource/       # ROS2 resource
│           └── tests/         # 单元/集成测试
│
├── LICENSE                # MIT许可证
└── NOTICE                 # 第三方依赖声明
```

## 快速开始

### ROS2 安装（git clone）

```bash
cd ros2_ws/src
git clone https://gitee.com/ericbrunt/linkerhand_telop_python.git
mv linkerhand_telop_python/ros2/src/linkerhand_retarget ./
rm -rf linkerhand_telop_python
cd ..
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

### ROS2 安装（下载压缩包）

```bash
cd ros2_ws/src
wget -O linkerhand_telop_python.zip https://gitee.com/ericbrunt/linkerhand_telop_python/repository/archive/master.zip
unzip linkerhand_telop_python.zip
mv linkerhand_telop_python/ros2/src/linkerhand_retarget ./
rm -rf linkerhand_telop_python.zip linkerhand_telop_python/
cd ..
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

### ROS1 安装（git clone）

```bash
cd catkin_ws/src
git clone https://gitee.com/ericbrunt/linkerhand_telop_python.git
mv linkerhand_telop_python/ros1/src ./
rm -rf linkerhand_telop_python
cd ..
catkin_make install
source install/setup.bash
```

### ROS1 安装（下载压缩包）

```bash
cd catkin_ws/src
wget -O linkerhand_telop_python.zip https://gitee.com/ericbrunt/linkerhand_telop_python/repository/archive/master.zip
unzip linkerhand_telop_python.zip
mv linkerhand_telop_python/ros1/src ./
rm -rf linkerhand_telop_python.zip linkerhand_telop_python/
cd ..
catkin_make install
source install/setup.bash
```

## 配置说明

配置文件位于 `config/base_config.yml`，主要配置项如下：

### system 系统配置

| 配置项 | 说明 | 可选值 |
|--------|------|--------|
| `motion_type` | 数据手套类型 | `linkerforce`, `vtrdyn`, `udexreal`, `udexrealv2t`, `linkermcg`, `linkereg1`, `linkereg2` |
| `datasource_type` | 数据源类型 | `motion` |
| `retargeting_type` | 重定向类型 | `projection` |
| `robotname_r` | 右手机械手型号 | `o6`, `l6`, `l7`, `l10`, `l20`, `l21`, `l25`, `g20` |
| `robotname_l` | 左手机械手型号 | 同上 |
| `motion_device` | 运动设备标识（仅 LinkerTG 有效） | 如 `eric` |

### serial 串口配置（LinkerFFG）

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `auto_scan` | 是否自动扫描串口 | `false` |
| `baudrates` | 候选波特率列表 | `[2000000, 460800, 1000000, 921600]` |
| `left.port` / `right.port` | 左右手串口路径 | 如 `/dev/ttyUSB0` |
| `left.baudrate` / `right.baudrate` | 左右手波特率 | `460800` |
| `exclude_ports` | 排除的串口列表 | `[]` |
| `serial_debug` | 串口调试开关 | `false` |

### udp 网络配置（LinkerTG / LinkerMCG / VTR-DYN）

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `ip` | 手套 UDP 服务器 IP | `192.168.11.88` |
| `port` | 手套 UDP 端口 | `8888` |
| `serverport` | 本机监听端口 | `5551` |

### calibration 标定配置（LinkerFFG）

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `show_fist` | 是否显示握拳标定 | `true` |
| `fist_extend_ratio` | 握拳延伸比例 | `0.5` |

### debug 调试配置

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `mapper_debug` | 映射器调试开关 | `false` |
| `joint_motor_debug_r` | 右手关节电机调试 | `false` |
| `joint_motor_debug_l` | 左手关节电机调试 | `false` |
| `joint_pub_debug` | 话题发布调试 | `false` |

### humanset 人体姿态配置（VTR-DYN）

| 配置项 | 说明 |
|--------|------|
| `bodyfile` | 人体姿态文件名（无扩展名） |
| `targethandfile` | 目标手机械手信息文件名（无扩展名） |

### LinkerEG 专用配置

```yaml
linkereg:
  port: null        # 串口路径，为 null 则自动扫描
  password: "i"    # sudo 密码，用于自动修复串口权限
```

### 配置示例

```yaml
system:
  motion_type: linkerforce    # 使用 LinkerFFG 机械手
  robotname_r: l25            # 右手 L25
  robotname_l: l25            # 左手 L25

serial:
  auto_scan: false
  baudrates: [2000000, 460800, 1000000, 921600]
  left:
    port: /dev/ttyUSB1
    baudrate: 460800
  right:
    port: /dev/ttyUSB0
    baudrate: 460800
```

## 仓库地址

```bash
git clone https://gitee.com/ericbrunt/linkerhand_telop_python.git
```

## 许可证

MIT License - 详见 [LICENSE](LICENSE)