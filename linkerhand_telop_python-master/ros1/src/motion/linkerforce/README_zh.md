# LinkerFFG 机械手驱动模块 (ROS1)

LinkerFFG (O6/L7/L10/G20/R20/L25) 机械手的 ROS1 驱动模块，通过串口控制机械手，支持数据手套实时映射。

---

## 快速入门

### 步骤 1：安装

```bash
cd ~/catkin_ws/src
git clone https://gitee.com/ericbrunt/linkerhand_telop_python.git
mv linkerhand_telop_python/ros1/src ./
rm -rf linkerhand_telop_python
cd ..
catkin_make install
source install/setup.bash
```

### 步骤 2：连接串口

将 LinkerFFG 机械手通过 USB 连接到电脑，确认串口权限：

```bash
# 添加当前用户到 dialout 组（需要重新登录生效）
sudo usermod -a -G dialout $USER

# 查看串口设备
ls -l /dev/ttyUSB*
```

### 步骤 3：配置机械手型号

编辑 `config/base_config.yml`，设置机械手型号：

```yaml
system:
  motion_type: linkerforce    # 数据手套类型：linkerforce（必须）
  robotname_r: l25            # 右手机械手型号：o6 / l7 / l10 / g20 / r20 / l25
  robotname_l: l25            # 左手机械手型号

serial:
  auto_scan: false            # 是否自动扫描串口
  baudrates: [2000000, 460800, 1000000, 921600]  # 波特率列表，2000000 优先
  left:
    port: /dev/ttyUSB1        # 左手套接的串口
    baudrate: 460800          # 无线460800  有线2000000
  right:
    port: /dev/ttyUSB0        # 右手套接的串口
    baudrate: 460800          # 无线460800  有线2000000
```

### 步骤 4：启动

**方式一：直接运行节点**

```bash
rosrun linkerhand_retarget handretarget
```

**方式二：指定串口启动（不修改配置文件）**

```bash
rosrun linkerhand_retarget handretarget ports:="/dev/ttyUSB0,/dev/ttyUSB1"
```

### 步骤 5：标定（如需要）

```bash
rosrun linkerhand_retarget handretarget calibration:=auto_calibrate
```

标定时按照提示做三个动作：
1. **五指张开** → 保持 5 秒
2. **握紧拳头** → 保持 5 秒
3. **O 型手势** → 保持 5 秒

---

## 机械手型号说明

| 型号 | 关节数 | 说明 |
|------|--------|------|
| O6 | 6 | 6 自由度基础款 |
| L7 | 7 | 7 自由度（拇指增加横滚） |
| L10 | 10 | 10 自由度工业款 |
| G20 | 20 | 20 自由度工业款 |
| R20 | 20 | 20 自由度研究款 |
| L25 | 25 | 25 自由度全功能款 |

配置中的 `robotname_r` 和 `robotname_l` 必须与实际连接的机械手型号匹配。

---

## 串口参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `ports` | 候选串口列表（逗号分隔） | 空（使用配置文件） |
| `baudrate` | 指定波特率（优先级高于配置） | 使用配置文件 |
| `auto_scan` | 预设失败后是否自动扫描 | `false` |

---

## 自动标定详解

### 标定配置

在 `config/base_config.yml` 中：

```yaml
calibration:
  show_fist: true       # 是否显示握拳标定步骤
  fist_extend_ratio: 0.5  # 握拳延伸比例（仅 show_fist=false 时生效）
```

### 启动标定

```bash
rosrun linkerhand_retarget handretarget calibration:=auto_calibrate
```

### 标定流程

1. **五指张开** → 保持 5 秒（电机值 255）
2. **O 型手势** → 保持 5 秒（电机中间值）

如果 `show_fist: true`，还会出现第三步：

3. **握紧拳头** → 保持 5 秒（电机值 0）

### show_fist=true 与 show_fist=false 的区别

| 对比项 | `show_fist: true` | `show_fist: false` |
|--------|-------------------|-------------------|
| 标定步骤 | 张开 → O 型 → 握拳（共 3 步） | 张开 → O 型（共 2 步） |
| 握拳数据来源 | 用户实际做握拳动作采集 | 从 O 型按比例延伸计算 |
| 握拳延伸公式 | 无 | `fist = opose + (opose - original) × fist_extend_ratio` |
| 映射精度 | 三段线性插值，最精确 | 两段插值，依赖延伸估算 |
| fist_extend_ratio | 不生效 | 控制延伸比例（默认 0.5） |

### fist_extend_ratio 详解

仅在 `show_fist: false` 时生效，用于从 O 型自动计算握拳值：

- `fist_extend_ratio = 0.5`（默认）：O 型向握拳方向延伸 50%
- `fist_extend_ratio = 0.0`：握拳值 = O 型值（无延伸）
- `fist_extend_ratio = 1.0`：握拳值 = O 型 + (O 型 - 张开) 的全量延伸
- 推荐值范围：`0.3 ~ 0.7`，需要根据实际效果调整

### 稳定性检测

- 标定时自动检测手势稳定性（方差 < 0.03）
- 需**连续稳定 5 秒**才完成采集
- 不稳定时清空重来，无超时限制
- 终端显示进度条，实时反馈稳定时长

### 标定数据存储

- 存储位置：`motion/linkerforce/tmp/jointangle_data.tmp`
- 格式：JSON
- 首次使用自动加载内置样本数据
- 重新标定会覆盖旧数据

---

## ROS 话题参考 (ROS1)

### 发布话题

| 话题名 | 消息类型 | 说明 |
|-------|---------|------|
| `/cb_right_hand_control_cmd` | `sensor_msgs/JointState` | 右手控制数据 |
| `/cb_left_hand_control_cmd` | `sensor_msgs/JointState` | 左手控制数据 |

### 订阅话题

| 话题名 | 消息类型 | 说明 |
|-------|---------|------|
| `/hand_teleop_param` | `std_msgs/String` | 动态参数控制（JSON） |

### 话题参数控制（运行时动态调整）

通过 `/hand_teleop_param` 话题动态调整运行参数，无需重启节点。

```bash
# 开启全部手指的映射器调试
rostopic pub /hand_teleop_param std_msgs/String '{data: "{\"mapper_debug\": true}"}' -1

# 用标定数据替代手套数据（用于测试）
rostopic pub /hand_teleop_param std_msgs/String '{data: "{\"force_glove_pose\": \"opose\"}"}' -1

# 恢复正常数据源
rostopic pub /hand_teleop_param std_msgs/String '{data: "{\"force_glove_pose\": \"none\"}"}' -1
```

### 可调参数

| 参数 | 说明 | 示例值 |
|------|------|--------|
| `mapper_debug` | 映射器调试开关 | `true` / `false` / `["thumb_rotate"]` |
| `mapper_exp_factor` | 延伸指数因子 | `2.0`（全局）或 `{"thumb_rotate": 2.0}`（单指） |
| `mapper_scale_factor` | 缩放因子 | `1.5`（全局）或 `{"index_root_flexion": 1.5}`（单指） |
| `force_glove_pose` | 强制手套数据源 | `open` / `fist` / `opose` / `none` |

---

## 故障排除

### 串口无法打开

```bash
# 检查串口权限
ls -l /dev/ttyUSB*
# 添加权限
sudo chmod 666 /dev/ttyUSB0
```

### 机械手无反应（重点监测话题）

按以下顺序逐项检查：

**步骤 1：检查 LinkerFFG 输出话题**

```bash
# 查看 LinkerFFG 驱动发布的话题
rostopic list | grep cb_

# 查看是否有数据输出（机械手节点是否正常发布数据）
rostopic echo /cb_right_hand_control_cmd -n 1
rostopic echo /cb_left_hand_control_cmd -n 1
```

**步骤 2：检查 LinkerFFG 关节控制话题**

```bash
# 查看关节控制指令是否有输出
rostopic echo /cb_right_hand_control_cmd -n 1
rostopic echo /cb_left_hand_control_cmd -n 1

# 查看话题频率是否正常（应该 50Hz 左右）
rostopic hz /cb_right_hand_control_cmd
```

**步骤 3：启用调试打印**

```bash
# 开启全部调试
rostopic pub /hand_teleop_param std_msgs/String '{data: "{\"mapper_debug\": true}"}' -1

# 查看终端输出，是否有手套数据变化
# 如果数据不变，说明手套未连接或话题未发布
```

**步骤 4：检查标定数据**

```bash
# 查看当前使用的标定数据
rostopic pub /hand_teleop_param std_msgs/String '{data: "{\"force_glove_pose\": \"opose\"}"}' -1
# 观察机械手是否有反应

rostopic pub /hand_teleop_param std_msgs/String '{data: "{\"force_glove_pose\": \"fist\"}"}' -1
# 观察机械手是否握紧

rostopic pub /hand_teleop_param std_msgs/String '{data: "{\"force_glove_pose\": \"none\"}"}' -1
# 恢复正常数据源
```

**步骤 5：检查串口连接**

1. 确认波特率配置与机械手一致（无线 460800，有线 2000000）
2. 尝试使用 `auto_scan: true` 自动检测
3. 检查机械手电源是否正常

**步骤 6：检查日志**

```bash
# 查看节点日志
rosrun linkerhand_retarget handretarget
# 观察终端输出的 debug 信息
```

**常见问题快速定位**

| 现象 | 可能原因 | 解决方法 |
|------|----------|----------|
| 话题无数据 | 手套未连接或话题名错误 | 检查手套连接，确认话题名 |
| 数据一直是 0 或 255 | 标定数据异常 | 重新标定或删除 `tmp/jointangle_data.tmp` |
| 数据跳变剧烈 | 串口信号干扰 | 检查串口线，使用屏蔽线 |
| 终端无输出 | 节点未启动成功 | 检查是否报错，检查依赖是否安装 |
| 话题有数据但机械手不动 | 机械手SDK未收到指令 | 检查机械手SDK连接，确认话题被正确订阅 |

---

## 文件结构

```
motion/linkerforce/
├── config/              # 手型配置文件
│   ├── o6_config.py     # O6 手型配置
│   ├── l6_config.py     # L6 手型配置
│   ├── l7_config.py     # L7 手型配置
│   ├── l10_config.py    # L10 手型配置
│   ├── l20_config.py    # L20 手型配置
│   ├── g20_config.py    # G20 手型配置
│   └── o7_config.py     # O7 手型配置
├── hand/                # 机械手驱动
│   ├── linkerforce_o6.py
│   ├── linkerforce_l6.py
│   ├── linkerforce_l7.py
│   ├── linkerforce_l10.py
│   ├── linkerforce_l20.py
│   └── linkerforce_g20.py
├── tmp/                 # 临时文件（标定数据等）
├── retarget.py          # ROS1 集成层
└── README.md            # 本文档
```