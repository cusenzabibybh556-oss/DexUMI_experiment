# DEXUMI 工程内容整理

本文档基于当前 `Dexumi` 文件夹内的工程内容整理，目标是说明整个工程文件夹在做什么、采用了什么流程、最终产出是什么。本文尽量覆盖工程结构、数据流、训练/采集/验证流程和各模块职责，但不展开具体代码实现。

## 1. 工程总体定位

`Dexumi` 不是一个单一脚本工程，而是一个围绕“人手到灵巧机器人手”的多源数据采集、遥操作、重定向、模仿学习训练和真机验证的综合工程目录。

从功能上看，它包含四条主要线索：

1. DexUMI 原始工程线：使用人手/外骨骼作为通用操作接口，采集外骨骼、相机、触觉/力传感、手部动作等数据，经过重放、分割、去除外骨骼、生成训练集，最终训练 Diffusion Policy，并在真机灵巧手上评估。

2. TransHandR 工程线：基于 Vision Pro 的手部 25 点数据，训练或运行一个从人手关键点到机器人灵巧手关节角的 Transformer 模型，实现实时重定向和 HDF5 数据集测试。

3. TeleOperation 工程线：使用操纵杆、机械臂、灵巧手和 RealSense 相机进行远程操控与数据采集，将目标/实际状态、RGB-D 图像等保存成 HDF5 episode。

4. 多源采集整合线：在 TransHandR 实时重定向基础上，引入 RealSense D405/D435i、灵巧手真实反馈、最终 action、VIVE Tracker 6DoF 位姿，并以统一时间轴写入质量优先的 HDF5 episode。

整体目标可以概括为：

> 采集人手、灵巧手、相机和 Tracker 等多源同步数据；将人手运动重定向到真实灵巧手；沉淀可训练的数据集；训练策略模型；最后用于真机灵巧操作任务。

## 2. 顶层目录概览

当前工程的主要目录和文件如下：

| 路径 | 作用 |
|---|---|
| `docs/` | 项目设计文档，尤其是 TransHandR 多源数据采集需求设计。 |
| `project1yu/DexUMI-main/` | DexUMI 原始工程，包含外骨骼采集、数据生成、Diffusion Policy 训练、真机评估、硬件优化等完整流程。 |
| `project1yu/TeleOperation/` | 机械臂/灵巧手遥操作与数据采集工程，包含操纵杆进程、控制进程、存储进程和 HDF5 采集脚本。 |
| `project1yu/tele/` | dex-retargeting/AnyTeleop 风格的人手到机器人手重定向工程，含手部检测、优化器、URDF 资产、LinkerHand 控制代码。 |
| `project1yu/act-main/` | ACT 模仿学习工程，包含 Action Chunking Transformer 训练、仿真环境、HDF5 episode 可视化工具。 |
| `project1yu/SAM2/` | SAM2 视频/网页分割辅助脚本，用于图像/视频分割相关处理。 |
| `project1yu/pyvut/`、`project1yu/pyvut-main/` | VIVE Ultimate Tracker 的 6DoF 位姿读取与可视化工具，支持 USB/dongle 模式。 |
| `TransHandR0725/TransHandR/` | 当前重点整合工程，包含 TransHandR 模型、训练、实时推理、灵巧手控制、多源采集器、校验脚本和文档。 |
| `project1yu/*.pdf` | 参考论文，包括 DexGrasp-VLA、AnyTeleop、TransRetarget 等相关工作。 |
| `*.zip` | 原始工程压缩包备份，如 DexUMI、ACT、tele 等。 |

## 3. 关键子工程说明

### 3.1 DexUMI-main：外骨骼到策略学习的完整流程

`project1yu/DexUMI-main` 是 DexUMI 原始工程，定位是 “Universal Manipulation Interface for Dexterous Manipulation”。它的核心思想是把人手或人手外骨骼作为通用操作接口，通过外骨骼采集演示数据，再转化成机器人灵巧手可学习的数据。

主要模块包括：

| 模块 | 作用 |
|---|---|
| `dexumi/camera/` | 相机抽象与具体设备接入，如 iPhone、OAK、RealSense、调试相机、二维码位姿估计。 |
| `dexumi/data_recording/` | 数据缓冲、数值记录、视频记录、record manager。 |
| `dexumi/encoder/` | 数值编码、力/触觉传感器、UART 读取、XHand 触觉数据。 |
| `dexumi/diffusion_policy/` | Diffusion Policy 策略模型实现。 |
| `dexumi/real_env/` | 真实环境运行策略、ring buffer、SpaceMouse 等真机评估组件。 |
| `dexumi/hand_sdk/` | 灵巧手 SDK 封装。 |
| `real_script/data_collection/` | 外骨骼采集入口。 |
| `real_script/data_generation_pipeline/` | 数据插值、重放、转图片、分割、去外骨骼、合成视频、生成最终训练数据集。 |
| `real_script/policy_training/` | Diffusion Policy 训练脚本。 |
| `real_script/eval_policy/` | XHand/Inspire Hand 真机策略评估脚本。 |
| `linkage_optimization/` | 外骨骼/连杆结构优化、轨迹匹配、URDF 写入和可视化。 |

DexUMI 的标准流程是：

1. 搭建外骨骼硬件，例如 Inspire Hand 或 XHand 外骨骼。
2. 使用外骨骼、相机、力/触觉传感器等采集 episode。
3. 将外骨骼轨迹重放到真实灵巧手，录制机器人手视角数据。
4. 对视频帧进行分割、去除外骨骼或人手遮挡、合成对比视频。
5. 生成策略训练用的最终数据集。
6. 使用 Diffusion Policy 训练策略。
7. 启动真机服务器并在 XHand/Inspire Hand 上评估策略。

典型产出包括：

- 原始采集 episode。
- replay 后的机器人手数据。
- 分割 mask、去外骨骼图像、合成 debug 视频。
- 策略训练数据集。
- Diffusion Policy 模型 checkpoint。
- 真机评估结果。
- 硬件连杆优化参数和可视化结果。

### 3.2 TransHandR：Vision Pro 手部关键点到灵巧手角度

`TransHandR0725/TransHandR` 是当前目录里非常重要的本地开发工程。它的目标是利用 Vision Pro 或 HDF5 数据中的人手关键点，预测机器人灵巧手的关节角，从而完成实时重定向。

主要模块包括：

| 模块 | 作用 |
|---|---|
| `config/` | 全局变量、相机参数、数据集配置、训练参数、骨架/四元数/可视化工具。 |
| `dataset/` | 数据集加载、生成器、HDF5 数据处理。 |
| `model/` | PoseTransformer/TransHandR 模型、运动学、角度转真实手控制值、损失函数。 |
| `scripts/` | 训练、测试、实时推理、可视化、采集、校验等脚本入口。 |
| `LinkerHand/` | LinkerHand 灵巧手控制接口。 |
| `inspire_hands/` | Inspire Hand 控制相关包。 |
| `yumi_gym/` | 仿真环境接口。 |
| `logs/` | 训练日志。 |
| `checkpoint/` | 模型 checkpoint 目录，当前文件列表中主要通过代码路径引用。 |

TransHandR 模型大致流程是：

1. 输入 Vision Pro 或 HDF5 中的人手 25 个关键点，每个关键点为三维坐标。
2. 模型先做空间编码，理解同一帧内各手部关节之间的关系。
3. 再做时序编码，利用多帧上下文提升预测稳定性。
4. 输出机器人灵巧手关节角或控制向量。
5. 通过运动学、角度限制、碰撞约束等损失，让预测结果更符合真实机械手结构。
6. 实时模式下将预测结果发送到 Inspire Hand、LinkerHand L10/L21 或仿真环境。

TransHandR 的常见脚本用途：

| 脚本/文档 | 作用 |
|---|---|
| `main_train.py`、`scripts/main_train.py` | 训练 TransHandR/PoseTransformer 模型。 |
| `scripts/main_test.py`、`scripts/main_test_cpu.py` | 模型测试。 |
| `scripts/main_h5_simulate.py` | 使用 HDF5 数据在仿真中测试。 |
| `scripts/main_h5_realtime.py` | 使用 HDF5 数据做实机测试。 |
| `scripts/main_visionpro_realtime_*` | Vision Pro 实时输入到不同灵巧手或仿真手的重定向。 |
| `scripts/main_visionpro_multi_hand.py` | 多手型实时重定向入口，支持 telemetry 发布给采集器。 |
| `scripts/check_gym.py` | 检查仿真环境和 URDF 路径。 |
| `scripts/check_hand_connect.py` | 检查灵巧手连接。 |
| `scripts/check_model.py` | 检查模型参数是否正常。 |
| `scripts/check_visionpro_connect.py` | 检查 Vision Pro 网络连接。 |

TransHandR 的产出包括：

- 训练日志。
- 模型 checkpoint。
- 预测的灵巧手关节角。
- 仿真/实机重定向结果。
- HDF5 数据集测试结果。
- 论文绘图用的 loss、attention、轨迹等图表。

### 3.3 多源采集器：quality-first V1 HDF5 episode

`TransHandR0725/TransHandR/scripts/collect_transhandr_multisource.py` 是当前工程里最完整的新采集器。它不是简单地按固定 30Hz 人工写入，而是以 D405 RGB-D frameset 作为主时间轴，每个 D405 有效帧生成一个 timestep，再把其他源对齐到这个时间点。

它要解决的问题是：

> 在 TransHandR 控制灵巧手运行时，同时采集 Vision Pro 右手数据、真实灵巧手反馈、最终 action、D405 手腕相机、可选 D435i 头部相机、可选 VIVE Tracker，并保存为结构化 HDF5 episode。

数据源包括：

| 数据源 | 内容 | 说明 |
|---|---|---|
| D405 | 手腕视角 RGB-D | 主时间轴，每个 frameset 产生一行。 |
| D435i | 头部 RGB-D | 可选，按 D405 时间戳做 nearest 对齐。 |
| Vision Pro / TransHandR | 右手 25 点三维关键点 | 只采右手，不用左手兜底。 |
| 灵巧手 state | 真实设备反馈角度/位置 | 区分真实反馈 state 和下发 action。 |
| action | 实际成功下发的最终控制命令 | 不保存模型中间输出，只保存成功发送后的 final command。 |
| VIVE Tracker | 手腕 Tracker 位置和四元数 | 可选，支持未启用时写 invalid 占位。 |

当前实现的关键同步策略：

| 源 | 对齐方式 |
|---|---|
| D405 | 主源，不对齐到别人。 |
| D435i | nearest 最近邻匹配。 |
| right hand | nearest 最近邻匹配。 |
| hand state | 线性插值。 |
| tracker position | 线性插值。 |
| tracker quaternion | SLERP 球面线性插值。 |
| action | latest-before，只使用主时间戳之前最近一次成功下发 action，避免 future action 泄漏。 |

它支持两种运行模式：

| 模式 | 作用 |
|---|---|
| `mock` | 无硬件验收模式，生成模拟相机、手部、手状态、action、Tracker 数据，可用于检查 HDF5 schema 和 Dashboard。 |
| `hardware` | 真机模式，接入 D405/D435i、Vision Pro、灵巧手、VIVE Tracker 等真实设备。 |

它还包含以下工程能力：

- 支持网页 Dashboard，默认 `127.0.0.1:8080`。
- 支持网页按钮或键盘 `R/S/Q` 控制开始、停止和退出。
- 支持录制通道选择，D405 必选，其他源可选。
- 支持 episode 自动递增命名。
- 支持最大录制时长和最大文件大小限制。
- 支持保存后生成质量摘要。
- 支持独立 HDF5 校验脚本 `validate_multisource_episode.py`。

新采集器产出的 HDF5 schema 版本为 `quality-first-v1.1`。主要字段包括：

| HDF5 分组 | 内容 |
|---|---|
| `timestamps/master_timestamp_ns` | D405 主时间轴。 |
| `observations/images/d405_rgb` | D405 RGB 图像。 |
| `observations/depth/d405_depth` | D405 深度图。 |
| `observations/images/d435i_rgb` | 可选 D435i RGB 图像。 |
| `observations/depth/d435i_depth` | 可选 D435i 深度图。 |
| `observations/right_hand/world_keypoints` | Vision Pro/TransHandR 右手世界坐标关键点。 |
| `observations/right_hand/relative_keypoints` | 右手相对 wrist 坐标关键点。 |
| `dexterous_hand/joint_positions` | 灵巧手真实反馈关节位置。 |
| `action/command` | 实际成功下发的 action。 |
| `vive_tracker/pos` | Tracker 位置。 |
| `vive_tracker/quat` | Tracker 四元数，顺序为 `[w, x, y, z]`。 |
| `camera/d405/*`、`camera/d435i/*` | 相机帧号、设备时间戳、接收时间戳、valid 标志。 |
| `sync/*` | 各源同步误差、插值 gap 或 action age。 |
| `calibration/*` | D405/D435i 外参预留，未标定时为单位矩阵并标记未校准。 |
| `quality` | 保存后的帧数、时长、频率、有效率、同步误差等质量指标。 |

与早期 `docs/TransHandR_多源数据采集需求设计.md` 相比，新采集器做了几处重要调整：

- 早期设计是固定 30Hz 主时间轴；新实现以 D405 frameset 作为主时间轴。
- 早期设计保存 raw matrices、Euler、完整 SDK state 等；quality-first V1 更强调稳定可训练字段。
- 早期设计不保存 action；新实现明确保存成功下发后的 final action。
- 早期设计对所有源使用 nearest 和统一 50ms 阈值；新实现按源分别设置阈值，并对 state/tracker 使用插值。
- 新实现要求 invalid 帧使用统一占位：float 为 NaN，timestamp/int 为 -1，图像为 0，并配合 valid 字段读取。

### 3.4 TeleOperation：机械臂和灵巧手遥操作采集

`project1yu/TeleOperation` 是一个更传统的遥操作采集工程。它的主要目标是通过操纵杆读取目标关节/夹爪状态，控制机械臂和灵巧手，同时采集相机数据和实际状态，保存为 HDF5。

主要组件：

| 文件 | 作用 |
|---|---|
| `main.py` | 多进程入口，创建共享内存、队列，并启动 joystick/control/storage 三个进程。 |
| `teleop_collect_hdf5.py` | 较完整的 HDF5 采集入口，集成机械臂、灵巧手和 RealSense 双相机。 |
| `shared_memory.py` | 跨进程共享 6 个机械臂关节目标、夹爪目标、退出标志。 |
| `joystick_process.py` | 读取操纵杆输入并写入共享内存。 |
| `control_process.py` | 从共享内存读取目标，控制机械臂/灵巧手，并把数据放入队列。 |
| `storage_process.py` | 从队列读取数据，缓存后写入磁盘。 |
| `robot_interface.py` | RM65 机械臂、LinkerHand L10、Inspire Hand 控制封装。 |
| `config.py` | 机械臂 IP、操纵杆串口、控制频率、相机序列号、灵巧手类型、保存路径等配置。 |

TeleOperation 的典型流程：

1. 主进程创建共享内存。
2. 操纵杆进程以高频读取目标关节和夹爪命令。
3. 控制进程以固定控制频率读取目标命令。
4. 控制进程向 RM65 机械臂、L10/Inspire 灵巧手发送命令。
5. 采集实际机械臂关节、灵巧手状态、腕部/前方 RealSense RGB-D。
6. 存储进程将数据按 episode 写入 HDF5。

`teleop_collect_hdf5.py` 的 HDF5 产出字段主要包括：

- `timestamps`
- `observations/arm_joints`
- `observations/hand_joints`
- `observations/images/wrist_RGB`
- `observations/images/front_RGB`
- `observations/depth/wrist_depth`
- `observations/depth/front_depth`

它的当前特点和限制：

- 支持 episode 自动扫描已有文件并递增编号。
- 支持腕部相机和前方相机。
- 可以保存实际机械臂状态和灵巧手状态。
- Inspire Hand 的实际 state 在部分封装中会回退为最近下发 command，这也是后来 quality-first V1 明确要改掉的问题。
- 它更像早期遥操作数据采集系统，不完全满足最新多源采集规范。

### 3.5 tele：dex-retargeting 与 LinkerHand 控制实验

`project1yu/tele` 基于 dex-retargeting/AnyTeleop 思路，负责把人手运动转换为机器人灵巧手运动。

主要内容包括：

| 模块 | 作用 |
|---|---|
| `dex_retargeting/` | 重定向优化器、运动学适配、URDF wrapper、配置和序列重定向。 |
| `example/vector_retargeting/` | 从手部视频或摄像头检测人手并实时重定向。 |
| `example/position_retargeting/` | 从手-物体位姿数据重定向到机器人手。 |
| `Robot_control/` | 机械臂、RealSense、LinkerHand、路径规划、位置处理等控制实验。 |
| `linker_hand_python_sdk/` | LinkerHand SDK。 |
| `assets/` | 多种机器人手 URDF/模型资产和渲染/碰撞处理工具。 |

它的作用更偏底层算法和实验验证：

- 人手关键点检测。
- 人手到机器人手的向量/位置重定向。
- 机器人手 URDF 解析和关节顺序适配。
- 使用 SAPIEN/IsaacGym/可视化工具验证重定向结果。
- LinkerHand 控制实验。

### 3.6 act-main：ACT 模仿学习基线

`project1yu/act-main` 是 Action Chunking with Transformers 工程。它与 DexUMI/TransHandR 的关系主要是提供模仿学习训练和 HDF5 episode 可视化参考。

主要模块：

| 文件/目录 | 作用 |
|---|---|
| `imitate_episodes.py` | 训练和评估 ACT 策略。 |
| `policy.py` | ACT policy 适配层。 |
| `detr/` | ACT 使用的 DETR/Transformer 模型定义。 |
| `sim_env.py`、`ee_sim_env.py` | MuJoCo/DM Control 仿真环境。 |
| `record_sim_episodes.py` | 生成仿真 episode。 |
| `visualize_episodes.py` | 从 HDF5 数据集导出视频/可视化结果。 |

它的典型产出：

- 仿真 HDF5 episode。
- ACT 模型 checkpoint。
- rollout 评估视频。
- episode 可视化视频。

### 3.7 pyvut：VIVE Ultimate Tracker 6DoF 接入

`project1yu/pyvut` 和 `project1yu/pyvut-main` 是 VIVE Ultimate Tracker 的 Linux 读取工具，用于在没有 HMD 的情况下通过 USB 或 dongle 读取 Tracker 的 6DoF 位姿。

主要能力：

- 直接 USB tracker 连接。
- wireless dongle 模式。
- Tracker 配对、SLAM host/role 设置、Wi-Fi ACK。
- 实时读取位置、旋转四元数、tracking status。
- pygame 简单 3D 可视化。
- 向多源采集器提供手腕 Tracker 位姿。

在本工程中，它的定位是为多源采集提供 `vive_tracker/pos` 和 `vive_tracker/quat`。

## 4. 整体数据与控制流程

整个工程可以抽象成以下几层：

### 4.1 人手输入层

来源包括：

- Vision Pro 手部 25 点 tracking。
- 外骨骼关节/传感器数据。
- 普通摄像头/视频中的人手关键点检测。
- 操纵杆输入。

这些输入提供“人想怎么动”的信息。

### 4.2 重定向/映射层

主要由以下模块完成：

- TransHandR 模型：将 Vision Pro 右手 25 点关键点映射到灵巧手关节角。
- dex-retargeting 优化器：基于手部关键点、向量、位置目标和 URDF 运动学求解机器人手 qpos。
- 手型专用映射：将模型输出转换为 Inspire、LinkerHand L10、L21 等设备可接收的控制值。

这一层产出机器人手目标关节角或设备控制命令。

### 4.3 真机控制层

涉及设备：

- RM65 机械臂。
- LinkerHand L10/L21。
- Inspire Hand。
- XHand/DexHand。
- RealSense D405/D435i。
- VIVE Ultimate Tracker。

控制层负责：

- 建立机械臂/灵巧手连接。
- 设置工作坐标系、工具坐标系、速度、力、串口/CAN 参数。
- 下发关节目标或灵巧手 command。
- 读取真实设备反馈 state。
- 将成功下发的 command 发布给采集器。

### 4.4 数据采集与同步层

不同工程有不同采集方式：

- TeleOperation 早期系统：多进程 + 队列 + HDF5 缓存写入。
- DexUMI 原始流程：外骨骼 episode 采集，然后 replay 和后处理。
- TransHandR 多源采集器：D405 主时间轴 + 多源缓存 + 最近邻/插值/latest-before 对齐 + HDF5 quality-first schema。

当前最完整、最面向后续训练的数据采集路径是 TransHandR 多源采集器。

### 4.5 数据后处理层

主要包括：

- 数据插值。
- 图像帧提取。
- SAM2 分割。
- ProPainter 去除外骨骼/手部遮挡。
- 视频合成。
- HDF5 schema 校验。
- 质量统计。
- 数据格式转换成策略训练格式。

### 4.6 策略训练与评估层

包含两类路线：

- DexUMI 的 Diffusion Policy 路线：使用最终数据集训练扩散策略，然后真机评估。
- ACT 路线：使用 HDF5 episode 训练 Action Chunking Transformer，作为模仿学习基线或参考。

## 5. 主要产出物

整个工程最终会产生多类产出。

### 5.1 数据文件

| 类型 | 说明 |
|---|---|
| 原始 episode | 外骨骼或遥操作采集的原始数据。 |
| replay episode | 将外骨骼动作重放到机器人手后的数据。 |
| 多源 HDF5 episode | 包含 RGB-D、右手关键点、灵巧手 state、action、Tracker、sync、quality 的 episode。 |
| 训练数据集 | 面向 Diffusion Policy 或 ACT 的最终数据组织。 |
| 中间图像帧 | 从视频拆出的 `jpg/png` 帧。 |
| 分割 mask | SAM2 生成的手/外骨骼/机器人手 mask。 |
| 去遮挡图像 | ProPainter 或相关流程生成的 inpainted 图像。 |

### 5.2 模型文件

| 类型 | 说明 |
|---|---|
| TransHandR checkpoint | 从手部关键点到灵巧手角度的模型权重。 |
| Diffusion Policy checkpoint | DexUMI 策略模型权重。 |
| ACT checkpoint | Action Chunking Transformer 策略权重。 |

### 5.3 可视化与报告

| 类型 | 说明 |
|---|---|
| combined/debug 视频 | 展示外骨骼、机器人手、mask、inpaint 等处理效果。 |
| rollout 视频 | 策略评估过程可视化。 |
| loss 曲线 | 训练过程损失图。 |
| attention 图 | 模型注意力分析图。 |
| HDF5 校验报告 | episode 是否符合 schema、各源有效率和同步质量。 |
| Dashboard 实时页面 | 多源采集期间显示相机、手状态、action、Tracker 状态。 |

### 5.4 硬件优化结果

主要来自 DexUMI 的 linkage optimization：

- 仿真连杆设计搜索结果。
- 与 mocap 指尖轨迹匹配的最优连杆参数。
- URDF 或可视化轨迹。
- Inspire/XHand 外骨骼结构优化参考。

## 6. 当前最推荐的数据采集流程

如果目标是为后续 TransHandR/ACT/策略训练沉淀高质量真机数据，当前最推荐的路径是：

1. 先检查基础设备：
   - `check_gym.py`：确认仿真环境和 URDF 路径。
   - `check_hand_connect.py`：确认灵巧手连接。
   - `check_model.py`：确认模型权重和参数正常。
   - `check_visionpro_connect.py`：确认 Vision Pro IP 和数据流正常。

2. 启动多源采集器：
   - 无硬件时用 `mock` 模式做验收。
   - 真机时用 `hardware` 模式接入 D405、Vision Pro、灵巧手、可选 D435i 和 VIVE Tracker。

3. 如果 TransHandR 同时控制 Inspire Hand，推荐两个终端：
   - 终端 A：运行 `main_visionpro_multi_hand.py`，负责 Vision Pro 推理、灵巧手控制、发布真实 state/final action。
   - 终端 B：运行 `collect_transhandr_multisource.py`，负责采集 D405/D435i/VIVE，并通过 UDP 接收右手数据、hand state 和 action。

4. 录制 episode：
   - Dashboard 点击开始/停止，或键盘 `R` 开始、`S` 停止、`Q` 退出。
   - 每个 episode 保存成 `episode_000000.hdf5`、`episode_000001.hdf5` 这类文件。

5. 保存后校验：
   - 使用 `validate_multisource_episode.py` 检查字段完整性、时间戳单调性、valid 帧比例、同步误差、future action 泄漏、quaternion norm、invalid 占位等。

6. 根据训练目标转换数据：
   - 如果做 TransHandR 训练，重点使用右手关键点和对应机器人手 state/action。
   - 如果做 ACT 或 Diffusion Policy，进一步转换为相应训练格式。

## 7. 当前多源采集 HDF5 的质量重点

quality-first V1/V1.1 采集器非常强调以下几点：

1. 真实 state 和 action 必须区分。
   - `dexterous_hand/joint_positions` 是设备真实反馈。
   - `action/command` 是成功下发给设备的最终命令。

2. 不允许把模型中间输出当作真实 action。
   - 只有实际调用设备发送接口成功后，command 才应进入 action buffer。

3. 不允许 future action 泄漏。
   - 对齐到某个 D405 时间点时，只能使用这个时间点之前已经成功发送的 action。

4. Vision Pro 只采右手。
   - 右手缺失时应写 invalid，不能自动用左手替代。

5. 每个源都必须有 valid 标志。
   - valid=false 的数据不能直接当有效值使用。

6. invalid 占位要统一。
   - float 用 NaN。
   - timestamp/int 用 -1。
   - 图像用 0。

7. Tracker 四元数必须保持单位长度。

8. 外参即使未标定，也要保留字段。
   - 未标定时写单位矩阵，并标记 `calibrated=false`。

## 8. 与早期/旧格式的关系

当前目录里存在多套数据格式和工程阶段：

1. 旧 HDF5 数据集可能只有 `frame_ids/right_hand/left_hand`，或只有 glove position，不包含 D405、真实手反馈、action、VIVE、统一时间戳。

2. `TeleOperation` 的 HDF5 格式包含机械臂、灵巧手、双相机 RGB-D，但它不是 quality-first V1 schema。

3. `docs/TransHandR_多源数据采集需求设计.md` 是早期需求文档，字段更全，但有些内容与新实现不同。

4. `TransHandR0725/TransHandR/docs/multisource_collection.md` 和 `scripts/collect_transhandr_multisource.py` 代表当前更接近实际可运行的新实现。

因此，如果后续要做训练或论文实验，应优先确认使用哪一种 schema。当前工程最清晰的新标准是：

> `schema_version=quality-first-v1.1` 的 TransHandR 多源 HDF5 episode。

## 9. 环境与依赖概览

不同子工程依赖不同环境：

| 子工程 | 主要环境 |
|---|---|
| DexUMI-main | Ubuntu 22.04、Python/conda、SAM2、ProPainter、Record3D、RealSense/iPhone/灵巧手 SDK。 |
| TransHandR | Python 3.10，文档中提到 Windows 11；训练依赖 PyTorch、timm、einops、tensorboardX 等。 |
| TransHandR 多源采集器 | 轻量采集依赖，见 `requirements-collector.txt`；硬件模式还需要 RealSense、Vision Pro、pyvut、灵巧手 SDK。 |
| TeleOperation | Python、多进程、h5py、numpy、pyrealsense2、RM65/Alicia-D-SDK、LinkerHand SDK。 |
| tele/dex-retargeting | Python、pinocchio、SAPIEN/渲染依赖、URDF 资产、手部检测依赖。 |
| ACT | Python 3.8/conda、PyTorch、MuJoCo、dm_control、h5py、einops 等。 |
| pyvut | Python、hid、numpy、pygame，可选 adb/udev 配置。 |

## 10. 工程当前状态判断

从文件结构和文档看，当前工程处于“多来源工程整合 + 真机采集系统迭代”的状态，而不是一个已经完全统一、干净发布的软件包。

比较成熟的部分：

- DexUMI-main 原始流程文档完整。
- TransHandR 模型、训练和实时脚本较完整。
- 新多源采集器已经有 mock 模式、hardware 模式、Dashboard、schema writer、validator。
- TeleOperation 已有可运行的多进程/双相机 HDF5 采集思路。
- pyvut 已经作为 VIVE Tracker 接入基础。

需要特别注意的部分：

- 目录里存在多个重复/备份版本，如 `tele`、`tele copy`、`pyvut`、`pyvut-main`、多个 zip 包。
- 不同子工程的 HDF5 schema 不统一。
- 旧数据格式不能直接当作 quality-first V1 的 ACT 多源 episode 使用。
- 真机运行时设备独占很重要：灵巧手串口/CAN、VIVE dongle、RealSense 序列号都不能被多个进程同时占用。
- Vision Pro 的 source clock 语义在文档中标记为仍需实测确认。
- Inspire Hand 的真实反馈读取必须避免 command echo，否则不满足 quality-first 要求。
- URDF 路径和 mesh 路径在 Windows/Linux 下可能因斜杠方向导致问题。

## 11. 一句话总结

这个 `Dexumi` 文件夹整体是在做一个“人手/Vision Pro/外骨骼到机器人灵巧手”的研究与工程系统：前端采集人手和多传感器数据，中间通过 TransHandR 或 dex-retargeting 映射到灵巧手动作，后端控制真实机械臂/灵巧手并同步保存高质量 HDF5 episode，最终产出可用于 TransHandR、ACT 或 Diffusion Policy 训练的数据集、模型 checkpoint、可视化结果和真机策略评估结果。

