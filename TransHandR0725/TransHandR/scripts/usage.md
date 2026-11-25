# TransHandR 多模态采集使用说明

首次部署、安装命令和 GitHub 迁移要求见 [根目录 README](../README.md)；
具体缺项与验证范围见 [迁移核验报告](../docs/migration_audit.md)。
下文 `/home/zmy/...` 路径、环境名、序列号、IP 和 CAN 名称为原机器示例，新机器须替换。
统一平台使用 Python 3.10；`versionprodemo` 是原机器 Python 3.8 的 Vision Pro 推理环境，不能当作整个平台的安装基线。

## 0. 统一平台入口（推荐）

平台常驻运行，网页负责选择控制源、手型、手侧和保存通道，并管理采集器与设备进程：

```bash
cd /home/zmy/Projects/Dexumi/TransHandR0725/TransHandR
python3 scripts/run_teleoperation_platform.py --mode hardware --host 0.0.0.0 --port 8080 --telemetry-port 8765
```

打开 `http://127.0.0.1:8080/overview`。第一版合法组合：

Vision Pro 切换网络后，可点击“系统配置”右上角的设置按钮修改 IP。设备设置同时收纳相机序列号、手套串口、真机串口和 CAN 接口，并且只能在系统停止时修改。

| 控制源 | 目标手型 | 手侧 |
|---|---|---|
| Vision Pro | Inspire / L10 / L21 | 左 / 右 |
| LinkerForce 外骨骼 | O6 | 左 / 右 |
| LinkerForce 外骨骼 | Inspire | 左 / 右 |

外骨骼 + O6 使用前先启动 CAN：

```bash
cd /home/zmy/Projects/Dexumi/TransHandR0725/TransHandR
bash scripts/start_can.sh can1
```

这台机器当前 O6 接在 `can1`；设备设置中的 CAN 接口也必须填写 `can1`。有多块 USB-CAN 时脚本不再默认选择 `can0`。重新插拔适配器后先用 `ip -br link show` 核对接口名，再启动系统。

网页不会执行 `sudo`。O6 链路依次启动采集器、ROS telemetry adapter、`o6_bridge` 和 `handretarget`；Inspire 链路将 bridge 替换为 `inspire_bridge.py`。O6 默认速度为 `255`、力矩为 `40`，六通道真机限位读取：

```text
/home/zmy/Projects/Dexumi/linkerhand_telop_python-master/ros2/src/linkerhand_retarget/config/o6_command_limits.yaml
```

外骨骼遥操作 Inspire 前，在设备设置中分别填写左手套、右手套和 Inspire Hand 的三个独立 `/dev/serial/by-path/...` 路径。两只手套可以接同一个 USB Hub，但不能指向同一设备节点。选择 `LinkerForce + Inspire + 手侧` 后点击“重新标定”，依次完成张手、O 型和握拳；缺少该手型、该手侧的独立标定时，平台会拒绝启动真机。

Inspire 当前配置速度为 `800`、力阈值为 `500`，六通道限位读取：

```text
/home/zmy/Projects/Dexumi/TransHandR0725/TransHandR/config/inspire_command_limits.yaml
```

“数据保存通道”是平台唯一的数据源选择入口。D405 固定启用；选择 D435i 或 Tracker 会在下次启动系统时同时启用设备和 HDF5 保存，其他通道控制 HDF5 是否记录。配置按“控制源 + 手型 + 手侧”分别保存，只能在系统停止时修改。

Vision Pro 模式显示 25 点手部；LinkerForce + O6 显示五指触觉阵列、B1 掌心阵列和三维左手；LinkerForce + Inspire 显示所选左右手 Inspire 模型、Action/State 和 Tracker，不显示触觉。停止系统时，平台会保存当前 episode，并清理当前用户下残留的本项目 bridge、`handretarget`、ROS adapter 和 Vision Pro 控制进程。

O6 B1 的五指有效区域为每指 10×4（HDF5 仍以 12×6 保存并保留 `-1` padding），掌心为 14×15。勾选“触觉”后，两路原始事件分别写入 `dexterous_hand/tactile_stream` 和 `dexterous_hand/palm_stream`；D405 主表仅保存对应的 `index/age_ms/valid` 引用。掌心 taxel 的物理行列方向尚需单点按压标定。

CLI 方式继续保留，以下章节用于单独调试各数据源。

## 1. 当前硬件

| 设备 | 用途 | 序列号 / 端口 |
|---|---|---|
| Intel RealSense D405 | 主时间轴、近景 RGB-D | `352122271077` |
| Intel RealSense D435i | 头部 RGB-D | `141722077699` |
| Inspire Hand | 灵巧手控制与真实反馈 | `/dev/ttyUSB0` |
| Vision Pro | 可选左/右手 25 点追踪 | `172.20.10.9` |
| VIVE Tracker | 位置与四元数旋转 | `DONGLE_USB` |

两台 RealSense 都使用 `640x480@30Hz`。D405 是 dataset master；D435i 与其他模态在 Linux `CLOCK_MONOTONIC` 时间域中对齐。相同帧率不代表硬件同时曝光。

### TRON2 Top 相机用于推理：匹配旧 D435i 视场

2026-10-22 从实机只读查询确认：TRON2 Top 是 **Intel RealSense D455**，序列号
`261822301393`，彩色输出 `640×480@30Hz`；不是上表中的独立采集用 D435i。
从 `/camera/top/color/camera_info` 得到当前彩色内参：

```text
K = [[387.528625,   0,        323.965607],
     [  0,        386.698151, 239.892380],
     [  0,          0,          1       ]]
D = [-0.0543384254, 0.0592227876, -0.0006739546, 0.0002846827, -0.0188229475]
distortion_model = plumb_bob
```

旧 `o6/episode_000065.hdf5` 的 D435i 彩色 `fx/fy` 约为 `606.986/606.105`。
两者在此分辨率下的针孔视场近似分别为 `79.1°×63.7°` 和 `55.6°×43.2°`。
Top 因此需要约 `1.57` 倍的视场收窄，并按 D455 的畸变校正；单纯将整图缩放到
640×480 不会完成这一步。该结果仅适用于当前分辨率和具体设备，不能代替其他设备的实测内参。

从 Dexumi 根目录运行独立实时预览。它订阅 Top，不启动 Dexumi 采集或遥操，也不改变机器人和相机设置：

```bash
cd /home/zmy/Projects/Dexumi
source /opt/ros/humble/setup.bash
/usr/bin/python3 tools/preview_tron2_top_camera.py \
  --reference-hdf5 /home/zmy/workplace/insert-data/transhandr_demo/o6/episode_000065.hdf5
```

左侧是 Top 原图，黄线是校正后实际采样的边界；右侧是匹配参考 D435i 内参的
640×480 图像。按 `S` 保存当前画面和参数，`Q`/Esc 退出。
加入 `--snapshot` 可在无显示窗口时保存一组图像后退出。
输出位于根目录的 `camera_alignment_results/top_<日期时间>/`。
可以用 `--reference-frame 73` 选择同一数据文件的其他参考帧；参考帧只用于保存对照图，
整个文件的相机内参相同。

如需微调取景方向，`--look-down-deg 3` 表示虚拟视角向下看 3°，
`--look-right-deg 2` 表示向右看 2°，负数方向相反。它们只改变图像采样光线，不转动头部；
默认均为 0。若采样超出 Top 原图，工具会报错，避免输出黑边图像作为推理输入。
应在相同桌面布置下根据标定板/固定标记选择方向，不能把这些试调值当成已测得的安装外参。

首轮实机结果保存在
[Top D455 标定参数](../../../camera_alignment_results/top_d455_20261022/calibration.json) 和
[真实校正前后对比](../../../camera_alignment_results/top_d455_20261022/comparison.png)。
此次所有输出像素均落在 Top 原图内；实测头部俯仰约 `50.29°`、偏航约 `0.81°`。
这完成了内参/光学视场匹配，尚未标定旧 D435i 与 Top 的安装位置关系。
若桌边倾斜、目标位置仍对不上，需要固定头部角度、桌面位置，并用同一标定板求外参。
平面透视变换只能精确对齐该平面，不能同时消除离桌面有高度的手和物体的平移视差。

推理中可以直接复用已保存的映射。以下 `top_rgb` 必须是 640×480 的原始 RGB 数组：

```python
from tools.preview_tron2_top_camera import CameraViewAlignment

aligner = CameraViewAlignment.from_json(
    "camera_alignment_results/top_d455_20261022/calibration.json")
cam_high_rgb = aligner.apply(top_rgb)
# 随后接 ACT 原有的图像归一化和张量转换；apply 保留 RGB 通道顺序，不上下翻转。
```

标定配置仅供上面的预览和显式调用；尚未自动接入任何 ACT 推理服务。

实机头部可通过根目录 `tools/tron2_head_pose.py` 单独读取/调整。使用 `dexumi`
环境的 Python（该环境带 `websocket-client`），不需要启动机械臂回放：

```bash
cd /home/zmy/Projects/Dexumi
conda activate dexumi
python tools/tron2_head_pose.py
```

如需执行头部粗调，先在机器人端切换至 `High-Level Dev`，再运行：

```bash
python tools/tron2_head_pose.py --pitch-deg 54.3 --duration 5 --execute
```

`54.3°` 是本机 2026-10-22 根据 episode 65 第 0 帧黄/粉夹子中心的初步俯仰目标，
并非通用标定值。未指定 `--yaw-deg` 时保持读取到的偏航；去掉 `--execute` 仅检查。
工具只调用头部 `MoveH`，不会发送双臂轨迹。当前模式不支持时机器人返回
`fail_is_not_develop_mode`；工具不会自行切换整机模式或重发运动。

## 2. 单独启动采集器

先启动采集器。命令默认只预览，不会自动录制。

```bash
conda activate versionprodemo
cd /home/zmy/Projects/Dexumi/TransHandR0725/TransHandR

python3 scripts/collect_transhandr_multisource.py \
  --mode hardware \
  --hand-type inspire \
  --hand-source udp \
  --right-hand-source udp \
  --vision-hand-side right \
  --avp-ip 172.20.10.9 \
  --d405-serial 352122271077 \
  --width 640 \
  --height 480 \
  --camera-fps 30 \
  --enable-head-camera \
  --head-camera-serial 141722077699 \
  --head-width 640 \
  --head-height 480 \
  --head-camera-fps 30 \
  --head-camera-max-gap-ms 45 \
  --enable-vive \
  --tracker-mode DONGLE_USB \
  --tracker-index 0 \
  --action-port 8765 \
  --max-episode-duration 300 \
  --max-episode-size-gb 5
```

不要同时运行另一份 `stream_poses.py`，否则它会和采集器争用 VIVE Dongle。

## 3. 启动 Vision Pro 与灵巧手控制

在第二个终端运行：

```bash           
conda activate versionprodemo
cd /home/zmy/Projects/Dexumi/TransHandR0725/TransHandR

python3 scripts/main_visionpro_multi_hand.py \
  --hand inspire \
  --avp_ip 172.20.10.9 \
  --vision_hand_side right \
  --serial_port /dev/ttyUSB0 \
  --telemetry_host 127.0.0.1 \
  --telemetry_port 8765 \
  --telemetry_state_hz 60
```

## 4. 网页与录制

打开 DexUMI 数据采集管理页：

```text
http://127.0.0.1:8080/overview
```

- 主视觉区是 2×2 等尺寸布局：上排为 D435i RGB / RGB-D，下排为 D405 RGB / RGB-D。
- 网页中的“RGB-D”窗口显示深度数据的伪彩色图，HDF5 中仍保存原始 `uint16` 深度值。
- 网页“开始录制”或网页快捷键 `R`：开始新的 episode。
- 网页“停止并保存”或网页快捷键 `S`：停止并安全保存。
- 终端键盘 `Q` 或 `Ctrl+C`：退出；正在录制时会先保存。
- 不要在普通预览命令中加入 `--record`，避免忘记停止后持续占用磁盘。

右侧“数据保存通道”可以决定当前 episode 录制哪些模态。D405 是主时间轴，因此固定必选。统一平台会按控制源过滤合法通道，并将配置持久化到 `data/multisource/.recording_channels.json`；CLI 采集器仍将配置保存在当前任务目录。系统运行或录制期间不可修改通道。网页“设备设置”中的新录制保存目录仅影响之后的 Episode，旧文件不会迁移。

HDF5 的 group/dataset 结构始终保持稳定。未选择的通道不会保存真实数据，其数值位置写入 `NaN` 或零占位，同时 `valid=false`；文件根属性 `selected_channels` 记录本次实际选择。这样训练和读取代码无需为不同通道组合维护多套 schema。

顶部“数据”页会列出已有 episode、采样频率、帧数、时长和初步质量结果，可直接进行四路图像的离线逐帧回放或下载 HDF5。该回放由服务端读取 HDF5 并转换图像，不依赖 HDF5 查看器的 WebGL Heatmap。

同一页的“真机回放”是独立操作，调用 `replay_tron2_fixed_start.py`；仅硬件模式、从机器人旁的本机 `http://127.0.0.1:8080/data` 可提交真机操作，远程浏览器只能看状态。2026-10-22 已取消右臂自动归位：准备函数只恢复左臂，MoveJ 中右臂目标保持本次准备前实测关节角；右臂不再接受历史固定起点验收。ServoP 仍成对发送双腕目标，右腕保持回放前实测位姿，这不是右臂七关节的硬件锁死。

**当前默认准备姿态已改回小臂上仰 5°，本次从 10° 复归且已实测到位。** 大臂竖直向下、手与小臂对直，左腕目标位置为 `[0.449428671, 0.253076873, -0.266192769] m`，目标左臂关节角为 `[0, 0, 0.183222811, -1.658062789, -1.570796327, 0, 0] rad`；四元数与此前 +5° 完全相同。肘关节 `−95°`，腕 yaw/pitch/roll `−90°/0°/0°`，不叠加额外滚转。相对 10° 模型目标，SDK 目标降低约 `3.9426 cm`。

最新画面显示桌面位置变化后，先保留目标配置，等待用户明确确认手下“有至少 5 cm 空隙，可以调整”，随后按最新反馈执行 1 段、3 秒的左臂 MoveJ 从 10° 复归 5°。五次静止复核实际仰角为 `5.4504°–5.4562°`、大臂竖直误差约 `0.3704°`、手/小臂轴线夹角约 `0.0690°`、额外滚转约 `0.0643°`，目标位置误差最大 `4.2543 mm`、朝向误差最大 `0.4628°`，全部关节速度为零。右臂保持本次实测位置，编码器最大变化约 `0.03440°`，头部变化 `0°`。没有开始 replay 或发送 O6 手指指令。

当前 `FIXED_LEFT_START_VALIDATED = True`、`FIXED_LEFT_START_MEASURED = True`；`FIXED_LEFT_PREPARE_PATH_VALIDATED = False` 仍独立保留。当前记录见 [+5° 本次复归档案](../../../validation_results/forearm_up5_return_measured_20261028/fixed_start_profile.json)、[五次复核](../../../validation_results/forearm_up5_return_measured_20261028/verification.json)、[动作后真实画面](../../../camera_alignment_results/forearm_up5_return_after_20261028/top_raw.png) 与 [准备姿态记录](../../../tron2_fixed_start_profiles.md)。先前等待净空时的 [仅配置目标](../../../validation_results/forearm_up5_return_target_20261028/fixed_start_profile.json) 保留为历史，不再代表当前待执行状态。

**回放离线 IK 可复用同目标的 5° 结果。** 当前目标 Q/XYZ/四元数与此前 +5° 完全一致；episode 65 使用 `scale=1.0`、`orientation-scale=0.5`、`speed=0.2` 时仍为 7453 帧、24.839 秒，最大相对位移约 `0.3177 m`、半幅最大转角约 `18.91°`，目标范围为 `[0.4491,-0.0377,-0.2957]` 至 `[0.5940,0.2710,-0.2161] m`。原 [5° IK 报告](../../../validation_results/forearm_up5_replay_ik_20261028/result.json) 中 4 个候选有效，选中 ID `2`、权重 `1.0`，最小关节余量 `10°`、最大关节速度约 `0.263168 rad/s`；该模型结果不验证实际固件分支、整段碰撞、动力学或跟踪。

此前 10° 姿态及 [10° 实测档案](../../../validation_results/forearm_up10_measured_20261028/fixed_start_profile.json)、[10° IK 报告](../../../validation_results/forearm_up10_replay_ik_20261028/result.json) 保留为历史，其目标与当前 5° 锚点不同。

**改变准备姿态本身无需重新标定 Tracker。** 地图、机器人基座和 Tracker 刚性安装关系保持一致时，仍使用原标定矩阵；本轮原采集进程和标定 SHA256 保留。真实头部画面可见左腕 O6 安装，但本轮没有发送手指指令，同步手指回放仍须确认 CAN 可用。

**此前 +5° 姿态保留为历史。** 当时按用户选择先分 18 段恢复水平中立腕并验收，再分 2 段到 +5°，小臂实测仰角 `4.6729°–4.6844°`，目标位置 `[0.449428671, 0.253076873, -0.266192769] m`。记录见 [+5° 两阶段实测档案](../../../validation_results/forearm_up5_two_stage_measured_20261028/fixed_start_profile.json)；该相同 5° 目标的 [IK 报告](../../../validation_results/forearm_up5_replay_ik_20261028/result.json) 可复用；本次实际到位依据为上方复归档案，10° 的报告仍属于不同锚点。

**此前已实测的水平中立腕准备点保留为历史。** 该版位置为 `[0.450789437, 0.253329025, -0.305890000] m`，曾完成 5 段、每段 3 秒的左臂 MoveJ，复核大臂竖直误差约 `0.453°`、小臂仰角约 `0.715°`、手与小臂轴线夹角约 `0.470°`、掌心朝下误差约 `1.150°`。其五次静止反馈和运动结果不能代替当前活动准备姿态的到位核验，档案见 [水平中立腕历史记录](../../../validation_results/neutral_start_measured_20261028/fixed_start_profile.json)。该旧锚点的 [离线 IK 报告](../../../validation_results/neutral_replay_ik_20261028/result.json) 仍保留，但不覆盖当前活动锚点。本轮早先的 [最小清理记录](../../../validation_results/neutral_start_measured_20261028/cleanup.json) 同样保留，历史证据没有删除。

**更早的历史目标：**10 月 22 日 20:51:19 原掌心向下参考 `[0.497985154, 0.186801851, -0.266951889] m`、后来的纯 Z +8 cm 目标 `[0.497985154, 0.186801851, -0.186951889] m`，以及 10 月 27 日保留完整腕部朝向的水平小臂目标 `[0.447066782, 0.174041590, -0.264423426] m` 均原样保留，不再作为活动准备目标。

机械臂已切回 ServoP（`--arm-control servop`），CLI 与网页显式使用同一接口：左腕发送目标位姿，右腕保持本次启动时实测位姿，不发送头部目标。ServoJ 因实际电机目标被异常裁剪到约 `±0.15 rad` 已停用，旧 ServoJ 命令会被拒绝；切回 ServoP 不代表底层异常已修复。2026-10-23 按只规划左臂的要求，右腕不再做 MoveP 工作空间验收，不自动抬右臂、归位、修改负载或换接口；仍核对反馈有效性、异常偏移和超时。左腕运动目标继续使用保守参考范围，它不是已确认的 ServoP 固件边界。取消右腕参考范围检查不代表真机保持已通过验证。当前先用 episode 65 的 `scale=0.1`、`max-displacement=0.04` 做离线/只读检查；ServoP 不做本机 IK，离线通过不证明控制器可达或碰撞安全。训练和部署应遵循日志中的 `SERVOP_ACTION_CONTRACT`：基座坐标、米、WXYZ 绝对腕目标，不能把真人 Tracker 当机器人实测状态。命令与完整语义见根目录 `tron2_tracker_replay_commands.md`。`--check-robot` 只读显示双腕与头部，不发送运动。

测试流程为：先选择含有效 Tracker（可选 O6 左手实测状态）的 episode，调整平移倍率、速度、轴映射、最大相对位移、倒计时及是否同步 O6，再点击“离线预检”。勾选同步 O6 时可选择 `can0`、`can1` 等专用接口；采集系统须为“已停止”，现场核实 CAN 接线。输入 `PREPARE` 后，任务先连接 TRON2，只读核对双腕反馈有效性及左臂起点；已在起点则不发送 MoveJ，偏离时仍受恢复路径检查限制。全部准备条件通过后才独立启动 O6 桥接，不启动数据手套或相机；输入 `REPLAY` 后倒计时，重新核对最新双腕反馈、静止状态及左腕目标参考范围，再共享同一单调时钟发送 ServoP/O6 轨迹。不生成本机 IK 关节轨迹，也不自动恢复头部视角。结束或报错时退出 O6 回放模式并关闭专用桥接，关闭后是否持位须现场确认。禁止并发占用同一 CAN。桥接日志与 JSONL 同名，后缀为 `.o6_bridge.log`。“仅恢复准备位置（不回放）”只处理左臂，已到位时零运动成功退出，不依赖 episode、不会进入回放。恢复和回放共用互斥锁，不能并发。左臂恢复路径及真实控制器行为尚未验证时，应停留在离线和只读检查阶段。

若离线预检显示“参数超出界限”，先看报错指向的输入框；原页面的“最大相对位移”默认值 `0.35` 与步长 `0.01` 不对齐，现已改为允许任意范围内的小数。网页允许 scale `0–1`、speed `0.05–1`、最大相对位移 `0.001–0.35 m`、倒计时 `5–30 s`；episode 65 在 scale=1 时的最大相对位移约 `0.3177 m`，默认 `0.35 m` 能通过这项检查。如果是实际轨迹位移超过设置的上限，应核对数据、轴映射和 scale，或选择位移较小的 episode；不要把“调大上限”当成可达性或碰撞验证。

关闭网页弹窗不会停止任务；“停止后续指令”不是硬件急停，MoveJ 可能仍在机器人端继续执行。必须在现场清空运动范围并持有实体急停，不可将离线工作空间检查视为碰撞或 IK 可达性证明。网页和固定起点 CLI 使用同一互斥锁，防止它们同时运行。网页启动的平台 Python 必须有回放所需依赖；若不同于 `dexumi` 环境，启动平台前可设置 `DEXUMI_REPLAY_PYTHON` 为该环境的 Python 绝对路径。修改网页代码后需要重启平台进程并刷新浏览器。

准备阶段遇到单次 `response_get_joint_state` 超时，现在只重试读取反馈，不重发 MoveJ；若整个准备窗口内仍不能确认关节到位，程序仍中止，不会进入下一阶段。这不能替代排查机器人控制器或网络长期不回包的根因。

若仅需现场试验左臂恢复动作，在工程根目录运行 `python3 restore_tron2_fixed_start.py --test-path --execute --countdown 5 --trial-step-duration 3`。先重新检查裸臂模型，输入大写 `RESTORE` 后分段 MoveJ，每段计划关节变化不超过 5°，逐段验收；不启动 Tracker/ServoP/O6 回放，不修改头部、负载或模式，也不自动批准正式恢复路径。该模型缺少实际 O6、转接件、桌面和线缆，现场须先确认整个扫掠空间及实体急停，软件停止不能撤回已接受的 MoveJ。仅允许交互终端启动，不能从网页后台触发。详细命令和限制见根目录 `tron2_tracker_replay_commands.md` 的 9.2 节。

### 数据源有效性
“数据源有效性”是实时初检，不只是检查设备线程是否启动。每个 D405 主帧到来时，采集器按以下条件判断：

- D405：当前主帧由相机成功返回且 `valid=true`；
- D435i 与 Vision Pro：存在距离当前 D405 时间戳不超过配置阈值的最近有效样本；
- Hand State：当前 D405 时间点能在阈值内完成有效插值；
- Action：引用时间不晚于当前 D405 帧的最后成功命令；当前默认持续保持至被新命令替换（`action_max_age_ms=0`），禁止使用未来指令；
- Tracker：原始追踪状态为 `OK (2)`，且能在阈值内完成 position 插值和 quaternion SLERP。

网页状态含义：`有效` 表示当前主帧能取得合格样本，`无有效样本` 表示已启用但当前帧未通过上述检查，`未启用` 表示启动命令未启用该源。这只是录制前的实时检查；episode 保存后的完整合格性仍以校验脚本输出为准。

默认输出目录：

```text
/home/zmy/workplace/transhandr_episodes/transhandr_demo/inspire/
```

双路 RGB-D 数据量较大。单个 episode 达到 300 秒或 5 GiB 时会自动停止并保存，以先到者为准。

## 5. 时间对齐规则

| 模态 | 对齐方法 | 默认阈值 |
|---|---|---:|
| D405 RGB-D | master timestep | — |
| D435i RGB-D | nearest | 45 ms |
| Vision Pro 所选手 | nearest | 45 ms |
| Inspire Hand State | linear interpolation | 35 ms |
| VIVE position | linear interpolation | 30 ms |
| VIVE quaternion | SLERP | 30 ms |
| Action | latest-before，禁止未来指令 | 默认保持最后成功命令，直至替换 |

对设置了时间阈值的模态，超过对应阈值时，该 timestep 会设置 `valid=false`。Action 的默认保持策略不设置过期时间；只有显式设置正的 `action_max_age_ms` 时才按该值过期。

## 6. Tracker 状态排查

网页会显示 Tracker 的 tracking state、map state、收包数和完整位姿数。只有 `OK (2)` 会被当作有效位姿；`NOT_INITIALIZED (1)`、`LOST (3)` 等状态不会写成有效数据。
命令启用 `--enable-vive` 时，如果 Tracker 尚未进入 `OK (2)`，开始录制会被拒绝，防止产生 Tracker 全空的 episode。

如果显示 `NO_POSE_STREAM`：

1. 确保没有另一个 `stream_poses.py` 或采集器占用 Dongle。
2. 把 Tracker 正面摄像头朝向纹理明显的环境，保持光线充足，缓慢移动完成 SLAM 初始化。
3. 等终端的 `CURRENT_TRACKING_STATE` 从 `NOT_INITIALIZED (1)` 变为 `OK (2)` 后再开始录制。
4. 单 Tracker 使用 `--tracker-index 0`；这是按连接顺序的逻辑编号。

用单独命令验证 Tracker 时，必须先退出采集器：

```bash
cd /home/zmy/Projects/Dexumi/project1yu/pyvut
/home/zmy/anaconda3/envs/versionprodemo/bin/python scripts/test_position.py
```

左手模式下，发送给右手训练模型的腕部局部关键点会先执行 XYZ 中心反演。网页与 HDF5 仍保留 Vision Pro 原始左手的世界/相对关键点，不会被这个推理转换改写。

`world_keypoints` 保留世界坐标姿态；`relative_keypoints` 使用 Vision Pro 腕部局部坐标，网页骨架读取后者。网页会根据腕部和四个掌指关节实时建立手掌平面，以掌心正视角投影骨架；整体转动手腕不会再改变骨架形状。左右手只在网页显示方向上分别镜像，不会改写保存的关键点或模型输入。

### 6.1 左腕 Tracker / TRON2 标定位姿采集

入口：本机打开 `http://127.0.0.1:8080/wrist-calibration`，或点击顶栏“腕部标定”。
下面的网页/普通 CLI 入口会自行初始化 Dongle。**已有 Tracker 采集正在运行且需保持连接时，改用 6.3 的只读旁路方式，不退出或重启现有采集。**
更新后需要重启数采平台服务，浏览器再刷新。使用自主管理 Dongle 的入口时，先停止概览页的采集系统，并退出独立的
`stream_poses.py` / `test_position.py`，避免重复占用 Dongle。此入口不依赖 D405、头部相机、
数据手套或 O6 桥接，只查询 TRON2 左腕实测位姿，不发送 MoveJ、ServoP 或其它运动请求。

1. 填写机器人 IP（默认 `10.192.1.2`）、Tracker 编号（默认 `0`）、地图和安装备注，点击“连接设备”。
2. 通过机器人已有控制方式调整姿态，点击“采集一组”；默认倒计时 3 秒，再静止采样 2 秒。
3. 换不同姿态重复采样。页面显示两路接收帧龄、原始跟踪状态、配对数和静止波动；可排除异常组。
4. 点击“结束采集并断开”，再下载原始样本。排除操作追加记录，不删除原始数据。

也可使用独立命令行（只读，不需要启动网页）：

```bash
cd /home/zmy/Projects/Dexumi
conda activate dexumi
python3 capture_tron2_tracker_calibration.py \
  --robot-ip 10.192.1.2 --tracker-index 0 \
  --map-label "2026-10-27地图" --mount-label "左腕固定支架" \
  --countdown 3 --duration 2
```

终端按 Enter 采一组；输入 `status` 查看两路最新反馈；`undo` 排除末组；`q` 结束。
网页与 CLI 默认保存到 `/home/zmy/Projects/Dexumi/tracker_calibrations/<会话>/samples.jsonl`。
网页可通过环境变量 `WRIST_CAPTURE_PYTHON` 指定采集进程 Python；默认优先使用 dexumi 环境。
`--mock` 仅供离线界面验证，生成带明确模拟标记的 status=20 样本，不连接硬件。

文件第一行为会话信息：地图/安装备注、机器人身份、米制位置、WXYZ 四元数和坐标约定
`T_WT = T_WB @ T_BE @ T_ET`。每个 `sample` 保存采样窗口内全精度 Tracker/机器人帧和
按主机接收时间最近邻匹配的索引；最大接收时间差为 100 ms。另存机器人请求耗时与两端时间字段。
这不是硬件同步；当前 pyvut 的 `timestamp_ms` 来自主机解析时钟，因此另命名为
`parser_host_timestamp_ms`，不伪装成传感器时间。

2026-10-27 已实测发现：拖动示教中 `get_move_pose` / `get_servo_pose` 的回复时间戳
在更新，但位姿可能冻结在进入模式前。因此新采集使用 `response_get_joint_state` 的实测关节角，
通过官方模型正运动学计算左腕 SDK TCP（不是 IK，不发送运动），保存为 `tracker-tron2-pairs-v2`。
每帧保留 16 维原始关节角，会话头保留模型 SHA256、关节顺序和 TCP 偏移；重复关节时间戳拒绝使用。
这是基于编码器和模型的位姿估计，误差含模型/零位偏差，不能当作外部测量真值。
旧版 v1 样本仍可读取，但在拖动模式中采到的冻结腕位姿组必须排除重采。
更新采集程序后结束旧会话、重新连接即可加载；不要混用两种来源的样本。

`status=20` 等非 `OK (2)` 状态仍原样保存，标记 `eligible_for_calibration=false`，不阻止继续采集。
丢帧、查询过慢、样本不足或静止波动过大也会标记原因；断连不复用旧位姿冒充新样本。
`eligible_for_calibration=true` 只表示本组基础采样检查通过，**所有样本的
`calibration_validated` 均为 false**。结束采集后按下一节离线求解，结果另存，不改写原始记录或回放参数。
重建地图或重新安装 Tracker 后结束旧会话、新建会话，不混合不同参考系的样本。

接口（本机同源 JSON POST）：`/api/wrist-calibration/start`、`sample`、`exclude`、`stop`。
`sample` 参数为 `countdown`、`duration`、`label`，`exclude` 参数为 `id`；
GET `/api/wrist-calibration/status` 查看状态，GET `/api/wrist-calibration/download` 下载当前会话。
首次建议采 12–20 个分布在不同旋转轴和位置的稳定姿态，并额外采几组供独立验证。

### 6.2 求解腕部映射与误差报告

网页：在“腕部标定”页结束采集并断开后，点击“求解并查看误差”，完成后可下载结果 ZIP。
求解只读取已保存文件，无需重新连接 Tracker 或机器人。旧会话也可以用命令行求解：

```bash
cd /home/zmy/Projects/Dexumi
conda activate dexumi

# 将“实际会话目录”替换成采集时输出的目录名，也可直接传 samples.jsonl 文件路径。
python3 solve_tron2_tracker_calibration.py \
  --input "/home/zmy/Projects/Dexumi/tracker_calibrations/实际会话目录"
```

默认只使用 `status=2` 的稳定组。如果数据主要为 `status=20`，可在网页勾选“仅诊断：纳入 status=20”，
或给上面命令添加 `--include-status-20`。这样可以先计算和查看误差，但结果始终标记为待确认，
不会因残差小就认定该跟踪状态有效。其它无效状态不被纳入。

每次求解在会话目录下新建 `solutions/<唯一编号>/`，不覆盖已有结果：

| 文件 | 内容 |
| --- | --- |
| `calibration.json` | `T_ET`、`T_WB`、`T_BW` 三个 4×4 变换、源文件校验值、会话身份和结果状态 |
| `report.txt` | 中文报告：矩阵、逐组误差、总体统计、样本排除原因和姿态变化检查 |
| `report.json` | 同一报告的完整数值，含每组腕部 XYZ 误差，供后续程序读取 |
| `errors.csv` | 每组编号、拟合/验证用途、配对数、位置误差和旋转误差，可用表格软件打开 |

求解失败时仍保存报告，但不生成 `calibration.json`。终端也会直接打印误差与结果目录。

坐标定义：`T_AB` 把 B 系坐标变换到 A 系；W 为 Tracker 地图，B 为机器人基座，
E 为机器人反馈的腕部坐标系，T 为 Tracker 自身坐标系。程序联合求解：

```text
T_WT = T_WB @ T_BE @ T_ET
T_BW = inverse(T_WB)
T_BE_estimated = T_BW @ T_WT @ inverse(T_ET)
```

因此 `T_ET` 表示 Tracker 到机器人腕部的安装旋转和平移，`T_WB` 表示机器人基座到 Tracker 地图的关系。
所有矩阵中的平移均为米。旧 episode 使用这些结果的前提是地图和 Tracker 安装关系匹配；
换了安装位置，不能直接把本次安装变换套到旧数据。

每个静止姿态组只贡献一个平均位姿，组内重复帧不算多个独立姿态。
至少需要 3 组、且存在绕不同轴的旋转；纯平移、反复采同一姿态或只绕一个轴转会报告不可充分观测。
建议 12–20 组。达到 9 个合格组时，程序自动固定留出至少 3 个整组作为独立验证，不参与拟合；
也可用 `--validation-ids 3,7,12` 指定验证组。少于 3 个独立验证组时仍可输出估计，但状态为待复核。
这一手眼标定的运动要求可参考 [OpenCV 标定文档](https://docs.opencv.org/4.13.0/d9/d0c/group__calib3d.html)。
程序先由 `AX=XB` 初始化，再用 [SciPy 鲁棒最小二乘](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html)
联合优化两个刚体变换，不自动删除大误差组来改善报告。

报告分别统计拟合组和独立验证组的均值、RMSE、中位数、P95、最大误差：

- **腕部位置误差（mm）**：由 Tracker 反推的腕部原点与 TRON2 实测腕部原点之间的距离。
- **Tracker 位置误差（mm）**：由机器人腕部正推的 Tracker 原点与 Tracker 实测原点之间的距离。
- **旋转误差（度）**：两种朝向之间的最小旋转角，不是直接相减欧拉角。

默认数值复核参考为位置 RMSE ≤5 mm、最大 ≤15 mm，旋转 RMSE ≤2°、最大 ≤6°。
这是本项目报告阈值，不是厂家限位或精度保证；可用 `--position-tolerance-mm`、`--rotation-tolerance-deg`
调整 RMSE 参考值，最大误差参考值为其 3 倍。姿态变化不足、拟合/验证误差过大及模拟/status=20 输入都会明确提示。
这些残差衡量两套测量的一致性，不等同于外部真值精度。求解结果不会自动接入回放。

新增接口：本机同源 POST `/api/wrist-calibration/solve`，JSON 参数 `{"include_status20": false}`；
GET `/api/wrist-calibration/status` 的 `solution` 字段返回进度及报告；
GET `/api/wrist-calibration/solution-download` 下载结果 ZIP。

### 6.3 保留 Tracker 连接的自动标定与 DexUMI 原目标求解

`tracker_passive_reader.py` 使用 Linux `hidraw` 的 `O_RDONLY` 输入队列读取副本，
不初始化 Dongle、不发送配对/建图请求，结束时仅关闭自己的句柄。
现有 `stream_poses.py` 继续运行。设备节点和 MAC 以当前连接为准，重新插拔后可能改变。

```bash
cd /home/zmy/Projects/Dexumi
conda activate dexumi

# 只读取当前关节并规划，不运动。以下设备/MAC 为 2026-10-27 本次连接。
python3 auto_calibrate_tron2_tracker.py \
  --passive-hid /dev/hidraw6 --tracker-mac 23:32:ae:d4:f6:a1

# 已确认 High-Level Dev、Tracker 刚性固定且运动范围清空后使用。
python3 auto_calibrate_tron2_tracker.py \
  --passive-hid /dev/hidraw6 --tracker-mac 23:32:ae:d4:f6:a1 \
  --tracker-quaternion raw-xyzw \
  --execute --countdown 5
```

自动采集 16 组，仅变化左腕末端三个关节，目标偏转最多 12°；靠近限位时减少该方向角度。
每段 MoveJ 最多 5°、计划 2 秒；右臂目标保持启动关节角，不发送头部目标。
每组停稳后记录 2 秒，最后一组回到本次起点用于重复性检查。
模型检查不含 Tracker 支架、线缆或桌面；Ctrl+C 只能停止后续请求，不能撤回已接受的 MoveJ。

当前设备曾返回原始 `status=18`。需要诊断这类数据时可显式加 `--diagnostic-status 18`，
原始状态和非正常标记仍保留，不改成 `2`；已知丢失状态不会放行。
即使几何残差合格，含未知状态的结果仍为 `review_required`，不会用于自动回放。

本入口使用 `dexumi_wrist_solver.py`，与 6.2 的鲁棒联合求解器分开。
直接参考本地 `project1yu/DexUMI-main/real_script/calibration/compute_ur5_iphone_offset.py`：

```text
D_i = inverse(T_WT_0) @ T_WT_i
T_BE_pred_i = T_BE_0 @ T_ET @ D_i @ inverse(T_ET)
loss = mean(norm([position, rotation_vector]_pred - [position, rotation_vector]_measured))
```

平移和旋转六个参数一起用 L-BFGS-B 优化。为对应原版 UR5 的零 TCP 偏移，
先在 TRON2 模型腕法兰处拟合，再将结果转换回 SDK TCP 定义。
保留原版安装平移各轴 ±0.2 m、旋转向量各分量 ±π 的优化边界；它们不是机械臂关节限位。
额外使用几何初值，并固定留出第 4、8、12、16 组验证，不用验证组挑选参数。
静止窗口代替原版动态数据的固定延迟裁剪，未套用 iPhone/UR5 的延迟常数。
原目标混合米和弧度，报告另外给出毫米位置误差和 SO(3) 角度误差。

完整采样后在本次 `samples.jsonl` 同级的 `solutions/dexumi_*/` 保存矩阵、误差和原始文件校验值。
结果适用于本次 Tracker 安装及地图；更换安装或重建地图后不能直接套到旧 episode。

**2026-10-27 实测结论：** 原解码把 HID 四元数按 `WZYX` 重排，与位置数据的刚体关系不一致。
旁路记录保留了原始报文，按原始 `XYZW` 解释后，同一批数据的留出位置误差显著减小。
使用 15 组 `status=2` 数据（11 组拟合、4 组留出；首组 `status=18` 排除），
位置验证 RMSE `1.685 mm`、旋转 `0.201°`。
随后另采 8 组肩/肘 ±2° 姿态，保持标定矩阵不变，验证得到位置 RMSE `2.149 mm`、
最大 `3.506 mm`，旋转 RMSE `0.157°`、最大 `0.199°`。
这是当前安装及采样范围内与机器人编码器/模型的一致性，不是整个工作空间的精度保证。

最终文件见工程根目录 `tracker_calibrations/left_mapping_20261027_220113/`。
当前采集进程和旧 HDF5 未修改；读取原 pyvut 的存储四元数 `q=[w,x,y,z]` 时，
本次映射必须先做 **`q_corrected=[q[1],q[0],q[3],q[2]]`**，位置保留 XYZ。
这项转换由标定文件的 `tracker_quaternion_convention=raw-xyzw` 明确记录；
不得先在采集端改顺序后又重复转换。

```bash
# 从原始旁路样本重新求解（离线，不连接设备）
python3 dexumi_wrist_solver.py \
  --input tracker_calibrations/auto_20261027_215417_aad49c1d/20261027_215418_7f9d8fe8/samples.jsonl \
  --tracker-quaternion raw-xyzw
```

`dexumi_wrist_solver.wrist_from_tracker(T_WT_stored, calibration)` 已实现该转换及
`T_BE = T_BW @ T_WT_corrected @ inverse(T_ET)`，仅计算矩阵，不发送机器人指令。
本次 `T_ET` 是 Tracker 与裸左腕支架的安装关系；以前在人腕/另一支架上采集的 episode
还需核对当时的安装变换，不能直接套用。重新安装 O6 也需核对 TCP 与 Tracker 安装是否变化。

**同日补充复核：当前结果仅在小范围验证通过，尚未完成任务空间的精度验收。**
15 组样本里 Tracker 原点最大两点间距为 `31.65 mm`，一个方向的跨度仅 `2.64 mm`；
末端姿态最大两组夹角为 `27.07°`。逐一删掉一组拟合样本重新计算时，`T_ET` 的平移
最多变化 `8.70 mm`，但原留出组的位置 RMSE 仍在 `1.37–1.87 mm`。
因此低局部残差不等于外参参数已经稳定；应扩展多位置、多旋转轴采样，并留出更大范围姿态验证。
数值与方法见 `tracker_calibrations/left_mapping_20261027_220113/sampling_sensitivity_audit.txt` 和同名 JSON。
本次审计未改变矩阵、原始样本或 Tracker 连接，也未发送扩大范围的动作。

### 6.4 多位置、大范围标定

`auto_calibrate_tron2_tracker_wide.py` 在零 TCP 法兰的四个空间中心改变末端朝向，
通过全左臂协调得到 32 组姿态。前三个中心拟合，第四中心全部留出验证；
保留 DexUMI 原始六参数目标函数。右臂保持启动目标，头部不下发目标，Tracker 沿用现有连接。

```bash
cd /home/zmy/Projects/Dexumi
conda activate dexumi

# 只读规划；设备节点和 MAC 必须对应当前连接。
python3 auto_calibrate_tron2_tracker_wide.py \
  --passive-hid /dev/hidraw6 --tracker-mac 23:32:ae:d4:f6:a1

# 确认 High-Level Dev、支架固定及整个计划范围清空后才执行。
python3 auto_calibrate_tron2_tracker_wide.py \
  --passive-hid /dev/hidraw6 --tracker-mac 23:32:ae:d4:f6:a1 \
  --diagnostic-status 18 --execute --countdown 5
```

`--diagnostic-status 18` 只允许保留该原始状态采样，主求解仍排除含 18 的整组；
不会把 18 改写为 2。若剩余独立验证组不足 3 组，求解不通过。

**2026-10-27 已实测完成：** 32 组采样，正常组中 12 组拟合、6 组独立验证。
零偏移法兰跨度约 15.88 cm、Tracker 跨度约 18.67 cm、朝向跨度约 44.45°。
独立验证位置 RMSE 为 **8.627 mm**、朝向 RMSE 为 **0.764°**，主结果为
`review_required`，未通过原定 5 mm 位置 RMSE 标准，未应用于回放。
新增尺度自由度的诊断提示约 5% 的尺度差值得核查，但未自动添加任何缩放。

矩阵、误差图和后续复核事项见
[大范围标定报告](../../../tracker_calibrations/left_mapping_wide_20261027_221722/README.md)。
可用 `audit_dexumi_wrist_calibration.py --calibration <calibration.json>`
离线复查覆盖范围、删组敏感性及更换参考帧的敏感性。

进一步扩大整臂采样可使用 `--sampling-range large`。它重新读取当前姿态，规划四个
法兰中心、每中心 5 个朝向，并增加返回起点复测；若当前姿态不可达则在运动前拒绝整段计划。

```bash
# 只读规划；加 --execute 才会在模式和反馈检查通过后执行。
python3 auto_calibrate_tron2_tracker_wide.py --sampling-range large \
  --passive-hid /dev/hidraw6 --tracker-mac 23:32:ae:d4:f6:a1
```

2026-10-27 22:35 已执行该方案：实测法兰跨度 **31.92 cm**、肘关节跨度 **90.58°**、
朝向跨度 **69.59°**。本次刚体独立验证位置 RMSE **12.215 mm**，尚未通过。
相同腕朝向的两组正常状态数据中，模型法兰移动 192.911 mm，Tracker 经微小旋转杠杆
校正后移动 183.958 mm；约 4.64% 的差异说明还需核对物理尺度，不能仅靠继续增大动作解决。
完整记录见 [整臂大范围报告](../../../tracker_calibrations/left_mapping_large_20261027_223541/README.md)。

### 6.5 2026-10-27 全局刚体映射与新姿态验收

已从用户恢复的起点完成七个空间位置的广域采样（36 组），并在冻结矩阵后另采
9 个全新姿态做验收。肩肘与三个腕关节均参与；结束后已回到起点，原 Tracker 连接保留。

最终矩阵整合四轮的 47 个正常拟合组，保留 DexUMI 核心刚体关系，将单参考姿态
扩展为多参考姿态；不增加 scale，所有旧留出组及新验收组均不参与拟合。
新姿态中 7 组正常跟踪用于验收：位置 RMSE **7.642 mm**、最大 **12.099 mm**；
旋转 RMSE **0.625°**、最大 **0.898°**。这是约 1 cm 级的系统一致性，未达到此前
5 mm 的严格标准。回放入口现可通过 `--tracker-calibration` 显式使用该文件。

用户随后确认“1cm我觉得可以接受”，已将本组矩阵记录为当前地图与安装下接受的标定结果：
[用户验收记录](../../../tracker_calibrations/left_mapping_final_20261027/acceptance.json)。
实际最大位置误差仍为 12.099 mm；这次确认不改写冻结矩阵或通用求解阈值。

结果位于 `tracker_calibrations/left_mapping_final_20261027/`：
[矩阵、坐标定义和误差报告](../../../tracker_calibrations/left_mapping_final_20261027/README.md)。
转换公式为 `T_BE = T_BW @ T_WT @ inv(T_ET)`，其中位置单位为米，旋转使用完整矩阵。
安装偏移造成的圆弧位移已包含；不能直接要求 Tracker 与腕部两个原点的移动距离相等。
本次 HID 四元数保存格式的转换及地图/安装适用范围也在报告中说明。

对于历史真人左腕 episode 65，用户已确认地图及底座朝向未变。回放命令使用新矩阵的
`R_BW` 转换相对位置与相对旋转，首帧仍锚定准备姿态，不套用机器人法兰 `T_ET` 的安装平移。
使用 `--tracker-calibration tracker_calibrations/left_mapping_final_20261027/calibration.json`
替换 `--axis-map y,z,x`，并显式指定 `--tracker-quaternion pyvut-legacy`。
完整命令见 [回放命令第 12、13 节](../../../tron2_tracker_replay_commands.md#12-恢复查询超时后重新恢复再回放)。
网页现有换轴参数仍走旧路径；本次更新的是命令行入口，未自动修改网页或触发真机运动。

## 7. 检查保存文件

找到最近的 episode：

```bash
cd /home/zmy/Projects/Dexumi/TransHandR0725/TransHandR

latest_episode=$(find /home/zmy/workplace/transhandr_episodes/transhandr_demo/inspire \
  -type f -name 'episode_*.hdf5' | sort | tail -n 1)

python3 scripts/validate_multisource_episode.py "$latest_episode"
```

JSON 报告：

```bash
python3 scripts/validate_multisource_episode.py "$latest_episode" --json
```

报告会显示：

- D405 实际采样频率；
- 每个模态的有效帧比例；
- 相对 D405 的平均、P95 和最大绝对时间误差；
- 时间戳是否单调；
- Action 是否存在 future leakage；
- Tracker 四元数是否归一化。
- 本次未选择保存的通道会显示 `SKIPPED (not selected)`，不会被误认为采集失败。

只有真实采集并保存 episode 后，才能对 Vision Pro、Hand State、Action 和 Tracker 的真机对齐质量作最终判断。


## 8. SAM3及Diffueraser处理 生成mask视频

此节依赖外部 DiffuEraser 工程及其独立环境/权重，当前采集仓库未包含；仅迁移本仓库不能复现本节。
## full-black(sam3&diffueraser)
/home/zmy/Projects/DiffuEraser/.venv/bin/python \
/home/zmy/Projects/DiffuEraser/run_hdf5_sam3_diffueraser_batch.py \
--input-dir /home/zmy/workplace/insert-data/transhandr_demo/VR \
--output-dir /home/zmy/workplace/insert-data/transhandr_demo/VRFULLBACK_1 \
--auto-dark-glove \
--dark-policy full-black \
--overwrite \
--keep-work \
 --work-dir /home/zmy/workplace/insert-data/transhandr_demo/vrwork
    
如果只想先试一个文件
--limit-files 1
保留中间的 raw.mp4 和 human_hand_mask.mp4 方便检查:
  --keep-work \
  --work-dir /home/zmy/workplace/insert-data/transhandr_demo/vrwork

## blackground(only sam3 )版本1
# 机械手实例完整时，保留黑色拇指和关节。(VR)
# 与大块黑色外骨骼合并时，自动启用白色主体筛选。（linker）
 cd /home/zmy/Projects/sam3

.venv/bin/python scripts/segment_d435i_objects_auto.py \
  --input /home/zmy/workplace/insert-data/transhandr_demo/o6 \
          /home/zmy/workplace/insert-data/transhandr_demo/VR \
  --output-dir /home/zmy/workplace/insert-data/transhandr_demo/sam3use_auto

## 版本2（diffueraser）
cd /home/zmy/Projects/sam3
.venv/bin/python scripts/segment_d435i_objects_v2.py

## 机器人 Wi-Fi 与 VS Code/Codex 代理同时使用

已于2026-11-02配置双网络：电脑 Wi-Fi 连接 TRON2，手机 USB 网络共享提供互联网，
Clash Verge 保持现有代理配置。使用时保持手机移动数据和 USB 网络共享开启。
仅连接机器人 Wi-Fi、没有第二条互联网链路时，代理无法凭空提供互联网出口。

机器人两个 Wi-Fi 配置（`TRON2A_253_2.4G` / `TRON2A_253_5G`）均已持久设置：
IPv4/IPv6 不提供默认路由、不采用其自动 DNS，路由 metric 为700。
手机 USB 当前默认路由 metric 为100；机器人 `10.192.1.0/24` 走 Wi-Fi 直连。
没有更改代理节点、账号或端口，也没有修改机器人控制参数。

```bash
# 先连接手机并开启 USB 网络共享，再连接机器人 Wi-Fi。
nmcli --wait 40 connection up id TRON2A_253_2.4G

# 应显示机器人走 wlo1 直连；默认网关应来自手机 USB 网卡。
ip route get 10.192.1.2
ip route show table main
```

拔掉手机后，在机器人 Wi-Fi 上无法继续使用互联网代理；重新插入并开启 USB 网络共享即可。
当前代理端口为40209，以 Clash Verge 的实际设置为准。
已验证机器人 WebSocket 身份为 `DACH_TRON2A_253`，同时经代理访问 OpenAI HTTPS 成功；
未带密钥的 `/v1/models` 返回401是预期结果，只说明网络/TLS可达，不代表账号认证测试。

修改前配置及验收记录：
[网络配置备份](../../../validation_results/dual_network_20261102_091747/network_profiles.json)、
[双网络验收](../../../validation_results/dual_network_20261102_091747/verification.json)。

## CLI回放：接管后先恢复Z再启动轨迹（2026-11-03）

在根目录固定起点回放命令中显式增加 `--z-compensation --settle-handover`。
接管后先在原名义位置缓慢学习最多20mm的Z修正，连续1秒Z误差≤1mm才启动Tracker/O6共同时间轴；
30秒未收敛则不开始轨迹。共同起点等待期间保持连续发送，不重锚、不清零修正。
这是位置恢复，仍不能阻止首次瞬间下沉，也没有修复底层力矩/重力前馈或姿态误差。

原地真机验证：约14.72秒通过门禁，停发后Z误差−0.63mm。
随后已完成episode91与O6同步真机回放：整段Z误差RMSE从上次带手的20.80mm降到11.90mm，
目标最高点仍低约19.9mm，不能把起点≤1mm的收敛条件当成全轨迹精度。
原直接模式和网页默认行为保持；此新增模式目前通过CLI显式选择。
准备姿态及恢复路径检查不变；完整命令见
[回放命令末节](../../../tron2_tracker_replay_commands.md)，
[验证记录](../../../validation_results/settle_handover_20261103_173744/README.md)。
整段对比见[episode91验证](../../../validation_results/episode91_settle_handover_20261103_174958/README.md)。

CLI可增加 `--interpolation none` 对照回放原始Tracker目标：按原帧时间戳到期更新XYZ/朝向，
帧间重复上一帧，ServoP发送频率仍为300Hz；默认 `linear` 保持位置线性/旋转SLERP。
O6共同时间轴不变，该选项不消除首次接管下坠。episode91原帧最大目标阶跃约35.67mm/5.24°，
19:04已完成实机对照，目标最高点仍低约20mm；关闭脚本插值未解决高度问题。完整命令和与Z外环共同使用的解释见
[不插值命令](../../../tron2_tracker_replay_commands.md#不插值回放对照2026-11-03)。

