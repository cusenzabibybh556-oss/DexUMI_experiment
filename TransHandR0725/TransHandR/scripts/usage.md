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
| LinkerForce 外骨骼 | O6 | 左 |
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

同一页的“真机回放”是独立操作，调用 `replay_tron2_fixed_start.py`；仅硬件模式、从机器人旁的本机 `http://127.0.0.1:8080/data` 可提交真机操作，远程浏览器只能看状态。先选择含有效 Tracker（可选 O6 左手实测状态）的 episode，调整平移倍率、速度、轴映射、最大相对位移、倒计时及是否同步 O6，再点击“离线预检”。勾选同步 O6 时，可在回放窗口选择 `can0`、`can1` 等回放专用接口；采集系统须处于“已停止”，并须现场核实 CAN 接线。输入 `PREPARE` 后，回放任务只启动一个 O6 桥接并确认其回放服务已就绪，不启动数据手套或相机，然后才连接 TRON2、恢复双臂固定起点；验证到位后，页面再次要求输入 `REPLAY`，之后才发送 ServoP/O6 轨迹。任务结束或报错时会先退出 O6 回放模式，再关闭本任务启动的桥接；关闭后 O6 是否持续保持位置取决于设备本身，须现场确认。禁止同时运行另一个占用同一 CAN 的桥接。回放专用桥接日志与 JSONL 轨迹日志同名，后缀为 `.o6_bridge.log`。若只需回到保存的双臂准备姿态，可在同一窗口输入 `PREPARE` 后点击“仅恢复准备位置（不回放）”：此操作不依赖 episode 预检、不读取轨迹，只执行必要的 MoveJ 并验证双臂到位，绝不会自动进入 REPLAY。两种操作共用互斥锁，不能并发。

若离线预检显示“参数超出界限”，先看报错指向的输入框；原页面的“最大相对位移”默认值 `0.35` 与步长 `0.01` 不对齐，现已改为允许任意范围内的小数。网页允许 scale `0–1`、speed `0.05–1`、最大相对位移 `0.001–0.35 m`、倒计时 `5–30 s`；episode 65 在 scale=1 时的最大相对位移约 `0.3177 m`，默认 `0.35 m` 能通过这项检查。如果是实际轨迹位移超过设置的上限，应核对数据、轴映射和 scale，或选择位移较小的 episode；不要把“调大上限”当成可达性或碰撞验证。

关闭网页弹窗不会停止任务；“停止后续指令”不是硬件急停，MoveJ 可能仍在机器人端继续执行。必须在现场清空运动范围并持有实体急停，不可将离线工作空间检查视为碰撞或 IK 可达性证明。网页和固定起点 CLI 使用同一互斥锁，防止它们同时运行。网页启动的平台 Python 必须有回放所需依赖；若不同于 `dexumi` 环境，启动平台前可设置 `DEXUMI_REPLAY_PYTHON` 为该环境的 Python 绝对路径。修改网页代码后需要重启平台进程并刷新浏览器。

准备阶段遇到单次 `response_get_joint_state` 超时，现在只重试读取反馈，不重发 MoveJ；若整个准备窗口内仍不能确认关节到位，程序仍中止，不会进入下一阶段。这不能替代排查机器人控制器或网络长期不回包的根因。

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

## houbu
/home/zmy/Projects/DiffuEraser/.venv/bin/python \
/home/zmy/Projects/DiffuEraser/run_hdf5_sam3_diffueraser_batch.py \
  --input-dir /home/zmy/workplace/insert-data/transhandr_demo/o6 \
  --output-dir /home/zmy/workplace/insert-data/transhandr_demo/o6complete_batch \
  --auto-dark-glove \
  --dark-policy full-black \
  --diffueraser-mask-dilation 5 \
  --overwrite

## full-black(completed)
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
