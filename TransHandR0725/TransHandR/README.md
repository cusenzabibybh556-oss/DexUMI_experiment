# TransHandR 多源数据采集与遥操作

当前采集平台以 **Ubuntu 22.04、Python 3.10、ROS 2 Humble** 为部署基线。Vision Pro 的旧模型推理另用 Python 3.8 子进程。原 Windows/训练环境说明不能用于部署整条采集链路。

**迁移状态（2026-10-16）：可以按下述目录和步骤准备部署，尚不能只 clone 当前 TransHandR Git 仓库就直接使用真机。** ROS 源码、Tracker 本地修改、模型权重、设备设置和标定都需要额外交付。详见 [迁移核验报告](docs/migration_audit.md)。硬件安装步骤是部署配方；本次实际验收范围为独立 Python 3.10 环境的离线测试和 mock 录制，不包含新机器的真机验收。

## 环境对照与教程覆盖范围

**当前 README 覆盖采集主链路的安装步骤，并未覆盖整个工作目录中所有项目的完整环境。** 下表将现场已有环境与迁移安装方案分开列出：

| 用途 | 现场环境 / 迁移方案 | 本 README 覆盖情况 |
|---|---|---|
| 采集、网页、HDF5、RealSense、Tracker | 迁移时新建 Python 3.10 `.venv-collector`；原 `usage.md` 还保留 `versionprodemo` 的现场命令 | 第 2–3 节提供安装和验证命令；独立环境已通过离线验收。`.venv-collector` 是新建方案，当前源码目录里并未创建它 |
| LinkerForce、ROS adapter、O6/Inspire bridge | 系统 `/usr/bin/python3` 3.10.12 + ROS 2 Humble；不属于 Conda 环境 | 第 4 节提供 ROS 软件源、依赖、构建、权限和 CAN 步骤；新机器真机验收待完成 |
| Vision Pro 模型推理 | 现有 `versionprodemo` 为 Python 3.8.20；迁移示例名 `transhandr-vision` | 第 5 节提供重建命令；只核验了旧环境导入，未全量重建和真机验收；权重另行交付 |
| TRON2 真机回放 | 原回放命令使用 `dexumi`，Python 3.10.16；当前可读取到 `limxsdk` 4.1.1 | 第 7 节仅列依赖、脚本和解释器设置，**尚缺厂商 SDK 的完整安装/交付教程** |
| SAM3 / DiffuEraser 后处理 | `usage.md` 实际调用外部 `/home/zmy/Projects/DiffuEraser/.venv`，Python 3.10，Torch 2.3.1 | **未覆盖完整安装**；外部工程自身有 README，但其原始 Conda 示例与当前 `.venv` 不一致，仍需针对本地批处理脚本和模型单独整理 |

本机另有 `aloha`、`env_isaaclab`、`handsformer`、`openvla`、`tron2`、`diffueraser` 等 Conda 环境。它们的存在不代表采集必须全部安装；本 README 不为这些其他训练/实验项目提供完整教程。Conda 的 `diffueraser` 与外部工程的 `.venv` 也是两个独立环境。

还核实到旧环境会读取用户级依赖：`dexumi` 的 NumPy、h5py、limxsdk 来自 `~/.local/lib/python3.10/site-packages`，`versionprodemo` 的 h5py 来自 `~/.local/lib/python3.8/site-packages`。因此只复制 Conda 环境目录不能完整迁移当前依赖，应按步骤重新安装并验证。

## 1. 数据链路与交付目录

```text
Vision Pro → avp_stream → 模型推理 → LinkerHand / Inspire → UDP Action、State、25 点手部
LinkerForce → ROS handretarget → O6 / Inspire bridge → ROS telemetry adapter → UDP
D405 主相机 + 可选 D435i + 可选 pyvut Tracker + 上述 UDP
  → collect_transhandr_multisource → 主时钟对齐 → HDF5
  → 网页实时预览、录制管理、离线回放 → validate_multisource_episode
```

统一平台入口是 `scripts/run_teleoperation_platform.py`。支持 Vision Pro + Inspire/L10/L21 左右手，以及 LinkerForce + O6 左手或 Inspire 左右手。D405 必选，其他保存通道在网页选择。

以下目录关系是当前代码的要求，`Dexumi` 根目录可放在任意位置：

```text
Dexumi/                                      # 部署根目录，当前本身不是 Git 仓库
├── TransHandR0725/TransHandR/                 # 当前 Git 仓库
│   ├── scripts/、config/、model/、tests/
│   ├── LinkerHand/                           # O6 bridge 使用的本地 SDK
│   ├── linkerhand_20_lite_test/linkerhand_python_sdk/  # Vision Pro 使用的 SDK
│   ├── inspire_hands/inspire_hand/            # 本地 Inspire SDK
│   ├── dataset/robot/inspire_hand/            # 网页 URDF、mesh
│   └── checkpoint/models/Optimized/           # 独立下载/复制模型权重
├── linkerhand_telop_python-master/ros2/src/linkerhand_retarget/
│   ├── linkerhand_retarget/                   # 必须交付当前修改过的版本
│   ├── config/、resource/、launch/
│   └── setup.py、setup.cfg、package.xml
├── project1yu/pyvut/                          # 必须交付当前本地源码及 pyproject.toml
└── replay_*.py 等                            # 仅 TRON2 真机回放需要，见第 7 节
```

即使是统一平台的 mock 模式，也会导入 ROS 源目录里的 `calibration_store.py`，并从该目录读取 O6 展示模型；因此平台 mock 也要保留 ROS **源码和资源**，但不需要安装 ROS。独立采集器 CLI 的 mock 不需要 ROS 源码。

上游 LinkerHand/pyvut 仓库不能直接替代本机修改后的代码。不要复制旧 `ros2/build/`、`install/`、`log/`，到新路径后重新编译。

## 2. 基础环境与无硬件验收

以下示例使用 Bash；先将三部分源码放到第 1 节目录，然后在 `Dexumi` 根目录执行：

```bash
export DEXUMI_ROOT="$PWD"
export TRANSHANDR_ROOT="$DEXUMI_ROOT/TransHandR0725/TransHandR"
sudo apt update
sudo apt install -y git curl ca-certificates build-essential python3-pip python3-venv \
  zsh iproute2 can-utils libhidapi-hidraw0 libhidapi-libusb0 libusb-1.0-0

cd "$TRANSHANDR_ROOT"
/usr/bin/python3 -m venv .venv-collector
source .venv-collector/bin/activate
python -m pip install -r requirements-collector.txt
python -m pip install 'pytest>=7,<9'
python -m pip check
python -m pytest -q
```

`requirements-collector.txt` 用于采集、网页、HDF5 与基础 SDK；不要用旧 `requirements.txt` 替代它。旧文件属于历史训练环境，包含 Torch nightly，且缺少当前采集所需依赖。

不连接任何硬件，录制一个 3 秒 mock episode：

```bash
python scripts/collect_transhandr_multisource.py \
  --mode mock --control-source linkerforce --hand-type o6 --vision-hand-side left \
  --enable-exoskeleton --enable-tactile --enable-head-camera --enable-vive \
  --record --duration 3 --no-dashboard --action-port 0 \
  --output-dir /tmp/transhandr-mock

latest_episode=$(find /tmp/transhandr-mock -type f -name 'episode_*.hdf5' | sort | tail -n 1)
python scripts/validate_multisource_episode.py "$latest_episode" --json
```

启动统一网页（保持另一个终端没有采集器占用端口）：

```bash
python scripts/run_teleoperation_platform.py \
  --mode mock --host 127.0.0.1 --port 8080 --output-dir /tmp/transhandr-platform
```

打开 `http://127.0.0.1:8080/overview`，启动系统、开始录制、停止并保存，再到“数据”页回放。mock 数据只能证明软件流程，不能证明硬件时序质量。

## 3. 相机与 Tracker（采集 Python 3.10 环境）

保持 `.venv-collector` 激活：

```bash
python -m pip install pyrealsense2==2.58.3.10794
python -m pip install -e "$DEXUMI_ROOT/project1yu/pyvut"
python -c 'import pyrealsense2, hid; from pyvut import UltimateTrackerAPI; print("imports OK")'
```

该 RealSense 版本来自本机 Python 3.10 环境；头显旧环境实际使用 `2.55.1.6486`。Python wheel 不会替代 Linux 设备权限配置。新机器还需按 [RealSense 官方 Linux 安装说明](https://github.com/realsenseai/librealsense/blob/master/doc/distribution_linux.md) 配置软件源、udev 和适配内核的驱动。Ubuntu 22.04 软件源与工具安装示例：

```bash
sudo apt install -y gnupg apt-transport-https
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://librealsense.realsenseai.com/Debian/librealsenseai.asc \
  | gpg --dearmor | sudo tee /etc/apt/keyrings/librealsenseai.gpg >/dev/null
echo 'deb [signed-by=/etc/apt/keyrings/librealsenseai.gpg] https://librealsense.realsenseai.com/Debian/apt-repo jammy main' \
  | sudo tee /etc/apt/sources.list.d/librealsense.list
sudo apt update
sudo apt install librealsense2-utils librealsense2-udev-rules
realsense-viewer
python -c 'import pyrealsense2 as rs; print([(d.get_info(rs.camera_info.name), d.get_info(rs.camera_info.serial_number)) for d in rs.context().query_devices()])'
```

退出 viewer 后再启动采集器。若官方说明要求当前内核安装 `librealsense2-dkms`，按官方内核兼容要求处理。D405/D435i 需在网页填写本机序列号，使用 640×480@30 Hz。

串口与 Tracker HID 权限：

```bash
sudo usermod -aG dialout "$USER"
sudo tee /etc/udev/rules.d/70-transhandr-tracker.rules >/dev/null <<'EOF'
SUBSYSTEM=="hidraw", ATTRS{idVendor}=="0bb4", ATTRS{idProduct}=="0350", MODE="0660", GROUP="plugdev", TAG+="uaccess"
SUBSYSTEM=="hidraw", ATTRS{idVendor}=="0bb4", ATTRS{idProduct}=="06a3", MODE="0660", GROUP="plugdev", TAG+="uaccess"
EOF
sudo usermod -aG plugdev "$USER"
sudo udevadm control --reload-rules
sudo udevadm trigger
```

重新登录并重新插拔设备后生效。规则中的设备 ID 来自本地 `pyvut/pyvut/enums_usb.py`。Tracker 还需完成配对、建图和 SLAM 初始化，具体按 [本地 pyvut 教程](../../project1yu/pyvut/README.md)；这里的相对链接按完整部署目录解析。单独验证时先关闭采集器：

```bash
python "$DEXUMI_ROOT/project1yu/pyvut/scripts/stream_poses.py" --mode DONGLE_USB
```

只有 tracking state `OK (2)` 才能用于有效录制。不要让调试脚本和采集器同时占用 Dongle。

## 4. LinkerForce + O6 / Inspire（ROS 系统 Python 3.10）

此分支不需要 Torch 或 Vision Pro。统一平台固定调用 `/usr/bin/python3`、`/usr/bin/zsh` 和 `/opt/ros/humble/setup.zsh`，因此只在 Conda/采集 venv 中安装 ROS 依赖是不够的。

按 [ROS 2 Humble 官方 Ubuntu 安装说明](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html)，在 **Ubuntu 22.04、未激活 Conda/venv 的新终端**配置 locale 与 ROS apt 软件源。已有 Humble 安装可跳过此段：

```bash
sudo apt update
sudo apt install -y locales software-properties-common curl ca-certificates
sudo locale-gen en_US en_US.UTF-8
export LANG=en_US.UTF-8
sudo add-apt-repository -y universe
ROS_APT_SOURCE_VERSION=$(curl -fsSL https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest \
  | /usr/bin/python3 -c 'import json,sys; print(json.load(sys.stdin)["tag_name"])')
curl -fL -o /tmp/ros2-apt-source.deb \
  "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.jammy_all.deb"
sudo dpkg -i /tmp/ros2-apt-source.deb
```

安装 ROS、Python 依赖并编译本地源码：

```bash
sudo apt update
sudo apt install -y ros-humble-ros-base ros-dev-tools \
  ros-humble-sensor-msgs ros-humble-geometry-msgs ros-humble-std-msgs ros-humble-std-srvs \
  python3-colcon-common-extensions python3-rosdep python3-pip zsh

export DEXUMI_ROOT=/absolute/path/to/Dexumi
export TRANSHANDR_ROOT="$DEXUMI_ROOT/TransHandR0725/TransHandR"
/usr/bin/python3 -m pip install --user -r "$TRANSHANDR_ROOT/requirements-ros.txt"
/usr/bin/python3 -m pip install --user -e "$TRANSHANDR_ROOT/inspire_hands"
source /opt/ros/humble/setup.bash

# 仅首次初始化 ROS 机器时执行；已初始化则跳过这一条
sudo rosdep init
rosdep update
cd "$DEXUMI_ROOT/linkerhand_telop_python-master/ros2"
rosdep install --from-paths src --ignore-src -r -y
/usr/bin/python3 -m colcon build --symlink-install --packages-select linkerhand_retarget
source install/setup.bash
ros2 pkg executables linkerhand_retarget
/usr/bin/python3 -c 'import rclpy, serial, can; from inspire_hand import InspireHand; print("ROS imports OK")'
```

预期能看到 `handretarget`、`o6_bridge`。平台使用源码相对位置定位 SDK，须使用 `--symlink-install` 并保持第 1 节目录。`requirements-ros.txt` 根据当前 import 和本机版本整理，仍需在新机器验证构建与设备连接。

O6 使用前，在停止系统的状态下显式指定实际 CAN 口：

```bash
ip -br link show
bash "$TRANSHANDR_ROOT/scripts/start_can.sh" can1
```

`can1` 是本机接线示例，新机器根据枚举结果替换；网页设备设置需填写同一个接口。脚本使用 PEAK USB 驱动和 1 Mbit/s SocketCAN，其他适配器需先装对应驱动。Inspire 使用串口，不需要 CAN。

手套与 Inspire 真机分别使用独立的 `/dev/serial/by-path/...`。在网页选择控制源、手型、手侧并重新标定。标定默认保存到 `~/.config/transhandr/linkerforce_calibration.json`，也可用 `TRANSHANDR_CALIBRATION_STORE` 指定位置；本次迁移不会自动带走该文件。限位在 `config/inspire_command_limits.yaml` 和 ROS 源目录 `config/o6_command_limits.yaml`。

## 5. Vision Pro + Inspire / L10 / L21（独立推理环境）

本机 `versionprodemo` 实测为 Python **3.8.20**、Torch **1.12.1+cu113**、Torchvision **0.13.1+cu113**、NumPy **1.24.4**、`avp_stream` **2.51**。下面按旧推理环境重建，采集平台仍运行第 2 节的 Python 3.10。

已安装 Miniconda/Anaconda 后：

```bash
conda create -n transhandr-vision python=3.8 -y
conda activate transhandr-vision
cd /absolute/path/to/Dexumi/TransHandR0725/TransHandR

# 根据本机 NVIDIA GPU/驱动核对兼容性；命令对应旧环境 CUDA 11.3 wheel
python -m pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 \
  --extra-index-url https://download.pytorch.org/whl/cu113
python -m pip install -r requirements-vision.txt
python -m pip install -e ./inspire_hands
python -m pip check
python -c 'import torch, avp_stream, timm; from model.model_poseformer import PoseTransformer; from inspire_hand import InspireHand; print(torch.__version__, torch.cuda.is_available())'
```

CPU 调试可以将上面的 Torch/Torchvision 后缀换成 `+cpu`，并改用 `https://download.pytorch.org/whl/cpu`；实时推理速度需另测。安装组合依据 [PyTorch 官方历史版本说明](https://pytorch.org/get-started/previous-versions/)。该配方未在本次从零安装和真机验收，不能视为完整锁文件；新 GPU 需要新版 Torch 时应单独验证权重加载与推理，不能直接混用历史 `requirements.txt`。

头显端还需安装并启动 [Tracking Streamer / VisionProTeleop](https://github.com/Improbable-AI/VisionProTeleop)，与采集电脑在同一网络，网页填写 IP。平台预检访问 gRPC TCP `12345`。

必须另行取得以下权重，路径固定；每份约 186 MB，当前被 `checkpoint/` 忽略，仓库内尚无下载地址：

| 选择手型 | 需要的权重（相对 TransHandR） |
|---|---|
| Inspire | `checkpoint/models/Optimized/inspire/model_final.pth` |
| L10（控制器内部映射到 l20） | `checkpoint/models/Optimized/l20_lite_right/model_final.pth` |
| L21 | `checkpoint/models/Optimized/linker/model_final.pth` |

建议用独立模型制品下载或 Git LFS 管理，并校验报告中的 SHA-256。缺权重时 mock 仍可运行，Vision Pro 真机推理不能运行。

记录推理环境解释器的绝对路径，下一节使用：

```bash
python -c 'import sys; print(sys.executable)'
```

## 6. 启动真机平台与保存验收

回到采集环境。在第 5 节实际解释器路径基础上设置变量：

```bash
cd /absolute/path/to/Dexumi/TransHandR0725/TransHandR
source .venv-collector/bin/activate
# 仅 Vision Pro 分支需要；必须替换为上一节输出的路径
export TRANSHANDR_VISION_PYTHON=/absolute/path/to/envs/transhandr-vision/bin/python
python scripts/run_teleoperation_platform.py \
  --mode hardware --host 127.0.0.1 --port 8080 --telemetry-port 8765
```

打开 `http://127.0.0.1:8080/overview`。在系统停止时，先通过设备设置修改**录制保存目录**、D405/D435i 序列号、Vision Pro IP、手套和真机串口、CAN 口，再选择保存通道与所需标定。硬件模式默认保存目录仍硬编码为 `/home/zmy/workplace/transhandr_episodes`，新机器必须改为本机可写的绝对路径；`--output-dir` 只设置平台配置/日志目录，不能替代网页的录制保存目录。

默认端口为 HTTP TCP `8080`、本机 telemetry UDP `8765`；ROS 设备进程由平台管理。启动系统后检查每个已选数据源有效，录制短 episode，停止并保存，然后校验：

```bash
python scripts/validate_multisource_episode.py /absolute/path/to/episode_000000.hdf5 --json
```

检查主帧频率、有效率、时间戳、Action 无未来泄漏及 Tracker 四元数。D405 是主时间轴，软件对齐不等于相机硬件同步。当前 Hand State 默认插值阈值是 **35 ms**；Action 默认保持最后成功命令直至被替换，禁止使用未来指令。

详细操作见 [scripts/usage.md](scripts/usage.md) 和 [LinkerForce → Inspire 说明](docs/linkerforce_inspire_teleoperation.md)。其中旧机器的绝对路径、设备序列号和 IP 都是现场示例。

## 7. 可选真机回放与视频后处理

采集不要求 TRON2 SDK。网页的 TRON2 真机回放还需要在 `Dexumi` 根目录交付：

```text
replay_tron2_fixed_start.py    replay_tron2_tracker.py
restore_tron2_fixed_start.py  prepare_tron2_arms.py
o6_hand_replay.py            o6_replay_bridge.py
replay_o6_hand_ros.py
```

回放 Python 环境需安装 `numpy`、`h5py` 和厂商 `limxsdk`；该 SDK 当前不在采集仓库中，需按厂商针对机器人/系统架构的安装说明交付。用 `DEXUMI_REPLAY_PYTHON=/absolute/path/to/python` 指定解释器。回放准备姿态、机器人网络和运动参数需重新现场验收，按使用说明执行 PREPARE/REPLAY。

`scripts/usage.md` 的 SAM3 / DiffuEraser 步骤调用 `/home/zmy/Projects/DiffuEraser` 外部项目，该项目、权重和环境均不在当前交付目录；因此**后处理链路尚不具备独立迁移条件**，也不属于以上采集安装命令的验收范围。

## 8. 上传 GitHub 前

### 多个 `.gitignore` 与嵌套 `.git`

2026-10-16 对整个 Dexumi 工作目录扫描，排除 IDE、缓存和虚拟环境后，发现 **15 份项目 `.gitignore` 和 8 个 `.git` 仓库目录**。计数覆盖历史项目与副本，不代表当前采集需要全部项目。

- `.gitignore` 是忽略规则，同一仓库允许多份，子目录规则作用于对应目录。保留它们不会妨碍上传源码；删除后不会合并仓库，反而可能把 episode、权重、虚拟环境和构建产物纳入提交。
- `.git` 保存仓库历史、分支和远程信息。若要将多个项目作为普通源码目录合并到一个新仓库，应在**独立发布副本**中处理嵌套 `.git`，保留原工作区的历史。
- 最简单的交付方案是准备一个保留第 1 节相对路径的发布目录，复制所需源码和资源时排除 `.git`、虚拟环境、采集数据及旧构建产物，再为发布目录配置根 `.gitignore` 并检查最终提交清单。

当前采集相关的嵌套仓库还包括 `inspire_hands/`、`linkerhand_20_lite_test/linkerhand_python_sdk/` 和 `project1yu/pyvut/`。本次没有删除任何 `.git` 或 `.gitignore`，也没有执行整合仓库发布。

当前 Git 根目录是 `TransHandR0725/TransHandR`，并非整个 `Dexumi`。`scripts/`、`tests/`、采集配置和依赖文件当前有大量未跟踪项；只 push 已有提交不会上传这些新增代码。先用 `git status --short` 检查，再选择提交范围。

完整交付可采用新建整合仓库并保留第 1 节目录，或将修改后的 ROS/pyvut/Inspire 各自发布为固定提交的子模块。当前尚未建立这套整合关系；现有 `inspire_hands/.git`、`project1yu/pyvut/.git` 等嵌套 Git 目录不能在未配置子模块时当普通目录直接 `git add`。

上传范围应包含源码、网页 vendor 文件、URDF/mesh、限位配置、依赖和文档；数据 episode、数据集压缩包、模型权重、旧构建产物与本机环境应独立管理。公开发布前检查 SDK `setting*.yaml` / ROS `base_config.yml` 的密码字段以及 `pyvut/pyvut/wifi_info.json` 的网络配置，使用脱敏模板交付，并保留第三方许可证。详细缺项、证据及验证结果见 [迁移核验报告](docs/migration_audit.md)。
