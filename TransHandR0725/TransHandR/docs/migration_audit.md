# 数据采集链路迁移核验

核验日期：2026-10-16。范围：当前 Dexumi 工作目录下的采集平台、Vision Pro 控制器、LinkerForce ROS 链路、Tracker、HDF5 与网页，另记录真机回放和视频后处理的外部依赖。

## 结论

**现有安装教程不足以直接迁移；只上传当前 TransHandR 仓库，不能在新机器复现整条真机链路。** 补齐的安装命令见 [README](../README.md)。本次已经证明，在独立 Python 3.10 环境中，以及保持目录关系的临时源码副本中，离线测试、mock 录制、网页资源与 HDF5 回放可运行。尚未证明新机器上的相机/手套/灵巧手/头显/Tracker 真机链路可运行。

## 现有文档与修订

| 原有材料 | 核验结果 |
|---|---|
| 根 README | 原写 Windows 11 + Python 3.10，且引用已移走的根目录检查脚本；已改为当前采集部署说明 |
| `requirements.txt` | 包含 Torch/Torchvision nightly 和训练依赖，不匹配采集运行环境；保留原文件，README 明确不将其用于采集安装 |
| `requirements-collector.txt` | 原缺 SDK 实际 import 的 `PyYAML`；独立环境测试确认后已补齐，并修正“硬件依赖已随仓库附带”的错误注释 |
| `scripts/usage.md` | 原有较完整的现场启动/录制操作，但没有系统安装、跨仓库交付步骤；已补安装入口并修正过时的 State/Action 阈值、Inspire 速度和一处续行错误 |
| ROS 上游 README | 有通用 ROS 构建说明，未覆盖本地 O6/标定/遥测修改和 TransHandR 相对路径要求 |
| pyvut README | 有 HID、安装、建图与使用说明，但位于当前 Git 根以外 |
| 本次新增依赖文件 | `requirements-ros.txt`、`requirements-vision.txt`，按进程解释器拆分；它们是重建配方，尚非完整依赖锁文件 |

## 上传与迁移缺项

| 级别 | 缺项及证据 | 迁移时需要完成 |
|---|---|---|
| 阻断 | `Dexumi/` 本身没有 `.git`；当前 Git 根在 `TransHandR0725/TransHandR/` | 确定完整交付仓库布局，保留 README 中三部分源码的相对位置 |
| 阻断 | `scripts/`、`tests/`、采集配置、依赖文件在核验时仍有大量未跟踪内容 | 将实际新增代码纳入交付提交；只推已有提交会遗漏采集链路 |
| 阻断 | `scripts/run_teleoperation_platform.py` 在模块级导入相邻 ROS 源目录的 `CalibrationStore`，mock 也不例外 | 一并交付当前 `linkerhand_telop_python-master/ros2/src/linkerhand_retarget/` 源码和资源 |
| 阻断 | O6 bridge 与手套节点位于修改过的 ROS 包中，网页 O6 URDF/mesh 也在该包下 | 不能用上游默认版本替代；到新路径重新 `colcon build --symlink-install` |
| 阻断（Tracker） | 采集器通过相对路径导入 `project1yu/pyvut`；它是独立 Git 仓库 | 发布本地修改的固定版本，或作为整合交付的一部分；复制旧虚拟环境不能代替源码/安装 |
| 阻断（Vision Pro） | 三个推理权重均被 `checkpoint/` 忽略，未见下载地址 | 独立上传模型制品并提供校验，或采用 Git LFS；安装教程不能代替缺失权重 |
| 必配 | `TRANSHANDR_VISION_PYTHON` 默认是 `/home/zmy/anaconda3/envs/versionprodemo/bin/python` | 显式设置成新机器推理解释器绝对路径 |
| 必配 | 硬件默认录制目录是 `/home/zmy/workplace/transhandr_episodes` | 在网页“设备设置”修改；`--output-dir` 只设置平台配置/日志目录 |
| 必配 | 默认 IP、相机序列号、手套/真机串口、CAN 接口均为本机值 | 按新机器枚举结果填写，并配置串口/HID 权限、Tracker 配对和建图 |
| 必配 | LinkerForce 标定在 `~/.config/transhandr/linkerforce_calibration.json`，仓库之外 | 新现场重新标定，或经验证后迁移；左右手与目标手型分别管理 |
| 交付结构 | `inspire_hands/.git`、`project1yu/pyvut/.git` 是嵌套仓库；当前 TransHandR 没有已跟踪的 `.gitmodules` 配置 | 正确发布固定提交的子模块，或在独立交付副本中将源码纳入整合仓库；本次未改动这些 Git 元数据 |
| 上传体积 | 本地 `dataset.zip` 约 228 MiB，多个 HDF5 超过 100 MiB；三个权重各约 177 MiB | 数据与模型独立管理；已在 `.gitignore` 补采集输出、制品、本地 venv 和根数据压缩包排除规则 |
| 发布配置 | SDK `setting*.yaml` / ROS `base_config.yml` 有密码字段，pyvut `wifi_info.json` 有网络配置 | 发布前逐项脱敏并提供配置模板；本次仅记录文件位置，不展示或修改现场配置值 |
| 可选回放 | `scripts/robot_replay_web.py` 调用 Dexumi 根目录回放脚本，`replay_tron2_tracker.py` 动态导入 `limxsdk` | 将 README 第 7 节列出的回放文件和厂商 SDK 一并交付；本次不连接机器人 |
| 后处理 | `scripts/usage.md` 调用 `/home/zmy/Projects/DiffuEraser/run_hdf5_sam3_diffueraser_batch.py` | 单独交付外部工程、环境、SAM3/DiffuEraser 权重与安装说明；当前仓库不包含此链路 |

`.gitignore` 不会取消已经跟踪的文件；仍应人工核对最终提交范围。原工作区在核验前已有大量修改/删除，不能用一次无差别 `git add -A` 来代替审核。

本次没有初始化整合仓库、修改子模块、提交或推送 GitHub，也没有替用户选择数据/模型的公开发布范围。

## 解释器与依赖证据

| 运行部分 | 本机实际情况 | 迁移基线 |
|---|---|---|
| 操作系统 | Ubuntu 22.04.5 LTS | Ubuntu 22.04 |
| 系统/ROS Python | `/usr/bin/python3` = 3.10.12 | ROS Humble + 系统 Python 3.10 |
| Vision Pro Python | `versionprodemo` = 3.8.20 | 独立 Python 3.8 推理子进程，避免混用采集/ROS 依赖 |
| Vision Pro 模型 | Torch 1.12.1+cu113、Torchvision 0.13.1+cu113、timm 1.0.28、einops 0.8.1 | README 提供匹配旧环境的安装命令；未完成新环境全量重建 |
| Vision Pro 数值/通信 | NumPy 1.24.4、SciPy 1.10.1、avp_stream 2.51、grpcio 1.70.0 | 见 `requirements-vision.txt`；部分间接依赖尚未锁定 |
| RealSense | 系统 Python 中 2.58.3.10794；旧推理环境中 2.55.1.6486 | 采集 Python 3.10 示例采用前者，设备权限/驱动另装 |
| ROS shell | 平台固定调用 `/usr/bin/zsh` 并 source Humble 与本地 workspace 的 `setup.zsh` | 新机器必须安装 zsh，不只是能在 Bash 手工运行 ROS |

平台的 `termios`、`fcntl`、SocketCAN 与 Humble 路径表明整套平台以 Linux 为运行前提。原 Windows 文档不构成 Windows 迁移支持。

## 本次实际验证

所有真机相关检查均为静态读取或 Python 模块导入；未启动 hardware 模式，未发送机器人或灵巧手指令。

1. 原系统 Python 下关闭无关 pytest 插件自动加载后，现有测试 **60 passed**。直接执行原系统的 pytest 曾因系统 pytest 6.2.5 与用户级 anyio 插件不兼容而失败，这也是采用独立采集环境的原因。
2. 新建 `/tmp/dexumi-migration-audit-20261016` Python 3.10 venv，仅安装采集依赖与 pytest。第一次测试定位到缺 `PyYAML`；补齐依赖文件后，`pip check` 无冲突，**60 passed**。
3. 使用该独立环境执行 README 的 O6 左手 mock CLI（包括双相机、Tracker、外骨骼、触觉）录制 3 秒，保存 **83 帧**、30 个触觉事件；校验器返回 **`ok=true`、`errors=[]`、`warnings=[]`**。这是合成数据验收，不是真机精度证据。
4. 将相关源码、ROS 包和模型展示资源复制到临时 `relocated/Dexumi/`，没有复制旧 `.git`、ROS 构建目录、模型权重或本机虚拟环境。使用同一个独立 venv，在新路径再次运行测试，**60 passed**。
5. 在上述新路径启动 mock 平台和真实 HTTP 服务，验证 `/overview`、Three.js、O6/Inspire URDF、状态接口均返回 **HTTP 200**；完成录制保存、Episode 列表/详情读取和 JPEG 帧回放，HDF5 校验通过。未进行浏览器视觉检查；HTTP 可读取不代表所有模型的渲染效果已验收。
6. 在原有 Vision Pro 环境中确认 `torch`、`avp_stream`、`timm`、`PoseTransformer` 和本地 `InspireHand` 可导入。未重新安装整个推理环境，也未做新 GPU 兼容性/真实头显输入/模型权重推理验收。

独立采集环境实际解析版本：Python 3.10.12、NumPy 2.2.6、h5py 3.16.0、Pillow 12.3.0、hid 1.0.9、pyserial 3.5、python-can 4.6.1、PyYAML 6.0.3、pytest 8.4.2。这些版本代表本次成功安装的结果，并非未来安装的严格锁定。

## Vision Pro 权重校验

路径均相对 `TransHandR`；下载/复制后运行 `sha256sum <path>` 校验。此表仅确认当前本地文件身份，不提供下载渠道。

| 手型 | 路径 | 字节数 | SHA-256 |
|---|---|---:|---|
| Inspire | `checkpoint/models/Optimized/inspire/model_final.pth` | 185987322 | `350b68e0d4d3c0f057423f4925fa0fa589f87cae14a455cd6d2ba70095b3ef91` |
| L10 | `checkpoint/models/Optimized/l20_lite_right/model_final.pth` | 186054970 | `60eb37eac3c7c4804e770983e63c96e5c191c16118b2f586a0feb37e8b2b295f` |
| L21 | `checkpoint/models/Optimized/linker/model_final.pth` | 186035770 | `782bc91532fe5cdff7afd51c26052640139858236e95e8ef346d322914b58693` |

## 判定“可直接迁移”的后续验收条件

1. 从最终发布仓库/固定子模块提交取得源码，能够恢复 README 所列布局；新增脚本、SDK 修改、网页与模型资源没有遗漏。
2. 从发布的模型渠道取得所选手型权重，校验通过；目标环境安装与 ROS 构建通过。
3. 新机器无需依赖原机器目录，即可完成 mock 流程；填写设备配置并重新标定。
4. 逐个检查真实相机、Tracker、控制源与灵巧手，录制短 episode，HDF5 校验和实际有效率达到使用要求。
5. 若交付范围含 TRON2 回放或 SAM3/DiffuEraser，分别补充其外部交付和验收；采集 mock 通过不能覆盖这两项。
