# 实验室工程案例记录：TransHandR 多源采集与 O6 左手遥操作整合

> 整理日期：2026-09-22  
> 资料来源：本仓库已有工程整理、TransHandR 多源采集文档、O6 左手外骨骼遥操作手册、数据格式核对报告、使用说明，以及本机 Codex 会话记录摘要。本文是实验室工程案例记录底稿，用于复盘工程处理过程与适用边界，不是对外项目介绍。

## 一、案例与披露范围

- 案例名称：TransHandR 多源数据采集与 O6 左手外骨骼遥操作链路整合。
- 发生时间：具体日期待确认；从现有日志和文档看，工程迭代集中在 2026 年 6 月至 7 月之后。
- 本人参与的工作范围：整理和联调多源采集、ROS 遥操作、O6 真机控制、HDF5 episode 保存与校验相关流程；具体个人贡献边界待导师或项目记录确认。
- 为什么选择这个案例：该案例同时涉及人手输入、灵巧手控制、RGB-D 相机、Tracker、ROS/CAN、HDF5 数据格式和质量校验，能体现实验室机器人系统工程中“多设备、多进程、多时钟、多格式”的典型问题。
- 允许对外介绍哪些内容：可概括介绍系统目标、公开设备类别、通用数据采集设计、调试思路和验证方法。
- 明确不能披露哪些内容：未经确认不披露导师项目源码细节、原始采集数据、内部日志、未公开指标、设备标定参数、任务对象或实验对象的敏感信息。
- 哪些信息尚需导师确认：本人实际完成的具体代码修改、真机测试时长、真实采集成功率、数据是否进入后续训练、哪些素材可对外展示。

本次补充读取到的相关本机会话包括：

| 会话主题 | 与本案例相关的信息 |
|---|---|
| 定位 TransHandR0725 重定向程序 | 需求从“保存重定向输出”收敛为“优先保存灵巧手真实反馈”，并形成多源采集需求文档。 |
| 查找 VIVE Tracker 采集脚本 | 确认 Tracker 只有 `status=2` 时可当作有效 6DoF；`status=3` 或 `NOT_INITIALIZED` 不应写成有效位姿。 |
| 核对数据格式并制作可视化界面 | 对比旧 HDF5 与 quality-first V1，推进采集器、网页 Dashboard 和 validator；曾定位 D435i 未被系统枚举的问题。 |
| 查找外骨骼手套重定向代码 | 阅读 `linkerhand_telop_python-master`，整合 LinkerForce 外骨骼、O6、ROS2 与统一平台；曾定位 VR 控制进程缺 `torch` 的环境问题。 |
| 打开 `ship1.hdf5` | 确认某次采集文件内含 D405/D435i 两路 RGB 帧序列，并成功导出为 MP4 视频用于离线查看。 |

## 二、原本要实现什么

原有模块分别能完成的事情：

| 模块 | 原有能力 |
|---|---|
| TransHandR | 从 Vision Pro 或 HDF5 中的人手 25 点关键点预测机器人灵巧手关节角，用于实时重定向或数据集测试。 |
| TeleOperation | 通过操纵杆、机械臂、灵巧手和 RealSense 采集 HDF5 episode，已有多进程和双相机采集思路。 |
| dex-retargeting / tele | 基于人手关键点、向量或位置目标进行机器人手重定向，并包含 LinkerHand 控制实验。 |
| VIVE Tracker 工具 | 读取 VIVE Ultimate Tracker 位置和四元数，为手腕 6DoF 位姿提供数据源。 |
| ROS2 O6 链路 | 通过 LinkerForce 左手套发布控制命令，经 `o6_bridge` 控制 O6 左手仿真或真机。 |

希望整合后的系统完成的流程：

1. 使用 Vision Pro 或 LinkerForce 外骨骼获取操作者手部输入。
2. 将人手动作重定向到 Inspire、LinkerHand L10/L21 或 O6 左手。
3. 用 D405 作为主时间轴采集手腕视角 RGB-D，可选 D435i 头部 RGB-D。
4. 同步保存右手或指定手侧关键点、真实灵巧手反馈、实际成功下发 action、VIVE Tracker 位姿和各源有效位。
5. 将 episode 写成稳定 HDF5 schema，并用校验脚本检查时间戳、有效率、同步误差、action future leakage 和四元数归一化。
6. 通过统一网页平台管理采集器和设备进程，降低手动启动多个终端的出错概率。

当时的必要条件或限制：

- D405 是采集主时间轴，D405 必须可用。
- 设备独占很重要：灵巧手串口/CAN、VIVE Dongle、RealSense 设备不能被多个进程同时打开。
- O6 真机运行前必须确认 `can0` 已启动、手套话题有变化，并先用 `dry_run` 检查下发姿态。
- VIVE Tracker 可以关闭并写 invalid 占位；但如果启用 Tracker，必须等追踪状态有效后再录制。
- HDF5 group/dataset 结构应稳定，未选择或无效模态写占位并配合 `valid=false`，方便训练代码统一读取。

怎样才算完成：

- 真机或 mock 模式能产生 episode。
- Dashboard 可显示数据源状态并控制开始、停止、保存。
- 保存后的 HDF5 能通过 `validate_multisource_episode.py` 校验。
- O6 左手链路能按 LinkerForce 手套变化产生 `/robot/o6/left/action_applied`、`/robot/o6/left/joint_states` 和触觉相关 topic。
- state 与 action 在数据中明确区分，action 只来自成功下发后的最终命令。

## 三、实际问题与影响

实际观察到或文档明确指出的问题：

| 问题 | 影响 |
|---|---|
| 旧 HDF5 数据只有 `frame_ids/right_hand/left_hand` 或 glove position | 缺少 D405、真实手反馈、action、VIVE、统一时间戳和 valid 标志，不能直接作为 quality-first V1 episode。 |
| 早期设计使用人工固定 30Hz 主时间轴 | 与实际相机 frameset 和多源异步到达不完全匹配，难以判断每帧 RGB-D 与其他源的真实对应关系。 |
| 旧设计未严格保存实际下发 action | 无法区分模型中间输出、目标命令和设备成功执行后的最终 action。 |
| 部分旧封装可能把 command echo 当作 hand state | 会把“想让手到哪里”误写成“真实设备反馈在哪里”，影响训练和故障分析。 |
| 多设备存在资源争用 | 同时运行多个采集器、`stream_poses.py` 或手部控制进程会争用串口、CAN、VIVE Dongle。 |
| O6 真机链路存在安全风险 | 如果手套话题固定、限位错误或 CAN 状态异常，直接控制真机会导致手停在单一姿态或产生不安全动作。 |
| D435i 头部相机曾启动失败 | 会话记录中出现过 `self.head_camera.start()` 报 `RuntimeError: No device connected`，当时 `lsusb` 和 RealSense 枚举只看到 D405，说明不是采集命令本身错误，而是 D435i 没有被系统识别。 |
| VR/TransHandR 控制进程曾因环境缺依赖失败 | 会话记录中平台启动后相机识别正常，但 VR 控制进程报 `ModuleNotFoundError: No module named 'torch'`，导致平台进入 `ERROR` 并安全停止采集器。 |
| Tracker 位姿曾处于不可用状态 | 会话记录中 Tracker 能输出姿态包，但状态为 `status=3`、`CURRENT_TRACKING_STATE=NOT_INITIALIZED` 时不能算有效 6DoF；直到 `status=2` 且 position 连续变化才可接入采集。 |

在什么条件下出现：

- 数据格式问题出现在使用旧数据集或旧采集脚本时。
- 资源争用通常出现在多个终端同时打开同一串口、VIVE Dongle 或 CAN 接口时。
- O6 风险出现在跳过手套话题检查、跳过 `dry_run`、未设置六通道限位或未确认 `can0` 状态时。
- D435i 启动失败出现在启用头部相机参数、但系统 USB/RealSense 枚举中没有 D435i 时。
- VR 控制进程依赖问题出现在平台用不含 `torch` 的 Python 环境启动 TransHandR/Vision Pro 控制链路时。
- Tracker 不可用出现在配对或 SLAM 初始化未完成、tracking state 不是 OK 的阶段。

对工作造成的影响：

- 旧 episode 无法直接用于 quality-first V1 训练或 ACT 多源 episode。
- 难以回溯每一帧图像对应的真实手状态与最终 action。
- 真机联调需要更严格的启动顺序和设备所有权管理。
- 采集后的数据必须通过独立校验脚本确认，而不能只看网页预览。
- 头部相机未被枚举时，启用 D435i 会导致采集器启动失败；临时方案是删除 `--enable-head-camera` 及相关参数，只先用 D405 采集。
- 控制进程缺依赖时，即使相机可用，平台也会因为控制链路失败而进入错误状态，网页会显示各源无有效样本。

尚未确认的内容：

- 具体某次真机故障是否由设备争用、时钟漂移、串口异常或代码逻辑导致，需要结合当时日志确认。
- Vision Pro source clock 的语义仍需实测；无法确认时只能退化到本机 `time.monotonic_ns()`。

## 四、如何排查与定位

| 步骤 | 当时怀疑的原因 | 做了什么检查或尝试 | 实际观察结果 | 能支持或排除什么 | 尚不能确定什么 |
|---|---|---|---|---|---|
| 1 | 旧数据格式是否满足训练需求 | 对已有 HDF5 文件做结构核对 | 旧文件只有手部关键点或 glove position | 支持“旧数据不能直接当 V1 episode 使用” | 旧数据是否可转换为某个简化训练格式 |
| 2 | 主时间轴是否应该固定 30Hz | 对比早期需求与当前实现文档 | 新实现改为 D405 frameset 驱动，每个有效 RGB-D frameset 产生一行 | 支持以相机主源建立 episode 的方案 | 两台 RealSense 的硬件曝光同步仍未解决 |
| 3 | hand state 与 action 是否混淆 | 检查采集文档中对 state/action 的定义 | V1 要求 state 是真实反馈，action 是成功下发后的 final command | 支持把模型输出、设备反馈、最终命令分开保存 | 部分旧 SDK 接口的真实反馈可靠性需设备验证 |
| 4 | Tracker 数据是否可靠 | 查看使用说明中的 Tracker 状态条件 | 只有 `OK (2)` 被视为有效位姿，未初始化或丢失不写 valid | 支持以 valid 标志和状态机保护数据质量 | Tracker 在具体实验场景中的丢帧率待实测 |
| 5 | O6 真机是否可以启动 | 手册要求先查串口、topic、`can0`、`dry_run` | 需要先看到手套六通道随动作变化，再启动真机 | 支持“先仿真/干跑，再真机”的排障顺序 | 某次异常姿态的根因需结合日志确认 |
| 6 | D435i 报错是否由代码或参数导致 | 查看 `rs-enumerate-devices`、`lsusb` 和 RealSense 枚举 | D405 `352122271077` 可见，D435i `141722077699` 不可见；错误发生在 `head_camera.start()` | 支持“D435i 未连接/线缆/USB 枚举问题”，不是序列号格式或 Python 逻辑问题 | 换线、换 USB 口后是否长期稳定 |
| 7 | 平台显示无有效样本是否一定是相机问题 | 对比相机预检和控制进程错误 | 两台相机可识别时，VR 控制进程仍可能因缺 `torch` 失败，平台进入 `ERROR` | 支持“控制链路环境失败会连带导致平台停止采集器” | 其他依赖或网络错误是否也会触发同样表现 |
| 8 | 已保存 HDF5 是否能离线检查图像 | 打开 `ship1.hdf5` 并查看数据内容 | 文件内不是封装 MP4，而是 D405/D435i 两路 RGB 帧序列；导出视频均为 `640x480`、`236` 帧、约 `29.99 FPS` | 支持用离线视频复核 episode 图像质量和帧率 | 该 episode 的其他模态有效率仍需 validator 检查 |

最关键的一条判断依据：

- 数据能不能进入后续训练，不取决于“有没有保存成 HDF5”，而取决于是否有稳定 schema、主时间轴、真实 state、最终 action、valid 标志和同步误差记录。

最终原因是否确认：

- 数据格式差异：已确认。
- 多设备资源争用风险：工程上明确存在，但具体某次故障根因需日志确认。
- O6 真机异常风险：已通过手册中的启动顺序和干跑流程进行规避，具体异常根因视当次情况而定。
- D435i 某次启动失败：已部分确认是系统未枚举到 D435i；具体是线缆、接口、供电或设备状态导致，需现场复测。
- VR 控制环境问题：已确认曾出现缺 `torch`，修复方向是固定使用 `versionprodemo` 环境。
- VIVE Tracker 有效性判定：已确认采集侧应以 `tracking_status == 2` 作为有效条件之一。

## 五、采取了什么处理方法

最后修改或整理的层面：

- 数据格式层：从旧格式转向 `quality-first-v1/v1.1` HDF5 schema。
- 同步策略层：从人工固定 30Hz 改为 D405 frameset 主时间轴。
- 数据语义层：明确区分真实 hand state 与最终 action。
- 设备进程层：统一设备所有权，避免多个进程同时打开同一硬件。
- 操作流程层：提供统一网页入口和 O6 左手真机手册，减少手动启动顺序错误。
- 环境启动层：统一平台中的 VR/TransHandR 控制进程固定到包含 `torch` 的 `versionprodemo` 环境，避免相机正常但控制进程因 Python 环境错误失败。
- 硬件排障层：对 D435i 这类可选头部相机增加“先枚举、再启用”的检查思路；当 D435i 不可见时，可先关闭头部相机只跑 D405 主链路。

修改前是什么方式：

- 旧设计使用固定 30Hz 对齐，全源 nearest，统一 50ms 阈值。
- 旧数据缺少完整 RGB-D 元数据、source/unified/receive 时间戳、真实 action、valid 标志和校验报告。
- O6/外骨骼链路需要手动启动多个 ROS 节点，并靠人工记忆检查步骤。

修改后是什么方式：

- D405 每个有效 RGB-D frameset 生成一个 timestep。
- D435i 和 Vision Pro 使用 nearest 对齐。
- hand state 使用线性插值。
- Tracker position 使用线性插值，quaternion 使用 SLERP。
- action 使用 latest-before，只允许使用当前 D405 时间点之前已经成功下发的命令，避免 future action 泄漏。
- invalid 帧使用统一占位：float 写 NaN，timestamp/int 写 -1，图像写 0，同时设置 `valid=false`。
- 统一平台可在网页选择 Vision Pro/Inspire/L10/L21 或 LinkerForce/O6 组合，并管理采集器和 ROS 进程。
- 平台预检区分“设备源未运行”和“运行中但无有效样本”，避免系统停止时误把所有相机都显示成采集失败。
- 对 VIVE Tracker，只有 `tracking_status == 2` 且位置/姿态连续更新时写 `valid=true`；未初始化、丢失或地图重建状态写 invalid。

为什么选择这项修改：

- 用 D405 作为主时间轴可以让图像数据成为 episode 的稳定锚点。
- state/action 分离能避免训练时把“目标命令”误当作“真实反馈”。
- latest-before action 可以避免未来信息泄漏。
- stable schema 可以让不同通道开关不影响读取代码结构。
- O6 真机前加入 topic 检查、限位和 dry-run，能降低硬件风险。
- 固定运行环境可以把“硬件没数据”和“控制进程没启动”区分开，减少误判。
- 对可选头部相机采用可关闭参数，可以在 D435i 硬件未就绪时保留 D405 主链路，避免整个采集流程被非主源阻塞。

修改带来的限制、额外成本或新问题：

- D405 失效时无法产生有效主时间轴。
- 两台 RealSense 仍没有硬件触发同步，软件对齐不能等价于同时曝光。
- Tracker 需要完成 SLAM 初始化，环境纹理和光照会影响有效率。
- Vision Pro source clock 语义还需要实测和 affine clock mapping。
- 数据文件更大，需要 episode 时长和文件大小上限。
- D435i 作为可选源时，软件对齐依赖它能被系统稳定枚举；USB 线、接口、扩展坞、供电都可能变成采集前置问题。
- 固定 conda 环境能解决依赖一致性，但也要求后续安装包时维护该环境，避免版本漂移。

这是临时绕过还是持续方案：

- `quality-first-v1/v1.1` schema、D405 主时间轴、state/action 分离、validator 属于持续方案。
- Vision Pro source clock 退化到本机时间、未标定外参写单位矩阵、关闭 Tracker 写 invalid 属于工程上可运行的阶段性方案。

## 六、如何验证结果

| 验证项目 | 测试条件与操作 | 实际次数/时长 | 观察结果 | 结果依据 | 未覆盖的条件 |
|---|---|---|---|---|---|
| mock 采集是否可运行 | `collect_transhandr_multisource.py --mode mock --record` 后运行 validator | 次数待确认 | 文档说明可生成并校验 episode | `multisource_collection.md` | 不能代表真机同步质量 |
| 真机多源采集是否可运行 | 启动采集器、Vision Pro/灵巧手控制、D405/D435i/VIVE | 待确认 | 文档给出硬件命令与 Dashboard 流程 | `usage.md` | 真实有效率、长时间稳定性待实测 |
| HDF5 schema 是否合格 | 对 episode 运行 `validate_multisource_episode.py` | 待确认 | validator 会检查采样频率、有效帧比例、同步误差、future leakage、quat norm | 使用说明 | 只验证已保存文件，不替代现场安全检查 |
| D435i 枚举问题定位 | `rs-enumerate-devices -s`、`lsusb`、RealSense 枚举 | 至少一次会话记录 | D405 可见，D435i 不可见，采集报 `No device connected` | 本机 Codex 会话记录 | 重新插拔、换线、换 USB 口后的长期稳定性 |
| VR 控制环境验证 | 检查平台启动日志和 Python 环境中的 `torch` | 至少一次会话记录 | 缺 `torch` 会使 VR 控制进程失败；固定到 `versionprodemo` 后预检通过 | 本机 Codex 会话记录 | Vision Pro 网络链路、模型权重和长期运行 |
| VIVE Tracker 有效性 | 观察 tracking status、position、quaternion | 至少一次会话记录 | `status=3`/`NOT_INITIALIZED` 不可靠；`status=2` 且 position 连续变化时可用 | 本机 Codex 会话记录 | 不同环境光照、纹理和遮挡下的有效率 |
| 已保存 episode 图像回放 | 从 `ship1.hdf5` 导出 D405/D435i RGB MP4 | 至少一次文件检查 | 两路视频均为 `640x480`、`236` 帧、约 `29.99 FPS`、时长约 `7.87s` | 本机 Codex 会话记录 | 深度图、hand state、action、Tracker 质量需另行校验 |
| O6 手套输入是否正常 | `ros2 topic hz/echo /cb_left_hand_control_cmd` | 待确认 | 六个 `position` 应随手套变化 | O6 手册 | 未覆盖真机负载和长时间温升 |
| O6 真机前干跑 | `o6_bridge -p dry_run:=True` | 待确认 | `dry_run pose` 应随手套变化 | O6 手册 | 干跑不访问真实手动力学和负载 |
| CAN 状态 | `ip -br link show type can` | 待确认 | `can0` 应显示 `UP` | O6 手册 | CAN 统计错误率需另查 |

哪些结果有记录，哪些来自回忆：

- 本文引用的结构、命令、字段和校验项来自仓库文档。
- D435i 枚举、VR 环境缺 `torch`、VIVE `status=2`、`ship1.hdf5` 视频导出来自本机 Codex 会话记录摘要。
- 具体长时间稳定运行时长、真机任务成功率、最终数据是否进入训练没有在本次整理中找到明确记录，标记为待确认。

与修改前相比能确认的变化：

- 新采集器明确生成 quality-first schema。
- 新采集器明确保存真实 state 与最终 action。
- 新采集器提供 Dashboard、通道选择、自动保存上限和独立 validator。
- O6 左手链路有完整的编译、手套检查、仿真、限位、CAN、dry-run、真机和停止流程。
- 平台排障从“只看网页有效/无效”细化为同时检查设备枚举、控制进程环境、Tracker tracking state 和保存后的离线回放。

## 七、本人、AI 与其他人的贡献

| 工作环节 | 本人实际完成的工作 | AI/同学/导师/开源资料提供的帮助 | 本人如何检查结果 | 仍无法解释或独立完成的部分 |
|---|---|---|---|---|
| 问题定位 | 待本人补充具体参与细节 | 仓库文档和工程代码提供证据 | 对比旧格式、需求文档和新实现说明 | 当时真机日志和具体故障链路 |
| 方案选择 | 待本人补充 | DexUMI、TransHandR、TeleOperation、pyvut、ROS2/O6 资料 | 检查方案是否保留 valid、timestamp、state/action 语义 | 是否为最优长期架构 |
| 修改与联调 | 待本人补充 | 现有脚本、SDK、设备手册、AI 整理辅助 | mock/hardware 启动、topic 检查、validator | 真实设备底层 SDK 行为 |
| 验证 | 待本人补充 | validator、Dashboard、ROS topic 工具 | 查看校验报告、topic 频率和 HDF5 字段 | 长时间稳定性和跨任务泛化 |
| 会话复盘 | 本人授权读取本机 Codex 会话记录并整理证据 | AI 辅助从 JSONL 中提取可读摘要，过滤系统提示和内部推理 | 只采用可和仓库文档、命令输出或文件检查相互印证的结论 | Copilot/其他账号云端会话未纳入 |

对这次修改，我能用自己的话解释到什么程度：

- 可以解释为什么要区分 state 和 action。
- 可以解释为什么 D405 适合作为主时间轴。
- 可以解释为什么 action 要 latest-before，不能用未来命令。
- 可以解释为什么 invalid 数据必须保留占位和 valid 标志。
- 可以解释为什么 D435i “No device connected” 要先查 USB 枚举，而不是先改代码。
- 可以解释为什么 Tracker 有姿态包不等于可用 6DoF，必须看 `tracking_status == 2` 和 position 连续变化。
- 对底层 SDK、Vision Pro 时钟语义、Tracker SLAM 初始化和真机控制安全策略，还需要结合设备文档和导师经验继续确认。

## 八、适用边界与复用依据

这次方法依赖的前提：

- 能访问 D405/D435i、灵巧手、Vision Pro 或外骨骼、VIVE Tracker 等设备接口。
- 采集程序运行在可以稳定读取设备并写入 HDF5 的 Linux 环境。
- 各设备进程遵守独占规则，不能多进程争用串口、CAN 或 Dongle。
- 后续训练代码能读取 quality-first schema 或经过明确转换。
- 运行环境需要固定并可复现，例如控制进程使用包含 `torch` 的 `versionprodemo`。
- 可选相机和 Tracker 需要先通过系统枚举或状态机确认，再作为有效模态参与录制。

可能适用于类似问题的排查步骤：

- 先核对数据 schema，而不是只看文件扩展名。
- 把真实反馈、目标命令和最终 action 分开记录。
- 为每个数据源保存 timestamp、valid、同步误差。
- 先 mock 验收 schema，再真机验收设备。
- 真机控制前先做 topic、dry-run、限位和设备状态检查。
- 遇到相机报 `No device connected`，先查 `rs-enumerate-devices -s` 和 `lsusb`，确认是不是系统未识别设备。
- 遇到 Dashboard 全部无有效样本，不只看相机，也检查控制进程是否因 Python 环境、模型依赖或网络失败退出。
- 遇到 Tracker 有数据但轨迹不可信，先看 tracking status；未到 OK 状态不能当有效样本。
- 保存后用离线回放或导出视频检查图像帧率和内容，再用 validator 检查结构化模态。

只适用于当前配置的修改：

- D405/D435i 序列号、Vision Pro IP、Inspire 串口、O6 CAN 接口、O6 六通道顺序和限位。
- 当前平台合法组合：Vision Pro 到 Inspire/L10/L21，LinkerForce 外骨骼到 O6 左手。
- `LinkerForce -> /cb_left_hand_control_cmd -> O6` 这条 ROS topic 约定。
- D435i `141722077699` 和 D405 `352122271077` 是当前记录中的设备序列号，换设备必须更新。
- `versionprodemo` 是当前记录中已用于控制链路的环境名，换机器时不应默认存在。

换设备或运行环境后，需要重新确认：

- 相机内参、外参、序列号和 depth scale。
- 相机是否被系统枚举为 USB 3.x 设备，是否被其他进程占用。
- 灵巧手真实 state 读取接口是否不是 command echo。
- action 维度、单位、关节顺序和安全限位。
- Tracker 坐标系、四元数顺序和追踪状态定义。
- Tracker 是否完成配对和 SLAM 初始化，`tracking_status` 的数值定义是否与当前 pyvut 版本一致。
- ROS topic 名称、消息类型、频率和发布/订阅者数量。
- 时钟域是否一致，是否需要 affine clock mapping。

哪些问题不在这次方法解决范围内：

- 两台 RealSense 的硬件级同步。
- Vision Pro source clock 的最终标定。
- 灵巧手控制策略本身的任务成功率提升。
- 数据后处理中的分割、去遮挡和策略训练效果。
- 未经实测的跨实验室硬件复用。

是否有第二个真实案例验证过复用：

- 本次整理未找到第二个完整真机案例记录；暂写“仅在本工程案例中整理和验证流程，复用效果待更多实验确认”。

## 九、可选素材清单

| 素材 | 能说明什么 | 属于谁的成果 | 是否已确认允许对外展示 | 是否需遮挡/裁剪 | 是否保留 |
|---|---|---|---|---|---|
| Dashboard 局部截图 | 说明多源状态和录制控制 | 实验室项目 | 待确认 | 需遮挡 IP、序列号、路径 | 可保留但暂不外发 |
| HDF5 validator 摘要 | 说明数据质量检查方式 | 实验室项目 | 待确认 | 可只保留字段名和统计项 | 可保留 |
| ROS topic 检查输出 | 说明 O6 链路是否通 | 实验室项目 | 待确认 | 需去除设备路径和内部主机信息 | 可保留 |
| RealSense 枚举输出 | 说明 D435i 未识别或双相机识别状态 | 实验室项目 | 待确认 | 需遮挡无关 USB 设备信息 | 可保留 |
| Tracker 状态输出 | 说明 `status=2` 才是有效 6DoF | 实验室项目 | 待确认 | 需遮挡设备 MAC 或内部路径 | 可保留 |
| `ship1.hdf5` 导出视频摘要 | 说明 episode 可离线回放，D405/D435i 帧率约 30Hz | 实验室项目 | 待确认 | 不展示原始画面，先只保留统计摘要 | 可保留 |
| 本文整理的通用流程 | 说明本人复盘能力 | 本人整理 | 可内部使用 | 对外需导师确认 | 保留 |

## 十、最后压缩成六句话

1. **问题：**当时希望把人手输入、灵巧手控制、RGB-D 相机、Tracker 和 HDF5 采集整合为可训练的数据链路，但旧数据格式缺少真实反馈、最终 action、统一时间戳、valid 标志和同步质量信息。
2. **判断：**通过对比旧 HDF5、早期需求文档、新采集器说明和会话排障记录，发现问题核心不是单个脚本能否运行，而是多源数据语义、时间对齐、设备枚举和进程环境是否可信；Vision Pro 时钟语义和长时间真机指标仍需确认。
3. **处理：**工程上采用 D405 frameset 作为主时间轴，区分真实 hand state 与成功下发后的 final action，并用 nearest、插值、SLERP 和 latest-before 分别处理不同模态；同时把 VR 控制进程固定到正确 conda 环境，并把 D435i、VIVE Tracker 作为可独立验证的可选源。
4. **结果：**新流程可以通过 Dashboard 录制 quality-first HDF5 episode，并用 validator 检查采样频率、有效率、同步误差、future action leakage 和四元数归一化；会话中还验证过 D435i 未枚举的定位方法、Tracker `status=2` 的有效性规则，以及 `ship1.hdf5` 两路 RGB 视频可离线导出。
5. **边界：**该方法目前依赖当前设备、串口/CAN/相机序列号、ROS topic、conda 环境和采集 schema，换设备或场地时需要重新确认外参、时钟、SDK 反馈语义、USB 枚举、Tracker 状态定义和安全限位。
6. **待了解：**这段经历让我想继续了解其他机器人团队在多源真机采集、action/state 语义、时钟同步和数据质量校验上是否遇到类似问题，但目前不能据此推断外部需求或商业价值。

## 带回前检查

- [x] 至少一个案例说清楚了问题、处理过程、结果和边界。
- [x] 观察事实、原因推测和未来计划已区分。
- [x] 没有补写未做过的检查、测试或对照实验。
- [x] 没有把 AI 或他人的贡献写成本人独立完成。
- [x] 没有把导师团队成果写成个人独立产品。
- [x] 未经确认可披露的内容已标记待确认或省略。
- [x] 待确认事项已单独标记，没有因材料完整性写成确定事实。
