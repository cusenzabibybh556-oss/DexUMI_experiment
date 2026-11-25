# Tracker → ACT 数据与训练框架

包含数据准备、训练、离线预测评估和显式启动的左臂＋O6实机部署。
实机入口、命令和当前验证范围见 [DEPLOYMENT.md](DEPLOYMENT.md)。
沿用 TronCamp ACT 的 Transformer/CVAE 思路，具体上游版本见
[vendor/UPSTREAM.md](vendor/UPSTREAM.md)。原有 replay、O6 bridge 和采集程序不改动。

可选 `prepare --foreground-root <目录>` 读取逐episode的 `foreground.hdf5` 前景缓存。
缓存必须通过图像检查并标记 `review_status=accepted`，同时匹配原始文件指纹、
RGB尺寸、全部帧号、有效位和时间戳；缺失或未通过的缓存会被拒绝，不回退到原图。
manifest记录前景文件指纹，准备后修改缓存也会拒绝加载。
此选项只替换RGB，腕历史、O6实测状态和手套控制命令仍从原始文件读取。
2026-11-16早期SAM3跟踪试验缓存标记为rejected；后续加入了
[标签修正和轻量分割链路](../tron2_vision/README.md)。新分割器的缓存默认标记为
experimental，仅在显式添加 `--allow-experimental-foreground` 时用于离线链路检查。
该选项会记录进manifest，仍拒绝rejected、来源变化或时间戳错位的缓存。
它不表示图像泛化、任务成功率或机器人部署已经验证通过。

## 数据和模型约定

- 每个任务独立准备数据、训练模型。同一任务允许物体和唯一目标在一定范围内变化；
  机器人使用固定准备位姿。目录中的通用 `task_name` 不能证明所有文件属于同一实际任务。
  请用配置的 `episodes` 或命令行 `--episodes` 明确选择同一任务的文件。
- 输入：真实 D435i/Top RGB、一段相对腕历史、当前 O6 六通道反馈。默认历史含过去
  100 ms、50 ms 两个位姿，各为 XYZ + rotation6D，总状态维度为 `2×9+6=24`。
  无触觉；不输入机械臂关节角；无需人工逐帧标注任务开始。
- 输出：未来 50 步，每步为 `XYZ位置增量3 + 相对旋转6D + O6绝对目标6`，共15维。
  rotation6D 为旋转矩阵前两列依次连接，不是六个角度。
- 动作和历史均采用基座轴向的空间增量：`dp=R_BW(p_k-p_now)`，
  `dR=(R_BW R_k)(R_BW R_now)^T`。部署阶段应左乘锚点朝向，不能用腕局部增量解码。
- 使用已有标定的 `R_BW`，不使用世界原点平移或 `T_ET`。保留当前 replay 的
  Tracker 原点代理语义；它不等价于已经完成 Tracker—SDK 腕参考点的安装补偿。
  相对腕历史训练／部署参考点的差异仍需要后续实验验证。
- `pyvut-legacy` 仅用于已确认的旧采集格式，显式将保存的 WXYZ 数组重新排列；
  对正确的新格式应配置 `wxyz`，不要重复修正。
- O6 `q_pos` 输入为 `dexterous_hand/joint_positions`，即真机 SDK 实测反馈。
  O6 `action` 标签为 `action/command`，即人手手套经过重定向、桥接限位后发送给 O6 的
  六通道控制目标；不是未来实测状态，也不是 `operator/exoskeleton/joint_positions` 中的原始21维手套角度。
  顺序：拇指弯曲、拇指外展、食指、中指、无名指、小指。

## 时间与图像

默认 `time_scale=0.4` 是自由空间仿真的折中候选，尚未获得相应真机任务成功率验证。
`action_hz=20` 表示每步50 ms**物理时间**，50步覆盖2.5秒；每一步在原始演示中推进20 ms。
以后部署不能再按0.4额外慢放一次。

样本锚点直接选真实相机帧的 `unified_timestamp_ns`，去除重复的相机帧。
历史、当前状态和未来标签在原时间轴上按 `source_offset=physical_offset×time_scale`
查询：位置和O6实测状态线性插值，旋转SLERP；O6动作按 `action/command_timestamp_ns`
选择目标时刻之前最新的命令，零阶保持，不把两条指令线性混合。
保留原文件 `action_max_age_ms` 的有效期规则；0表示持续保持最后一条指令。
不生成图像中间帧，也不强制图像按20 Hz更新。
原30 FPS视频重定时后约12张不同图像/物理秒，这不等于部署图像频率已确定。

时间戳默认要求严格递增。对已确认的采集重复行，可配置
`deduplicate_identical_timestamps: true` 或 prepare时添加
`--deduplicate-identical-timestamps`。程序先核对同一master时间的相邻行中，
图像、相机帧号、Tracker、O6状态和动作命令是否完全相同；通过后只在内存中
合并重复状态时间，图像仍使用原始行索引。审计写入每段的timestamp_audit。
同一时刻内容冲突、时间倒退仍会拒绝；不排序、不改写原始时间戳或HDF5。

排除NaN和无效反馈，不填零假装有效。有效数值端点间隔超过默认100 ms原记录时间时
不插值；腕历史不足的样本跳过，未来标签遇到缺口即截断，只在有效前缀上计算损失。
录制边界作为默认任务边界，episode尾部padding被屏蔽，不跨episode拼接，也不训练停止头。
当前框架使用已有统一时间戳，不声称它们已补偿所有设备的曝光／接收延迟。

动作约定现为 `tron2-act-spatial-command-v2`。早期 v1 曾将未来实测手状态用作标签，
不符合本项目的 state/action 定义；旧 prepared 数据、统计和权重必须重新生成，不能恢复混用。
历史 replay 使用实测手状态作回放目标的行为保持不变，不将其作为 ACT 标签来源。

## 环境

本机已用 `dexumi` 环境的 Python 3.10、PyTorch 2.5.1+cu124、torchvision 0.20.1+cu124
和 RTX 3090 检查。无需安装 RoboTwin、ROS、厂商SDK或旧 ACT 的仿真环境。

```bash
cd /home/zmy/Projects/Dexumi
conda activate dexumi
python -m tron2_act --help
```

其他机器按 [requirements.txt](requirements.txt) 安装依赖，torch与torchvision需选择匹配CUDA的版本。
默认训练加载 torchvision ResNet18 ImageNet 权重，首次使用可能下载；`--no-pretrained` 用于
无网络环境或链路检查。测试不下载模型。

## 准备数据

先编辑 [默认配置](../configs/tron2_act.yaml)，设置当前任务名、episode列表及本机路径，再运行：

```bash
python -m tron2_act prepare \
  --config configs/tron2_act.yaml \
  --task pick_place \
  --episodes 33 65 91 \
  --output artifacts/tron2_act/pick_place/prepared
```

这里的3个编号仅用于检查整个流程，不表示已经验证它们属于同一任务，也不足以支撑正式任务训练。
实际训练应替换为整理后的完整任务数据。输出目录必须不存在，避免覆盖旧切分和统计。

输出包括：

- `audit.json`：逐episode接受／拒绝原因、有效样本数；不会修改原始HDF5。
- `manifest.json`：完整episode切分、动作约定、配置、标定哈希和源文件指纹。
- `stats.json`：仅使用训练集的归一化统计，排除padding。
- `summary.json`：数据规模、输入输出尺寸和物理预测时域。

按episode固定随机划分训练／验证／测试，目标80%／10%／10%，验证和测试各至少一个文件，
因此至少需要3个有效episode。相邻帧和同一episode不会跨集合。
如果同一演示存在多个复制版本，应事先只保留一个，避免跨episode内容泄漏。
不要求人工剪起始帧，但混合任务、失败和调试数据仍需要整理。

准备阶段不复制RGB视频；训练按需读取原始HDF5。原文件的大小或修改时间变化会触发拒绝，
需要重新准备。模型包记录原数据位置和指纹，因此移动数据后也需要更新并重新准备。

## 训练和恢复

```bash
python -m tron2_act train \
  --prepared artifacts/tron2_act/pick_place/prepared \
  --output artifacts/tron2_act/pick_place/run01
```

默认 ResNet18＋hidden512、8头、encoder4层、decoder7层、latent32、batch8，
学习率1e-5，KL权重10，20000步。使用AdamW、梯度裁剪，状态维度和动作维度分别配置。
重建损失按有效动作元素平均；验证始终使用零潜变量先验，不把未来标签提供给CVAE编码器。

```bash
python -m tron2_act train \
  --prepared artifacts/tron2_act/pick_place/prepared \
  --output artifacts/tron2_act/pick_place/run01 \
  --resume artifacts/tron2_act/pick_place/run01/policy_last.ckpt \
  --max-steps 40000
```

输出 `policy_best.ckpt`、`policy_last.ckpt`、`metrics.jsonl`、配置、切分、统计和运行信息。
best按独立验证集的先验预测L1选择。恢复会检查数据身份与配置，恢复优化器与随机状态，
但采样器从新的确定性遍历开始，不承诺逐minibatch与未中断训练完全一致。
仅加载自己信任的本地checkpoint。

`history_steps=0` 可做无腕历史对照，但需要重新prepare，不能将24维模型直接改成6维输入。
改变时间倍率、图像尺寸、相对运动定义或历史长度同样需要重新prepare和训练。

## 离线评估

```bash
python -m tron2_act evaluate \
  --prepared artifacts/tron2_act/pick_place/prepared \
  --checkpoint artifacts/tron2_act/pick_place/run01/policy_best.ckpt \
  --device cuda \
  --output artifacts/tron2_act/pick_place/test01
```

输出真实单位的位置RMSE/P95、旋转角误差、O6各通道MAE、保持当前腕和手状态的对照，
以及 `predictions.npz`、对应源帧索引和 `trajectory.png`。旋转先投影到SO(3)再计算角误差；
退化旋转单独计数。评估不加载预训练权重、不访问机器人，不代表抓取成功率。

## 小规模链路检查

不要直接把正式模型长训练当成安装测试。先在独立配置中使用小图像、hidden64、
encoder1/decoder1、batch2、`pretrained_backbone: false` 准备3个episode，再执行：

```bash
python -m tron2_act train --prepared /path/to/smoke/prepared \
  --output /path/to/smoke/run --max-steps 4 \
  --eval-every 2 --checkpoint-every 2 --eval-batches 2
python -m tron2_act evaluate --prepared /path/to/smoke/prepared \
  --checkpoint /path/to/smoke/run/policy_best.ckpt \
  --device cuda --max-batches 2 --output /path/to/smoke/eval
python -m unittest discover -s tests -p test_tron2_act.py -v
```

`--eval-batches` / `--max-batches` 只用于链路检查；正式训练评估应省略，使用完整验证／测试集。
四步训练仅验证数据读取、反向传播、保存、加载及评估，不能生成可部署的策略。

## 双杯仿真与 ServoP 执行层验证

任务闭环使用 `--scene-config configs/tron2_act_cup_task_sim.json --task-validation`。
该配置修正 O6 掌面朝向并近似匹配演示初始画面；ACT 使用当前仿真 RGB 和实测状态，物体位姿/接触力只给独立任务判定器。
判定要求真实抓取、抬升、放置并稳定，运行结果在 `artifacts/tron2_act/task_closed_loop_20261117/README.md`。
当前三组场景测试均未完成抓取，不能将闭环链路运行或接口测试当作任务成功。

`tools/simulate_tron2_act_isaac.py` 使用现有 TRON2＋可动 O6 USD、桌面、两个中空杯子和 Top 相机。
输入来自当前仿真渲染和物理关节反馈，通过训练时的同一个前景模型，再送入 ACT；不会读取演示图像作为在线观测。
它不连接机器人、WebSocket 或 CAN。需要 `env_isaaclab` 环境中的 Isaac Sim、PyTorch、OpenCV、SciPy、h5py 和 MuJoCo（后者只用于参考模型 IK/前馈）。

```bash
cd /home/zmy/Projects/Dexumi
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
  /home/zmy/anaconda3/envs/env_isaaclab/bin/python tools/simulate_tron2_act_isaac.py \
  --output artifacts/tron2_act/my_sim_run --mode act --seconds 15
```

输出目录必须不存在。默认加载本次杯子基线的 `policy_best.ckpt` 和冻结的前景模型。
`--mode hold` 单独检查起点保持，`--no-feedback` 关闭残差反馈用于对照，`--gui` 显示界面。

执行时 ACT 为 20 Hz、chunk=50，历史为物理时间前 100/50 ms；保持训练的 0.4 时间倍率语义，不再对输出二次慢放。
腕部空间旋转增量左乘当前旋转；O6 `q_pos` 来自仿真关节反馈，`action` 为预测的六通道 SDK 绝对指令。
O6 指令在其未来 50 ms 到期时执行，并保持到下一条指令，避免重规划把指令重新替换成当前实测状态。

ServoP 执行参考复用实机回放的 `PoseReferenceModel`、124 ms 预瞄、模型前馈、`PoseCorrection` 和修正幅度/速率保护；
仿真内环为本机 IK＋PhysX 关节驱动，并非厂商固件。在线重规划额外限制参考速度/加速度
（80 mm/s、0.3 m/s²、30°/s、90°/s²）；所有平滑偏差都会记录，IK/关节/补偿保护不会放宽。
起点先渐入前馈并保持 8 秒，误差不满足 3 mm/1° 时拒绝启动 ACT。

保存 `scene.usda`、原图/前景并排的 `top.mp4`、反馈与 O6 指令 `trace.jsonl`、预测 chunk、来源哈希和报告。
被保护停止会写入 `status.json` 的 `stopped_by_guard` 及 `stop_diagnostic.json`，不能当作任务成功。
杯子尺寸、摩擦、相机和 O6 SDK 到关节角的映射尚未实物标定，因此此场景首先用于发现集成问题。

`tools/validate_tron2_act_servop_local.py` 是另一项控制器接口检查：地址固定 `127.0.0.1:5000`、身份固定
`DACH_TRON2A_001`，执行一段保存的 ACT 预测并启用相同外环。它需要先启动本机逐际控制器/电机仿真并进入开发模式；
该项使用录制图像产生的预测，没有视觉重规划，原厂电机仿真也不执行 O6。
2026-11-17 的视频、误差、接口测试和失败原因见 `artifacts/tron2_act/sim_20261117/README.md`。
