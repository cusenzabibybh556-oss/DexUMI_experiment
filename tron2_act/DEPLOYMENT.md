# Top＋左臂＋O6 ACT 上机

入口是 `tools/deploy_tron2_act.py`，配置为 `configs/tron2_act_deploy.json`。
已完成离线程序和录制数据验证，尚未运行本入口的真机闭环。使用现有最佳权重，不重新训练。

## 启动

所有命令从本项目目录执行，输出目录必须不存在：

```bash
cd /home/zmy/Projects/Dexumi
conda activate dexumi

# 完全离线：真实录制画面 → 同一前景模型 → ACT → 第一帧目标解码
python tools/deploy_tron2_act.py \
  --output artifacts/tron2_act/offline_check_01
```

真机需在固定准备位姿、开发模式，Top ROS 图像和 CameraInfo 正在发布。
保持与相机相同的 ROS_DOMAIN_ID／ROS 网络环境；主程序使用 dexumi，ROS 子进程使用系统 Python。
本程序不会启动相机、手套或采集器。左臂未到固定起点会退出，恢复位置继续使用现有
`restore_tron2_fixed_start.py` 流程。

```bash
# 可选只读预测：需要已经启动的真实左 O6 bridge；不接管手、不发送手臂或手指目标
python tools/deploy_tron2_act.py --shadow --existing-o6-bridge \
  --o6-can can0 --max-duration 5 --monitor-vision \
  --output artifacts/tron2_act/shadow_01

# 首次短时上机：自动启动独占 O6 bridge；进入真实 ServoP／手指控制
python tools/reset_o6_act_start.py --can can0
python restore_tron2_fixed_start.py --test-path --execute --trial-step-duration 3 --countdown 5
python tools/deploy_tron2_act.py --execute \
  --o6-can can0 --max-duration 0 --execution-speed 1.0 --monitor-vision \
  --output artifacts/tron2_act/hardware_01
```

CAN 以现场实际接线为准；以上沿用最近 replay 命令的 `can0`，若本次接在 `can1` 就显式改成
`--o6-can can1`。若已有可用 bridge，执行模式也可加 `--existing-o6-bridge`，不会再启动第二个。
bridge 必须为真实左 O6、接口一致、非 dry_run；不能同时运行其他机械臂控制程序。
`--monitor-vision` 在独立进程实时显示“对齐后的 Top 原图｜ACT 黑底输入”，并按采集时间写入
`<output>/vision_record/act_vision.mp4`；逐帧时间、序号、推理耗时和掩码像素数写入同目录的
`frames.jsonl`。显示或磁盘处理落后时只丢弃监看帧，不等待模型或 ServoP。

`--execute` 的顺序是：锁定控制权 → 加载并校验权重 → 启动或连接 O6 bridge → 只读核对
机器人身份、静止状态、固定起点、负载、相机及真实 O6 初始手型 → 首次视觉预测 → 5秒倒计时 →
已有 ServoP 渐入接管和位姿稳定 → O6 独占接管 → 取得新预测 → 开始任务计时。
`--max-duration` 只计算 ACT 任务段，不包含模型加载、倒计时和接管稳定时间。

`--max-duration 0` 表示不设任务时限，由现场按 `Ctrl+C` 停止；正数表示自动停止秒数且不再设置
60秒上限。`--execution-speed 1.0` 是当前训练部署速度，`0.5` 为半速，`0.2` 为五分之一速度；
只允许 `0.1..1.0`，不会开放超速。重定时会同时延长50步动作块的2.5秒覆盖范围，但不改空间目标幅度。
本模型没有任务完成／停止头，不会自动判定杯子是否已放好。
`report.json.completed=true` 仅表示运行时长内没有底层硬故障，不能解释为任务成功。

## 输入输出与时钟

| 项目 | 当前设置 |
|---|---|
| 权重 | 杯子基线 `policy_best.ckpt`，step 18500；策略和分割器 SHA256 固定 |
| 图像 | 当前 Top RGB → 内参视场匹配 → 冻结的实时分割器 → 机械左手＋相关物体保留，背景置黑 |
| 状态 | 当前实测 O6 `q_pos` 六通道＋物理时间前100/50ms的相对腕部历史，共24维 |
| 输出 | 未来50×15：基座空间 `dp3 + dR6D + O6目标6` |
| 动作时间 | 20Hz，首步为图像时刻后50ms，50步覆盖2.5s；训练0.4时间尺度已在标签中体现 |
| I/O时效 | 机器人/O6反馈硬时效为300ms；图像或预测超过300ms时冻结上一有效目标，恢复后继续 |
| 执行 | GPU 独立进程；主进程以300Hz调度已有 SDK ServoP；O6以20Hz发送到期目标 |
| 重预测 | 每幅新图最多查询一次，查询间隔至少50ms；反馈、图像、推理均不允许无限积压 |

不再应用 replay 的 `scale=0.1` 或二次慢放。默认仍使用 replay 的 SDK 腕参考点代理
Tracker 原点语义。SDK 反馈已在基座轴向中，部署不再重复乘 `R_BW`。
空间旋转左乘当前旋转；不同观察时刻的 chunk 先解码为同一基座系下的绝对目标，再进行
时间融合（越新的预测权重越大，decay=0.1）。不是把相对增量平均后反复累加。

O6 输入来自 `/robot/o6/left/joint_states` 的实际设备反馈，输出对应训练中的手套重定向命令，
为六通道 SDK 绝对目标，顺序为拇指弯曲、拇指外展、食指、中指、无名指、小指。
执行限位为 `[80,0,0,0,0,0]..255`。`action_applied` 只确认桥接已发送匹配目标，不能冒充实测手状态。

腕部反馈按请求往返中点配时，O6源时间戳转换到主机单调时钟；只有实测样本包围图像时间时
才构造输入。历史位置线性插值、旋转SLERP，不做外推，不跨越长缺口。
默认图像时间为本机 ROS 回调接收时间；可配置 `camera_clock=header`，但必须有一致的设备／主机时钟。
这不代表相机曝光、网络传输和机器人内部反馈年龄已完成精确标定。
前景处理不会消除相机安装位置、透视和遮挡差异；目前内参匹配也不补偿平移视差。

## 执行层及退出

直接复用 `PoseReplaySession`：独立只读反馈连接、模型前馈、124ms预瞄、XYZ／SO(3)残差闭环，
修正幅度／速率、IK残差、关节限位、工作空间、右臂和头部偏移等检查保持原约束。
该控制器仍沿用既有实验参数，124ms时序假设没有因新增 ACT 接口而自动获得实机标定。
运行时保存当时实际选择的名义目标，反馈配对不重新用新 chunk 改写历史目标。

与逐际动力公开 ACT rollout 一致，单帧空前景、图像断流、预测过期/乱序或暂时没有 chunk 覆盖当前
时刻时，不结束 episode：腕部和 O6 重发最后一个有效目标，后续有效观察到达后自动继续。
机器人/O6实测反馈超过300ms、O6发送未获确认、CAN/控制进程退出、关节/工作空间越限等底层
硬故障仍停止后续指令。
首次启动允许丢弃一次过期的冷启动预测，取得新鲜预测后才进入任务。
ServoP调度迟到时跳过错过的周期，不密集追赶；跳过次数写入报告，实际发送频率需看运行日志。
机器人位姿查询若首次超时、随后重试成功，控制配时只使用成功尝试本身的RTT；完整重试耗时另记为
`telemetry_query_recovered`，不会再把恢复时间误判成单包RTT。连续查询失败和反馈失联保护仍然保留。
正常到时保持末帧1秒；退出 O6 replay 模式后保持最后目标，不自动张手、归位或恢复手套控制。
Ctrl+C／异常会走资源清理。软件停发不等价于实体急停。

如以后有经核对的 Tracker 安装变换，可在配置中设 `reference_point=mounted-tracker`，
`mount_transform_file` 指向包含 `T_ET` 的JSON。矩阵单位为米，描述 SDK TCP 到虚拟 Tracker 点的
固定刚体变换；程序会在输入历史及输出目标两端应用同一个参考点转换。默认没有启用此选项，
不能把轴映射或世界坐标标定直接当作安装变换。

## 结果检查

每次生成 `runtime_config.json`、`report.json`、`trace.jsonl`、约每秒一张的
`policy_previews/`（原始Top／处理后输入并排）；自动桥接另有 `o6_bridge.log`。

| 日志事件 | 用途 |
|---|---|
| `act_raw_prediction` | 实际24维输入、图像时刻、原始50步输出、锚点和处理耗时 |
| `pose_replay_command` | 模型名义腕目标、执行层补偿后目标、补偿量与计算耗时 |
| `pose_replay_feedback` | 对齐后的名义目标和实测腕反馈，用于判断执行误差 |
| `act_hand_scheduled`／`act_ros_hand_target`／`act_hand_applied` | O6计划、发布、桥接发送确认 |
| `act_hand_feedback` | 真实 O6 状态，与预测命令分开保存 |
| `act_scheduler_skipped`／`act_error` | 调度丢周期和停止原因 |

离线验证和训练评估见
[`deployment_review_20261117/README.md`](../artifacts/tron2_act/deployment_review_20261117/README.md)。

## 上机前调整 Top 视角，检查左手和两个杯子

先摆好任务用的两个杯子，左手位于准备位。头部在 ACT 开始之前调好，任务期间保持该角度。
`tools/tron2_head_pose.py` 可以读取当前 pitch/yaw，两个参数的单位均为度；现场确定机械头角后才用该工具的
`--execute` 调节。历史实测角度只描述旧场景。取景时把训练参考帧放在旁边，比较手相对两个杯子的方向、大小、工作区边缘，以及伸手和叠杯时会发生的遮挡。

下面两步分别用 ROS 系统 Python 和 dexumi 环境。第一步只订阅 Top 并保存一张原图、CameraInfo 和训练参考图；第二步在保存的原图上运行与部署共用的模型预处理。两个输出目录每次使用新名称。

```bash
cd /home/zmy/Projects/Dexumi
(source /opt/ros/humble/setup.bash && /usr/bin/python3 tools/preview_tron2_top_camera.py \
  --reference-hdf5 /home/zmy/workplace/insert-data/transhandr_demo/o6/episode_000074.hdf5 \
  --snapshot --output-dir camera_alignment_results/cup_setup_01)
/home/zmy/anaconda3/envs/dexumi/bin/python tools/check_tron2_act_camera.py \
  --snapshot-dir camera_alignment_results/cup_setup_01 \
  --output artifacts/tron2_act/cup_setup_01_vision
```

查看 `vision.jpg`。左栏是实际送去分割的视角，中栏青色为手、橙色为物体，右栏是送入 ACT 的黑底 RGB。
检查机械左手和两只杯子的可见部分，以及桌边、电线、灯具等是否被误收。模型给两个杯子共用一个物体类，
杯子靠拢时连成一个掩码也可能正确。退出码 0 只说明达到了最低像素数；报告不会给出两个杯子均正确的判定。
调角或重新布置后重新拍摄一张；需要观察连续帧时使用已有的 `--shadow` 模式。

若只做机械头部调整，部署配置中的 `virtual_look_down_deg` 和 `virtual_look_right_deg` 保持 0。
若预览时还使用 `--look-down-deg` / `--look-right-deg` 做了虚拟取景，把相同数值写入部署 JSON
的这两个字段，再运行快照检查和 ACT。两者都从原始 Top 图像重映射一次。软件取景量与机械头关节角含义不同。
