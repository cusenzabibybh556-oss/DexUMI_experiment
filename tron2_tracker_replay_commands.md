# TRON2 左臂＋O6：全位姿补偿回放

更新：2026-11-05。当前版本：`pose-reference-v1-experimental-20261105`。

本文仅保留当前使用流程：**episode91、scale=1、完整朝向、speed=0.2、O6/can0**。

## 模式说明

本模式通过 SDK ServoP 发送左腕位姿，右腕保持启动时实测参考，头部不发送运动目标；
O6 回放同一 episode 的手指实测状态，与机械臂共用回放起点时钟。

补偿由三部分组成：

- **模型前馈**：根据模型重力和参考运动速度，提前修正 XYZ 与朝向参考。
- **全位姿反馈**：比较反馈时刻的原始名义目标与实测位姿，逐步累积修正；运动中也更新。
  这是带限幅、限速的积分式外环，不是完整 PID。
- **轨迹预瞄**：读取未来 124 ms 的参考，尝试抵消响应滞后；该数值来自仿真，尚未完成实机时序标定。

参考发送频率为 300 Hz，误差学习随新反馈更新（本次约 20 Hz）。补偿修改 Cartesian ServoP 参考，
不直接写力矩、刚度或负载。XYZ 残差上限 10 mm、朝向残差上限 2°；
模型与残差合成修正上限 40 mm / 5°。

当前入口用于已有示教回放；策略模型的 action chunk 还需接入带时间戳的动作缓冲，才能复用此补偿环。

## 1. 环境与执行条件

```bash
cd /home/zmy/Projects/Dexumi
conda activate dexumi
```

机器人为 `DACH_TRON2A_253`，默认连接 `10.192.1.2`。左腕安装 O6＋转接件；
当前使用 `can0`，执行前该接口须已启用。程序自行启动回放专用桥接，无需启动手套或相机采集。
不要同时运行其他机械臂控制程序或另一套 O6 桥接。

回放前须处于固定准备姿态：小臂水平、中立腕、掌心向下。
偏离起点时按下节单独恢复；回放入口不会绕过未批准的自动恢复路径。
episode91 的名义最大相对位移约 28.63 cm，还需计入补偿范围，现场确认 O6、桌面、线缆净空并持实体急停。
软件停止仅停止后续发送，不是硬件急停。

## 2. 需要时：恢复固定起点

已在固定起点可跳过。本命令会执行左臂分段 MoveJ，不启动示教或手指轨迹：

```bash
python3 restore_tron2_fixed_start.py \
  --test-path --execute --countdown 5 --trial-step-duration 3
```

程序先读取当前姿态并规划，输入大写 `RESTORE` 后倒计时执行，每段验收到位后才继续。
右臂保持启动实测关节目标，头部不发送目标。模型检查不覆盖实际 O6、桌面和线缆；
恢复路径的现场净空需要单独确认。此入口不自动批准后续任意位置恢复路径。

## 3. 最新回放命令

```bash
python3 replay_tron2_fixed_start.py \
  --episode 91 \
  --arm-control servop --interpolation linear \
  --tracker-calibration tracker_calibrations/left_mapping_final_20261027/calibration.json \
  --tracker-quaternion pyvut-legacy \
  --scale 1.0 --orientation-scale 1.0 --speed 0.2 \
  --max-displacement 0.35 --countdown 5 \
  --pose-compensation --pose-preview-ms 124 --pose-feedback-age-ms 0 \
  --with-o6-hand --o6-can can0 \
  --execute
```

**只做离线预检：删除最后一行 `--execute`，同时删除上一行末尾的反斜杠。**
离线检查读取示教并检查整段名义/模型修正参考，不连接机器人、不打开 CAN。

实机执行顺序：

1. 核对显示版本为 `pose-reference-v1-experimental-20261105`，完成整段参考预检和桥接准备。
2. 输入大写 `REPLAY`，倒计时后接管，模型前馈至少用 5 秒渐入。
3. 起点位置误差 ≤3 mm、朝向误差 ≤1°，连续稳定 1 秒后安排机械臂/O6 公共起点；
   45 秒内无法稳定则中止，不开始示教轨迹。
4. 回放结束后检查终端验收信息及日志写入警告。

不要叠加旧 Z 补偿参数；新版会拒绝混用。

## 4. 常用参数

| 参数 | 当前值与含义 |
|---|---|
| `--episode` | `91`；数据默认为 `/home/zmy/workplace/insert-data/transhandr_demo/o6/episode_000091.hdf5` |
| `--scale` | `1.0`，保持原示教平移幅度 |
| `--orientation-scale` | `1.0`，保持完整相对朝向 |
| `--speed` | `0.2`，按原记录时间的 0.2 倍速度回放 |
| `--pose-compensation` | 显式启用当前全位姿模式 |
| `--pose-preview-ms` | `124`，未来目标预瞄时间；未完成硬件标定 |
| `--pose-feedback-age-ms` | `0`，按主机查询中点配对的年龄假设，并不表示反馈没有延迟 |
| `--with-o6-hand --o6-can` | 同步灵巧手，使用 `can0` |

更换 episode、speed 或 scale 会改变可达性和动态误差，应重新离线预检。
当前实验入口只接受 `0 < speed ≤ 0.5`、`0 < scale ≤ 1`；允许范围不等于每个组合都已实机验证。

## 5. 当前实测结果与限制

最近一次 episode91 实机已完成；终端报告末帧误差 **1.8 mm / 0.2°**。
已保存的前 30.51 秒中，Z RMSE **2.70 mm**、三维位置 P95 **7.70 mm**、朝向 P95 **0.66°**。
该窗口名义最大抬升 **82.24 mm**，实机最大抬升 **76.86 mm**，两个峰值相差约 **5.38 mm**。
接管仍曾下沉约 **9.75 mm**，尚未消除；位置 P95 也尚未达到 5 mm 的研究筛选线。

本次日志因 NumPy 布尔值序列化故障缺失最后约 3.45 秒及最终保持段，不能给出完整轨迹 RMSE。
该记录故障已修复并通过相关回归，补偿数值未改变；下一轮仍需核对完整日志。
真实反馈年龄、ServoP 等效增益和附件重心尚未完整识别，不能承诺复现仿真亚毫米精度。
SDK 腕反馈也不能替代外部位置测量或手指实际到位测量。

日志保存在 `/home/zmy/Projects/Dexumi/tron2_replay_traces/`，包括机械臂 `.jsonl` 和 O6 `.o6_bridge.log`。

- [最近一次实机评估](validation_results/pose_real_evaluation_20261105_201410/README.md)
- [模型配置与部署说明](validation_results/pose_reference_live_port_20261105/README.md)
