# LinkerForce → Inspire 遥操作

## 网页启动

```bash
cd /home/zmy/Projects/Dexumi/TransHandR0725/TransHandR
conda activate versionprodemo
python3 scripts/run_teleoperation_platform.py --mode hardware --host 0.0.0.0 --port 8080
```

打开 `http://127.0.0.1:8080/overview`，选择：

```text
控制源：LinkerForce 外骨骼
目标手型：Inspire
手侧：左手或右手
```

在设备设置中填写左手套、右手套和 Inspire Hand 的三个独立
`/dev/serial/by-path/...` 路径。Inspire 链路不使用 CAN。

首次使用某一手侧时点击“重新标定”，按网页提示依次保持张手、O 型和握拳。
标定只连接所选手套，不启动 Inspire bridge，也不会移动真机。标定保存在：

```text
~/.config/transhandr/linkerforce_calibration.json
```

数据按 `inspire:left`、`inspire:right`、`o6:left` 和 `o6:right` 隔离。

## 数据链路

```text
LinkerForce 手套
  → handretarget（21 维原始关节 + 6 维重定向命令）
  → inspire_bridge（重排、0–1000 缩放、限位、Modbus 批量写）
  → Action / State ROS topics
  → ROS telemetry adapter
  → D405 主时钟对齐
  → quality-first-v3.1 HDF5
```

| 数据 | Topic |
|---|---|
| 外骨骼原始关节 | `/operator/linkerforce/{side}/joint_states` |
| 重定向命令输入 | `/cb_{side}_hand_control_cmd` |
| 实际成功下发 Action | `/robot/inspire/{side}/action_applied` |
| Inspire 真实 State | `/robot/inspire/{side}/joint_states` |
| Inspire 六路实际力 | `/robot/inspire/{side}/force_states` |

LinkerForce 六维指令顺序为：拇指弯曲、拇指侧摆、食指、中指、无名指、小指。
Bridge 按 `[5, 4, 3, 2, 0, 1]` 重排；Action 和 State 顺序均为：
小指、无名指、中指、食指、拇指弯曲、拇指旋转。Inspire 链路不会继承
O6 专用的电机限位。
Inspire 无触觉数据，HDF5 的 `tactile_stream` 为空；D405 主表中的触觉引用
写 `index=-1`、`valid=false`。

在“设备设置”中可为 Inspire 左右手分别勾选“力反馈”；只在系统停止时修改，
保存后下次启动生效，默认关闭。该开关只控制手套阻力，不影响六路实际力的采集。
需要写入 Episode 时，在底部“数据保存通道”选择 `Inspire Force`。
原始读数以 `dexterous_hand/force_stream/{value,timestamp_ns,sequence}` 独立保存，
不复制到 D405 的 30 Hz 主表。手套五路反馈顺序为拇指弯曲、食指、中指、
无名指、小指；拇指侧摆力仅保留在六路原始流中。

`FORCE_ACT` 是设备报告的实际力（g），不是 `CURRENT` 电流；寄存器定义见
[RH56 用户手册](https://www.inspire-robots.com/support/download/dexterous%20hands/)（官网下载中心）。
反馈启动后的前 20 帧用于空载零点，期间保持手指无接触。首次启用请先不佩戴
手套，逐指验证方向与输出上限；当前软件限幅不能替代真机安全验收。

## 限位

编辑 [inspire_command_limits.yaml](../config/inspire_command_limits.yaml)：

```yaml
inspire_bridge_node:
  ros__parameters:
    command_min: [0, 0, 0, 0, 0, 0]
    command_max: [1000, 1000, 1000, 1000, 1000, 1000]
    speed: 800
    force: 500
    min_send_interval: 0.02
    state_hz: 60.0
    force_hz: 20.0
    command_deadband: [12, 12, 12, 12, 16, 60]
    command_filter_alpha_min: 0.25
    command_filter_alpha_max: 0.75
    command_write_epsilon: 5
```

修改后无需重编译主项目，停止并重新启动系统即可生效。
Bridge 仅在滤波后目标发生有效变化时写入 Modbus；日志每两秒输出输入、
写入频率以及 Action/State 峰峰值，用于区分输入噪声和真机接触振荡。
手套稳定不动时写入与 Action 事件频率可能降为 `0 Hz`，这表示最后一条成功
命令仍在生效；D405 对齐会持续引用该命令，不会把它误判为断流。
