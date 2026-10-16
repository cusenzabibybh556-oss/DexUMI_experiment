# TransHandR quality-first V3 采集器

## 安装轻量依赖

采集器不需要安装训练用的整套 Torch/CUDA 环境。首次运行先执行：

```bash
python3 -m pip install --user -r requirements-collector.txt
```

## 快速验收（无硬件）

```bash
cd TransHandR0725/TransHandR
python3 scripts/collect_transhandr_multisource.py \
  --mode mock --record --width 320 --height 240
python3 scripts/validate_multisource_episode.py \
  data/multisource/transhandr_demo/l21/episode_000000.hdf5
```

采集时访问 <http://127.0.0.1:8080>。网页的开始/停止按钮与 `--record` 均可控制 episode。
程序会持续运行到按下 `Q` 或 `Ctrl+C`。`--duration 10` 仅用于自动化测试，10 秒后采集器和网页服务都会关闭。

## 真机运行

当 TransHandR 同时控制 Inspire Hand 时，必须保证只有控制进程打开 `/dev/ttyUSB0`。使用两个终端：

终端 A（控制手并发布真实 state/final action）：

```bash
python3 scripts/main_visionpro_multi_hand.py \
  --hand inspire --avp_ip 172.20.10.9 \
  --serial_port /dev/ttyUSB0 \
  --telemetry_port 8765 --telemetry_state_hz 60
```

终端 B（采集器通过 UDP 接收手数据，不打开串口）：

```bash
python3 scripts/collect_transhandr_multisource.py \
  --mode hardware \
  --hand-type inspire --hand-source udp --right-hand-source udp \
  --d405-serial 352122271077 \
  --enable-head-camera --head-camera-serial 141722077699 \
  --head-width 640 --head-height 480 --head-camera-fps 30 \
  --enable-vive --tracker-mode DONGLE_USB
```

先启动 B 再启动 A，或之后重启 A 均可。不要同时运行另一份 `stream_poses.py`，因为 VIVE Dongle 也必须只有一个读取者。

```bash
python3 scripts/collect_transhandr_multisource.py \
  --mode hardware --hand-type inspire --hand-serial-port /dev/ttyUSB0 \
  --avp-ip 10.227.197.16 --d405-serial YOUR_SERIAL
```

只先接入 D405 + VIVE Tracker（其余模态保存为 invalid 占位）：

```bash
python3 scripts/collect_transhandr_multisource.py \
  --mode hardware \
  --d405-serial 352122271077 \
  --enable-vive --tracker-mode DONGLE_USB \
  --disable-vision --disable-hand \
```

默认单个 episode 最长 300 秒、最大 5 GiB，任一上限达到后自动停止并保存。启动命令默认不录制，请在网页点击开始或按 `R`。

D435i 头部相机通过 `--enable-head-camera` 开启。它独立采集 RGB-D，再以 D405 的主时间戳做 nearest 软件对齐；`sync/d435i_dt_ms` 保存配对时间差，超过 `--head-camera-max-gap-ms`（默认 45 ms）时该帧 `camera/d435i/valid=false`。当前硬件序列号为 `141722077699`，已验证 RGB8/Z16 `640x480@30Hz` 可同时运行。两台相机没有硬件触发同步，因此相同 30 Hz 不代表同时曝光。

真机模式中采集器独占灵巧手连接，同时在 `127.0.0.1:8765` 接收 TransHandR 最终映射、裁剪后的 command：

```python
import json, socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.sendto(json.dumps({"command": final_command.tolist()}).encode(),
            ("127.0.0.1", 8765))
```

只有实际调用设备发送接口成功后，command 才会写入 action buffer。不要从模型中间输出发送此消息。Linker L10/L21 保存 SDK `get_state()` 的真实反馈；Inspire 保存 `ANGLE_ACT` 读取结果，不使用 command echo。

VIVE 可通过 `--enable-vive --tracker-mode DONGLE_USB` 开启。未开启时仍写占位值且 `valid=false`。

如果 Dongle 日志显示 `Unpaired`，长按 Tracker 电源键直到 LED 快速闪烁蓝色，再重新启动上述命令。配对并完成 SLAM 初始化后，Dashboard 的 `TRACKER` 会从 `INVALID` 变为 `VALID`。不要运行会写固件或清除地图的实验性 pyvut 命令。

## 触觉异步流

O6 触觉不重采样到 D405 30 Hz。每个真实触觉事件只写入一次：

```text
dexterous_hand/tactile_stream/matrix
dexterous_hand/tactile_stream/timestamp_ns
dexterous_hand/tactile_stream/sequence
```

D405 主表只保存 `dexterous_hand/tactile/index`、`age_ms` 和 `valid`。
训练时也可以依据统一时间戳重新执行 `latest-before / max-age`。

## 与旧格式的差异

旧 `project1yu/TeleOperation/teleop_collect_hdf5.py` 和 `docs/TransHandR_多源数据采集需求设计.md` 不符合 quality-first V1：它们使用人工固定时间轴、缺少 source/unified/receive 三类时间戳、未区分真实 state 与最终 action，并保存了 V1 明确排除的 raw matrices/Euler/full SDK state。新采集器以每个 D405 RGB-D frameset 的 source time 产生一行，并采用 right-hand nearest、state linear interpolation、tracker position lerp + quaternion SLERP、action latest-before。
