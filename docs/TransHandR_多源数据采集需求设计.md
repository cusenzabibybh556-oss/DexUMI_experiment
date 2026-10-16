# TransHandR 多源数据采集需求设计

## 1. 目标

本文件定义一个新的多源数据采集程序，用于在 TransHandR 右手识别与灵巧手重定向运行时，按统一 30Hz 时间轴采集并保存 episode 数据。

采集程序参考：

- `project1yu/TeleOperation/teleop_collect_hdf5.py`：episode 管理、HDF5 写入、RealSense 采集结构。
- `TransHandR0725/TransHandR/scripts/main_visionpro_multi_hand.py`：Vision Pro 右手数据、TransHandR 推理、灵巧手控制流程。
- `project1yu/pyvut`：VIVE Tracker 6DoF 数据读取接口。

本采集程序必须保存以下数据：

- TransHandR 识别到的右手数据。
- 当前灵巧手真实反馈角度/状态，而不是只保存重定向输出。
- RealSense D405 手腕视角 RGB-D 数据。
- 1 个 VIVE Tracker 的完整 6DoF 数据：`pos`、`quat`、欧拉角。
- 所有数据源的时间戳，并保存 30Hz 对齐后的同步误差。

## 2. 已确定需求

### 2.1 灵巧手

采集程序必须支持多种灵巧手：

- Inspire Hand。
- LinkerHand L10。
- LinkerHand L21。

核心保存对象是灵巧手当前真实反馈角度/状态。重定向模型输出和下发命令不是第一版保存数据。

必须保存：

- `dexterous_hand/current_angles`：当前灵巧手实际角度或实际位置反馈。
- `dexterous_hand/current_state`：SDK 能读取到的完整状态。
- `dexterous_hand/state_valid`：该帧反馈是否读取成功。
- `dexterous_hand/state_timestamp_ns`：读取反馈时的本机时间戳。
- `dexterous_hand/hand_type`：写入 HDF5 attrs，例如 `inspire`、`l10`、`l21`。
- `dexterous_hand/hand_side`：写入 HDF5 attrs，例如 `right`。

读取失败时不能中断整个 episode。该帧必须写入占位值，并将 `state_valid=False`。

### 2.2 TransHandR 右手数据

数据来源为 Vision Pro/TransHandR 当前使用的右手关键点流。

程序必须只采集右手数据。不能在右手缺失时自动使用左手数据替代。

必须保存：

- `observations/right_hand/raw_matrices`：右手 25 个关键点原始 4x4 矩阵，shape `(T, 25, 4, 4)`。
- `observations/right_hand/world_keypoints`：转换后的右手 25 个三维关键点，shape `(T, 25, 3)`。
- `observations/right_hand/relative_keypoints`：以 wrist 为原点的右手相对关键点，shape `(T, 25, 3)`。
- `observations/right_hand/valid`：该帧右手数据是否有效。
- `observations/right_hand/system_timestamp_ns`：收到右手数据时的本机时间戳。

当前 TransHandR 坐标转换规则必须记录在 HDF5 attrs 中：

```python
x = -mat[1][3]
y =  mat[2][3]
z = -mat[0][3]
```

相对关键点计算方式：

```python
relative_keypoints = world_keypoints - world_keypoints[0]
```

### 2.3 RealSense D405 手腕视角数据

D405 相机安装在灵巧手手腕上，作为手腕视角相机。

必须保存：

- `observations/images/d405_rgb`：RGB 图像，shape `(T, H, W, 3)`，dtype `uint8`。
- `observations/depth/d405_depth`：深度图，shape `(T, H, W)`，dtype `uint16`。
- `camera/d405/frame_number`：RealSense frame number。
- `camera/d405/device_timestamp_ms`：RealSense 设备时间戳。
- `camera/d405/system_timestamp_ns`：本机收到图像时的时间戳。
- `camera/d405/valid`：该帧 RGB-D 是否有效。

HDF5 attrs 必须保存：

- D405 序列号。
- 图像宽高。
- 相机 FPS。
- depth scale。
- RGB 内参。
- depth 内参。

### 2.4 D405 外参

D405 是手腕视角，因此必须在数据设计中支持保存外参。外参用于描述 D405 相机坐标系相对于手腕/Tracker/机器人坐标系的位置关系。

第一版必须预留并保存以下外参字段；如果标定尚未完成，写入单位矩阵并标记 `calibrated=False`：

- `calibration/d405/T_wrist_camera`：手腕坐标系到 D405 相机坐标系的 4x4 齐次变换矩阵。
- `calibration/d405/T_tracker_camera`：VIVE Tracker 坐标系到 D405 相机坐标系的 4x4 齐次变换矩阵。
- `calibration/d405/calibrated`：外参是否已经完成真实标定。

外参包括：

- 平移：`x, y, z`，单位米。
- 旋转：3x3 旋转矩阵，或等价四元数。
- 4x4 齐次变换矩阵：同时包含平移和旋转。

### 2.5 VIVE Tracker 6DoF 数据

第一版只采集 1 个 VIVE Tracker。该 Tracker 安装在灵巧手手腕上。

由于当前 Tracker 尚未完全调通，采集程序必须允许 `enable_vive_tracker=False`。关闭时仍按 30Hz 写入占位数据，并将 `valid=False`。

必须保存：

- `vive_tracker/pos`：位置，shape `(T, 1, 3)`，单位米。
- `vive_tracker/quat`：四元数，shape `(T, 1, 4)`，顺序 `[w, x, y, z]`。
- `vive_tracker/euler_deg`：欧拉角，shape `(T, 1, 3)`，顺序 `[roll, pitch, yaw]`，单位度。
- `vive_tracker/tracking_status`：追踪状态，shape `(T, 1)`。
- `vive_tracker/valid`：Tracker 数据是否有效，shape `(T, 1)`。
- `vive_tracker/system_timestamp_ns`：本机收到或读取 Tracker 数据的时间戳，shape `(T, 1)`。

## 3. 采集频率与同步

### 3.1 主采集频率

主对齐频率固定为 30Hz。

每个 episode 最终保存的主时间轴为：

- `timestamps/monotonic_ns`：30Hz 对齐时间轴，dtype `int64`。
- `timestamps/unix_ns`：对应系统 Unix 时间，dtype `int64`。

### 3.2 数据源采样

各数据源可以高于 30Hz 运行，但写入 episode 的对齐帧固定为 30Hz。

- Vision Pro/TransHandR 右手数据：持续读取最新帧。
- 灵巧手反馈状态：持续读取最新状态。
- D405 RGB-D：按 30 FPS 读取。
- VIVE Tracker：调通后按 pyvut 实际输出频率读取。

### 3.3 对齐方法

每个采集源独立维护最近样本缓存。Recorder 进程按 30Hz 产生对齐帧，并为每个数据源选择离当前主时间戳最近的样本。

超过同步阈值时，该源本帧写入占位值，并将对应 `valid=False`。

同步阈值固定为：

- 右手数据：50 ms。
- 灵巧手反馈：50 ms。
- D405 RGB-D：50 ms。
- VIVE Tracker：50 ms。

必须保存同步误差：

- `sync/right_hand_dt_ms`
- `sync/hand_state_dt_ms`
- `sync/d405_dt_ms`
- `sync/vive_tracker_dt_ms`

## 4. HDF5 文件结构

每次录制保存一个 episode 文件：

```text
episode_000000.hdf5
episode_000001.hdf5
```

HDF5 schema：

```text
attrs:
  episode_idx
  created_at
  task_name
  capture_hz = 30
  hand_type
  hand_side
  avp_ip
  d405_serial
  tracker_count = 1
  enable_vive_tracker
  image_height
  image_width
  right_hand_coordinate_rule
  quat_order = "[w, x, y, z]"
  euler_order = "[roll, pitch, yaw]"

timestamps/
  monotonic_ns                    (T,) int64
  unix_ns                         (T,) int64

observations/
  right_hand/
    raw_matrices                  (T, 25, 4, 4) float32
    world_keypoints               (T, 25, 3) float32
    relative_keypoints            (T, 25, 3) float32
    valid                         (T,) bool
    system_timestamp_ns           (T,) int64

  images/
    d405_rgb                      (T, H, W, 3) uint8

  depth/
    d405_depth                    (T, H, W) uint16

dexterous_hand/
  current_angles                  (T, J_hand) float32
  current_state                   (T, J_state) float32
  state_valid                     (T,) bool
  state_timestamp_ns              (T,) int64

vive_tracker/
  pos                             (T, 1, 3) float32
  quat                            (T, 1, 4) float32
  euler_deg                       (T, 1, 3) float32
  tracking_status                 (T, 1) int32
  valid                           (T, 1) bool
  system_timestamp_ns             (T, 1) int64

camera/
  d405/
    frame_number                  (T,) int64
    device_timestamp_ms           (T,) float64
    system_timestamp_ns           (T,) int64
    valid                         (T,) bool

calibration/
  d405/
    T_wrist_camera                (4, 4) float32
    T_tracker_camera              (4, 4) float32
    calibrated                    bool

sync/
  right_hand_dt_ms                (T,) float32
  hand_state_dt_ms                (T,) float32
  d405_dt_ms                      (T,) float32
  vive_tracker_dt_ms              (T, 1) float32
```

第一版不保存以下字段：

- `retarget/model_angles`
- `retarget/control_values`
- `retarget/control_success`
- `retarget/inference_timestamp_ns`
- `retarget/command_timestamp_ns`
- 未对齐的 raw stream
- 多 Tracker 数据

## 5. 程序架构

程序采用多进程或多线程混合结构。

必须包含：

- `vision_process`：读取 Vision Pro 右手数据，生成右手 25 点数据。
- `transhandr_process`：维持 TransHandR 实时重定向运行，但第一版不保存模型输出。
- `hand_state_process`：读取当前灵巧手真实反馈角度/状态。
- `d405_process`：读取 RealSense D405 RGB-D。
- `tracker_process`：读取 1 个 VIVE Tracker 6DoF；未调通时可关闭。
- `recorder_process`：按 30Hz 对齐所有数据并写入 HDF5。

## 6. Episode 状态机

按键逻辑沿用 `TeleOperation`：

- `R`：开始录制一个新 episode。
- `S`：停止当前 episode 并保存。
- `Ctrl+C`：退出；如果正在录制，必须先保存当前 episode。

开始录制前必须检查：

- Vision Pro 右手流是否可读取。
- TransHandR 是否已正常运行。
- 灵巧手反馈读取接口是否可用。
- D405 是否能读取 RGB-D。
- 输出目录是否可写。

VIVE Tracker 当前未调通，因此不作为第一版开始录制的硬性阻塞项。

## 7. 配置项

必须支持配置：

```text
avp_ip
hand_type
hand_side
hand_serial_port
hand_can_interface
d405_serial
image_width
image_height
camera_fps = 30
capture_hz = 30
enable_vive_tracker
tracker_mode
tracker_index = 0
output_dir
task_name
sync_threshold_ms = 50
```

## 8. 数据质量检查

每个 episode 保存后必须输出质量摘要。

必须统计：

- 总帧数。
- 实际采集时长。
- 实际平均采集频率。
- 右手数据有效帧比例。
- 灵巧手状态有效帧比例。
- D405 有效帧比例。
- VIVE Tracker 有效帧比例。
- 各数据源最大同步误差。
- 各数据源平均同步误差。
- HDF5 文件大小。

必须保存：

```text
quality/right_hand_valid_ratio
quality/hand_state_valid_ratio
quality/d405_valid_ratio
quality/tracker_valid_ratio
quality/max_sync_error_ms
quality/mean_sync_error_ms
```

## 9. HDF5 查看与验证

第一版采集程序完成后，必须配套一个本地验证脚本，用于检查 HDF5 文件内容。

验证脚本必须支持：

- 打印 HDF5 文件层级和 attrs。
- 显示 D405 RGB 第一帧、最后一帧和任意指定帧。
- 显示 D405 depth 伪彩色图。
- 绘制灵巧手当前角度曲线。
- 绘制 VIVE Tracker `pos` 和 `euler_deg` 曲线。
- 绘制同步误差曲线。

可参考 `project1yu/act-main/visualize_episodes.py` 的视频和曲线导出方式，但需要按本项目 HDF5 schema 重新实现字段读取。

## 10. 推荐文件命名

采集脚本：

```text
TransHandR0725/TransHandR/scripts/collect_transhandr_multisource.py
```

配置文件：

```text
TransHandR0725/TransHandR/config/multisource_collection.py
```

验证脚本：

```text
TransHandR0725/TransHandR/scripts/visualize_multisource_episode.py
```

数据目录：

```text
/mnt/1c714c0e-2ec6-4b54-b86f-8ab77eaea676/study/data/robot_data/<task_name>/<hand_type>/
```

