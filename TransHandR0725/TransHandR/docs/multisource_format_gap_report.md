# 数据格式核对报告

核对基准：`TransHandR_multisource_data_spec_quality_first_v1.md`。

## 结论

当前仓库里的数据格式不符合 quality-first V1，不能直接作为该规范下的 ACT 多源 episode 使用。

实际检查 `dataset/data_collect/hand_data_0204.h5`，文件只有：

```text
frame_ids   (2400,)       int32
left_hand   (2400,25,3)   float32
right_hand  (2400,25,3)   float32
```

另一个已有的 `dataset/data/h5/glove_data_aligned.h5` 也只有 `frame_ids / l_glove_pos / r_glove_pos`。二者都缺少 D405、真实灵巧手反馈、最终 action、VIVE、统一时间戳、有效位和标定数据。

## 关键差异

| 项目 | 当前实现/旧设计 | quality-first V1 | 结果 |
|---|---|---|---|
| 主时间轴 | 人工固定 30 Hz 或普通 timestamp | D405 frameset 的统一采样时刻 | 不符合 |
| RGB-D | 旧 TeleOperation 有图像但不是完整 D405 元数据 | 同一 frameset 的 RGB + Depth | 不完整 |
| 右手 | 现有样本同时保存左右手；旧设计保存 raw matrices | 只保存右手 world/relative 25×3 | 不符合 |
| 时间戳 | 到达时间或无时间戳 | source / unified / host receive | 不符合 |
| Hand state | 旧代码 Inspire 会回退 command echo | 必须是真实设备反馈 | 不符合 |
| Action | 旧设计明确不保存 | 必须保存实际下发的 final command | 不符合 |
| State/action | 未严格区分 | 两者都保存，允许维度不同 | 不符合 |
| VIVE | 旧设计含 Euler | 只保留 pos + `[w,x,y,z]` quat | 不符合 |
| 对齐 | 全源 nearest、统一 50 ms | hand/tracker 插值，quat SLERP，action latest-before，独立阈值 | 不符合 |
| 无效值 | 缺少统一约定 | float NaN、int/timestamp -1、图像 0，并读 valid | 不符合 |
| 校验 | 无 V1 validator | 10 类结构与时序规则 | 不符合 |

## 新实现

`scripts/collect_transhandr_multisource.py` 生成 `schema_version=quality-first-v1` 的 HDF5：

- D405 每个有效 RGB-D frameset 产生一个 timestep；
- right hand 使用 nearest；
- hand state 使用线性插值；
- tracker position 使用线性插值，quaternion 使用 SLERP；
- action 使用 latest-before，拒绝 future action；
- state 是设备真实反馈，action 是设备发送成功后的最终 command；
- 各源保留独立 valid、时间戳和质量指标；
- Dashboard 与写盘独立，关闭网页不影响采集；
- 保存后的 episode 可由 `scripts/validate_multisource_episode.py` 独立校验。

注意：Vision Pro 当前 `avp_stream` 接口没有明确公开 source clock 语义。硬件模式会优先读取包内 timestamp；无法确认其为统一纳秒时，按规范退化到 `time.monotonic_ns()`。正式采集前应实测 Vision Pro 时钟并配置稳定的 affine clock mapping。
