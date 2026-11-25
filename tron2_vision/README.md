# Top 前景数据处理链路

链路：原始HDF5 → SAM3初始掩码 → 关键帧修正标签 → 轻量分割器训练 →
固定分割器逐帧自动处理 → 无损前景缓存 → ACT prepare与训练读取。
人工标签只用于分割器训练和检查，ACT读取分割器实际预测出来的RGB。

环境使用本机 `conda activate dexumi`，Python3.10、torch2.5.1、torchvision0.20.1、
Pillow、numpy、h5py、opencv。运行轻量模型不依赖SAM3虚拟环境。
ResNet18初始化用ImageNet权重，本机已有缓存；其他机器首次训练可能下载。
现有SAM3种子导出脚本仍使用 `/home/zmy/Projects/sam3/.venv/bin/python`。

## 导出关键帧

```bash
cd /home/zmy/Projects/Dexumi
conda activate dexumi
python -m tron2_vision export \
  --cache-root artifacts/top_foreground_tracking_20261116/trial_fused \
  --episodes 66 73 91 --samples 5 \
  --frames 66:133 66:166 73:140 73:157 91:125 \
  --output artifacts/top_foreground_new/annotations
```

每个输出目录必须不存在，避免覆盖已有标注、权重或数据。
保存原始分辨率PNG、初始标签、可编辑标签，以及原始文件指纹、行号、相机帧号、
纳秒时间戳。JSON中的纳秒时间戳以字符串保存，避免浏览器浮点精度丢失。
标签为单通道uint8：0背景、1机械左手O6、2任务物体（两个杯子取同类）。
只标注当前可见的表面，不补被支架或手遮住的物体。

## 修正和保存

```bash
python -m tron2_vision serve --project artifacts/top_foreground_new/annotations --port 8091
```

浏览器打开 http://127.0.0.1:8091 ，服务仅监听本机。
在原图叠加区用画笔或多边形修选区，右侧/下方显示黑背景效果。
Shift拖动临时擦除；多边形逐点点击后按Enter填充、Esc取消；Ctrl+Z撤销。
原图缩放显示时，保存坐标仍是原始像素坐标。
“保存草稿”与“保存并标记已检查”分开。训练只读取已检查标签。
每次保存记录修正者和修订版本；旧页面覆盖新标签会被拒绝，需重新载入。
原始HDF5永远以只读方式打开。

项目文件中的reviewer用于区分人工和助手修正。当前随项目提供的20张种子标签
由助手按原图检查，其中5张补了可见拇指/指节；它们不是独立人工标注真值。
坐标修正记录见实验目录的 `seed_corrections.json`，仍可通过页面进一步细修。

## 训练分割器

```bash
python -m tron2_vision train \
  --project artifacts/top_foreground_new/annotations \
  --validation-episodes 91 --steps 300 \
  --output artifacts/top_foreground_new/segmenter
```

默认使用ResNet18骨干＋32通道特征金字塔，约1127万参数，三类输出。
RGB内部输入240×320，ImageNet归一化；输出logits双线性恢复到原始图像尺寸后argmax。
训练使用加权交叉熵＋前景Dice；骨干学习率3e-5、其他参数1e-3，batch4，AdamW。
小样本训练固定骨干BN统计，只有轻微亮度增广，图像和标签几何关系不变。
按完整episode隔离验证帧，不以验证损失选择权重；固定训练步数后评估一次。
权重记录输入处理、类别、标注清单哈希、修正者和训练/验证帧ID。
报告IoU基于当前修正标签的质量，不能当作真实部署成功率。

## 自动处理和ACT读取

```bash
python -m tron2_vision process \
  --checkpoint artifacts/top_foreground_new/segmenter/foreground.pt \
  --episodes 66 73 91 \
  --output artifacts/top_foreground_new/foreground

python -m tron2_act prepare --episodes 66 73 91 \
  --foreground-root artifacts/top_foreground_new/foreground \
  --allow-experimental-foreground \
  --output artifacts/top_foreground_new/act_prepared
```

自动处理每次只输入当前RGB，无人工提示、未来帧或双向传播。
保留区域复制原图像素，背景置0；不生成RGB、不裁剪物体，也不修改腕、O6数据或时间戳。
缓存保存全部原始行索引、有效位、帧号、时间戳、模型哈希和预处理元数据。
MP4仅供预览，训练读取无损HDF5。

新模型输出默认标记为experimental。ACT默认不接受实验缓存；显式开启上述选项才可做
离线链路试验，manifest会记录这个选择。该选项仍拒绝rejected缓存、来源不符、
帧号/时间戳错位或prepare后被修改的文件，不会回退到原始RGB。
当前动作约定继续为 `tron2-act-spatial-command-v2`：O6输入是真机实测反馈，
动作标签是手套重定向后发送的六通道命令，腕历史和未来增量不受视觉处理影响。

未来实时输入复用同一接口：

```python
from tron2_vision.model import ForegroundPredictor
predictor = ForegroundPredictor("/absolute/path/foreground.pt", device="cuda")
processed = predictor.predict(rgb_uint8)  # 当前帧，RGB，H×W×3
act_rgb = processed["rgb"]
```

这里只提供相同权重和处理过程，不包含相机驱动、ACT调度或机器人控制。
实机相机视角下的泛化、采集延迟和闭环效果仍需另外验证。

## 验证

```bash
python -m unittest discover -s tests -p 'test_tron2*.py' -v
```

覆盖标注修订冲突、来源变化、标签类型、实验缓存的显式选择、帧号错位拒绝、
同一图像预测不依赖前一帧，以及ACT原有的时间、旋转、state/action和split检查。
