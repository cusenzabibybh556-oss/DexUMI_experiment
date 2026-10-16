#!/bin/bash
# setup_act_env.sh —— 一键创建 act-main 所需的 aloha conda 环境
# 用法:
#   bash setup_act_env.sh
# 等价于 act-main/README.md 的官方安装步骤（用 conda_env.yaml 统一管理依赖版本）。

set -e

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACT_DIR="$HERE/../act-main"

if ! command -v conda >/dev/null 2>&1; then
    echo "未找到 conda，请先安装 Anaconda/Miniconda。" >&2
    exit 1
fi

echo "==> 创建 conda 环境 aloha（来自 $ACT_DIR/conda_env.yaml）"
conda env create -f "$ACT_DIR/conda_env.yaml"

echo "==> 安装 detr（ACT 模型定义，editable 模式）"
# 需要先激活环境。用 -n 指定环境名执行 pip，避免污染 base。
conda run -n aloha pip install -e "$ACT_DIR/detr"

echo "==> 校验关键依赖"
conda run -n aloha python - <<'PY'
import importlib
for m in ['torch', 'torchvision', 'dm_control', 'mujoco', 'h5py', 'pyquaternion', 'einops', 'cv2', 'matplotlib', 'numpy']:
    try:
        mod = importlib.import_module(m)
        print(f"{m:12s} OK {getattr(mod, '__version__', '')}")
    except Exception as e:
        print(f"{m:12s} FAIL {type(e).__name__}: {str(e)[:100]}")
import torch
print("CUDA available:", torch.cuda.is_available())
PY

echo "==> 完成。激活环境后即可运行："
echo "    conda activate aloha"
echo "    python trush-bin/run_act.py demo --task sim_transfer_cube_scripted --save_video"
