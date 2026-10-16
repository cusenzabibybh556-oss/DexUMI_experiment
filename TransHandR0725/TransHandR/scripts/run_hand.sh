#!/bin/bash
# ============================================================
# Vision Pro → 灵巧手控制 一键启动
# 用法: bash run_hand.sh [手型]   默认全部
# 手型: l20, l21, inspire
# ============================================================

CONDA_ENV="yumi_pyg"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
HAND_TYPE="${*:-l21 l20 inspire}"

echo ""
echo "  ★ 即将启动手型:  ${HAND_TYPE}"
echo "  ★ 未连接的手型会自动跳过"
echo ""

# ---- 1. conda 环境 ----
echo "[1/4] 激活 conda: ${CONDA_ENV}"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}

# ---- 2. CAN 接口 (LinkerHand L20/L21 需要) ----
echo "[2/4] 启动 CAN 接口"
sudo bash "${SCRIPT_DIR}/../linkerhand_20_lite_test/start_can0.sh"

# ---- 3. 串口权限 (Inspire Hand 需要) ----
if [ -e /dev/ttyUSB0 ]; then
    echo "[3/4] 串口权限: /dev/ttyUSB0"
    sudo chmod 666 /dev/ttyUSB0
else
    echo "[3/4] 串口 /dev/ttyUSB0 不存在，跳过"
fi

# ---- 4. 启动 ----
echo "[4/4] 启动控制程序..."

cd "${SCRIPT_DIR}"
python main_visionpro_multi_hand.py --hand ${HAND_TYPE}
