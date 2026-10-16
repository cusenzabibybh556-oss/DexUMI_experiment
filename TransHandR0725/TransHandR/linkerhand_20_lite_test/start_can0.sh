#!/bin/bash
# 启动 LinkerHand CAN 接口

# can0
sudo ip link set can0 up type can bitrate 1000000
ip -br link show can0
sudo ip link set can0 txqueuelen 1000

# can1 (可能不存在，忽略错误)
sudo ip link set can1 up type can bitrate 1000000 2>/dev/null || echo "  can1 不存在，跳过"
ip link show can1 2>/dev/null || true
sudo ip link set can1 txqueuelen 1000 2>/dev/null || true
