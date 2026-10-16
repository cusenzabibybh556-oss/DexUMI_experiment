#!/usr/bin/env bash
# 初始化：
set -euo pipefail

sudo modprobe peak_usb
sudo modprobe can
sudo modprobe can_raw
sudo modprobe can_dev

can_iface="${1:-${CAN_INTERFACE:-}}"
if [[ -z "$can_iface" ]]; then
    mapfile -t can_ifaces < <(find /sys/class/net -maxdepth 1 -name 'can*' -printf '%f\n' | sort)
    if [[ ${#can_ifaces[@]} -ne 1 ]]; then
        printf 'Expected one CAN interface; found %s (%s). Pass the interface name explicitly.\n' \
            "${#can_ifaces[@]}" "${can_ifaces[*]:-none}" >&2
        exit 1
    fi
    can_iface="${can_ifaces[0]}"
fi

if [[ ! -d "/sys/class/net/$can_iface" || "$can_iface" != can* ]]; then
    printf 'CAN interface not found: %s\n' "$can_iface" >&2
    exit 1
fi

sudo ip link set "$can_iface" down 2>/dev/null || true
sudo ip link set "$can_iface" type can bitrate 1000000
sudo ip link set "$can_iface" up

printf 'Configured CAN interface: %s\n' "$can_iface"
ip -br link show "$can_iface"
