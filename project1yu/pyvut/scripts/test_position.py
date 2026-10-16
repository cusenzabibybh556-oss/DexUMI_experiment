#!/usr/bin/env python3
"""持续打印 VIVE Ultimate Tracker 的 XYZ 位置 (单位: 米) — 自动等待 SLAM 初始化.

用法:
    sudo /home/zmy/anaconda3/envs/versionprodemo/bin/python scripts/test_position.py

在 Tracker 开机 + 配对后运行. 脚本会自动等待 SLAM 地图建立,
期间请缓慢移动 Tracker 帮助建图. 收到有效位姿后自动开始打印.

按 Ctrl+C 停止.
"""

import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pyvut import ViveTrackerGroup, set_tracker_core_verbose

STATUS_MAP = {
    -1: "NOT_READY", 0: "NO_IMG", 1: "NOT_INIT",
    2: "OK", 3: "LOST", 4: "FROZEN",
}


def main():
    # 关闭底层库的调试输出, 只保留我们自己的状态信息
    set_tracker_core_verbose(False)

    print("正在初始化 Dongle 连接...")
    group = ViveTrackerGroup(mode="DONGLE_USB", debug=False)
    print("初始化完成.\n")

    # ---- 第一阶段: 等待 Tracker 出现 ----
    print("[阶段1] 等待 Tracker 配对...")
    while True:
        group.do_loop()
        if any(n > 0 for n in group.poses_recvd):
            break
        time.sleep(0.05)

    # ---- 第二阶段: 等待 SLAM 地图建立 (至少一个 slot 达到 OK 状态) ----
    # 只有 tracking_status == 2 (OK) 才算真正追踪成功.
    # 注意: 每次 Tracker 断电/重连都需要重新建图 (SLAM 视觉记忆不保存).
    print("[阶段2] 等待 SLAM 地图建立...")
    print("  提示: 把 Tracker 对着有纹理的地面缓慢移动, 确保光线充足")
    last_report = time.time()
    phase2_start = time.time()

    while True:
        group.do_loop()
        now = time.time()

        # 检查是否有 slot 进入 OK 状态 (且收到足够多的包, 排除残影)
        for i in range(5):
            if group.pose_tracking_status[i] == 2 and group.poses_recvd[i] > 10:
                print(f"\n✅ SLAM 初始化完成! Slot {i} 状态=OK\n")
                break
        else:
            # 没找到 OK slot, 继续等
            if now - last_report > 2:
                last_report = now
                elapsed = int(now - phase2_start)

                # 只显示"活跃"的 slot: 收到足够包 且 不是 FROZEN 残影
                active = [(i, n, group.pose_tracking_status[i], group.tracker_map_state[i])
                          for i, n in enumerate(group.poses_recvd)
                          if n > 5 and group.pose_tracking_status[i] != 4]  # 排除 FROZEN

                if active:
                    parts = []
                    for i, n, st, mp in active:
                        st_str = STATUS_MAP.get(st, str(st))
                        mp_str = {0: "?", 2: "NOEXIST", 9: "WAIT_STATIC",
                                  10: "CREATING", 3: "REBUILT", 1: "EXIST"}.get(mp, str(mp))
                        parts.append(f"s{i}={st_str}[m={mp_str}]")
                    print(f"  [{elapsed}s] {' | '.join(parts)}")
                else:
                    # 没有任何活跃 slot, 可能是 dongle 没插或 tracker 没开机
                    stale = [(i, n, group.pose_tracking_status[i])
                             for i, n in enumerate(group.poses_recvd) if n > 0]
                    if stale:
                        print(f"  [{elapsed}s] 全是 FROZEN 残影 (slot={[i for i,_,_ in stale]}), "
                              "拔插 Dongle 后重试")
                    else:
                        print(f"  [{elapsed}s] 无数据, 请检查 Dongle 和 Tracker 是否通电")

                # 超时警告: 超过 60 秒还没建图成功
                if elapsed > 60:
                    print("  ⚠ 超时 60s, 请确认 Tracker 摄像头朝下对着有纹理的地面, 且光线充足")

            time.sleep(0.05)
            continue
        break

    # ---- 第三阶段: 持续打印位姿 ----
    # 打印所有 OK 或 LOST 的 slot (LOST 也有位置数据, 只是精度可能下降).
    # 只有 FROZEN/NO_IMG/NOT_INIT 才跳过.

    last_print = 0

    try:
        while True:
            group.do_loop()
            now = time.time()
            if now - last_print < 0.05:  # ~20Hz
                continue
            last_print = now

            # OK=2 和 LOST=3 都有有效位姿数据; 排除 FROZEN=4 残影
            active_slots = [i for i in range(5)
                            if group.pose_tracking_status[i] in (2, 3)
                            and group.poses_recvd[i] > 10]

            if not active_slots:
                # 没有活跃 slot，显示诊断信息 (过滤掉 FROZEN 残影)
                any_data = [(i, group.pose_tracking_status[i])
                            for i, n in enumerate(group.poses_recvd) if n > 0
                            and group.pose_tracking_status[i] != 4]
                if any_data:
                    states = [f"s{i}={STATUS_MAP.get(st, '?')}" for i, st in any_data]
                    print(f"\r{time.strftime('%H:%M:%S')} 等待追踪... "
                          f"({', '.join(states)})  ", end="")
                continue

            # 打印每个活跃 Tracker 的位置
            parts = []
            for slot in active_slots:
                pose = group.get_pose(idx=slot)
                p = pose["position"]
                st = STATUS_MAP.get(group.pose_tracking_status[slot], "?")
                parts.append(
                    f"T{slot}[{st}]: X={p[0]:7.4f} Y={p[1]:7.4f} Z={p[2]:7.4f}"
                )
            print(f"\r{time.strftime('%H:%M:%S')} | {' | '.join(parts)}  ", end="")

    except KeyboardInterrupt:
        print(f"\n已停止. 各 slot 收到位姿包: {group.poses_recvd}")


if __name__ == "__main__":
    main()
