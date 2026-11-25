#!/usr/bin/env python3
"""读取或单次调整 TRON2 头部；角度参数单位为度，默认只读/预检。

依赖 dexumi 环境的 websocket-client。仅发送 get_joint_state 和显式指定的 MoveH。
MoveH 的 joint 顺序为 [pitch, yaw]，未指定的头部轴保持读到的当前角度。
"""

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import sys
import time
import uuid
from zoneinfo import ZoneInfo

import websocket


class HeadClient:
    def __init__(self, robot_ip):
        self.socket = websocket.create_connection(
            f"ws://{robot_ip}:5000", timeout=4, http_no_proxy=[robot_ip])
        self.accid = self.receive("notify_robot_info")["accid"]

    def receive(self, title, guid=None, timeout=5):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.socket.settimeout(max(0.1, deadline - time.monotonic()))
            message = json.loads(self.socket.recv())
            if message.get("title") == title and (
                    guid is None or message.get("guid") == guid):
                return message
        raise TimeoutError(f"未收到 {title}")

    def request(self, title, data, reply_title):
        guid = uuid.uuid4().hex
        self.socket.send(json.dumps({"accid": self.accid, "title": title,
                                    "guid": guid, "timestamp": int(time.time() * 1000),
                                    "data": data}))
        reply = self.receive(reply_title, guid)
        if reply.get("data", {}).get("result") != "success":
            raise RuntimeError(f"机器人未接受请求：{reply}")
        return reply

    def joints(self):
        data = self.request("request_get_joint_state", {}, "response_get_joint_state")["data"]
        q = data.get("q", [])
        if len(q) < 16 or not all(math.isfinite(v) for v in q[:16]):
            raise RuntimeError("机器人关节反馈缺少有效头部角度")
        return q


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--robot-ip", default="10.192.1.2")
    p.add_argument("--pitch-deg", type=float)
    p.add_argument("--yaw-deg", type=float)
    p.add_argument("--duration", type=float, default=5)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    if not math.isfinite(args.duration) or args.duration <= 0:
        p.error("--duration 必须是有限正数")
    if args.execute and args.pitch_deg is None and args.yaw_deg is None:
        p.error("--execute 至少需要一个目标角度")
    if args.output is not None and args.output.exists():
        p.error(f"记录文件已存在，请换一个新路径：{args.output}")
    client = None
    report = {"captured_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
              "robot_ip": args.robot_ip, "execute": args.execute}
    try:
        client = HeadClient(args.robot_ip)
        before = client.joints()
        report.update(accid=client.accid, before_head_rad=before[14:16],
                      before_arm_rad=before[:14])
        print(f"机器人：{client.accid}；当前俯仰/偏航（度）："
              f"{[round(math.degrees(v), 3) for v in before[14:16]]}")
        if args.pitch_deg is None and args.yaw_deg is None:
            return
        target = [before[14] if args.pitch_deg is None else math.radians(args.pitch_deg),
                  before[15] if args.yaw_deg is None else math.radians(args.yaw_deg)]
        for name, q, lo, hi in zip(["pitch", "yaw"], target, [-0.78, -1.57], [1.04, 1.57]):
            if not math.isfinite(q) or not lo <= q <= hi:
                raise ValueError(f"{name} 超出官方关节范围 [{lo}, {hi}] rad")
        payload = {"time": args.duration, "joint": target}
        report["moveh_payload"] = payload
        print("MoveH 目标（度）：", [round(math.degrees(v), 3) for v in target])
        if not args.execute:
            print("只读预检完成，未发送 MoveH。")
            return
        print(f"开始头部 MoveH，计划 {args.duration:g} 秒。", flush=True)
        report["moveh_reply"] = client.request("request_moveh", payload, "response_moveh")
        deadline = time.monotonic() + args.duration + 5
        earliest_completion = time.monotonic() + args.duration
        while time.monotonic() < deadline:
            after = client.joints()
            report["after_head_rad"] = after[14:16]
            report["after_arm_rad"] = after[:14]
            if (time.monotonic() >= earliest_completion and
                    max(abs(a - b) for a, b in zip(after[14:16], target)) < 0.015):
                report["result"] = "completed"
                print("实测俯仰/偏航（度）：", [round(math.degrees(v), 3) for v in after[14:16]])
                return
            time.sleep(0.2)
        raise TimeoutError("头部目标尚未由反馈确认；没有重发 MoveH")
    except Exception as exc:
        report.update(result="error", error=str(exc))
        raise
    finally:
        if client is not None:
            client.socket.close()
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            # 单次运行的独立记录；不覆盖之前已有的记录文件。
            with args.output.open("x", encoding="utf-8") as stream:
                json.dump(report, stream, ensure_ascii=False, indent=2)
                stream.write("\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"错误：{exc}", file=sys.stderr)
        sys.exit(1)
