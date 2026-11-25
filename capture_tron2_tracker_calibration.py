#!/usr/bin/env python3
"""只读采集左腕 Tracker/TRON2 标定位姿；Enter 采一组，status 查看，undo 排除末组，q 退出。"""

from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import threading
import time
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parent
EVENT_PREFIX = "WRIST_CAPTURE "
STATUS_NAMES = {0: "NO_IMAGES_YET", 1: "NOT_INITIALIZED", 2: "OK", 3: "LOST",
                4: "RECENTLY_LOST", 5: "SYSTEM_NOT_READY"}
JOINT_FK_SOURCE = "response_get_joint_state+official_model_fk"


class MeasuredWristFK:
    """从实测编码器角度计算 SDK TCP，不使用拖动模式中冻结的末端接口缓存。"""

    def __init__(self):
        from tron2_left_ik import LeftArmIK, JOINT_NAMES, SDK_TCP_OFFSET
        self.model = LeftArmIK()
        self.metadata = {
            "model_path": str(self.model.model_path),
            "model_sha256": hashlib.sha256(self.model.model_path.read_bytes()).hexdigest(),
            "base_frame": "base_Link", "wrist_frame": "wrist_roll_L_Link",
            "tcp_offset_m": SDK_TCP_OFFSET.tolist(), "joint_names": JOINT_NAMES,
            "note": "实测关节角的模型正解；不是直接测量的笛卡尔真值，需独立验证",
        }

    def pose(self, q):
        return self.model.pose(q)


def number(value, minimum, maximum, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} 必须是数值")
    value = float(value)
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"{name} 必须在 {minimum:g} 至 {maximum:g} 之间")
    return value


def acquire_capture_lock():
    """与现有恢复/回放、平台设备启动共用进程锁，避免采样期间发生控制切换。"""
    handle = (ROOT / ".tron2_replay.lock").open("a+")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise RuntimeError("已有恢复、回放或腕部采集任务，请先结束该任务")
    return handle


def pose_values(position, quaternion):
    p, q = np.asarray(position, float), np.asarray(quaternion, float)
    if p.shape != (3,) or q.shape != (4,) or not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError("位姿不是有限的 XYZ / WXYZ")
    if not .9 <= np.linalg.norm(q) <= 1.1:
        raise ValueError("四元数模长异常")
    # 原始浮点数完整保存；只在统计时归一化，不修正数据或猜测坐标轴。
    return {"position_m": p.tolist(), "quaternion_wxyz": q.tolist()}


def pose_spread(frames):
    if not frames:
        return None
    p = np.asarray([f["position_m"] for f in frames])
    q = np.asarray([f["quaternion_wxyz"] for f in frames])
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    _, vectors = np.linalg.eigh(q.T @ q)
    mean_q = vectors[:, -1]
    if mean_q @ q[0] < 0:
        mean_q = -mean_q
    return {"mean_position_m": p.mean(axis=0).tolist(),
            "mean_quaternion_wxyz": mean_q.tolist(),
            "position_radius_mm": float(np.linalg.norm(p - p.mean(axis=0), axis=1).max() * 1000),
            "rotation_radius_deg": float(np.degrees(2 * np.arccos(np.clip(abs(q @ mean_q), 0, 1))).max())}


def summarize_sample(trackers, robots):
    """仅按主机接收时间配对；保留请求耗时，绝不声称硬件同步或标定已经有效。"""
    pairs = []
    for ri, robot in enumerate(robots):
        if trackers:
            ti = min(range(len(trackers)), key=lambda i: abs(trackers[i]["received_monotonic_ns"] - robot["received_monotonic_ns"]))
            gap_ms = abs(trackers[ti]["received_monotonic_ns"] - robot["received_monotonic_ns"]) / 1e6
            if gap_ms <= 100:
                pairs.append({"robot_index": ri, "tracker_index": ti, "receive_gap_ms": gap_ms})
    reasons = []
    statuses = sorted({t["tracking_status"] for t in trackers})
    if not trackers:
        reasons.append("采样窗口内没有 Tracker 新帧")
    if not robots:
        reasons.append("采样窗口内没有机器人新回复")
    if any(s != 2 for s in statuses):
        reasons.append("含非 OK 跟踪状态：" + ", ".join(map(str, statuses)) + "（原样保存，未确认有效）")
    if len({t["mac"] for t in trackers}) > 1:
        reasons.append("Tracker 身份在本组内变化")
    if len(pairs) < 3:
        reasons.append("时间接近的位姿对不足 3 组")
    if any(r["request_duration_ms"] > 250 for r in robots):
        reasons.append("机器人查询耗时超过 250 ms，需检查时间对应关系")
    ts, rs = pose_spread(trackers), pose_spread(robots)
    for name, stats in (("Tracker", ts), ("机器人", rs)):
        if stats and (stats["position_radius_mm"] > 3 or stats["rotation_radius_deg"] > 1):
            reasons.append(f"{name} 在采样窗口内未保持稳定")
    return {"pairs": pairs, "tracking_statuses": statuses, "tracker_frames": len(trackers),
            "robot_frames": len(robots), "pair_count": len(pairs),
            "tracker_statistics": ts, "robot_statistics": rs, "reasons": reasons,
            "eligible_for_calibration": not reasons, "calibration_validated": False}


class HardwareSource:
    """只连接读取接口；白名单从发送入口阻止任何运动请求。"""

    def __init__(self, robot_ip, tracker_index, kinematics=None, passive_hid=None, tracker_mac=None):
        from replay_tron2_tracker import Tron2JsonClient

        class ReadOnlyClient(Tron2JsonClient):
            def send(self, title, data, **kwargs):
                if title != "request_get_joint_state" or data:
                    raise RuntimeError("腕部标定采集只允许查询实测关节角")
                return super().send(title, data, **kwargs)

        self.kinematics = kinematics if kinematics is not None else MeasuredWristFK()
        self.pose_metadata = {"schema": "tracker-tron2-pairs-v2", "robot_pose_source": JOINT_FK_SOURCE,
                              "robot_kinematics": self.kinematics.metadata}
        self.last_robot_timestamp = None
        if passive_hid:
            # 独立标定采用直接请求/回复连接，避免原生 SDK 的退出析构崩溃。
            from tools.tron2_head_pose import HeadClient
            class ReadOnlySocketClient(HeadClient):
                def request(self, title, data, reply_title, timeout=4.):
                    if title != "request_get_joint_state" or data:
                        raise RuntimeError("腕部标定采集只允许查询实测关节角")
                    return super().request(title, data, reply_title)
            self.client = ReadOnlySocketClient(robot_ip)
        else:
            self.client = ReadOnlyClient(robot_ip)
            self.client.wait_for_identity()
        self.robot_id = self.client.accid
        self.tracker_index = tracker_index
        self.passive_hid, self.tracker_mac = passive_hid, tracker_mac
        if passive_hid:
            if not tracker_mac:
                raise ValueError("只读旁路采样必须指定 Tracker MAC，不能猜测设备编号")
            self.pose_metadata['tracker_transport'] = {'kind':'passive_hidraw', 'path':passive_hid,
                                                       'mac':tracker_mac, 'device_writes':False}
        self.api = None

    def start(self, callback):
        if self.passive_hid:
            from tracker_passive_reader import PassiveTrackerReader
            self.api = PassiveTrackerReader(self.passive_hid, self.tracker_mac, self.tracker_index)
            self.api.start(callback)
            return
        sys.path.insert(0, str(ROOT / "project1yu/pyvut"))
        from pyvut import UltimateTrackerAPI
        self.api = UltimateTrackerAPI(mode="DONGLE_USB")

        def on_pose(pose):
            if pose.tracker_index != self.tracker_index:
                return
            received = time.monotonic_ns()
            try:
                frame = pose_values(pose.position, pose.rotation)
            except ValueError:
                return
            frame.update(received_monotonic_ns=received, mac=pose.mac,
                         tracker_index=pose.tracker_index, tracking_status=int(pose.tracking_status),
                         # 当前 pyvut 字段来自主机解析时钟，不是硬件采样时钟。
                         parser_host_timestamp_ms=pose.timestamp_ms)
            callback(frame)

        self.api.add_pose_callback(on_pose)
        self.api.start()

    def robot_pose(self):
        if getattr(self.api, 'error', ''):
            raise RuntimeError(self.api.error)
        start = time.monotonic_ns()
        response = self.client.request("request_get_joint_state", {}, "response_get_joint_state", timeout=4.)
        received = time.monotonic_ns()
        data = response.get("data", {})
        if data.get("result") != "success":
            raise RuntimeError("未收到成功的实测关节反馈")
        q = np.asarray(data.get("q", []), dtype=float)
        if q.shape != (16,) or not np.isfinite(q).all():
            raise ValueError("实测关节反馈必须包含有限的左7、右7、头2共16维角度")
        stamp = number(data.get("timestamp"), 0, 1e16, "机器人关节时间戳")
        if self.last_robot_timestamp is not None and stamp <= self.last_robot_timestamp:
            raise RuntimeError("机器人关节时间戳未更新；不复用旧反馈")
        position, quaternion = self.kinematics.pose(q)
        self.last_robot_timestamp = stamp
        frame = pose_values(position, quaternion)
        frame.update(request_started_monotonic_ns=start, received_monotonic_ns=received,
                     request_duration_ms=(received - start) / 1e6,
                     robot_timestamp_ms=stamp, joint_position_rad=q.tolist(),
                     pose_source=JOINT_FK_SOURCE)
        return frame

    def close(self):
        if self.api is not None:
            self.api.stop()
        if self.passive_hid:
            self.client.socket.close()
        # SDK 连接由独立采集进程退出时释放；不发送退出模式/持位等命令。


class MockSource:
    robot_id = "MOCK_ONLY"

    def start(self, callback):
        self.callback = callback

    def robot_pose(self):
        now = time.monotonic_ns()
        self.callback({**pose_values([.1, .2, .3], [1, 0, 0, 0]),
                       "received_monotonic_ns": now, "mac": "MOCK", "tracker_index": 0,
                       "tracking_status": 20, "parser_host_timestamp_ms": int(time.time() * 1000)})
        return {**pose_values([.5, .2, -.2], [1, 0, 0, 0]), "received_monotonic_ns": now,
                "request_started_monotonic_ns": now, "request_duration_ms": 0., "robot_timestamp_ms": None}

    def close(self):
        pass


class CaptureSession:
    def __init__(self, source, output_dir, emit, metadata):
        self.source, self.emit = source, emit
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.trackers, self.robots = deque(maxlen=12000), deque(maxlen=1000)
        self.groups = []
        self.phase, self.error = "connecting", ""
        self.capture_error = ""
        self.remaining_s = 0.
        self.capture_thread = None
        self.poll_thread = None
        folder = Path(output_dir).expanduser().resolve() / (datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8])
        folder.mkdir(parents=True, exist_ok=False)
        self.path = folder / "samples.jsonl"
        self.file = self.path.open("x", encoding="utf-8")
        self.write({"event": "session", "schema": "tracker-tron2-pairs-v1", "arm": "left",
                    "created_at": datetime.now().astimezone().isoformat(), "robot_id": source.robot_id,
                    "position_unit": "m", "quaternion_order": "wxyz", "robot_pose_source": "response_get_servo_pose",
                    "transform_convention": "T_WT = T_WB @ T_BE @ T_ET",
                    "pairing": "host_receive_time_nearest; not hardware synchronized",
                    "calibration_validated": False, **metadata,
                    **getattr(source, "pose_metadata", {})})

    def write(self, value):
        self.file.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        self.file.flush()
        os.fsync(self.file.fileno())

    def on_tracker(self, frame):
        with self.lock:
            self.trackers.append(frame)

    def start(self):
        self.source.start(self.on_tracker)
        self.phase = "ready"
        self.poll_thread = threading.Thread(target=self.poll, daemon=True)
        self.poll_thread.start()

    def poll(self):
        while not self.stop_event.is_set():
            try:
                frame = self.source.robot_pose()
                with self.lock:
                    self.robots.append(frame)
                    self.error = ""
            except Exception as exc:
                with self.lock:
                    self.error = str(exc)
            self.stop_event.wait(.1)

    def status(self):
        with self.lock:
            now = time.monotonic_ns()
            def latest(frames):
                return {**frames[-1], "age_ms": (now - frames[-1]["received_monotonic_ns"]) / 1e6} if frames else None
            return {"phase": self.phase, "error": self.capture_error or self.error, "file": str(self.path),
                    "robot_id": self.source.robot_id, "remaining_s": self.remaining_s,
                    "tracker": latest(self.trackers), "robot": latest(self.robots), "groups": list(self.groups)}

    def sample(self, countdown=3., duration=2., label=""):
        countdown = number(countdown, 0, 30, "倒计时")
        duration = number(duration, .5, 10, "采样秒数")
        if not isinstance(label, str) or len(label) > 120:
            raise ValueError("样本备注最长 120 个字符")
        with self.lock:
            if self.phase != "ready":
                raise RuntimeError("当前正在采样或未连接，请等待")
            self.capture_error = ""
            self.phase = "countdown"
            self.capture_thread = threading.Thread(target=self._capture, args=(countdown, duration, label), daemon=True)
            self.capture_thread.start()

    def _wait(self, seconds, phase):
        deadline = time.monotonic() + seconds
        with self.lock:
            self.phase = phase
        while not self.stop_event.is_set():
            self.remaining_s = max(0., deadline - time.monotonic())
            if self.remaining_s <= 0:
                return True
            self.stop_event.wait(min(.1, self.remaining_s))
        return False

    def _capture(self, countdown, duration, label):
        try:
            if not self._wait(countdown, "countdown"):
                return
            start = time.monotonic_ns()
            if not self._wait(duration, "sampling"):
                return
            end = time.monotonic_ns()
            with self.lock:
                trackers = [f for f in self.trackers if start <= f["received_monotonic_ns"] <= end]
                robots = [f for f in self.robots if start <= f["request_started_monotonic_ns"] and f["received_monotonic_ns"] <= end]
                summary = summarize_sample(trackers, robots)
                group = {"id": len(self.groups) + 1, "label": label, "excluded": False,
                         **{k: v for k, v in summary.items() if k != "pairs"}}
                self.write({"event": "sample", **group, "started_monotonic_ns": start, "ended_monotonic_ns": end,
                            "tracker": trackers, "robot": robots, "pairs": summary["pairs"]})
                self.groups.append(group)
            self.emit({"event": "sample_saved", **group, "file": str(self.path)})
        except Exception as exc:
            with self.lock:
                self.capture_error = str(exc)
            self.emit({"event": "error", "error": str(exc)})
        finally:
            with self.lock:
                self.phase = "ready" if not self.stop_event.is_set() else "stopped"

    def exclude(self, group_id):
        with self.lock:
            if self.phase != "ready":
                raise RuntimeError("请等待当前采样结束")
            group = next((g for g in self.groups if g["id"] == group_id), None)
            if group is None:
                raise ValueError("样本编号不存在")
            self.write({"event": "exclude", "id": group_id})
            group["excluded"] = True

    def close(self):
        self.stop_event.set()
        if self.capture_thread:
            self.capture_thread.join(timeout=2)
        if self.poll_thread:
            self.poll_thread.join(timeout=5)
        self.source.close()
        with self.lock:
            self.phase = "stopped"
            self.write({"event": "session_end", "groups": len(self.groups)})
            self.file.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-ip", default="10.192.1.2")
    parser.add_argument("--tracker-index", type=int, choices=range(5), default=0)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "tracker_calibrations")
    parser.add_argument("--map-label", default="", help="地图/建图批次备注；不是设备自动返回的地图 ID")
    parser.add_argument("--mount-label", default="", help="Tracker 安装方式备注")
    parser.add_argument("--countdown", type=float, default=3.)
    parser.add_argument("--duration", type=float, default=2.)
    parser.add_argument("--mock", action="store_true", help="仅模拟设备，不连接硬件")
    parser.add_argument("--json-control", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--lock-fd", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    number(args.countdown, 0, 30, "倒计时")
    number(args.duration, .5, 10, "采样秒数")
    output_lock = threading.Lock()

    def emit(event):
        with output_lock:
            if args.json_control:
                print(EVENT_PREFIX + json.dumps(event, ensure_ascii=False, allow_nan=False), flush=True)
            elif event["event"] != "status":
                print(json.dumps(event, ensure_ascii=False), flush=True)

    handle, session = None, None
    try:
        if not args.mock:
            handle = os.fdopen(args.lock_fd, "a+") if args.lock_fd is not None else acquire_capture_lock()
        source = MockSource() if args.mock else HardwareSource(args.robot_ip, args.tracker_index)
        session = CaptureSession(source, args.output_dir, emit, {
            "mock": args.mock, "robot_ip": args.robot_ip, "tracker_index": args.tracker_index,
            "map_label": args.map_label, "mount_label": args.mount_label})
        session.start()
        emit({"event": "status", **session.status()})

        def report():
            while not session.stop_event.wait(.5):
                emit({"event": "status", **session.status()})

        if args.json_control:
            threading.Thread(target=report, daemon=True).start()
        else:
            print(f"只读采集已连接，文件：{session.path}\nEnter：采一组；status：查看；undo：排除末组；q：结束。status=20 原样保存并标记待确认。", flush=True)
        signal.signal(signal.SIGTERM, lambda *_: session.stop_event.set())
        # 输入线程避免浏览器断开连接时卡在 stdin；SIGTERM 仍可退出并落盘。
        import queue
        commands = queue.Queue()
        def read_input():
            for line in sys.stdin:
                commands.put(line)
            commands.put(None)
        threading.Thread(target=read_input, daemon=True).start()
        while not session.stop_event.is_set():
            try:
                line = commands.get(timeout=.2)
            except queue.Empty:
                continue
            if line is None:
                break
            try:
                if args.json_control:
                    command = json.loads(line)
                else:
                    word = line.strip()
                    command = {"action": {"": "sample", "q": "stop", "quit": "stop", "undo": "exclude"}.get(word, word)}
                    if word == "undo":
                        command["id"] = len(session.groups)
                action = command.get("action")
                if action == "stop":
                    break
                if action == "sample":
                    session.sample(command.get("countdown", args.countdown), command.get("duration", args.duration), command.get("label", ""))
                elif action == "exclude":
                    session.exclude(command.get("id"))
                elif action == "status":
                    if not args.json_control:
                        print(json.dumps(session.status(), ensure_ascii=False), flush=True)
                else:
                    raise ValueError("支持 sample/status/exclude/stop")
                emit({"event": "status", **session.status()})
            except (ValueError, TypeError, RuntimeError) as exc:
                emit({"event": "error", "error": str(exc)})
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        emit({"event": "fatal", "error": str(exc)})
        return 1
    finally:
        if session is not None:
            session.close()
            emit({"event": "status", **session.status()})
        if handle is not None:
            handle.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
