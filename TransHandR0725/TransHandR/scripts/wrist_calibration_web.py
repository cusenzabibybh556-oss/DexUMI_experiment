"""腕部标定采集进程：网页与独立 CLI 使用同一份只读采集程序。"""

from __future__ import annotations

import ipaddress
import io
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import uuid
import zipfile

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "capture_tron2_tracker_calibration.py"
SOLVER = ROOT / "solve_tron2_tracker_calibration.py"
PREFIX = "WRIST_CAPTURE "


class WristCalibrationManager:
    def __init__(self, can_connect, output_dir, mock=False, python=None):
        self.can_connect, self.output_dir, self.mock = can_connect, Path(output_dir).resolve(), mock
        default_python = Path("/home/zmy/anaconda3/envs/dexumi/bin/python")
        self.python = python or os.environ.get("WRIST_CAPTURE_PYTHON") or os.environ.get("DEXUMI_REPLAY_PYTHON") or (
            str(default_python) if default_python.exists() else sys.executable)
        self.lock = threading.RLock()
        self.process = None
        self.solution_process = None
        self.solution = {"phase": "idle"}
        self.state = {"phase": "idle", "groups": [], "file": "", "error": ""}

    def status(self):
        with self.lock:
            return {**self.state, "enabled": True, "mock": self.mock,
                    "solution": dict(self.solution),
                    "active": self.process is not None and self.process.poll() is None}

    def start(self, payload):
        if set(payload) - {"robot_ip", "tracker_index", "map_label", "mount_label"}:
            raise ValueError("连接参数仅支持 robot_ip、tracker_index、map_label、mount_label")
        ip = str(ipaddress.ip_address(payload.get("robot_ip", "10.192.1.2")))
        index = payload.get("tracker_index", 0)
        if isinstance(index, bool) or not isinstance(index, int) or index not in range(5):
            raise ValueError("Tracker 编号须为 0 至 4 的整数")
        for key in ("map_label", "mount_label"):
            if not isinstance(payload.get(key, ""), str) or len(payload.get(key, "")) > 120:
                raise ValueError("地图/安装备注最长 120 个字符")
        with self.lock:
            if self.solution.get("phase") == "running":
                raise RuntimeError("正在求解当前会话，请等待完成再新建采集")
            if self.process is not None and self.process.poll() is None:
                raise RuntimeError("腕部采集已连接，请先结束当前会话")
            # 先取得恢复/回放共用锁，再检查平台状态；设备启动也使用这个锁。
            handle = None
            if not self.mock:
                import fcntl
                handle = (ROOT / ".tron2_replay.lock").open("a+")
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    handle.close()
                    raise RuntimeError("已有恢复、回放或腕部采集任务")
            try:
                self.can_connect()
                command = [self.python, "-u", str(SCRIPT), "--json-control", "--robot-ip", ip,
                           "--tracker-index", str(index), "--output-dir", str(self.output_dir),
                           "--map-label=" + payload.get("map_label", ""),
                           "--mount-label=" + payload.get("mount_label", "")]
                if self.mock:
                    command.append("--mock")
                if handle is not None:
                    command += ["--lock-fd", str(handle.fileno())]
                process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, text=True, bufsize=1,
                                           start_new_session=True,
                                           pass_fds=(handle.fileno(),) if handle else ())
                self.process = process
                self.state = {"phase": "connecting", "groups": [], "file": "", "error": ""}
                self.solution = {"phase": "idle"}
                threading.Thread(target=self._watch, args=(process,), daemon=True).start()
            finally:
                if handle is not None:
                    handle.close()  # 子进程持有同一把锁，直至设备连接关闭。
            return self.status()

    def command(self, action, payload):
        if action == "start":
            return self.start(payload)
        if action == "solve":
            return self.solve(payload)
        with self.lock:
            if action not in {"sample", "exclude", "stop"}:
                raise ValueError("未知腕部采集操作")
            allowed = {"sample": {"countdown", "duration", "label"}, "exclude": {"id"}, "stop": set()}[action]
            if set(payload) - allowed:
                raise ValueError("不支持的采集参数")
            if self.process is None or self.process.poll() is not None:
                raise RuntimeError("请先连接 Tracker 和 TRON2")
            if action == "sample":
                if self.state["phase"] != "ready":
                    raise RuntimeError("请等待连接或当前采样完成")
                for key, lower, upper, default in (("countdown", 0, 30, 3), ("duration", .5, 10, 2)):
                    value = payload.get(key, default)
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not lower <= value <= upper:
                        raise ValueError(f"{key} 必须在 {lower:g} 至 {upper:g} 之间")
                if not isinstance(payload.get("label", ""), str) or len(payload.get("label", "")) > 120:
                    raise ValueError("样本备注最长 120 个字符")
            if action == "exclude" and (isinstance(payload.get("id"), bool) or not isinstance(payload.get("id"), int)):
                raise ValueError("样本编号必须是整数")
            self.process.stdin.write(json.dumps({"action": action, **payload}, ensure_ascii=False) + "\n")
            self.process.stdin.flush()
            if action == "sample":
                self.state.update(phase="countdown", error="")
            if action == "stop":
                self.state["phase"] = "stopping"
                # 同时中断连接/等待，避免 stdin 尚未开始读取时无法退出。
                self.process.send_signal(signal.SIGTERM)
            return self.status()

    def solve(self, payload):
        if set(payload) - {"include_status20"} or not isinstance(payload.get("include_status20", False), bool):
            raise ValueError("求解仅支持布尔参数 include_status20")
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                raise RuntimeError("请先结束采集并断开，再求解完整会话")
            if self.solution.get("phase") == "running":
                raise RuntimeError("正在求解，请等待")
            path = Path(self.state.get("file", "")).resolve()
            if self.output_dir not in path.parents or not path.is_file():
                raise FileNotFoundError("当前会话还没有采集文件")
            output = path.parent / "solutions" / uuid.uuid4().hex
            command = [self.python, "-u", str(SOLVER), "--input", str(path), "--output-dir", str(output)]
            if payload.get("include_status20"):
                command.append("--include-status-20")
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, start_new_session=True)
            self.solution_process = process
            self.solution = {"phase": "running", "output_dir": str(output)}
            threading.Thread(target=self._watch_solution, args=(process, output), daemon=True).start()
            return self.status()

    def _watch_solution(self, process, output):
        try:
            log, _ = process.communicate(timeout=60)
            report_path = output / "report.json"
            report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else None
            result = {"phase": report.get("status", "failed") if report else "failed",
                      "output_dir": str(output), "report": report,
                      "text": (output / "report.txt").read_text(encoding="utf-8") if (output / "report.txt").exists() else log[-8000:]}
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            result = {"phase": "failed", "output_dir": str(output), "text": "离线求解超过 60 秒，请检查样本量和数据，未应用回放配置"}
        except (ValueError, OSError) as exc:
            result = {"phase": "failed", "output_dir": str(output), "text": str(exc)}
        with self.lock:
            if process is self.solution_process:
                self.solution = result

    def export_solution(self):
        with self.lock:
            if self.solution.get("phase") not in {"passed", "review_required", "failed"}:
                raise FileNotFoundError("尚无已完成的求解报告")
            folder = Path(self.solution.get("output_dir", "")).resolve()
            if self.output_dir not in folder.parents or not (folder / "report.json").is_file():
                raise FileNotFoundError("求解报告尚未生成")
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
                for name in ("calibration.json", "report.json", "report.txt", "errors.csv"):
                    if (folder / name).is_file():
                        archive.write(folder / name, name)
            return buffer.getvalue()

    def _watch(self, process):
        last_log = ""
        for line in process.stdout:
            if not line.startswith(PREFIX):
                last_log = line.strip()[-1000:]
                continue
            try:
                event = json.loads(line[len(PREFIX):])
            except ValueError:
                continue
            with self.lock:
                if process is not self.process:
                    return
                if event.get("event") == "status":
                    self.state.update({k: v for k, v in event.items() if k != "event"})
                elif event.get("event") in {"fatal", "error"}:
                    self.state["error"] = event.get("error", "采集失败")
                    if event["event"] == "fatal":
                        self.state["phase"] = "failed"
        code = process.wait()
        with self.lock:
            if self.process is process:
                intentional = self.state["phase"] in {"stopping", "stopped"}
                self.state["phase"] = "failed" if code and not intentional else "stopped"
                if code and not intentional and not self.state.get("error"):
                    self.state["error"] = last_log or f"采集进程退出 ({code})"

    def export(self):
        with self.lock:
            path = Path(self.state.get("file", "")).resolve()
            if self.output_dir not in path.parents or not path.is_file():
                raise FileNotFoundError("当前会话尚未生成采集文件")
            data = path.read_bytes()
            # 正在采样时允许下载已完整落盘的记录，不导出末尾尚未写完的 JSON 行。
            return data[:data.rfind(b"\n") + 1]

    def shutdown(self):
        with self.lock:
            solver = self.solution_process
            if solver is not None and solver.poll() is None:
                solver.terminate()
        if solver is not None and solver.poll() is None:
            try:
                solver.wait(timeout=3)
            except subprocess.TimeoutExpired:
                solver.kill()
                solver.wait(timeout=3)
        with self.lock:
            process = self.process
            if process is None or process.poll() is not None:
                return
            self.state["phase"] = "stopping"
            process.send_signal(signal.SIGTERM)
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3)
