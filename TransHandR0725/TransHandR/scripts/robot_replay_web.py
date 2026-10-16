"""把固定起点回放接入数据页；本模块本身不直接调用机器人 SDK。"""

from __future__ import annotations

import json
import math
import os
import re
import signal
import subprocess
import sys
import threading
from collections import deque
from pathlib import Path
from typing import Any, Callable


REPLAY_SCRIPT = Path(__file__).resolve().parents[3] / "replay_tron2_fixed_start.py"
RESTORE_SCRIPT = Path(__file__).resolve().parents[3] / "restore_tron2_fixed_start.py"
PARAMETERS = {
    "scale": (0.0, 1.0, 1.0),
    "speed": (0.05, 1.0, 0.2),
    "max_displacement": (0.001, 0.35, 0.35),
    "countdown": (5, 30, 5),
}


def replay_options(payload: dict[str, Any]) -> dict[str, Any]:
    """网页只暴露已经使用过的平移回放参数；其他 CLI 参数不接受透传。"""
    allowed = set(PARAMETERS) | {"id", "axis_map", "with_o6_hand", "o6_can", "confirmation"}
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError("不支持的回放参数: " + ", ".join(sorted(unknown)))
    values: dict[str, Any] = {}
    for name, (minimum, maximum, default) in PARAMETERS.items():
        raw = payload.get(name, default)
        if isinstance(raw, bool):
            raise ValueError(f"{name} 必须是数值")
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} 必须是数值") from exc
        if not math.isfinite(value) or not minimum <= value <= maximum:
            raise ValueError(f"{name} 必须在 {minimum:g} 至 {maximum:g} 之间")
        if name == "countdown" and not value.is_integer():
            raise ValueError("countdown 必须是整数秒")
        values[name] = int(value) if name == "countdown" else value
    axis_map = str(payload.get("axis_map", "y,z,x")).replace(" ", "").lower()
    tokens = axis_map.split(",")
    axes = [token.lstrip("+-") for token in tokens]
    if (len(tokens) != 3 or set(axes) != {"x", "y", "z"}
            or any(token not in {"x", "y", "z", "+x", "+y", "+z",
                                     "-x", "-y", "-z"} for token in tokens)):
        raise ValueError("axis_map 必须是 x,y,z 的带符号排列，例如 y,z,x")
    values["axis_map"] = axis_map
    hand = payload.get("with_o6_hand", True)
    if not isinstance(hand, bool):
        raise ValueError("with_o6_hand 必须是布尔值")
    values["with_o6_hand"] = hand
    o6_can = payload.get("o6_can", "")
    if not isinstance(o6_can, str):
        raise ValueError("O6 CAN 接口必须是字符串")
    o6_can = o6_can.strip()
    if o6_can and re.fullmatch(r"can[0-9]+", o6_can) is None:
        raise ValueError("O6 CAN 接口须为 can0、can1 等 SocketCAN 名称")
    if o6_can and not hand:
        raise ValueError("未启用 O6 同步时不接受 O6 CAN 接口")
    values["o6_can"] = o6_can
    return values


class RobotReplayManager:
    """单任务进程管理；网页确认与 CLI 的 REPLAY 输入保持两阶段。"""

    def __init__(self, resolve_episode: Callable[[str], Path],
                 can_execute: Callable[[], None], python: str | None = None,
                 can_start_o6_bridge: Callable[[], None] | None = None):
        self.resolve_episode = resolve_episode
        self.can_execute = can_execute
        self.can_start_o6_bridge = can_start_o6_bridge
        self.python = python or os.environ.get("DEXUMI_REPLAY_PYTHON") or sys.executable
        self.lock = threading.RLock()
        self.process: subprocess.Popen[str] | None = None
        self.phase = "idle"
        self.task_kind = ""
        self.o6_can = ""
        self.lines: deque[str] = deque(maxlen=80)
        self.episode_id = ""
        self.trace_file = ""
        self.exit_code: int | None = None
        self.preview_signature: tuple[Any, ...] | None = None

    def _selection(self, payload: dict[str, Any]) -> tuple[str, Path, dict[str, Any], tuple[Any, ...]]:
        episode_id = payload.get("id")
        if not isinstance(episode_id, str) or not episode_id:
            raise ValueError("请选择 Episode")
        path = self.resolve_episode(episode_id)
        options = replay_options(payload)
        stat = path.stat()
        signature = (str(path), stat.st_size, stat.st_mtime_ns,
                     json.dumps(options, sort_keys=True))
        return episode_id, path, options, signature

    def _command(self, path: Path, options: dict[str, Any], execute: bool) -> list[str]:
        cmd = [self.python, "-u", str(REPLAY_SCRIPT), "--file", str(path),
               "--axis-map", options["axis_map"], "--scale", str(options["scale"]),
               "--speed", str(options["speed"]),
               "--max-displacement", str(options["max_displacement"]),
               "--countdown", str(options["countdown"])]
        if options["with_o6_hand"]:
            cmd.append("--with-o6-hand")
            if options["o6_can"]:
                cmd.extend(("--o6-can", options["o6_can"]))
        if execute:
            cmd.extend(("--execute", "--web-control"))
        return cmd

    def preview(self, payload: dict[str, Any]) -> dict[str, Any]:
        episode_id, path, options, signature = self._selection(payload)
        try:
            result = subprocess.run(self._command(path, options, False),
                                    stdin=subprocess.DEVNULL, capture_output=True,
                                    text=True, timeout=30, check=False)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("离线预检超过 30 秒，未连接机器人") from exc
        message = (result.stdout + result.stderr).strip()
        if result.returncode != 0:
            raise RuntimeError(message[-4000:] or f"离线预检失败 ({result.returncode})")
        with self.lock:
            self.preview_signature = signature
        return {"ok": True, "id": episode_id, "options": options,
                "summary": message[-4000:]}

    def status(self) -> dict[str, Any]:
        with self.lock:
            return {"phase": self.phase, "kind": self.task_kind,
                    "o6_can": self.o6_can,
                    "id": self.episode_id,
                    "exit_code": self.exit_code, "trace_file": self.trace_file,
                    "lines": list(self.lines)}

    def _launch(self, command: list[str], episode_id: str, kind: str) -> dict[str, Any]:
        process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
            start_new_session=True,
        )
        self.process = process
        self.phase = "preparing"
        self.task_kind = kind
        self.o6_can = command[command.index("--o6-can") + 1] if "--o6-can" in command else ""
        self.episode_id = episode_id
        self.trace_file = ""
        self.exit_code = None
        self.lines.clear()
        self.preview_signature = None
        threading.Thread(target=self._watch, args=(process,),
                         name="robot-replay-web", daemon=True).start()
        return self.status()

    def start(self, payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("confirmation") != "PREPARE":
            raise ValueError("启动固定起点准备前，请输入 PREPARE")
        episode_id, path, options, signature = self._selection(payload)
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                raise RuntimeError("已有真机回放任务，不能并发启动")
            if signature != self.preview_signature:
                raise RuntimeError("数据或参数已改变，请重新执行离线预检")
            self.can_execute()
            if options["o6_can"]:
                if self.can_start_o6_bridge is None:
                    raise RuntimeError("未配置回放专用 O6 桥接的运行状态检查")
                self.can_start_o6_bridge()
            return self._launch(self._command(path, options, True), episode_id,
                                "replay")

    def restore(self, payload: dict[str, Any]) -> dict[str, Any]:
        """只恢复保存姿态；不要求 episode 预检，也绝不启动轨迹。"""
        if set(payload) - {"confirmation", "countdown"}:
            raise ValueError("恢复准备位置不接受 episode 或轨迹参数")
        if payload.get("confirmation") != "PREPARE":
            raise ValueError("恢复准备位置前，请输入 PREPARE")
        countdown = replay_options({"countdown": payload.get("countdown", 5)})["countdown"]
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                raise RuntimeError("已有真机恢复或回放任务，不能并发启动")
            self.can_execute()
            command = [self.python, "-u", str(RESTORE_SCRIPT), "--execute",
                       "--web-control", "--countdown", str(countdown)]
            return self._launch(command, "", "restore")

    def confirm(self, payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("confirmation") != "REPLAY":
            raise ValueError("固定起点到位后，请输入 REPLAY")
        with self.lock:
            if self.phase != "awaiting_replay" or self.process is None:
                raise RuntimeError("当前未等待回放确认")
            if self.process.poll() is not None or self.process.stdin is None:
                raise RuntimeError("回放进程已退出")
            self.can_execute()
            if self.o6_can:
                assert self.can_start_o6_bridge is not None
                self.can_start_o6_bridge()
            self.process.stdin.write("REPLAY\n")
            self.process.stdin.flush()
            self.phase = "armed"
            return self.status()

    def cancel(self) -> dict[str, Any]:
        with self.lock:
            if self.process is None or self.process.poll() is not None:
                raise RuntimeError("没有运行中的任务")
            self.process.send_signal(signal.SIGINT)
            self.lines.append("已请求停止后续指令；MoveJ/ServoP 可能仍在机器人端执行，这不是实体急停。")
            self.phase = "cancelling"
            return self.status()

    def _watch(self, process: subprocess.Popen[str]) -> None:
        assert process.stdout is not None
        for line in process.stdout:
            line = line.strip()
            if not line:
                continue
            with self.lock:
                if self.process is not process:
                    continue
                self.lines.append(line[-1000:])
                if line.startswith("ROBOT_REPLAY_EVENT "):
                    try:
                        phase = json.loads(line.split(" ", 1)[1])["phase"]
                    except (ValueError, KeyError, TypeError):
                        continue
                    if (phase in {"awaiting_replay", "replaying", "completed"}
                            and self.phase != "cancelling"):
                        self.phase = phase
                elif line.startswith("记录文件: "):
                    self.trace_file = line.removeprefix("记录文件: ")
        code = process.wait()
        with self.lock:
            if self.process is process:
                self.exit_code = code
                if code == 0:
                    self.phase = "restored" if self.task_kind == "restore" else "completed"
                else:
                    self.phase = "failed"

    def shutdown(self) -> None:
        with self.lock:
            process = self.process
            if process is not None and process.poll() is None:
                process.send_signal(signal.SIGINT)
        if process is not None and process.poll() is None:
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                # 不能把 Python 进程退出与机器人急停混为一谈；不强杀正在
                # 等待原生 SDK/MoveJ 的进程，现场应依靠实体急停确认状态。
                pass
