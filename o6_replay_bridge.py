"""回放专用 O6 桥接进程；只由回放拥有，不启动数据手套或采集器。"""

from __future__ import annotations

import os
import re
import shlex
import signal
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ROS_SETUP = ROOT / "linkerhand_telop_python-master/ros2/install/setup.bash"
BRIDGE_EXECUTABLE = (
    ROOT / "linkerhand_telop_python-master/ros2/install/linkerhand_retarget"
    "/lib/linkerhand_retarget/o6_bridge"
)
LIMITS = (
    ROOT / "linkerhand_telop_python-master/ros2/src/linkerhand_retarget"
    "/config/o6_command_limits.yaml"
)


def validate_o6_can(interface: str) -> str:
    if not isinstance(interface, str) or re.fullmatch(r"can[0-9]+", interface) is None:
        raise ValueError("O6 回放 CAN 接口须为 can0、can1 等 SocketCAN 名称")
    return interface


def existing_o6_bridges() -> list[int]:
    """保守检查旧版/手动/平台桥接，避免两个进程同时持有 O6 CAN。"""
    found: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            args = [part.decode("utf-8", "replace") for part in
                    (entry / "cmdline").read_bytes().split(b"\0") if part]
        except (OSError, PermissionError):
            continue
        if any(Path(arg).name in {"o6_bridge", "o6_bridge_node"} for arg in args):
            found.append(int(entry.name))
    return found


class O6ReplayBridge:
    def __init__(self, interface: str, log_path: Path, trace=None):
        self.interface = validate_o6_can(interface)
        self.log_path = Path(log_path)
        self.trace = trace
        self.process: subprocess.Popen[bytes] | None = None
        self.log_handle = None

    def command(self) -> list[str]:
        args = [str(BRIDGE_EXECUTABLE), "--ros-args", "--params-file", str(LIMITS),
                "-p", "topic:=/cb_left_hand_control_cmd",
                "-p", "hand_type:=left", "-p", f"can:={self.interface}",
                "-p", "speed:=[255,255,255,255,255,255]",
                "-p", "torque:=[40,40,40,40,40,40]",
                "-p", "min_send_interval:=0.005"]
        shell = ("source /opt/ros/humble/setup.bash && "
                 f"source {shlex.quote(str(ROS_SETUP))} && "
                 f"exec {shlex.join(args)}")
        return ["bash", "-lc", shell]

    def start(self, hand_replay) -> None:
        if self.process is not None:
            raise RuntimeError("回放专用 O6 桥接已经启动")
        if not ROS_SETUP.is_file() or not BRIDGE_EXECUTABLE.is_file() or not LIMITS.is_file():
            raise RuntimeError("O6 ROS 桥接安装或配置文件不存在")
        result = subprocess.run(["ip", "-br", "link", "show", self.interface],
                                capture_output=True, text=True, timeout=3, check=False)
        words = result.stdout.split()
        if result.returncode or len(words) < 2 or words[1] != "UP":
            raise RuntimeError(f"{self.interface} 未启动；回放不会打开 O6 桥接")
        existing = existing_o6_bridges()
        if existing:
            raise RuntimeError(
                "检测到已有 O6 桥接进程 " + ", ".join(map(str, existing))
                + "；请先停止平台/独立桥接，不能重复占用 CAN"
            )
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_handle = self.log_path.open("xb", buffering=0)
        try:
            self.process = subprocess.Popen(
                self.command(), cwd=ROOT, stdout=self.log_handle,
                stderr=subprocess.STDOUT, start_new_session=True,
            )
            if self.trace is not None:
                self.trace.record("o6_bridge_started", can=self.interface,
                                  pid=self.process.pid, log=str(self.log_path))
            deadline = time.monotonic() + 16.0
            last_error = ""
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    break
                try:
                    hand_replay.probe_bridge()
                except RuntimeError as exc:
                    last_error = str(exc)
                    continue
                if self.process.poll() is not None:
                    break
                others = [pid for pid in existing_o6_bridges()
                          if pid != self.process.pid]
                if others:
                    raise RuntimeError(
                        "启动期间又出现 O6 桥接进程 "
                        + ", ".join(map(str, others)) + "；拒绝重复持有 CAN"
                    )
                ready_marker = f"hand_type=left, can={self.interface}"
                log_deadline = time.monotonic() + 5.0
                while True:
                    log = self.log_path.read_text(encoding="utf-8", errors="replace")
                    if "Hardware version number not recognized" in log:
                        raise RuntimeError(
                            f"{self.interface} 上的 O6 硬件版本未识别；"
                            "请核对 USB-CAN 与灵巧手连接"
                        )
                    if "O6 bridge ready:" in log and ready_marker in log:
                        break
                    if time.monotonic() >= log_deadline:
                        raise RuntimeError(
                            "O6 回放服务出现，但本任务桥接未确认正确的左手/CAN 身份；"
                            f"详见 {self.log_path}"
                        )
                    time.sleep(0.05)
                print(f"回放专用 O6 桥接已就绪：{self.interface}；未启动手套或相机。")
                return
            try:
                detail = self.log_path.read_text(
                    encoding="utf-8", errors="replace"
                ).splitlines()[-5:]
            except OSError:
                detail = []
            raise RuntimeError(
                "回放专用 O6 桥接未就绪: "
                + (" | ".join(detail) or last_error or "进程提前退出")
            )
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        process = self.process
        self.process = None
        if process is not None:
            for sig, timeout in ((signal.SIGINT, 4.0), (signal.SIGTERM, 2.0),
                                 (signal.SIGKILL, 1.0)):
                if process.poll() is not None:
                    break
                try:
                    os.killpg(process.pid, sig)
                except ProcessLookupError:
                    break
                try:
                    process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    continue
            if self.trace is not None:
                self.trace.record("o6_bridge_stopped", can=self.interface,
                                  pid=process.pid, exit_code=process.poll())
        if self.log_handle is not None:
            self.log_handle.close()
            self.log_handle = None
