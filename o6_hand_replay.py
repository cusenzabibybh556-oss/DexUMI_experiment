"""Parent-side O6 replay client; the ROS sidecar never opens a second CAN owner."""

from __future__ import annotations

import json
import queue
import shlex
import subprocess
import threading
import time
from pathlib import Path

from replay_o6_hand_ros import load_states


ROOT = Path(__file__).resolve().parent
ROS_SETUP = ROOT / "linkerhand_telop_python-master/ros2/install/setup.bash"
ROS_SIDECAR = ROOT / "replay_o6_hand_ros.py"


class O6HandReplaySession:
    def __init__(self, episode: Path, speed: float, origin_timestamp_ns: int,
                 trace=None):
        self.episode = Path(episode).resolve()
        self.speed = speed
        self.origin_timestamp_ns = int(origin_timestamp_ns)
        self.trace = trace
        self.due_ns, self.states = load_states(
            self.episode, speed, self.origin_timestamp_ns
        )
        self.process: subprocess.Popen[str] | None = None
        self.status: queue.Queue[dict] = queue.Queue()
        self.reader: threading.Thread | None = None
        self.error: str | None = None

    def _ros_command(self, arguments: str) -> list[str]:
        command = (
            "source /opt/ros/humble/setup.bash && "
            f"source {shlex.quote(str(ROS_SETUP))} && "
            f"exec /usr/bin/python3 {shlex.quote(str(ROS_SIDECAR))} {arguments}"
        )
        return ["bash", "-lc", command]

    def probe_bridge(self) -> None:
        """Before any arm MoveJ, verify the updated O6 bridge is discoverable."""
        try:
            result = subprocess.run(
                self._ros_command("--probe"), capture_output=True, text=True,
                timeout=9.0,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("O6 replay bridge probe timed out") from exc
        if result.returncode != 0 or '"event": "probe_ready"' not in result.stdout:
            detail = (result.stdout + result.stderr).strip().splitlines()
            raise RuntimeError(
                "O6 replay bridge is not ready: " + (detail[-1] if detail else "no response")
            )

    def acquire(self) -> None:
        if self.process is not None:
            raise RuntimeError("O6 replay session already started")
        arguments = (
            f"--file {shlex.quote(str(self.episode))} --speed {self.speed:g} "
            f"--origin-timestamp-ns {self.origin_timestamp_ns}"
        )
        self.process = subprocess.Popen(
            self._ros_command(arguments), stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        self.reader = threading.Thread(target=self._read_output, daemon=True)
        self.reader.start()
        deadline = time.monotonic() + 12.0
        while time.monotonic() < deadline:
            try:
                message = self.status.get(timeout=0.2)
            except queue.Empty:
                if self.process.poll() is not None:
                    break
                continue
            if message.get("event") == "ready":
                print(f"O6 左手实测状态回放已接管桥接输入：{message['samples']} 帧；"
                      "数据手套输入暂被忽略。")
                return
            if message.get("event") == "error":
                raise RuntimeError(f"O6 replay initialization failed: {message.get('message')}")
        raise RuntimeError("O6 replay sidecar did not acquire the bridge")

    def _read_output(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        for line in self.process.stdout:
            line = line.strip()
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                if line and self.trace is not None:
                    self.trace.record("o6_ros_log", line=line)
                continue
            if not isinstance(message, dict):
                continue
            if self.trace is not None:
                self.trace.record("o6_replay_status", status=message)
            if message.get("event") in ("error", "release_error"):
                self.error = str(message.get("message", "O6 replay failed"))
            self.status.put(message)

    def start_at(self, start_ns: int) -> None:
        self.require_healthy()
        assert self.process is not None and self.process.stdin is not None
        self.process.stdin.write(json.dumps({"command": "start", "start_ns": start_ns}) + "\n")
        self.process.stdin.flush()
        if self.trace is not None:
            self.trace.record("o6_replay_scheduled", start_monotonic_ns=start_ns,
                              state_count=len(self.states),
                              source_dataset="dexterous_hand/joint_positions")

    def require_healthy(self) -> None:
        if self.error:
            raise RuntimeError(f"O6 replay failed: {self.error}")
        if self.process is not None and self.process.poll() is not None:
            raise RuntimeError(f"O6 replay sidecar exited: {self.process.returncode}")

    def close(self) -> None:
        if self.process is None:
            return
        process = self.process
        if process.poll() is None and process.stdin is not None:
            try:
                process.stdin.write('{"command":"stop"}\n')
                process.stdin.flush()
            except (BrokenPipeError, OSError):
                pass
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2.0)
            raise RuntimeError(
                "O6 sidecar could not release replay mode; bridge may still hold replay input"
            )
        finally:
            if self.reader is not None:
                self.reader.join(timeout=1.0)
            if process.stdin is not None:
                process.stdin.close()
            if process.stdout is not None:
                process.stdout.close()
        if self.error:
            raise RuntimeError(f"O6 replay failed: {self.error}")
