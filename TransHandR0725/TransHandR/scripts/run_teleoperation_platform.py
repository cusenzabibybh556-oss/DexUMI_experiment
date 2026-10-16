#!/usr/bin/env python3
"""Persistent web platform for VR and LinkerForce teleoperation capture."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import ipaddress
import json
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional


ROOT = Path(__file__).resolve().parents[1]
DEXUMI_ROOT = ROOT.parents[1]
ROS_WS = DEXUMI_ROOT / "linkerhand_telop_python-master" / "ros2"
ROS_PACKAGE_SRC = ROS_WS / "src/linkerhand_retarget"
CALIBRATION_STORE_PATH = Path(os.environ.get(
    "TRANSHANDR_CALIBRATION_STORE",
    "~/.config/transhandr/linkerforce_calibration.json",
)).expanduser()
VISION_PYTHON = Path(os.environ.get(
    "TRANSHANDR_VISION_PYTHON",
    "/home/zmy/anaconda3/envs/versionprodemo/bin/python",
))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "config"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROS_PACKAGE_SRC))

from multisource_collection import CollectionConfig
from teleoperation_profiles import get_profile, public_profiles
from collect_transhandr_multisource import RECORD_CHANNELS, RecorderApp
from realtime_multisource_dashboard import DashboardServer
from robot_replay_web import RobotReplayManager
from linkerhand_retarget.motion.linkerforce.calibration_store import CalibrationStore


DEFAULT_RECORDING_DIR = Path("/home/zmy/workplace/transhandr_episodes")
DEFAULT_CONFIG = {
    "control_source": "linkerforce",
    "target_hand_model": "o6",
    "hand_side": "left",
    "task_name": "transhandr_demo",
    "recording_dir": str(DEFAULT_RECORDING_DIR),
    "avp_ip": "172.20.10.9",
    "left_glove_serial": "/dev/serial/by-path/pci-0000:00:14.0-usb-0:5.1:1.0-port0",
    "right_glove_serial": "/dev/serial/by-path/pci-0000:00:14.0-usb-0:5.2:1.0-port0",
    "hand_serial": "/dev/serial/by-path/pci-0000:00:14.0-usb-0:5.3:1.0-port0",
    "can_interface": "can0",
    "d405_serial": "352122271077",
    "d435i_serial": "141722077699",
    "feedback_enabled_profiles": {},
}

LEGACY_CHANNEL_FIELDS = {
    "enable_head_camera", "enable_tracker", "save_hand_state", "save_action",
    "save_tactile", "save_operator", "save_vision_hand",
}
CHANNEL_IDS = tuple(item["id"] for item in RECORD_CHANNELS)
MANAGED_SCRIPT_PATHS = {
    str((ROOT / "scripts/main_visionpro_multi_hand.py").resolve()),
    str((ROOT / "scripts/ros_telemetry_adapter.py").resolve()),
    str((ROOT / "scripts/inspire_bridge.py").resolve()),
}
MANAGED_ROS_EXECUTABLES = {"handretarget", "o6_bridge"}
O6_BRIDGE_SPEED = [255] * 6
O6_BRIDGE_TORQUE = [40] * 6
O6_BRIDGE_SEND_INTERVAL_S = 0.005
VISION_STREAM_PORT = 12345


class TeleoperationPlatform:
    def __init__(self, mode: str, host: str, port: int, telemetry_port: int,
                 output_dir: Path):
        self.mode, self.host, self.port = mode, host, port
        self.telemetry_port = telemetry_port
        self.output_dir = Path(output_dir).resolve()
        self.config_path = self.output_dir / ".platform_config.json"
        self.channel_config_path = self.output_dir / ".recording_channels.json"
        self.lock = threading.RLock()
        self._loaded_config: Dict[str, Any] = {}
        self.config = self._load_config()
        self.channel_profiles = self._load_channel_profiles()
        self._save_channel_profiles()
        if LEGACY_CHANNEL_FIELDS & set(self._loaded_config):
            self._save_config()
        self.recorder: Optional[RecorderApp] = None
        self.recorder_thread: Optional[threading.Thread] = None
        self.processes: Dict[str, subprocess.Popen] = {}
        self.log_handles = []
        self.expected_stop = False
        self.state = "STOPPED"
        self.error = ""
        runtime_calibration_store = (
            CALIBRATION_STORE_PATH if mode == "hardware"
            else self.output_dir / ".mock_linkerforce_calibration.json"
        )
        self.calibration_store = CalibrationStore(runtime_calibration_store)
        if mode == "hardware":
            self.calibration_store.migrate_legacy(
                ROS_PACKAGE_SRC / "linkerhand_retarget/motion/linkerforce/tmp/jointangle_data.tmp")
        self.calibration_status = {
            "state": "IDLE", "step": "", "progress": 0,
            "instruction": "", "error": "",
        }
        self.calibration_thread: Optional[threading.Thread] = None
        self.dashboard = DashboardServer(
            host, port, self.public_status, self.record_control,
            self.recording_config, self.update_recording_config,
            Path(self.config["recording_dir"]), self.system_config, self.update_system_config,
            self.system_control, self.calibration_control,
        )
        if mode == "hardware":
            self.dashboard.robot_replay = RobotReplayManager(
                self.dashboard._resolve_episode, self._allow_robot_replay,
                can_start_o6_bridge=self._allow_replay_only_o6_bridge,
            )

    def _allow_robot_replay(self) -> None:
        with self.lock:
            if self.state not in {"STOPPED", "RUNNING"}:
                raise RuntimeError("平台正在切换或处于异常状态，不能启动真机回放")
            if self.recorder is not None and self.recorder.public_status().get("recording"):
                raise RuntimeError("正在录制 Episode，请先结束录制再回放真机")

    def _allow_replay_only_o6_bridge(self) -> None:
        with self.lock:
            if self.state != "STOPPED":
                raise RuntimeError(
                    "回放专用 O6 桥接要求采集系统处于已停止状态；"
                    "请先停止平台系统，避免重复占用 CAN"
                )

    def _replay_boundary(self):
        """平台启动/清理不得抢占正在回放的进程或杀掉其专用 O6 桥接。"""
        if self.mode == "mock":
            return contextlib.nullcontext()
        handle = (DEXUMI_ROOT / ".tron2_replay.lock").open("a+")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.close()
            raise RuntimeError("真机恢复/回放正在运行，不能启动或清理平台设备进程") from exc
        return handle

    def _load_config(self) -> Dict[str, Any]:
        value = dict(DEFAULT_CONFIG)
        if self.mode == "mock":
            value["recording_dir"] = str(self.output_dir)
        try:
            loaded = json.loads(self.config_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                self._loaded_config = loaded
                value.update({key: loaded[key] for key in DEFAULT_CONFIG if key in loaded})
                if not isinstance(value["feedback_enabled_profiles"], dict):
                    value["feedback_enabled_profiles"] = {}
                if "left_glove_serial" not in loaded and loaded.get("glove_serial"):
                    value["left_glove_serial"] = loaded["glove_serial"]
        except (OSError, TypeError, json.JSONDecodeError):
            pass
        return value

    @staticmethod
    def _recording_path(raw: Any) -> Path:
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("数据保存目录不能为空")
        path = Path(raw.strip()).expanduser()
        if not path.is_absolute():
            raise ValueError("数据保存目录必须是绝对路径")
        return path.resolve()

    @staticmethod
    def _check_recording_dir(path: Path) -> None:
        disk = DEFAULT_RECORDING_DIR.parent
        if path == disk or disk in path.parents:
            if not disk.is_mount():
                raise RuntimeError(f"数据盘未挂载: {disk}；已阻止写入系统盘")
        try:
            path.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryFile(dir=path):
                pass
        except OSError as exc:
            raise RuntimeError(f"数据保存目录不可写: {path}: {exc}") from exc

    def _save_config(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        temporary = self.config_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.config, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.config_path)

    @staticmethod
    def _profile_key(control_source: str, target_hand_model: str, hand_side: str) -> str:
        return f"{control_source}:{target_hand_model}:{hand_side}"

    def _profile_identity(self, payload: Optional[Dict[str, Any]] = None) -> tuple[str, str, str]:
        payload = payload or {}
        source = str(payload.get("control_source", self.config["control_source"]))
        model = str(payload.get("target_hand_model", self.config["target_hand_model"]))
        side = str(payload.get("hand_side", self.config["hand_side"]))
        get_profile(source, model, side)
        return source, model, side

    @staticmethod
    def _legal_channels(control_source: str, target_hand_model: str = "") -> set[str]:
        common = {"d405", "d435i", "hand_state", "action", "vive_tracker"}
        if control_source == "vision_pro":
            return common | {"vision_hand"}
        channels = common | {"operator_exoskeleton"}
        if target_hand_model == "o6":
            channels.add("tactile")
        if target_hand_model == "inspire":
            channels.add("inspire_force")
        return channels

    def _default_channels(self, control_source: str, target_hand_model: str = "",
                          use_legacy: bool = False) -> list[str]:
        legacy = self._loaded_config if use_legacy else {}
        selected = {"d405", "hand_state", "action"}
        if control_source == "vision_pro":
            if legacy.get("save_vision_hand", True):
                selected.add("vision_hand")
        else:
            if legacy.get("save_operator", True):
                selected.add("operator_exoskeleton")
            if target_hand_model == "o6" and legacy.get("save_tactile", True):
                selected.add("tactile")
            if target_hand_model == "inspire":
                selected.add("inspire_force")
        if legacy.get("enable_head_camera", False):
            selected.add("d435i")
        if legacy.get("enable_tracker", False):
            selected.add("vive_tracker")
        if legacy.get("save_hand_state", True) is False:
            selected.discard("hand_state")
        if legacy.get("save_action", True) is False:
            selected.discard("action")
        return [channel for channel in CHANNEL_IDS if channel in selected]

    def _load_channel_profiles(self) -> Dict[str, list[str]]:
        profiles: Dict[str, list[str]] = {}
        current = self._profile_identity()
        current_key = self._profile_key(*current)
        try:
            saved = json.loads(self.channel_config_path.read_text(encoding="utf-8"))
            if isinstance(saved.get("profiles"), dict):
                for key, channels in saved["profiles"].items():
                    if isinstance(channels, list):
                        profiles[str(key)] = [name for name in CHANNEL_IDS if name in channels]
            elif isinstance(saved.get("channels"), list):
                profiles[current_key] = [name for name in CHANNEL_IDS if name in saved["channels"]]
        except (OSError, AttributeError, TypeError, json.JSONDecodeError):
            pass
        if current_key not in profiles:
            profiles[current_key] = self._default_channels(
                current[0], current[1],
                use_legacy=bool(LEGACY_CHANNEL_FIELDS & set(self._loaded_config)))
        return profiles

    def _save_channel_profiles(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        temporary = self.channel_config_path.with_suffix(".tmp")
        payload = {"version": 2, "profiles": self.channel_profiles}
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.channel_config_path)

    def system_config(self) -> Dict[str, Any]:
        with self.lock:
            process_status = {
                name: {"running": process.poll() is None, "pid": process.pid,
                       "returncode": process.poll()}
                for name, process in self.processes.items()
            }
            source, model, side = self._profile_identity()
            calibration = dict(self.calibration_status)
            calibration.update(self.calibration_store.status(model, side))
            calibration["required"] = get_profile(source, model, side).requires_calibration
            return {"config": dict(self.config), "profiles": public_profiles(),
                    "state": self.state, "error": self.error,
                    "processes": process_status, "mode": self.mode,
                    "calibration": calibration}

    def update_system_config(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        allowed = set(DEFAULT_CONFIG)
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError("未知配置字段: " + ", ".join(sorted(unknown)))
        with self.lock:
            if self.state != "STOPPED":
                raise RuntimeError("系统运行中不能切换设备配置")
            updated = {**self.config, **payload}
            recording_path = self._recording_path(updated["recording_dir"])
            self._check_recording_dir(recording_path)
            updated["recording_dir"] = str(recording_path)
            updated["avp_ip"] = str(updated["avp_ip"]).strip()
            get_profile(updated["control_source"], updated["target_hand_model"], updated["hand_side"])
            try:
                ipaddress.ip_address(updated["avp_ip"])
            except ValueError:
                labels = updated["avp_ip"].split(".")
                valid_hostname = (
                    0 < len(updated["avp_ip"]) <= 253
                    and all(
                        0 < len(label) <= 63
                        and re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?", label)
                        for label in labels
                    )
                )
                if not valid_hostname:
                    raise ValueError("Vision Pro IP 或主机名不合法")
            for key in ("left_glove_serial", "right_glove_serial", "hand_serial"):
                if updated[key] and not str(updated[key]).startswith("/dev/"):
                    raise ValueError(f"{key} 必须是 /dev/ 下的设备路径")
            if (updated["control_source"] == "linkerforce"
                    and updated["target_hand_model"] == "inspire"):
                configured = {
                    key: self._device_identity(str(updated[key]))
                    for key in ("left_glove_serial", "right_glove_serial", "hand_serial")
                    if updated[key]
                }
                duplicates = {
                    identity for identity in configured.values()
                    if list(configured.values()).count(identity) > 1
                }
                if duplicates:
                    names = [key for key, identity in configured.items() if identity in duplicates]
                    raise ValueError("手套与灵巧手串口不能重复: " + ", ".join(names))
            requested_feedback = payload.get("feedback_enabled_profiles", {})
            if not isinstance(requested_feedback, dict):
                raise ValueError("力反馈设置必须是按手侧保存的配置")
            for key, enabled in requested_feedback.items():
                if key not in {"linkerforce:inspire:left", "linkerforce:inspire:right"} or not isinstance(enabled, bool):
                    raise ValueError("力反馈设置只支持 Inspire 左右手布尔值")
            updated["feedback_enabled_profiles"] = {
                **self.config.get("feedback_enabled_profiles", {}), **requested_feedback}
            if not str(updated["can_interface"]).replace("_", "").isalnum():
                raise ValueError("CAN 接口名不合法")
            self.config = updated
            self._save_config()
            self.dashboard.episode_root = recording_path
        return self.system_config()

    def _empty_status(self) -> Dict[str, Any]:
        return {
            "recording": False, "episode_index": 0, "elapsed_s": 0.0,
            "fps": 0.0, "frames": 0, "hand_type": self.config["target_hand_model"],
            "control_source": self.config["control_source"], "hand_side": self.config["hand_side"],
            "vision_hand_side": self.config["hand_side"], "schema_version": "quality-first-v3.2",
            "valid": {}, "sync": {}, "streams": {}, "last_file": "", "rgb": None, "depth": None,
            "head_rgb": None, "head_depth": None,
            "head_preview": {"enabled": False, "valid": False, "error": ""},
            "right_hand": {}, "operator": [],
            "hand_state": [], "action": [], "tactile": {}, "force": [], "tracker": {"valid": False},
        }

    def public_status(self) -> Dict[str, Any]:
        with self.lock:
            status = self.recorder.public_status() if self.recorder else self._empty_status()
            status["system"] = {
                "state": self.state, "error": self.error,
                "force_feedback_enabled": bool(self.config.get("feedback_enabled_profiles", {}).get(
                    self._profile_key(*self._profile_identity()), False)),
                "processes": {name: process.poll() is None for name, process in self.processes.items()},
            }
            return status

    def recording_config(self, profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        with self.lock:
            source, model, side = self._profile_identity(profile)
            selected = set(self._selected_channels({
                "control_source": source, "target_hand_model": model, "hand_side": side,
            }))
            availability = self._channel_availability(source, model)
            runtime = self.recorder.recording_config() if self.recorder else {}
            return {
                "channels": [{**item, "selected": item["id"] in selected,
                              "available": availability.get(item["id"], False)}
                             for item in RECORD_CHANNELS],
                "output_dir": self.config["recording_dir"], "task_name": self.config["task_name"],
                "hand_type": model, "control_source": source, "hand_side": side,
                "next_file": runtime.get("next_file", "系统启动后确定"),
                "max_duration_s": runtime.get("max_duration_s", 300.0),
                "max_size_gb": runtime.get("max_size_gb", 5.0),
            }

    def update_recording_config(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        with self.lock:
            if self.state != "STOPPED":
                raise RuntimeError("系统运行中不能修改数据保存通道")
            source, model, side = self._profile_identity(payload)
            known = set(CHANNEL_IDS)
            selected = set(payload.get("channels", [])) & known
            selected.add("d405")
            unavailable = selected - self._legal_channels(source, model)
            if unavailable:
                raise RuntimeError("当前遥操作配置不支持: " + ", ".join(sorted(unavailable)))
            key = self._profile_key(source, model, side)
            self.channel_profiles[key] = [name for name in CHANNEL_IDS if name in selected]
            self._save_channel_profiles()
            return self.recording_config({
                "control_source": source, "target_hand_model": model, "hand_side": side,
            })

    def record_control(self, action: str) -> None:
        with self.lock:
            if not self.recorder or self.state != "RUNNING":
                raise RuntimeError("采集系统尚未运行")
            self.recorder.control(action)

    def _channel_availability(self, control_source: Optional[str] = None,
                              target_hand_model: Optional[str] = None) -> Dict[str, bool]:
        exo = (control_source or self.config["control_source"]) == "linkerforce"
        model = target_hand_model or self.config["target_hand_model"]
        return {"d405": True, "d435i": True,
                "vision_hand": not exo, "operator_exoskeleton": exo,
                "hand_state": True, "action": True,
                "tactile": exo and model == "o6",
                "inspire_force": exo and model == "inspire", "vive_tracker": True}

    def _selected_channels(self, config: Optional[Dict[str, Any]] = None) -> list[str]:
        source, model, side = self._profile_identity(config)
        key = self._profile_key(source, model, side)
        if key not in self.channel_profiles:
            self.channel_profiles[key] = self._default_channels(source, model)
        legal = self._legal_channels(source, model)
        selected = set(self.channel_profiles[key]) & legal
        selected.add("d405")
        return [name for name in CHANNEL_IDS if name in selected]

    @staticmethod
    def _device_identity(path: str) -> str:
        device = Path(path)
        try:
            return str(device.resolve(strict=True))
        except OSError:
            return str(device)

    def _selected_glove_serial(self) -> str:
        return str(self.config[f"{self.config['hand_side']}_glove_serial"])

    @staticmethod
    def _check_vision_stream(ip: str) -> None:
        probe = (
            "import grpc, sys; "
            "channel = grpc.insecure_channel(sys.argv[1]); "
            "grpc.channel_ready_future(channel).result(timeout=3); "
            "channel.close()"
        )
        try:
            result = subprocess.run(
                [str(VISION_PYTHON), "-c", probe, f"{ip}:{VISION_STREAM_PORT}"],
                capture_output=True, text=True, timeout=5, check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(
                f"Vision Pro {ip}:{VISION_STREAM_PORT} 的 gRPC 数据流未就绪；"
                "请确认电脑和头显处于同一网络，手部采集 App 已点击 Start"
            ) from exc
        if result.returncode:
            raise RuntimeError(
                f"Vision Pro {ip}:{VISION_STREAM_PORT} 的 gRPC 数据流未就绪；"
                "请确认电脑和头显处于同一网络，手部采集 App 已点击 Start"
            )

    def _preflight(self) -> None:
        c = self.config
        self._check_recording_dir(self._recording_path(c["recording_dir"]))
        selected = set(self._selected_channels())
        profile = get_profile(c["control_source"], c["target_hand_model"], c["hand_side"])
        test = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            test.bind(("127.0.0.1", self.telemetry_port))
        except OSError as exc:
            raise RuntimeError(f"UDP telemetry 端口 {self.telemetry_port} 已占用") from exc
        finally:
            test.close()
        if self.mode == "mock":
            return
        try:
            import pyrealsense2 as rs
            serials = {device.get_info(rs.camera_info.serial_number) for device in rs.context().query_devices()}
        except Exception as exc:
            raise RuntimeError(f"RealSense 枚举失败: {exc}") from exc
        required = [c["d405_serial"]]
        if "d435i" in selected:
            required.append(c["d435i_serial"])
        missing = [serial for serial in required if serial and serial not in serials]
        if missing:
            raise RuntimeError("未找到相机: " + ", ".join(missing))
        if c["control_source"] == "linkerforce":
            glove_serial = self._selected_glove_serial()
            if not glove_serial or not Path(glove_serial).exists():
                side_label = "左" if c["hand_side"] == "left" else "右"
                raise RuntimeError(f"未找到{side_label}手套串口: {glove_serial or '未配置'}")
            for path in (Path("/opt/ros/humble/setup.zsh"), ROS_WS / "install/setup.zsh"):
                if not path.exists():
                    raise RuntimeError(f"ROS 环境不存在: {path}")
            if profile.requires_can:
                result = subprocess.run(["ip", "-br", "link", "show", c["can_interface"]],
                                        text=True, capture_output=True)
                if result.returncode or " UP " not in f" {result.stdout} ":
                    raise RuntimeError("CAN 未启动，请先运行 scripts/start_can.sh")
            if profile.requires_hand_serial:
                if not c["hand_serial"] or not Path(c["hand_serial"]).exists():
                    raise RuntimeError(f"未找到 Inspire Hand 串口: {c['hand_serial'] or '未配置'}")
                if self._device_identity(glove_serial) == self._device_identity(c["hand_serial"]):
                    raise RuntimeError("手套与 Inspire Hand 配置成了同一个串口")
            if profile.requires_calibration:
                status = self.calibration_store.status(c["target_hand_model"], c["hand_side"])
                if not status["valid"]:
                    raise RuntimeError(
                        f"缺少 {c['target_hand_model']}:{c['hand_side']} 标定，请先在设备设置中完成标定")
        else:
            self._check_vision_stream(c["avp_ip"])
            if not VISION_PYTHON.exists():
                raise RuntimeError(f"Vision Pro Python 环境不存在: {VISION_PYTHON}")
            result = subprocess.run(
                [str(VISION_PYTHON), "-c", "import torch"],
                text=True, capture_output=True,
            )
            if result.returncode:
                raise RuntimeError(
                    f"Vision Pro Python 环境缺少 torch: {VISION_PYTHON}"
                )

    def _collection_config(self) -> CollectionConfig:
        c = self.config
        exo = c["control_source"] == "linkerforce"
        profile = get_profile(c["control_source"], c["target_hand_model"], c["hand_side"])
        selected = self._selected_channels()
        return CollectionConfig(
            control_source=c["control_source"], target_hand_model=c["target_hand_model"],
            avp_ip=c["avp_ip"], enable_right_hand=not exo, right_hand_source="udp",
            vision_hand_side=c["hand_side"], enable_hand_io=True, hand_source="udp",
            hand_type=c["target_hand_model"], hand_side=c["hand_side"],
            hand_serial_port=c["hand_serial"], hand_can_interface=c["can_interface"],
            enable_exoskeleton_operator=exo, enable_tactile=profile.tactile_shape is not None,
            enable_inspire_force=exo and c["target_hand_model"] == "inspire",
            d405_serial=c["d405_serial"], enable_head_camera="d435i" in selected,
            head_camera_serial=c["d435i_serial"], enable_vive_tracker="vive_tracker" in selected,
            tracker_mode="DONGLE_USB", output_dir=c["recording_dir"], task_name=c["task_name"],
            enable_realtime_dashboard=False, selected_channels=selected,
            hand_state_max_gap_ms=35.0,
        )

    def _recorder_target(self, recorder: RecorderApp) -> None:
        try:
            recorder.run()
        except Exception as exc:
            with self.lock:
                self.error = f"采集器异常: {exc}"
                self.state = "ERROR"
            if not self.expected_stop:
                threading.Thread(target=self._stop_runtime, args=(True,), daemon=True).start()

    def _ros_command(self, args: list[str]) -> list[str]:
        command = shlex.join(args)
        setup = shlex.quote(str(ROS_WS / "install/setup.zsh"))
        return ["/usr/bin/zsh", "-lc",
                f"source /opt/ros/humble/setup.zsh && source {setup} && exec {command}"]

    def _spawn(self, name: str, command: list[str]) -> None:
        log_dir = self.output_dir / "platform_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        handle = (log_dir / f"{name}.log").open("ab", buffering=0)
        process = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        self.log_handles.append(handle)
        self.processes[name] = process
        time.sleep(.5)
        if process.poll() is not None:
            log_path = log_dir / f"{name}.log"
            try:
                tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-4:]
            except OSError:
                tail = []
            detail = " | ".join(line.strip() for line in tail if line.strip())
            raise RuntimeError(
                f"{name} 启动失败: {detail}" if detail
                else f"{name} 启动失败，查看 {log_path}")

    def _start_device_processes(self) -> None:
        c = self.config
        profile = get_profile(c["control_source"], c["target_hand_model"], c["hand_side"])
        if self.mode == "mock":
            return
        if c["control_source"] == "vision_pro":
            model = "l20" if c["target_hand_model"] == "l10" else c["target_hand_model"]
            self._spawn("vision_controller", [str(VISION_PYTHON), str(ROOT / "scripts/main_visionpro_multi_hand.py"),
                        "--hand", model, "--avp_ip", c["avp_ip"], "--vision_hand_side", c["hand_side"],
                        "--serial_port", c["hand_serial"], "--telemetry_host", "127.0.0.1",
                        "--telemetry_port", str(self.telemetry_port), "--telemetry_state_hz",
                        "60" if c["target_hand_model"] == "inspire" else "30"])
            return
        adapter = ["/usr/bin/python3", str(ROOT / "scripts/ros_telemetry_adapter.py"),
                   "--host", "127.0.0.1", "--port", str(self.telemetry_port),
                   "--hand-side", c["hand_side"], "--robot-model", c["target_hand_model"],
                   "--operator-topic", profile.topic("operator_topic", c["hand_side"]),
                   "--action-topic", profile.topic("action_topic", c["hand_side"]),
                   "--state-topic", profile.topic("state_topic", c["hand_side"])]
        tactile_topic = profile.topic("tactile_topic", c["hand_side"])
        if tactile_topic:
            adapter.extend(["--tactile-topic", tactile_topic])
        force_topic = profile.topic("force_topic", c["hand_side"])
        if force_topic:
            adapter.extend(["--force-topic", force_topic])
        self._spawn("ros_telemetry_adapter", self._ros_command(adapter))

        command_topic = profile.topic("command_topic", c["hand_side"])
        if c["target_hand_model"] == "o6":
            limits = ROS_WS / "src/linkerhand_retarget/config/o6_command_limits.yaml"
            bridge = ["ros2", "run", "linkerhand_retarget", "o6_bridge", "--ros-args",
                      "--params-file", str(limits), "-p", f"topic:={command_topic}",
                      "-p", f"hand_type:={c['hand_side']}", "-p", f"can:={c['can_interface']}",
                      "-p", f"speed:={O6_BRIDGE_SPEED}",
                      "-p", f"torque:={O6_BRIDGE_TORQUE}",
                      "-p", f"min_send_interval:={O6_BRIDGE_SEND_INTERVAL_S}"]
            self._spawn("o6_bridge", self._ros_command(bridge))
        elif c["target_hand_model"] == "inspire":
            bridge = ["/usr/bin/python3", str(ROOT / "scripts/inspire_bridge.py"), "--ros-args",
                      "--params-file", str(ROOT / "config/inspire_command_limits.yaml"),
                      "-p", f"topic:={command_topic}",
                      "-p", f"hand_side:={c['hand_side']}",
                      "-p", f"serial_port:={c['hand_serial']}"]
            self._spawn("inspire_bridge", self._ros_command(bridge))

        handretarget = ["ros2", "run", "linkerhand_retarget", "handretarget", "--ros-args",
                        "-p", f"ports:=['{self._selected_glove_serial()}']",
                        "-p", "baudrate:=2000000", "-p", "auto_scan:=True",
                        "-p", "calibration:=False",
                        "-p", f"calibration_profile:={c['target_hand_model']}",
                        "-p", f"active_hand_side:={c['hand_side']}",
                        "-p", "enable_force_feedback:=" + str(bool(c.get("feedback_enabled_profiles", {}).get(
                            self._profile_key(c["control_source"], c["target_hand_model"], c["hand_side"]), False))).lower(),
                        "-p", f"calibration_store:={CALIBRATION_STORE_PATH}"]
        self._spawn("handretarget", self._ros_command(handretarget))

    def _monitor(self) -> None:
        while True:
            time.sleep(.25)
            with self.lock:
                if self.expected_stop or self.state not in {"STARTING", "RUNNING"}:
                    return
                failed = next(((name, process.returncode) for name, process in self.processes.items()
                               if process.poll() is not None), None)
                recorder_failed = self.recorder_thread and not self.recorder_thread.is_alive()
            if failed or recorder_failed:
                with self.lock:
                    self.state = "ERROR"
                    self.error = (f"子进程 {failed[0]} 异常退出 ({failed[1]})" if failed
                                  else "采集器异常退出")
                self._stop_runtime(True)
                return

    def _calibration_command(self) -> list[str]:
        c = self.config
        return self._ros_command([
            "ros2", "run", "linkerhand_retarget", "handretarget", "--ros-args",
            "-p", f"ports:=['{self._selected_glove_serial()}']",
            "-p", "baudrate:=2000000", "-p", "auto_scan:=True",
            "-p", "calibration:=True", "-p", "calibration_only:=True",
            "-p", f"calibration_profile:={c['target_hand_model']}",
            "-p", f"active_hand_side:={c['hand_side']}",
            "-p", f"calibration_store:={CALIBRATION_STORE_PATH}",
        ])

    def _calibration_worker(self, process: subprocess.Popen) -> None:
        log_dir = self.output_dir / "platform_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        completed = False
        try:
            with (log_dir / "calibration.log").open("a", encoding="utf-8") as log:
                if process.stdout is not None:
                    for line in process.stdout:
                        log.write(line)
                        log.flush()
                        marker = "CALIBRATION_EVENT "
                        if marker not in line:
                            continue
                        try:
                            event = json.loads(line.split(marker, 1)[1].strip())
                        except (json.JSONDecodeError, ValueError):
                            continue
                        with self.lock:
                            kind = event.get("event", "")
                            completed = completed or kind == "complete"
                            self.calibration_status.update({
                                "state": "COMPLETE" if kind == "complete" else (
                                    "FAILED" if kind == "failed" else "RUNNING"),
                                "step": event.get("step", self.calibration_status.get("step", "")),
                                "progress": int(event.get("progress", 0)),
                                "instruction": event.get(
                                    "instruction", self.calibration_status.get("instruction", "")),
                                "error": event.get("error", ""),
                            })
            returncode = process.wait()
        except Exception as exc:
            returncode = process.poll() if process.poll() is not None else 1
            with self.lock:
                self.calibration_status.update(state="FAILED", error=str(exc))
        with self.lock:
            self.processes.pop("calibration", None)
            if self.state != "CALIBRATING":
                return
            if returncode == 0 and completed:
                self.state, self.error = "STOPPED", ""
                self.calibration_status.update(
                    state="COMPLETE", progress=100, instruction="标定完成，可以启动系统", error="")
            else:
                detail = self.calibration_status.get("error") or f"标定进程退出码 {returncode}"
                self.state, self.error = "ERROR", detail
                self.calibration_status.update(state="FAILED", error=detail)

    def start_calibration(self) -> None:
        with self.lock:
            if self.state != "STOPPED":
                raise RuntimeError("停止系统后才能标定")
            c = self.config
            profile = get_profile(c["control_source"], c["target_hand_model"], c["hand_side"])
            if c["control_source"] != "linkerforce" or not profile.requires_calibration:
                raise RuntimeError("当前配置不需要独立 LinkerForce 标定")
            glove = self._selected_glove_serial()
            if not glove or (self.mode != "mock" and not Path(glove).exists()):
                raise RuntimeError(f"未找到所选手套串口: {glove or '未配置'}")
            self.state, self.error = "CALIBRATING", ""
            self.calibration_status = {
                "state": "RUNNING", "step": "connecting", "progress": 0,
                "instruction": "正在连接所选外骨骼手套", "error": "",
            }
        if self.mode == "mock":
            with self.lock:
                self.state = "STOPPED"
                self.calibration_status.update(
                    state="COMPLETE", step="complete", progress=100,
                    instruction="模拟标定完成")
            return
        try:
            survivors = self._cleanup_stale_processes()
            if survivors:
                raise RuntimeError("标定前无法清理旧进程: " + ", ".join(survivors))
            for path in (Path("/opt/ros/humble/setup.zsh"), ROS_WS / "install/setup.zsh"):
                if not path.exists():
                    raise RuntimeError(f"ROS 环境不存在: {path}")
            process = subprocess.Popen(
                self._calibration_command(), cwd=ROOT, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                bufsize=1, start_new_session=True,
            )
            with self.lock:
                self.processes["calibration"] = process
            self.calibration_thread = threading.Thread(
                target=self._calibration_worker, args=(process,),
                name="linkerforce-calibration", daemon=True)
            self.calibration_thread.start()
        except Exception as exc:
            with self.lock:
                self.state, self.error = "ERROR", str(exc)
                self.calibration_status.update(state="FAILED", error=str(exc))
            raise

    def cancel_calibration(self) -> None:
        with self.lock:
            if self.state != "CALIBRATING":
                return
            process = self.processes.get("calibration")
            self.state = "STOPPING"
        if process:
            self._stop_process(process)
        with self.lock:
            self.processes.pop("calibration", None)
            self.state, self.error = "STOPPED", ""
            self.calibration_status.update(
                state="CANCELLED", instruction="标定已取消", error="")

    def start_system(self) -> None:
        with self._replay_boundary():
            self._start_system_unlocked()

    def _start_system_unlocked(self) -> None:
        with self.lock:
            if self.state != "STOPPED":
                raise RuntimeError("系统已启动或正在切换状态")
            self.state, self.error, self.expected_stop = "STARTING", "", False
        try:
            if self.mode != "mock":
                survivors = self._cleanup_stale_processes()
                if survivors:
                    raise RuntimeError("启动前无法清理旧进程: " + ", ".join(survivors))
            self._preflight()
            recorder = RecorderApp(self._collection_config(), self.mode, self.telemetry_port,
                                   enable_keyboard=False)
            recorder.c.selected_channels = self._selected_channels()
            thread = threading.Thread(target=self._recorder_target, args=(recorder,),
                                      name="transhandr-recorder", daemon=True)
            with self.lock:
                self.recorder, self.recorder_thread = recorder, thread
            thread.start()
            if not recorder.ready_event.wait(12.0):
                raise RuntimeError("采集器启动超时，请检查相机连接")
            self._start_device_processes()
            with self.lock:
                self.state = "RUNNING"
            threading.Thread(target=self._monitor, name="platform-monitor", daemon=True).start()
        except Exception as exc:
            with self.lock:
                self.state, self.error = "ERROR", str(exc)
            self._stop_runtime(True)
            raise

    @staticmethod
    def _stop_process(process: subprocess.Popen) -> None:
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGINT)
            process.wait(timeout=4)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try: process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                try: process.wait(timeout=1)
                except subprocess.TimeoutExpired: pass
        except ProcessLookupError:
            pass

    @staticmethod
    def _proc_snapshot() -> Dict[int, Dict[str, Any]]:
        snapshot: Dict[int, Dict[str, Any]] = {}
        uid = os.getuid()
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                status = (entry / "status").read_text(encoding="utf-8", errors="replace")
                real_uid = int(next(line for line in status.splitlines()
                                    if line.startswith("Uid:")).split()[1])
                if real_uid != uid:
                    continue
                stat = (entry / "stat").read_text(encoding="utf-8", errors="replace")
                tail = stat[stat.rfind(")") + 2:].split()
                if not tail or tail[0] == "Z":
                    continue
                command = (entry / "cmdline").read_bytes().split(b"\0")
                args = [value.decode("utf-8", "replace") for value in command if value]
                snapshot[int(entry.name)] = {"ppid": int(tail[1]), "args": args}
            except (OSError, StopIteration, ValueError):
                continue
        return snapshot

    @staticmethod
    def _is_managed_command(args: list[str]) -> bool:
        if not args:
            return False
        resolved_tokens = set()
        for token in args:
            if token.startswith("/"):
                try: resolved_tokens.add(str(Path(token).resolve()))
                except OSError: pass
        if resolved_tokens & MANAGED_SCRIPT_PATHS:
            return True
        has_package = "linkerhand_retarget" in args or any(
            "linkerhand_retarget" in token for token in args)
        has_executable = any(
            token in MANAGED_ROS_EXECUTABLES or Path(token).name in MANAGED_ROS_EXECUTABLES
            for token in args)
        return has_package and has_executable

    @classmethod
    def _managed_processes(cls) -> Dict[int, Dict[str, Any]]:
        snapshot = cls._proc_snapshot()
        protected = set()
        pid = os.getpid()
        while pid > 1 and pid not in protected:
            protected.add(pid)
            pid = snapshot.get(pid, {}).get("ppid", 0)
        selected = {pid for pid, info in snapshot.items()
                    if pid not in protected and cls._is_managed_command(info["args"])}
        changed = True
        while changed:
            changed = False
            for pid, info in snapshot.items():
                if pid not in protected and pid not in selected and info["ppid"] in selected:
                    selected.add(pid)
                    changed = True
        return {pid: snapshot[pid] for pid in selected}

    @classmethod
    def _cleanup_stale_processes(cls) -> list[str]:
        managed = cls._managed_processes()
        if not managed:
            return []
        remaining = set(managed)
        for sig, timeout in ((signal.SIGINT, 4.0), (signal.SIGTERM, 3.0),
                             (signal.SIGKILL, 1.0)):
            for pid in list(remaining):
                try: os.kill(pid, sig)
                except ProcessLookupError: remaining.discard(pid)
                except PermissionError: pass
            deadline = time.monotonic() + timeout
            while remaining and time.monotonic() < deadline:
                time.sleep(.05)
                alive = set(cls._proc_snapshot())
                remaining &= alive
            if not remaining:
                break
        return [f"{pid} ({' '.join(managed[pid]['args'][:4])})" for pid in sorted(remaining)]

    def _stop_runtime(self, preserve_error: bool = False) -> None:
        with self.lock:
            was_calibrating = self.state == "CALIBRATING"
            self.expected_stop = True
            if not preserve_error:
                self.state = "STOPPING"
            recorder = self.recorder
        if recorder:
            try: recorder.stop_recording()
            except Exception: pass
        for name in ("inspire_bridge", "o6_bridge", "calibration", "handretarget",
                     "ros_telemetry_adapter", "vision_controller"):
            process = self.processes.get(name)
            if process: self._stop_process(process)
        if recorder:
            recorder.stop_event.set()
        if self.recorder_thread and self.recorder_thread is not threading.current_thread():
            self.recorder_thread.join(timeout=4)
        for handle in self.log_handles:
            try: handle.close()
            except OSError: pass
        survivors = self._cleanup_stale_processes() if self.mode != "mock" else []
        with self.lock:
            self.processes.clear(); self.log_handles.clear()
            self.recorder = None; self.recorder_thread = None
            self.expected_stop = False
            if was_calibrating:
                self.calibration_status.update(
                    state="CANCELLED", instruction="标定已取消", error="")
            if survivors:
                cleanup_error = "遥操作进程未退出: " + ", ".join(survivors)
                self.error = f"{self.error}; {cleanup_error}" if self.error else cleanup_error
                self.state = "ERROR"
            elif not preserve_error:
                self.state, self.error = "STOPPED", ""

    def stop_system(self) -> None:
        with self._replay_boundary():
            self._stop_system_unlocked()

    def _stop_system_unlocked(self) -> None:
        with self.lock:
            stopped = self.state == "STOPPED"
        if stopped:
            survivors = self._cleanup_stale_processes() if self.mode != "mock" else []
            if survivors:
                with self.lock:
                    self.state = "ERROR"
                    self.error = "遥操作进程未退出: " + ", ".join(survivors)
                raise RuntimeError(self.error)
            return
        self._stop_runtime(False)
        with self.lock:
            if self.state == "ERROR":
                raise RuntimeError(self.error)

    def system_control(self, action: str) -> None:
        if action == "start":
            self.start_system()
        elif action == "stop":
            self.stop_system()
        else:
            raise ValueError(f"unsupported system action: {action}")

    def calibration_control(self, action: str) -> None:
        if action == "start":
            self.start_calibration()
        elif action == "cancel":
            self.cancel_calibration()
        else:
            raise ValueError(f"unsupported calibration action: {action}")

    def run(self) -> None:
        self.dashboard.start()
        print(f"[PLATFORM] http://{self.host}:{self.port}/overview")
        try:
            while True: time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            self.dashboard.stop()
            self.stop_system()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("mock", "hardware"), default="hardware")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--telemetry-port", type=int, default=8765)
    parser.add_argument("--output-dir", default="data/multisource")
    args = parser.parse_args()
    TeleoperationPlatform(args.mode, args.host, args.port, args.telemetry_port,
                          ROOT / args.output_dir).run()


if __name__ == "__main__":
    main()
