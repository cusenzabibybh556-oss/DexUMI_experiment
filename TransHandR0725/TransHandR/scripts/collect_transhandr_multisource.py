#!/usr/bin/env python3
"""D405-mastered TransHandR episode recorder and live dashboard.

Mock mode is a complete, hardware-free acceptance test.  Hardware mode reads
D405/Vision Pro/VIVE directly.  Final hand commands are accepted as UDP JSON on
--action-port, sent by the recorder-owned hand I/O worker, and stored only after
the device send call succeeds: {"command": [..]}.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import select
import socket
import sys
import termios
import threading
import time
import tty
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import numpy as np
except ImportError as exc:
    raise SystemExit("缺少采集依赖，请先运行: python3 -m pip install --user -r requirements-collector.txt") from exc

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "config"))
sys.path.insert(0, str(ROOT / "LinkerHand"))
sys.path.insert(0, str(ROOT / "inspire_hands"))
sys.path.insert(0, str(REPO_ROOT / "project1yu" / "pyvut"))

from multisource_collection import CollectionConfig
try:
    from multisource_core import (ClockMapper, EpisodeWriter, SampleBuffer, TimedSample,
                                  interpolate_pose, interpolate_vector, latest_before, nearest,
                                  visionpro_hand_points)
    from realtime_multisource_dashboard import DashboardServer
    from telemetry_protocol import PROTOCOL_V2, parse_v2_packet, tactile_summary
except ImportError as exc:
    if exc.name in {"h5py", "PIL"}:
        raise SystemExit("缺少采集依赖，请先运行: python3 -m pip install --user -r requirements-collector.txt") from exc
    raise


TRACKER_STATE_NAMES = {
    0: "NO_IMAGES_YET",
    1: "NOT_INITIALIZED",
    2: "OK",
    3: "LOST",
    4: "RECENTLY_LOST",
    5: "SYSTEM_NOT_READY",
}
TRACKER_MAP_STATE_NAMES = {
    0: "MAP_NOT_CHECKED",
    1: "MAP_EXIST",
    2: "MAP_NOT_EXIST",
    3: "MAP_REBUILT",
    4: "MAP_SAVE_OK",
    5: "MAP_SAVE_FAIL",
    6: "MAP_REUSE_OK",
}

RECORD_CHANNELS = [
    {"id": "d405", "label": "D405 RGB-D", "path": "observations/images + observations/depth", "description": "主时间轴，必选", "required": True, "recommended": True},
    {"id": "d435i", "label": "D435i RGB-D", "path": "observations/images/d435i_rgb + depth", "description": "头部相机", "recommended": True},
    {"id": "vision_hand", "label": "Vision Pro Hand", "path": "observations/right_hand/*", "description": "25 点三维手部", "recommended": True},
    {"id": "operator_exoskeleton", "label": "LinkerForce Raw", "path": "operator/exoskeleton/*", "description": "21 维外骨骼原始关节", "recommended": True},
    {"id": "hand_state", "label": "Hand State", "path": "dexterous_hand/*", "description": "灵巧手真实反馈", "recommended": True},
    {"id": "action", "label": "Action", "path": "action/*", "description": "实际下发命令", "recommended": True},
    {"id": "tactile", "label": "O6 Tactile", "path": "dexterous_hand/{tactile,palm}_stream/*", "description": "原频率五指及掌心矩阵触觉", "recommended": True},
    {"id": "inspire_force", "label": "Inspire Force", "path": "dexterous_hand/force_stream/*", "description": "原频率六路实际力", "recommended": True},
    {"id": "vive_tracker", "label": "VIVE Tracker Pose", "path": "vive_tracker/*", "description": "位置与旋转四元数", "recommended": True},
]


def monotonic_sample(value: Any, valid: bool = True) -> TimedSample:
    now = time.monotonic_ns()
    return TimedSample(value, float(now), now, now, valid)


class MockCamera:
    def __init__(self, config: CollectionConfig): self.c, self.frame, self.next_ns = config, 0, time.monotonic_ns()
    def start(self) -> Dict[str, Any]:
        return {"serial": "MOCK-D405", "depth_scale": .001, "rgb_intrinsics": {"fx": 600, "fy": 600, "ppx": self.c.image_width/2, "ppy": self.c.image_height/2}, "depth_intrinsics": {"fx": 600, "fy": 600}}
    def read(self) -> Dict[str, Any]:
        self.next_ns += int(1e9 / self.c.camera_fps)
        delay = self.next_ns - time.monotonic_ns()
        if delay > 0: time.sleep(delay / 1e9)
        h, w, n = self.c.image_height, self.c.image_width, self.frame
        yy, xx = np.mgrid[:h, :w]
        rgb = np.stack(((xx+n*3)%256, (yy*2+n)%256, ((xx+yy)//2+n*2)%256), axis=-1).astype(np.uint8)
        depth = (450 + 250*np.sin(xx/75+n/12) + 100*np.cos(yy/55)).clip(1, 2000).astype(np.uint16)
        host = time.monotonic_ns(); self.frame += 1
        return {"rgb": rgb, "depth": depth, "frame_number": self.frame, "device_timestamp": host/1e6, "unified_timestamp_ns": host, "host_receive_timestamp_ns": host, "valid": True}
    def stop(self) -> None: pass


class MockHeadCamera:
    def __init__(self, config: CollectionConfig):
        self.c, self.frame, self.next_ns = config, 0, time.monotonic_ns()
    def start(self) -> Dict[str, Any]:
        return {"serial": "MOCK-D435I", "depth_scale": .001,
                "rgb_intrinsics": {"fx": 610, "fy": 610, "ppx": self.c.head_image_width/2, "ppy": self.c.head_image_height/2},
                "depth_intrinsics": {"fx": 610, "fy": 610}}
    def read(self) -> Dict[str, Any]:
        self.next_ns += int(1e9 / self.c.head_camera_fps)
        delay = self.next_ns - time.monotonic_ns()
        if delay > 0: time.sleep(delay / 1e9)
        h, w, n = self.c.head_image_height, self.c.head_image_width, self.frame
        yy, xx = np.mgrid[:h, :w]
        rgb = np.stack(((yy+n*2)%256, (xx+n*4)%256, ((xx+2*yy)//3+n)%256), axis=-1).astype(np.uint8)
        depth = (700 + 180*np.sin(xx/90+n/15) + 80*np.cos(yy/60)).clip(1, 4000).astype(np.uint16)
        host = time.monotonic_ns(); self.frame += 1
        return {"rgb": rgb, "depth": depth, "frame_number": self.frame, "device_timestamp": host/1e6,
                "unified_timestamp_ns": host, "host_receive_timestamp_ns": host, "valid": True}
    def stop(self) -> None: pass


class D405Camera:
    def __init__(self, config: CollectionConfig): self.c, self.pipeline, self.align, self.mapper = config, None, None, ClockMapper("ms")
    def start(self) -> Dict[str, Any]:
        import pyrealsense2 as rs
        self.rs = rs; self.pipeline = rs.pipeline(); cfg = rs.config()
        if self.c.d405_serial: cfg.enable_device(self.c.d405_serial)
        cfg.enable_stream(rs.stream.depth, self.c.image_width, self.c.image_height, rs.format.z16, self.c.camera_fps)
        cfg.enable_stream(rs.stream.color, self.c.image_width, self.c.image_height, rs.format.rgb8, self.c.camera_fps)
        profile = self.pipeline.start(cfg); self.align = rs.align(rs.stream.color) if self.c.depth_aligned_to_color else None
        dev = profile.get_device(); sensor = dev.first_depth_sensor(); depth_scale = sensor.get_depth_scale()
        for _ in range(15): self.pipeline.wait_for_frames(1500)
        color = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        depth = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
        intr = lambda x: {"width": x.width, "height": x.height, "fx": x.fx, "fy": x.fy, "ppx": x.ppx, "ppy": x.ppy, "model": str(x.model), "coeffs": list(x.coeffs)}
        return {"serial": dev.get_info(rs.camera_info.serial_number), "depth_scale": depth_scale, "rgb_intrinsics": intr(color), "depth_intrinsics": intr(depth)}
    def read(self) -> Dict[str, Any]:
        frames = self.pipeline.wait_for_frames(1500)
        if self.align: frames = self.align.process(frames)
        color, depth = frames.get_color_frame(), frames.get_depth_frame(); host = time.monotonic_ns()
        if not color or not depth: raise RuntimeError("D405 returned incomplete RGB-D frameset")
        device_ts = float(color.get_timestamp())
        return {"rgb": np.asanyarray(color.get_data()).copy(), "depth": np.asanyarray(depth.get_data()).copy(), "frame_number": int(color.get_frame_number()), "device_timestamp": device_ts, "unified_timestamp_ns": self.mapper.map(device_ts, host), "host_receive_timestamp_ns": host, "valid": True}
    def stop(self) -> None:
        if self.pipeline: self.pipeline.stop()


class D435IHeadCamera:
    """Independent head-mounted RGB-D stream; synchronized to D405 in software."""
    def __init__(self, config: CollectionConfig):
        self.c, self.pipeline, self.align, self.mapper = config, None, None, ClockMapper("ms")
    def start(self) -> Dict[str, Any]:
        import pyrealsense2 as rs
        self.rs = rs; self.pipeline = rs.pipeline(); cfg = rs.config()
        if self.c.head_camera_serial: cfg.enable_device(self.c.head_camera_serial)
        cfg.enable_stream(rs.stream.depth, self.c.head_image_width, self.c.head_image_height,
                          rs.format.z16, self.c.head_camera_fps)
        cfg.enable_stream(rs.stream.color, self.c.head_image_width, self.c.head_image_height,
                          rs.format.rgb8, self.c.head_camera_fps)
        profile = self.pipeline.start(cfg)
        self.align = rs.align(rs.stream.color) if self.c.head_depth_aligned_to_color else None
        dev = profile.get_device()
        name = dev.get_info(rs.camera_info.name)
        if "D435" not in name.upper():
            self.pipeline.stop(); self.pipeline = None
            raise RuntimeError(f"head camera serial resolved to {name}, expected D435i")
        depth_scale = dev.first_depth_sensor().get_depth_scale()
        for _ in range(15): self.pipeline.wait_for_frames(1500)
        color = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        depth = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
        intr = lambda x: {"width": x.width, "height": x.height, "fx": x.fx, "fy": x.fy,
                          "ppx": x.ppx, "ppy": x.ppy, "model": str(x.model), "coeffs": list(x.coeffs)}
        return {"name": name, "serial": dev.get_info(rs.camera_info.serial_number),
                "depth_scale": depth_scale, "rgb_intrinsics": intr(color), "depth_intrinsics": intr(depth)}
    def read(self) -> Dict[str, Any]:
        frames = self.pipeline.wait_for_frames(1500)
        if self.align: frames = self.align.process(frames)
        color, depth = frames.get_color_frame(), frames.get_depth_frame(); host = time.monotonic_ns()
        if not color or not depth: raise RuntimeError("D435i returned incomplete RGB-D frameset")
        device_ts = float(color.get_timestamp())
        return {"rgb": np.asanyarray(color.get_data()).copy(), "depth": np.asanyarray(depth.get_data()).copy(),
                "frame_number": int(color.get_frame_number()), "device_timestamp": device_ts,
                "unified_timestamp_ns": self.mapper.map(device_ts, host),
                "host_receive_timestamp_ns": host, "valid": True}
    def stop(self) -> None:
        if self.pipeline: self.pipeline.stop()


class RecorderApp:
    def __init__(self, config: CollectionConfig, mode: str, action_port: int,
                 enable_keyboard: bool = True):
        self.c, self.mode, self.action_port = config, mode, action_port
        self.enable_keyboard = enable_keyboard
        self.buffers = {key: SampleBuffer() for key in (
            "right_hand", "operator", "hand_state", "action", "tactile", "palm", "force", "tracker", "head_camera"
        )}
        self.stop_event = threading.Event(); self.writer: Optional[EpisodeWriter] = None; self.writer_lock = threading.Lock()
        self.ready_event = threading.Event()
        self.camera = MockCamera(config) if mode == "mock" else D405Camera(config)
        self.head_camera = ((MockHeadCamera(config) if mode == "mock" else D435IHeadCamera(config))
                            if config.enable_head_camera else None)
        self.camera_meta: Dict[str, Any] = {}; self.head_camera_meta: Dict[str, Any] = {}
        self.episode_index = self._next_episode_index(); self.start_ns = 0
        self.recording_config_path = self.c.episode_dir / ".recording_channels.json"
        self.dashboard: Optional[DashboardServer] = None; self.threads = []
        self.status: Dict[str, Any] = {"recording": False, "episode_index": self.episode_index, "elapsed_s": 0.0, "fps": 0.0, "frames": 0, "hand_type": config.hand_type, "control_source": config.control_source, "hand_side": config.hand_side, "vision_hand_side": config.vision_hand_side, "schema_version": "quality-first-v3.2", "valid": {}, "sync": {}, "last_file": "", "rgb": None, "depth": None, "head_rgb": None, "head_depth": None, "head_preview": {"enabled": config.enable_head_camera, "valid": False, "error": ""}, "right_hand": {}, "operator": [], "hand_state": [], "action": [], "tactile": {}, "palm": {}, "force": [], "tracker": {"valid": False, "pos": [0,0,0], "state": "WAITING_FOR_TRACKER", "tracking_status": None, "map_state": "MAP_NOT_CHECKED", "packet_count": 0, "pose_count": 0}}
        self.status_lock = threading.Lock(); self._frame_times = []
        self._stream_times = {key: deque() for key in (
            "d405", "right_hand", "operator", "hand_state", "action",
            "tactile", "palm", "force", "tracker", "head_camera",
        )}
        self._stream_last = {key: math.nan for key in self._stream_times}
        self._stream_times_lock = threading.Lock()
        self.tracker_diagnostics = dict(self.status["tracker"])
        self._mock_tactile_sequence = 0
        self.c.selected_channels = self._load_selected_channels()

    def _next_episode_index(self) -> int:
        self.c.episode_dir.mkdir(parents=True, exist_ok=True); indices = []
        for path in self.c.episode_dir.glob("episode_*.hdf5"):
            try: indices.append(int(path.stem.split("_")[-1]))
            except ValueError: pass
        return max(indices, default=-1) + 1

    def _load_selected_channels(self) -> list:
        defaults = [item["id"] for item in RECORD_CHANNELS
                    if item.get("required") or self._channel_available(item["id"])]
        try:
            saved = json.loads(self.recording_config_path.read_text(encoding="utf-8"))["channels"]
            selected = set(saved) | {"d405"}
            return [item["id"] for item in RECORD_CHANNELS
                    if item["id"] in selected and self._channel_available(item["id"])]
        except (OSError, KeyError, TypeError, json.JSONDecodeError):
            return defaults

    def _save_selected_channels(self) -> None:
        temporary = self.recording_config_path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"channels": self.c.selected_channels}, ensure_ascii=False), encoding="utf-8")
        temporary.replace(self.recording_config_path)

    def start_recording(self) -> None:
        with self.writer_lock:
            if self.writer: return
            if self.mode == "hardware":
                required = []
                selected = set(self.c.selected_channels)
                if "vision_hand" in selected: required.append("right_hand")
                if "operator_exoskeleton" in selected: required.append("operator")
                if "hand_state" in selected: required.append("hand_state")
                if "action" in selected: required.append("action")
                if "tactile" in selected: required.append("tactile")
                if "d435i" in selected: required.append("head_camera")
                if "vive_tracker" in selected: required.append("tracker")
                missing = [name for name in required
                           if self.buffers[name].latest() is None or not self.buffers[name].latest().valid]
                if missing:
                    raise RuntimeError("录制前检查失败，尚无有效数据源: " + ", ".join(missing))
            path = self.c.episode_dir / f"episode_{self.episode_index:06d}.hdf5"
            metadata = dict(self.camera_meta)
            metadata["d435i"] = dict(self.head_camera_meta)
            self.writer = EpisodeWriter(path, self.c, metadata, self.episode_index); self.start_ns = time.monotonic_ns()
            with self.status_lock: self.status.update(recording=True, frames=0, elapsed_s=0.0)
            print(f"[REC] episode {self.episode_index:06d} -> {path}")

    def stop_recording(self) -> None:
        with self.writer_lock:
            if not self.writer: return
            path = str(self.writer.path); summary = self.writer.close(); self.writer = None
            print("[SAVED]", path, json.dumps(summary, ensure_ascii=False, indent=2))
            self.episode_index += 1
            with self.status_lock: self.status.update(recording=False, episode_index=self.episode_index, last_file=path)

    def control(self, action: str) -> None:
        (self.start_recording if action == "start" else self.stop_recording)()

    def _channel_available(self, channel: str) -> bool:
        return {
            "d405": True,
            "d435i": self.c.enable_head_camera,
            "vision_hand": self.c.enable_right_hand,
            "operator_exoskeleton": self.c.enable_exoskeleton_operator,
            "hand_state": self.c.enable_hand_io,
            "action": self.c.enable_hand_io,
            "tactile": self.c.enable_tactile,
            "inspire_force": self.c.enable_inspire_force,
            "vive_tracker": self.c.enable_vive_tracker,
        }.get(channel, False)

    def recording_config(self, _profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        selected = set(self.c.selected_channels)
        return {
            "channels": [{**item, "selected": item["id"] in selected,
                          "available": self._channel_available(item["id"])} for item in RECORD_CHANNELS],
            "output_dir": str(self.c.episode_dir),
            "task_name": self.c.task_name,
            "hand_type": self.c.hand_type,
            "next_file": f"episode_{self.episode_index:06d}.hdf5",
            "max_duration_s": self.c.max_episode_duration_s,
            "max_size_gb": self.c.max_episode_size_gb,
        }

    def update_recording_config(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        with self.writer_lock:
            if self.writer: raise RuntimeError("录制中不能修改保存通道")
            known = {item["id"] for item in RECORD_CHANNELS}
            selected = set(payload.get("channels", [])) & known
            selected.add("d405")
            unavailable = [name for name in selected if not self._channel_available(name)]
            if unavailable: raise RuntimeError("数据源未启用: " + ", ".join(sorted(unavailable)))
            self.c.selected_channels = [item["id"] for item in RECORD_CHANNELS if item["id"] in selected]
            self._save_selected_channels()
        return self.recording_config()

    def _record_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Keep the stable schema while replacing unselected streams with invalid placeholders."""
        selected, result = set(self.c.selected_channels), dict(row)
        if "d435i" not in selected and "camera/d435i/valid" in result:
            result.update({
                "observations/images/d435i_rgb": np.zeros_like(result["observations/images/d435i_rgb"]),
                "observations/depth/d435i_depth": np.zeros_like(result["observations/depth/d435i_depth"]),
                "camera/d435i/frame_number": -1, "camera/d435i/device_timestamp": np.nan,
                "camera/d435i/unified_timestamp_ns": -1, "camera/d435i/host_receive_timestamp_ns": -1,
                "camera/d435i/valid": False, "sync/d435i_dt_ms": np.nan,
            })
        if "vision_hand" not in selected:
            result.update({
                "observations/right_hand/world_keypoints": np.full((25, 3), np.nan, np.float32),
                "observations/right_hand/relative_keypoints": np.full((25, 3), np.nan, np.float32),
                "observations/right_hand/valid": False, "observations/right_hand/source_timestamp": np.nan,
                "observations/right_hand/unified_timestamp_ns": -1,
                "observations/right_hand/host_receive_timestamp_ns": -1, "sync/right_hand_dt_ms": np.nan,
            })
        if "hand_state" not in selected:
            result.update({
                "dexterous_hand/joint_positions": np.full(int(self.c.hand_spec["state_dim"]), np.nan, np.float32),
                "dexterous_hand/state_valid": False, "dexterous_hand/source_timestamp": np.nan,
                "dexterous_hand/unified_timestamp_ns": -1, "dexterous_hand/host_receive_timestamp_ns": -1,
                "sync/hand_state_interpolation_gap_ms": np.nan,
            })
        if "operator_exoskeleton" not in selected:
            result.update({
                "operator/exoskeleton/joint_positions": np.full(21, np.nan, np.float32),
                "operator/exoskeleton/valid": False,
                "operator/exoskeleton/source_timestamp_ns": -1,
                "operator/exoskeleton/unified_timestamp_ns": -1,
                "operator/exoskeleton/host_receive_timestamp_ns": -1,
                "sync/operator_interpolation_gap_ms": np.nan,
            })
        if "action" not in selected:
            result.update({
                "action/command": np.full(int(self.c.hand_spec["action_dim"]), np.nan, np.float32),
                "action/valid": False, "action/command_timestamp_ns": -1, "sync/action_age_ms": np.nan,
            })
        if "tactile" not in selected:
            result.update({
                "dexterous_hand/tactile/index": -1,
                "dexterous_hand/tactile/age_ms": np.nan,
                "dexterous_hand/tactile/valid": False,
                "dexterous_hand/palm/index": -1,
                "dexterous_hand/palm/age_ms": np.nan,
                "dexterous_hand/palm/valid": False,
            })
        if "vive_tracker" not in selected:
            result.update({
                "vive_tracker/pos": np.full((1, 3), np.nan, np.float32),
                "vive_tracker/quat": np.full((1, 4), np.nan, np.float32), "vive_tracker/valid": [False],
                "vive_tracker/source_timestamp": [np.nan], "vive_tracker/unified_timestamp_ns": [-1],
                "vive_tracker/host_receive_timestamp_ns": [-1], "sync/tracker_interpolation_gap_ms": np.nan,
            })
        return result

    def _note_stream(self, name: str) -> None:
        now = time.monotonic()
        with self._stream_times_lock:
            times = self._stream_times[name]
            times.append(now)
            self._stream_last[name] = now
            while times and now - times[0] > 1.0:
                times.popleft()

    def _add_sample(self, name: str, sample: TimedSample) -> None:
        self.buffers[name].add(sample)
        self._note_stream(name)

    def _stream_metrics(self) -> Dict[str, Dict[str, float]]:
        now = time.monotonic()
        result = {}
        with self._stream_times_lock:
            for name, times in self._stream_times.items():
                while times and now - times[0] > 1.0:
                    times.popleft()
                hz = ((len(times) - 1) / (times[-1] - times[0])
                      if len(times) > 1 and times[-1] > times[0] else 0.0)
                result[name] = {
                    "hz": hz,
                    "age_ms": ((now - self._stream_last[name]) * 1000.0
                               if math.isfinite(self._stream_last[name]) else math.nan),
                }
        return result

    def _add_tactile_sample(
        self, matrix: Any, source_timestamp_ns: int, timestamp_ns: int,
        host_receive_timestamp_ns: int, sequence: int,
    ) -> bool:
        """Store one real tactile event and expose its derived live summary."""
        summary = tactile_summary(matrix, self.c.tactile_contact_threshold)
        stream_index = -1
        stream_epoch = -1
        with self.writer_lock:
            if self.writer is not None and "tactile" in self.c.selected_channels:
                try:
                    stream_index = self.writer.append_tactile(
                        matrix, timestamp_ns, sequence)
                    stream_epoch = self.writer.episode_idx
                except ValueError as exc:
                    print(f"[TACTILE] rejected: {exc}")
                    return False
        self._add_sample("tactile", TimedSample(
            summary, float(source_timestamp_ns), int(timestamp_ns),
            int(host_receive_timestamp_ns), True, int(sequence),
            stream_index, stream_epoch,
        ))
        return True

    def _add_palm_sample(
        self, matrix: Any, source_timestamp_ns: int, timestamp_ns: int,
        host_receive_timestamp_ns: int, sequence: int,
    ) -> bool:
        stream_index = -1
        stream_epoch = -1
        with self.writer_lock:
            if self.writer is not None and "tactile" in self.c.selected_channels:
                try:
                    stream_index = self.writer.append_palm(matrix, timestamp_ns, sequence)
                    stream_epoch = self.writer.episode_idx
                except ValueError as exc:
                    print(f"[PALM] rejected: {exc}")
                    return False
        self._add_sample("palm", TimedSample(
            np.asarray(matrix, np.int16), float(source_timestamp_ns), int(timestamp_ns),
            int(host_receive_timestamp_ns), True, int(sequence), stream_index, stream_epoch,
        ))
        return True

    def _add_force_sample(self, message, host_receive_ns: int) -> None:
        with self.writer_lock:
            if self.writer is not None and "inspire_force" in self.c.selected_channels:
                try:
                    self.writer.append_force(message.values, message.unified_timestamp_ns,
                                             message.sequence)
                except ValueError as exc:
                    print(f"[FORCE] rejected: {exc}")
                    return
        self._add_sample("force", TimedSample(
            message.values, float(message.source_timestamp_ns),
            message.unified_timestamp_ns, host_receive_ns,
            sequence=message.sequence))

    def public_status(self) -> Dict[str, Any]:
        with self.status_lock:
            def clean(v: Any) -> Any:
                if isinstance(v, np.ndarray): return clean(v.tolist())
                if isinstance(v, np.generic): return clean(v.item())
                if isinstance(v, float) and not math.isfinite(v): return None
                if isinstance(v, dict): return {k: clean(x) for k, x in v.items()}
                if isinstance(v, (list, tuple)): return [clean(x) for x in v]
                return v
            raw = dict(self.status)
            rgb, depth = raw.pop("rgb", None), raw.pop("depth", None)
            head_rgb, head_depth = raw.pop("head_rgb", None), raw.pop("head_depth", None)
            result = clean(raw)
            # Image endpoints need ndarray objects; /api/status explicitly omits them.
            result["rgb"], result["depth"] = rgb, depth
            result["head_rgb"], result["head_depth"] = head_rgb, head_depth
            return result

    def _thread(self, target: Any, name: str) -> None:
        t = threading.Thread(target=target, name=name, daemon=True); t.start(); self.threads.append(t)

    def _keyboard_worker(self) -> None:
        """Single-key episode control: R=start, S=stop, Q=quit."""
        fd = sys.stdin.fileno(); previous = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while not self.stop_event.is_set():
                readable, _, _ = select.select([sys.stdin], [], [], .2)
                if not readable: continue
                key = sys.stdin.read(1).lower()
                if key == "r":
                    try: self.start_recording()
                    except Exception as exc: print(f"[REC] {exc}")
                elif key == "s": self.stop_recording()
                elif key == "q": self.stop_event.set()
        finally: termios.tcsetattr(fd, termios.TCSADRAIN, previous)

    def _mock_sources(self) -> None:
        phase = 0.0; js, ja = int(self.c.hand_spec["state_dim"]), int(self.c.hand_spec["action_dim"])
        next_tactile_ns = time.monotonic_ns()
        while not self.stop_event.is_set():
            phase += .035; now = time.monotonic_ns()
            base = np.zeros((25, 3), np.float32)
            for finger in range(5):
                for joint in range(5):
                    i = finger*5+joint
                    base[i] = [(finger-2)*.025, joint*.022, .015*math.sin(phase+finger+joint*.3)]
            base[:, 0] += .03*math.sin(phase*.4)
            if self.c.enable_right_hand:
                self._add_sample("right_hand", TimedSample({"world": base + [0.2, 0.1, 0.5], "relative": base-base[0]}, float(now), now, now))
            state = (np.sin(phase + np.arange(js)*.2)*.5+.5) * (1000 if self.c.hand_type == "inspire" else 255)
            self._add_sample("hand_state", TimedSample(state.astype(np.float32), float(now), now, now))
            if self.c.enable_exoskeleton_operator:
                operator = ((np.sin(phase + np.arange(21) * .17) + 1) * 127.5).astype(np.float32)
                self._add_sample("operator", TimedSample(operator, float(now), now, now))
            if int(phase*10) % 3 == 0:
                command = np.resize(state, ja).astype(np.float32)
                self._add_sample("action", TimedSample(command, float(now), now, now))
            if self.c.enable_tactile and now >= next_tactile_ns:
                matrix = np.maximum(0, 20 * np.sin(phase + np.arange(360).reshape(5, 12, 6) * .03))
                self._add_tactile_sample(
                    np.rint(matrix), now, now, now, self._mock_tactile_sequence)
                self._mock_tactile_sequence += 1
                while next_tactile_ns <= now:
                    next_tactile_ns += 100_000_000
            if self.c.enable_vive_tracker:
                angle = phase*.1; pose = {"pos": np.array([.15*math.sin(angle), .1, .15*math.cos(angle)], np.float32), "quat": np.array([math.cos(angle/2), 0, math.sin(angle/2), 0], np.float32)}
                self._add_sample("tracker", TimedSample(pose, float(now), now, now))
            time.sleep(1/90)

    def _vision_worker(self) -> None:
        from avp_stream import VisionProStreamer
        stream = VisionProStreamer(ip=self.c.avp_ip)
        fingers_key = f"{self.c.vision_hand_side}_fingers"
        wrist_key = f"{self.c.vision_hand_side}_wrist"
        while not self.stop_event.is_set():
            try:
                packet = stream.get_latest(); fingers = packet.get(fingers_key)
                if fingers is None or len(fingers) < 25: time.sleep(.002); continue
                host = time.monotonic_ns()
                finger_matrices = np.asarray(fingers[:25], np.float64)
                wrist = np.asarray(packet.get(wrist_key), np.float64)
                wrist_matrix = wrist[0] if wrist.shape == (1, 4, 4) else wrist
                if wrist_matrix.shape != (4, 4):
                    time.sleep(.002); continue
                world, relative = visionpro_hand_points(finger_matrices, wrist_matrix)
                raw_ts = packet.get(f"{self.c.vision_hand_side}_fingers_timestamp", packet.get("timestamp", host))
                # AVP library currently has no documented clock contract; host monotonic is used unless ns is explicit.
                unified = int(raw_ts) if isinstance(raw_ts, (int, np.integer)) and raw_ts > 1e15 else host
                self._add_sample("right_hand", TimedSample({"world": world, "relative": relative}, float(raw_ts), unified, host))
            except Exception as exc:
                print(f"[VISION] {exc}"); time.sleep(.1)

    def _head_camera_worker(self) -> None:
        """Continuously acquire D435i RGB-D frames into the alignment buffer."""
        assert self.head_camera is not None
        last_error_log = 0.0
        while not self.stop_event.is_set():
            try:
                frame = self.head_camera.read()
                self._add_sample("head_camera", TimedSample(
                    frame, float(frame["device_timestamp"]), frame["unified_timestamp_ns"],
                    frame["host_receive_timestamp_ns"], frame["valid"]
                ))
                with self.status_lock:
                    self.status["head_preview"]["error"] = ""
            except Exception as exc:
                with self.status_lock:
                    self.status["head_preview"]["error"] = str(exc)
                now = time.monotonic()
                if now - last_error_log >= 1.0:
                    print(f"[D435i] {exc}")
                    last_error_log = now
                time.sleep(.02)

    def _tracker_worker(self) -> None:
        from pyvut import UltimateTrackerAPI
        mapper = ClockMapper("ms")
        def pose_callback(pose: Any) -> None:
            if pose.tracker_index != self.c.tracker_index: return
            host = time.monotonic_ns(); valid = pose.tracking_status == 2
            value = {"pos": np.asarray(pose.position, np.float32), "quat": np.asarray(pose.rotation, np.float32)}
            self._add_sample("tracker", TimedSample(value, float(pose.timestamp_ms), mapper.map(pose.timestamp_ms, host), host, valid))
            with self.status_lock:
                self.tracker_diagnostics.update(
                    tracking_status=int(pose.tracking_status),
                    state=TRACKER_STATE_NAMES.get(int(pose.tracking_status), f"UNKNOWN_{pose.tracking_status}"),
                    mac=pose.mac,
                    pose_count=int(self.tracker_diagnostics.get("pose_count", 0)) + 1,
                )
        with UltimateTrackerAPI(mode=self.c.tracker_mode) as api:
            api.add_pose_callback(pose_callback)
            group = api.tracker_group
            while not self.stop_event.wait(.25):
                map_states = list(group.tracker_map_state)
                map_state = next((state for state in map_states if state), 0)
                packet_count = int(sum(group.poses_recvd))
                with self.status_lock:
                    self.tracker_diagnostics.update(
                        map_state=TRACKER_MAP_STATE_NAMES.get(map_state, f"MAP_UNKNOWN_{map_state}"),
                        packet_count=packet_count,
                    )
                    if packet_count and not self.tracker_diagnostics.get("pose_count"):
                        self.tracker_diagnostics["state"] = "NO_POSE_STREAM"

    def _hand_io_worker(self) -> None:
        """Direct mode: own the physical connection and accept command UDP."""
        if self.c.hand_type in ("l10", "l21"):
            from LinkerHand.linker_hand_api import LinkerHandApi
            hand = LinkerHandApi(hand_joint=self.c.hand_type.upper(), hand_type=self.c.hand_side, can=self.c.hand_can_interface)
            send, read, close = lambda x: hand.finger_move(list(map(float, x))), hand.get_state, lambda: getattr(hand, "close_can", lambda: None)()
        else:
            from inspire_hand import InspireHand
            hand = InspireHand(port=self.c.hand_serial_port); hand.open()
            send = lambda x: hand.modbus.write_multiple_registers(1486, list(map(int, x)))
            read, close = hand.get_finger_angles, hand.close
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); sock.bind(("127.0.0.1", self.action_port)); sock.setblocking(False)
        last_error_log = 0.0
        try:
            while not self.stop_event.is_set():
                readable, _, _ = select.select([sock], [], [], .005)
                if readable:
                    try:
                        command = np.asarray(json.loads(sock.recv(65535))["command"], np.float32)
                        if command.size != int(self.c.hand_spec["action_dim"]): raise ValueError("wrong action dimension")
                        send(command); sent = time.monotonic_ns()
                        self._add_sample("action", TimedSample(command, float(sent), sent, sent))
                    except Exception as exc: print(f"[HAND command] rejected: {exc}")
                try:
                    value = np.asarray(read(), np.float32); received = time.monotonic_ns()
                    if value.size != int(self.c.hand_spec["state_dim"]): raise ValueError(f"state dimension={value.size}")
                    self._add_sample("hand_state", TimedSample(value, float(received), received, received))
                except Exception as exc:
                    now = time.monotonic()
                    if now - last_error_log >= 1.0:
                        print(f"[HAND state] {exc}")
                        last_error_log = now
        finally: sock.close(); close()

    def _telemetry_worker(self) -> None:
        """Receive legacy VR and versioned device-owner telemetry."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("127.0.0.1", self.action_port)); sock.settimeout(.2)
        expected_state = int(self.c.hand_spec["state_dim"])
        expected_action = int(self.c.hand_spec["action_dim"])
        try:
            while not self.stop_event.is_set():
                try: packet = json.loads(sock.recv(65535))
                except socket.timeout: continue
                except Exception as exc:
                    print(f"[HAND telemetry] invalid packet: {exc}"); continue
                host = time.monotonic_ns()
                if packet.get("protocol") == PROTOCOL_V2:
                    try:
                        message = parse_v2_packet(packet, {
                            "operator": 21,
                            "action": expected_action,
                            "state": expected_state,
                            "force": 6,
                        })
                        if message.hand_side != self.c.hand_side:
                            continue
                        if message.robot_model and message.robot_model != self.c.hand_type.lower():
                            continue
                        timestamp = message.unified_timestamp_ns
                        value = message.values
                        if message.kind == "operator" and self.c.enable_exoskeleton_operator:
                            self._add_sample("operator", TimedSample(
                                value, float(message.source_timestamp_ns), timestamp, host))
                        elif message.kind == "action":
                            self._add_sample("action", TimedSample(
                                value, float(message.source_timestamp_ns), timestamp, host))
                        elif message.kind == "state":
                            self._add_sample("hand_state", TimedSample(
                                value, float(message.source_timestamp_ns), timestamp, host))
                        elif message.kind == "tactile" and self.c.enable_tactile:
                            self._add_tactile_sample(
                                value, message.source_timestamp_ns, timestamp, host,
                                message.sequence)
                        elif message.kind == "palm" and self.c.enable_tactile:
                            self._add_palm_sample(
                                value, message.source_timestamp_ns, timestamp, host,
                                message.sequence)
                        elif message.kind == "force" and self.c.enable_inspire_force:
                            self._add_force_sample(message, host)
                    except (KeyError, TypeError, ValueError) as exc:
                        print(f"[HAND telemetry v2] rejected: {exc}")
                    continue
                if packet.get("right_hand_world") is not None:
                    packet_side = packet.get("vision_hand_side", "right")
                    if packet_side == self.c.vision_hand_side:
                        world = np.asarray(packet["right_hand_world"], np.float32)
                        relative = np.asarray(packet.get("right_hand_relative", world - world[0]), np.float32)
                        if world.shape == (25, 3) and relative.shape == (25, 3):
                            source_ts = float(packet.get("right_hand_source_timestamp", host))
                            timestamp = int(packet.get("right_hand_unified_timestamp_ns", host))
                            self._add_sample("right_hand", TimedSample(
                                {"world": world, "relative": relative}, source_ts, timestamp, host
                            ))
                if packet.get("command") is not None:
                    command = np.asarray(packet["command"], np.float32)
                    if command.size == expected_action:
                        timestamp = int(packet.get("command_timestamp_ns", host))
                        self._add_sample("action", TimedSample(command, float(timestamp), timestamp, host))
                if packet.get("state") is not None:
                    state = np.asarray(packet["state"], np.float32)
                    if state.size == expected_state:
                        timestamp = int(packet.get("state_timestamp_ns", host))
                        self._add_sample("hand_state", TimedSample(state, float(timestamp), timestamp, host))
        finally: sock.close()

    def _aligned_row(self, camera: Dict[str, Any]) -> Dict[str, Any]:
        c, t = self.c, camera["unified_timestamp_ns"]
        rh, rh_dt = nearest(self.buffers["right_hand"].snapshot(), t, c.right_hand_max_gap_ms)
        operator, operator_gap = interpolate_vector(
            self.buffers["operator"].snapshot(), t, c.operator_max_gap_ms)
        state, state_gap = interpolate_vector(self.buffers["hand_state"].snapshot(), t, c.hand_state_max_gap_ms)
        action, action_age = latest_before(self.buffers["action"].snapshot(), t, c.action_max_age_ms)
        active_epoch = self.writer.episode_idx if self.writer is not None else -1
        tactile_samples = tuple(
            sample for sample in self.buffers["tactile"].snapshot()
            if sample.stream_index >= 0 and sample.stream_epoch == active_epoch
        )
        tactile, tactile_age = latest_before(
            tactile_samples, t, c.tactile_max_gap_ms)
        if tactile is None:
            tactile_age = math.nan
        palm_samples = tuple(
            sample for sample in self.buffers["palm"].snapshot()
            if sample.stream_index >= 0 and sample.stream_epoch == active_epoch
        )
        palm, palm_age = latest_before(palm_samples, t, c.tactile_max_gap_ms)
        if palm is None:
            palm_age = math.nan
        tracker, tracker_gap = interpolate_pose(self.buffers["tracker"].snapshot(), t, c.tracker_max_gap_ms) if c.enable_vive_tracker else (None, math.nan)
        head, head_dt = nearest(self.buffers["head_camera"].snapshot(), t, c.head_camera_max_gap_ms) if c.enable_head_camera else (None, math.nan)
        nan25 = np.full((25,3), np.nan, np.float32); js, ja = int(c.hand_spec["state_dim"]), int(c.hand_spec["action_dim"])
        rhv = rh.value if rh else {"world": nan25, "relative": nan25}
        tv = tracker.value if tracker else {"pos": np.full(3,np.nan,np.float32), "quat": np.full(4,np.nan,np.float32)}
        row = {
            "timestamps/master_timestamp_ns": t, "observations/images/d405_rgb": camera["rgb"], "observations/depth/d405_depth": camera["depth"],
            "observations/right_hand/world_keypoints": rhv["world"], "observations/right_hand/relative_keypoints": rhv["relative"], "observations/right_hand/valid": rh is not None,
            "observations/right_hand/source_timestamp": rh.source_timestamp if rh else np.nan, "observations/right_hand/unified_timestamp_ns": rh.unified_timestamp_ns if rh else -1, "observations/right_hand/host_receive_timestamp_ns": rh.host_receive_timestamp_ns if rh else -1,
            "operator/exoskeleton/joint_positions": operator.value if operator else np.full(21, np.nan, np.float32), "operator/exoskeleton/valid": operator is not None, "operator/exoskeleton/source_timestamp_ns": int(operator.source_timestamp) if operator else -1, "operator/exoskeleton/unified_timestamp_ns": operator.unified_timestamp_ns if operator else -1, "operator/exoskeleton/host_receive_timestamp_ns": operator.host_receive_timestamp_ns if operator else -1,
            "dexterous_hand/joint_positions": state.value if state else np.full(js,np.nan,np.float32), "dexterous_hand/state_valid": state is not None, "dexterous_hand/source_timestamp": state.source_timestamp if state else np.nan, "dexterous_hand/unified_timestamp_ns": state.unified_timestamp_ns if state else -1, "dexterous_hand/host_receive_timestamp_ns": state.host_receive_timestamp_ns if state else -1,
            "action/command": action.value if action else np.full(ja,np.nan,np.float32), "action/valid": action is not None, "action/command_timestamp_ns": action.unified_timestamp_ns if action else -1,
            "dexterous_hand/tactile/index": tactile.stream_index if tactile else -1, "dexterous_hand/tactile/age_ms": tactile_age, "dexterous_hand/tactile/valid": tactile is not None,
            "dexterous_hand/palm/index": palm.stream_index if palm else -1, "dexterous_hand/palm/age_ms": palm_age, "dexterous_hand/palm/valid": palm is not None,
            "vive_tracker/pos": tv["pos"][None], "vive_tracker/quat": tv["quat"][None], "vive_tracker/valid": [tracker is not None], "vive_tracker/source_timestamp": [tracker.source_timestamp if tracker else np.nan], "vive_tracker/unified_timestamp_ns": [tracker.unified_timestamp_ns if tracker else -1], "vive_tracker/host_receive_timestamp_ns": [tracker.host_receive_timestamp_ns if tracker else -1],
            "camera/d405/frame_number": camera["frame_number"], "camera/d405/device_timestamp": camera["device_timestamp"], "camera/d405/host_receive_timestamp_ns": camera["host_receive_timestamp_ns"], "camera/d405/valid": camera["valid"],
            "sync/right_hand_dt_ms": rh_dt, "sync/operator_interpolation_gap_ms": operator_gap, "sync/hand_state_interpolation_gap_ms": state_gap, "sync/tracker_interpolation_gap_ms": tracker_gap, "sync/action_age_ms": action_age,
        }
        if c.enable_head_camera:
            hv = head.value if head else {"rgb": np.zeros((c.head_image_height, c.head_image_width, 3), np.uint8),
                                         "depth": np.zeros((c.head_image_height, c.head_image_width), np.uint16),
                                         "frame_number": -1, "device_timestamp": np.nan,
                                         "unified_timestamp_ns": -1, "host_receive_timestamp_ns": -1}
            row.update({
                "observations/images/d435i_rgb": hv["rgb"], "observations/depth/d435i_depth": hv["depth"],
                "camera/d435i/frame_number": hv["frame_number"], "camera/d435i/device_timestamp": hv["device_timestamp"],
                "camera/d435i/unified_timestamp_ns": hv["unified_timestamp_ns"],
                "camera/d435i/host_receive_timestamp_ns": hv["host_receive_timestamp_ns"],
                "camera/d435i/valid": head is not None, "sync/d435i_dt_ms": head_dt,
            })
        return row

    def _update_status(self, row: Dict[str, Any]) -> None:
        now = time.monotonic(); self._frame_times.append(now); self._frame_times = [x for x in self._frame_times if now-x < 1]
        latest_tactile = self.buffers["tactile"].latest()
        latest_palm = self.buffers["palm"].latest()
        latest_force = self.buffers["force"].latest()
        tactile_live_age = ((time.monotonic_ns() - latest_tactile.unified_timestamp_ns) / 1e6
                            if latest_tactile else math.nan)
        tactile_live = (latest_tactile is not None and tactile_live_age >= 0
                        and tactile_live_age <= self.c.tactile_max_gap_ms)
        tactile_value = latest_tactile.value if tactile_live else {}
        palm_live_age = ((time.monotonic_ns() - latest_palm.unified_timestamp_ns) / 1e6
                         if latest_palm else math.nan)
        palm_live = (latest_palm is not None and 0 <= palm_live_age <= self.c.tactile_max_gap_ms)
        latest_action = self.buffers["action"].latest()
        action_live = bool(row["action/valid"]) or (
            latest_action is not None and self.c.action_max_age_ms <= 0
        )
        action_value = (row["action/command"] if bool(row["action/valid"])
                        else latest_action.value if latest_action is not None
                        else [])
        force_live = (latest_force is not None and
                      0 <= time.monotonic_ns() - latest_force.unified_timestamp_ns <= 150_000_000)
        valid = {"d405": bool(row["camera/d405/valid"]), "d435i": bool(row.get("camera/d435i/valid", False)), "right_hand": bool(row["observations/right_hand/valid"]), "operator": bool(row["operator/exoskeleton/valid"]), "hand_state": bool(row["dexterous_hand/state_valid"]), "action": action_live, "tactile": tactile_live, "palm": palm_live, "force": force_live, "tracker": bool(row["vive_tracker/valid"][0])}
        streams = self._stream_metrics()
        latest_head = self.buffers["head_camera"].latest()
        head_age = streams["head_camera"]["age_ms"]
        head_preview_valid = (self.c.enable_head_camera and latest_head is not None
                              and latest_head.valid and 0 <= head_age <= 250)
        with self.status_lock:
            tracker_status = dict(self.tracker_diagnostics)
            tracker_status.update(valid=valid["tracker"], pos=row["vive_tracker/pos"][0], quat=row["vive_tracker/quat"][0])
            self.status["head_preview"]["valid"] = head_preview_valid
            self.status.update(fps=float(len(self._frame_times)), streams=streams, rgb=row["observations/images/d405_rgb"], depth=row["observations/depth/d405_depth"], head_rgb=latest_head.value["rgb"] if head_preview_valid else None, head_depth=latest_head.value["depth"] if head_preview_valid else None, valid=valid,
                right_hand={"relative_keypoints": row["observations/right_hand/relative_keypoints"] if valid["right_hand"] else []}, operator=row["operator/exoskeleton/joint_positions"] if valid["operator"] else [], hand_state=row["dexterous_hand/joint_positions"] if valid["hand_state"] else [], action=action_value if valid["action"] else [], tactile={"matrix": tactile_value.get("matrix", []), "max": tactile_value.get("max", []), "mean": tactile_value.get("mean", []), "contact_area": tactile_value.get("contact_area", [])}, palm={"matrix": latest_palm.value if palm_live else []}, tracker=tracker_status,
                force=latest_force.value if force_live else [],
                sync={"d435i_dt_ms": row.get("sync/d435i_dt_ms", math.nan), "right_hand_dt_ms": row["sync/right_hand_dt_ms"], "operator_interpolation_gap_ms": row["sync/operator_interpolation_gap_ms"], "hand_state_interpolation_gap_ms": row["sync/hand_state_interpolation_gap_ms"], "tactile_dt_ms": tactile_live_age, "palm_dt_ms": palm_live_age, "tracker_interpolation_gap_ms": row["sync/tracker_interpolation_gap_ms"], "action_age_ms": row["sync/action_age_ms"]})
            if self.writer: self.status.update(frames=self.writer.count, elapsed_s=(time.monotonic_ns()-self.start_ns)/1e9)

    def _recording_limit_reason(self) -> Optional[str]:
        if not self.writer: return None
        elapsed = (time.monotonic_ns() - self.start_ns) / 1e9
        if self.c.max_episode_duration_s > 0 and elapsed >= self.c.max_episode_duration_s:
            return f"达到 episode 时长上限 {self.c.max_episode_duration_s:g}s"
        size_gb = self.writer.path.stat().st_size / 1024 ** 3
        if self.c.max_episode_size_gb > 0 and size_gb >= self.c.max_episode_size_gb:
            return f"达到 episode 文件上限 {self.c.max_episode_size_gb:g}GiB"
        return None

    def run(self, auto_record: bool = False, duration: float = 0.0) -> None:
        self.camera_meta = self.camera.start()
        if self.head_camera is not None:
            self.head_camera_meta = self.head_camera.start()
            self._thread(self._head_camera_worker, "d435i-head-camera")
        if self.mode == "mock": self._thread(self._mock_sources, "mock-sources")
        else:
            if self.c.enable_right_hand and self.c.right_hand_source == "direct":
                self._thread(self._vision_worker, "vision-pro")
            needs_telemetry = ((self.c.enable_right_hand and self.c.right_hand_source == "udp") or
                               (self.c.enable_hand_io and self.c.hand_source == "udp") or
                               self.c.enable_exoskeleton_operator or self.c.enable_tactile
                               or self.c.enable_inspire_force)
            if needs_telemetry:
                self._thread(self._telemetry_worker, "multisource-telemetry")
            if self.c.enable_hand_io and self.c.hand_source == "direct":
                self._thread(self._hand_io_worker, "hand-io")
            if self.c.enable_vive_tracker: self._thread(self._tracker_worker, "vive-tracker")
        if self.c.enable_realtime_dashboard:
            self.dashboard = DashboardServer(
                self.c.dashboard_host, self.c.dashboard_port, self.public_status, self.control,
                self.recording_config, self.update_recording_config, Path(self.c.output_dir)
            ); self.dashboard.start()
            print(f"[DASHBOARD] http://{self.c.dashboard_host}:{self.c.dashboard_port}")
        self.ready_event.set()
        if self.enable_keyboard and sys.stdin.isatty():
            self._thread(self._keyboard_worker, "keyboard"); print("[KEYS] R=start  S=stop/save  Q=quit")
        if auto_record: self.start_recording()
        pending = deque(); deadline = time.monotonic()+duration if duration else None
        alignment_lag_ns = int(self.c.alignment_lag_ms * 1e6)
        try:
            while not self.stop_event.is_set():
                current = self.camera.read()
                self._note_stream("d405")
                pending.append(current)
                while pending and current["unified_timestamp_ns"] - pending[0]["unified_timestamp_ns"] >= alignment_lag_ns:
                    row = self._aligned_row(pending.popleft()); self._update_status(row)
                    with self.writer_lock:
                        if self.writer: self.writer.append(self._record_row(row))
                    limit_reason = self._recording_limit_reason()
                    if limit_reason:
                        print(f"[REC SAFETY] {limit_reason}，自动停止并保存")
                        self.stop_recording()
                if deadline and time.monotonic() >= deadline: break
        except KeyboardInterrupt: pass
        finally:
            self.stop_event.set()
            cleanup = [("episode", self.stop_recording), ("D405", self.camera.stop)]
            if self.head_camera is not None: cleanup.append(("D435i", self.head_camera.stop))
            if self.dashboard: cleanup.append(("dashboard", self.dashboard.stop))
            for label, stop in cleanup:
                try: stop()
                except KeyboardInterrupt: print(f"[STOP] {label} 停止时收到二次 Ctrl+C，继续退出")
                except Exception as exc: print(f"[STOP] {label}: {exc}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("mock", "hardware"), default="mock")
    p.add_argument("--control-source", choices=("vision_pro", "linkerforce"), default="vision_pro")
    p.add_argument("--hand-type", choices=("inspire", "l10", "l21", "o6"), default="l21")
    p.add_argument("--hand-serial-port", default="/dev/ttyUSB0")
    p.add_argument("--hand-can-interface", default="can0")
    p.add_argument("--hand-source", choices=("udp", "direct"), default="udp", help="udp: receive from controller owner; direct: recorder opens device")
    p.add_argument("--right-hand-source", choices=("udp", "direct"), default="udp", help="udp: receive right_fingers from controller; direct: recorder opens AVP stream")
    p.add_argument("--vision-hand-side", choices=("left", "right"), default="right", help="Vision Pro input hand; output channels remain unchanged")
    p.add_argument("--avp-ip", default="10.227.197.16"); p.add_argument("--d405-serial", default="")
    p.add_argument("--width", type=int, default=640); p.add_argument("--height", type=int, default=480); p.add_argument("--camera-fps", type=int, default=30)
    p.add_argument("--enable-head-camera", action="store_true", help="Capture a head-mounted D435i RGB-D stream")
    p.add_argument("--head-camera-serial", default="", help="D435i serial number")
    p.add_argument("--head-width", type=int, default=640); p.add_argument("--head-height", type=int, default=480); p.add_argument("--head-camera-fps", type=int, default=30)
    p.add_argument("--head-camera-max-gap-ms", type=float, default=45.0, help="Maximum D435i-to-D405 software alignment gap")
    p.add_argument("--output-dir", default="data/multisource"); p.add_argument("--task-name", default="transhandr_demo")
    p.add_argument("--dashboard-host", default="127.0.0.1"); p.add_argument("--dashboard-port", type=int, default=8080); p.add_argument("--no-dashboard", action="store_true")
    p.add_argument("--enable-vive", action="store_true"); p.add_argument("--tracker-mode", choices=("DONGLE_USB", "TRACKER_USB"), default="DONGLE_USB")
    p.add_argument("--tracker-index", type=int, choices=range(5), default=0, help="VIVE logical index (first connected tracker is 0)")
    p.add_argument("--disable-vision", action="store_true", help="Staged bring-up: save right-hand as invalid")
    p.add_argument("--disable-hand", action="store_true", help="Staged bring-up: save hand state/action as invalid")
    p.add_argument("--enable-exoskeleton", action="store_true", help="Receive 21-D LinkerForce operator telemetry")
    p.add_argument("--enable-tactile", action="store_true", help="Receive O6 matrix tactile telemetry")
    p.add_argument("--action-port", type=int, default=8765, help="Hardware mode UDP port for final command JSON")
    p.add_argument("--max-episode-duration", type=float, default=300, help="Safety stop in seconds; 0 disables")
    p.add_argument("--max-episode-size-gb", type=float, default=5, help="Safety stop in GiB; 0 disables")
    p.add_argument("--hand-state-max-gap-ms", type=float, default=None, help="Maximum interpolation gap; default: 35 ms")
    p.add_argument("--record", action="store_true", help="Start recording immediately"); p.add_argument("--duration", type=float, default=0, help="Stop after N seconds; 0 runs until Ctrl+C")
    return p.parse_args()


def main() -> None:
    a = parse_args(); hand_gap = a.hand_state_max_gap_ms if a.hand_state_max_gap_ms is not None else 35.0
    if a.control_source == "linkerforce" and (
        a.hand_type not in {"o6", "inspire"}
        or (a.hand_type == "o6" and a.vision_hand_side != "left")
    ):
        raise SystemExit("LinkerForce 支持 Inspire 左右手，以及 O6 左手")
    c = CollectionConfig(control_source=a.control_source, target_hand_model=a.hand_type, avp_ip=a.avp_ip, enable_right_hand=not a.disable_vision, right_hand_source=a.right_hand_source, vision_hand_side=a.vision_hand_side, enable_hand_io=not a.disable_hand, hand_source=a.hand_source, hand_type=a.hand_type, hand_side=a.vision_hand_side, hand_serial_port=a.hand_serial_port, hand_can_interface=a.hand_can_interface, enable_exoskeleton_operator=a.enable_exoskeleton or a.control_source == "linkerforce", enable_tactile=a.enable_tactile, d405_serial=a.d405_serial, image_width=a.width, image_height=a.height, camera_fps=a.camera_fps, enable_head_camera=a.enable_head_camera, head_camera_serial=a.head_camera_serial, head_image_width=a.head_width, head_image_height=a.head_height, head_camera_fps=a.head_camera_fps, head_camera_max_gap_ms=a.head_camera_max_gap_ms, output_dir=a.output_dir, task_name=a.task_name, enable_realtime_dashboard=not a.no_dashboard, dashboard_host=a.dashboard_host, dashboard_port=a.dashboard_port, enable_vive_tracker=a.enable_vive, tracker_mode=a.tracker_mode, tracker_index=a.tracker_index, hand_state_max_gap_ms=hand_gap, max_episode_duration_s=a.max_episode_duration, max_episode_size_gb=a.max_episode_size_gb)
    RecorderApp(c, a.mode, a.action_port).run(a.record, a.duration)


if __name__ == "__main__": main()
