"""Realtime capture console and local episode browser."""

from __future__ import annotations

import io
import ipaddress
import json
import math
import shutil
import threading
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, Optional
from urllib.parse import parse_qs, urlparse

import h5py
import numpy as np
from PIL import Image


ASSET_DIR = Path(__file__).with_name("dashboard")
O6_MODEL_ROOT = (
    Path(__file__).resolve().parents[3]
    / "linkerhand_telop_python-master/ros2/src/linkerhand_retarget/linkerhand_retarget"
    / "assets/robots/hands/linker_hand"
).resolve()
INSPIRE_MODEL_ROOT = (
    Path(__file__).resolve().parents[1] / "dataset/robot/inspire_hand"
).resolve()
INSPIRE_MODEL_DIRS = {"left": INSPIRE_MODEL_ROOT, "right": INSPIRE_MODEL_ROOT}
STREAM_DATASETS = {
    "d405_rgb": "observations/images/d405_rgb",
    "d405_depth": "observations/depth/d405_depth",
    "d435i_rgb": "observations/images/d435i_rgb",
    "d435i_depth": "observations/depth/d435i_depth",
}


def _jpeg(array: Any, depth: bool = False) -> bytes:
    arr = np.asarray(array)
    if depth and arr.ndim == 2:
        valid = arr[arr > 0]
        hi = np.percentile(valid, 95) if valid.size else 1
        gray = np.clip(arr / max(float(hi), 1) * 255, 0, 255).astype(np.uint8)
        arr = np.stack([gray // 5, np.minimum(255, gray * 2), 255 - gray], axis=-1).astype(np.uint8)
    if arr.ndim != 3:
        arr = np.zeros((240, 320, 3), np.uint8)
    output = io.BytesIO()
    Image.fromarray(np.asarray(arr, np.uint8)).save(output, "JPEG", quality=80)
    return output.getvalue()


class DashboardServer:
    def __init__(
        self,
        host: str,
        port: int,
        state_provider: Callable[[], Dict[str, Any]],
        control: Callable[[str], None],
        config_provider: Callable[[], Dict[str, Any]],
        config_update: Callable[[Dict[str, Any]], Dict[str, Any]],
        episode_root: Path,
        system_config_provider: Optional[Callable[[], Dict[str, Any]]] = None,
        system_config_update: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        system_control: Optional[Callable[[str], None]] = None,
        calibration_control: Optional[Callable[[str], None]] = None,
        robot_replay: Any = None,
    ):
        self.host, self.port = host, port
        self.state_provider, self.control = state_provider, control
        self.config_provider, self.config_update = config_provider, config_update
        self.episode_root = Path(episode_root).resolve()
        self.system_config_provider = system_config_provider
        self.system_config_update = system_config_update
        self.system_control = system_control
        self.calibration_control = calibration_control
        self.robot_replay = robot_replay
        self.wrist_calibration = None
        self.httpd: Optional[ThreadingHTTPServer] = None

    def _resolve_episode(self, episode_id: str) -> Path:
        path = (self.episode_root / episode_id).resolve()
        if self.episode_root not in path.parents or path.suffix != ".hdf5":
            raise ValueError("非法 episode 路径")
        if not path.is_file():
            raise FileNotFoundError(episode_id)
        return path

    @staticmethod
    def _attr(value: Any) -> Any:
        if isinstance(value, np.generic): return value.item()
        if isinstance(value, bytes): return value.decode("utf-8", "replace")
        if isinstance(value, float) and not math.isfinite(value): return None
        return value

    def _episode_info(self, path: Path) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "id": path.relative_to(self.episode_root).as_posix(), "name": path.name,
            "size_bytes": path.stat().st_size,
            "modified": datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat(),
        }
        try:
            with h5py.File(path, "r") as f:
                frames = len(f.get("timestamps/master_timestamp_ns", []))
                selected_raw = self._attr(f.attrs.get("selected_channels", "[]"))
                try: selected = json.loads(selected_raw)
                except (TypeError, json.JSONDecodeError): selected = []
                quality_group = f.get("quality")
                quality = ({key: self._attr(value) for key, value in quality_group.attrs.items()}
                           if quality_group is not None else {})
                fps = self._attr(f.attrs.get("actual_capture_hz", quality.get("actual_capture_hz")))
                duration = quality.get("duration_s", 0.0)
                ratios = {key[:-12]: value for key, value in quality.items()
                          if key.endswith("_valid_ratio")}
                names = {"vision_hand": "right_hand", "vive_tracker": "tracker"}
                active_ratios = [ratios.get(names.get(name, name), 1.0) for name in selected if name != "d405"]
                if not selected: active_ratios = list(ratios.values())
                worst_ratio = min(active_ratios, default=1.0)
                target = float(f.attrs.get("camera_fps", 30))
                status = "bad" if not fps or fps < target * .8 or worst_ratio < .8 else (
                    "warn" if fps < target * .93 or worst_ratio < .95 else "ok")
                info.update(
                    frames=frames, duration_s=duration, fps=fps, selected_channels=selected,
                    valid_ratios=ratios, quality=status,
                    schema=self._attr(f.attrs.get("schema_version", "")),
                    hand_type=self._attr(f.attrs.get("hand_type", "")),
                    vision_hand_side=self._attr(f.attrs.get("vision_hand_side", "")),
                    streams=[key for key, name in STREAM_DATASETS.items() if name in f and (
                        not selected or key.startswith("d405") or "d435i" in selected)],
                )
        except (OSError, ValueError) as exc:
            info.update(frames=0, duration_s=0, fps=None, quality="bad", error=str(exc), streams=[])
        return info

    def _episodes(self) -> list[Dict[str, Any]]:
        if not self.episode_root.exists(): return []
        paths = sorted(self.episode_root.rglob("episode_*.hdf5"), key=lambda p: p.stat().st_mtime, reverse=True)
        return [self._episode_info(path) for path in paths]

    def start(self) -> None:
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_: Any) -> None: pass

            def _send(self, body: bytes, mime: str, status: int = 200,
                      extra: Optional[Dict[str, str]] = None) -> None:
                try:
                    self.send_response(status); self.send_header("Content-Type", mime)
                    self.send_header("Content-Length", str(len(body)))
                    headers = {"Cache-Control": "no-store", **(extra or {})}
                    for key, value in headers.items(): self.send_header(key, value)
                    self.end_headers(); self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError): pass

            def _json(self, value: Any, status: int = 200) -> None:
                self._send(json.dumps(value, ensure_ascii=False, allow_nan=False).encode(),
                           "application/json; charset=utf-8", status)

            def _payload(self) -> Dict[str, Any]:
                length = int(self.headers.get("Content-Length", 0))
                if length > 65536:
                    raise ValueError("请求内容过大")
                value = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(value, dict):
                    raise ValueError("请求必须是 JSON 对象")
                return value

            def _local_robot_request(self) -> bool:
                try:
                    local = ipaddress.ip_address(self.client_address[0]).is_loopback
                except ValueError:
                    local = False
                origin = urlparse(self.headers.get("Origin", ""))
                return bool(local and origin.scheme == "http"
                            and origin.netloc == self.headers.get("Host")
                            and self.headers.get("Content-Type", "").split(";", 1)[0]
                            == "application/json")

            def do_GET(self) -> None:
                parsed = urlparse(self.path); path, query = parsed.path, parse_qs(parsed.query)
                if path in ("/", "/overview", "/data", "/wrist-calibration"):
                    return self._send((ASSET_DIR / "index.html").read_bytes(), "text/html; charset=utf-8")
                assets = {"/assets/dashboard.css": ("dashboard.css", "text/css; charset=utf-8"),
                          "/assets/dashboard.js": ("dashboard.js", "application/javascript; charset=utf-8"),
                          "/assets/wrist_calibration.js": ("wrist_calibration.js", "application/javascript; charset=utf-8"),
                          "/assets/inspire_hand_viewer.js": ("inspire_hand_viewer.js", "application/javascript; charset=utf-8"),
                          "/assets/o6_hand_viewer.js": ("o6_hand_viewer.js", "application/javascript; charset=utf-8"),
                          "/assets/vendor/three.module.min.js": ("vendor/three.module.min.js", "application/javascript"),
                          "/assets/vendor/GLTFLoader.js": ("vendor/GLTFLoader.js", "application/javascript"),
                          "/assets/vendor/BufferGeometryUtils.js": ("vendor/BufferGeometryUtils.js", "application/javascript"),
                          "/assets/vendor/STLLoader.js": ("vendor/STLLoader.js", "application/javascript"),
                          "/assets/vendor/OrbitControls.js": ("vendor/OrbitControls.js", "application/javascript")}
                if path in assets:
                    name, mime = assets[path]
                    return self._send((ASSET_DIR / name).read_bytes(), mime, extra={
                        "Cache-Control": "public, max-age=86400",
                    })
                if path.startswith("/assets/o6/"):
                    relative = path.removeprefix("/assets/o6/")
                    side, separator, asset_name = relative.partition("/")
                    model_root = (O6_MODEL_ROOT / f"o6_{side}") if side in {"left", "right"} and separator else None
                    model_asset = (model_root / asset_name).resolve() if model_root else None
                    if (model_asset is None or model_root not in model_asset.parents
                            or not model_asset.is_file()):
                        return self._send(b"not found", "text/plain", HTTPStatus.NOT_FOUND)
                    mime = "model/stl" if model_asset.suffix.lower() == ".stl" else "application/xml"
                    return self._send(model_asset.read_bytes(), mime, extra={
                        "Cache-Control": "public, max-age=86400",
                    })
                if path.startswith("/assets/inspire/"):
                    relative = path.removeprefix("/assets/inspire/")
                    side, separator, asset_name = relative.partition("/")
                    model_root = INSPIRE_MODEL_DIRS.get(side)
                    model_asset = (model_root / asset_name).resolve() if model_root and separator else None
                    if (model_asset is None or model_root not in model_asset.parents
                            or not model_asset.is_file()):
                        return self._send(b"not found", "text/plain", HTTPStatus.NOT_FOUND)
                    suffix = model_asset.suffix.lower()
                    mime = {".glb": "model/gltf-binary", ".urdf": "application/xml"}.get(
                        suffix, "application/octet-stream")
                    return self._send(model_asset.read_bytes(), mime, extra={
                        "Cache-Control": "public, max-age=86400",
                    })
                state = owner.state_provider()
                if path == "/api/status":
                    return self._json({k: v for k, v in state.items()
                                       if k not in ("rgb", "depth", "head_rgb", "head_depth")})
                if path == "/api/recording/config":
                    profile = {key: values[0] for key, values in query.items() if values}
                    return self._json(owner.config_provider(profile))
                if path == "/api/system/config" and owner.system_config_provider:
                    return self._json(owner.system_config_provider())
                if path == "/api/episodes": return self._json({"episodes": owner._episodes()})
                if path == "/api/wrist-calibration/status":
                    if owner.wrist_calibration is None:
                        return self._json({"enabled": False, "phase": "unavailable"})
                    return self._json({"local": ipaddress.ip_address(self.client_address[0]).is_loopback,
                                       **owner.wrist_calibration.status()})
                if path == "/api/wrist-calibration/download":
                    if owner.wrist_calibration is None:
                        return self._json({"error": "腕部采集未启用"}, 404)
                    try:
                        return self._send(owner.wrist_calibration.export(), "application/x-ndjson", extra={
                            "Content-Disposition": 'attachment; filename="wrist_calibration_samples.jsonl"'})
                    except (ValueError, FileNotFoundError, OSError) as exc:
                        return self._json({"error": str(exc)}, 404)
                if path == "/api/wrist-calibration/solution-download":
                    if owner.wrist_calibration is None:
                        return self._json({"error": "腕部采集未启用"}, 404)
                    try:
                        return self._send(owner.wrist_calibration.export_solution(), "application/zip", extra={
                            "Content-Disposition": 'attachment; filename="wrist_calibration_solution.zip"'})
                    except (ValueError, FileNotFoundError, OSError) as exc:
                        return self._json({"error": str(exc)}, 404)
                if path == "/api/robot-replay/status":
                    if owner.robot_replay is None:
                        return self._json({"enabled": False, "phase": "unavailable"})
                    return self._json({"enabled": True,
                                       "local": ipaddress.ip_address(self.client_address[0]).is_loopback,
                                       **owner.robot_replay.status()})
                if path == "/api/disk":
                    base = owner.episode_root if owner.episode_root.exists() else owner.episode_root.parent
                    usage = shutil.disk_usage(base)
                    return self._json({"total": usage.total, "used": usage.used, "free": usage.free,
                                       "path": str(owner.episode_root)})
                live_images = {"/api/rgb.jpg": ("rgb", False), "/api/depth.jpg": ("depth", True),
                               "/api/d435i_rgb.jpg": ("head_rgb", False),
                               "/api/d435i_depth.jpg": ("head_depth", True)}
                if path in live_images:
                    key, depth = live_images[path]; arr = state.get(key)
                    if arr is None: arr = np.zeros((240, 320, 3), np.uint8)
                    return self._send(_jpeg(arr, depth), "image/jpeg")
                if path == "/api/episode":
                    try: return self._json(owner._episode_info(owner._resolve_episode(query.get("id", [""])[0])))
                    except (ValueError, FileNotFoundError) as exc: return self._json({"error": str(exc)}, 404)
                if path == "/api/episode/frame.jpg":
                    try:
                        episode = owner._resolve_episode(query.get("id", [""])[0])
                        stream = query.get("stream", [""])[0]; dataset = STREAM_DATASETS[stream]
                        with h5py.File(episode, "r") as f:
                            ds = f[dataset]; index = max(0, min(int(query.get("index", [0])[0]), len(ds) - 1))
                            body = _jpeg(ds[index], stream.endswith("depth"))
                        return self._send(body, "image/jpeg")
                    except (KeyError, ValueError, FileNotFoundError, OSError) as exc:
                        return self._json({"error": str(exc)}, 404)
                if path == "/api/episode/download":
                    try:
                        episode = owner._resolve_episode(query.get("id", [""])[0])
                        self.send_response(200); self.send_header("Content-Type", "application/x-hdf5")
                        self.send_header("Content-Length", str(episode.stat().st_size))
                        self.send_header("Content-Disposition", f'attachment; filename="{episode.name}"')
                        self.end_headers()
                        with episode.open("rb") as source:
                            while chunk := source.read(1024 * 1024): self.wfile.write(chunk)
                        return
                    except (ValueError, FileNotFoundError, BrokenPipeError): return
                self._send(b"not found", "text/plain", HTTPStatus.NOT_FOUND)

            def do_PUT(self) -> None:
                path = urlparse(self.path).path
                if path == "/api/system/config" and owner.system_config_update:
                    try: return self._json({"ok": True, "config": owner.system_config_update(self._payload())})
                    except Exception as exc: return self._json({"ok": False, "error": str(exc)}, 409)
                if path != "/api/recording/config":
                    return self._json({"error": "not found"}, 404)
                try: self._json({"ok": True, "config": owner.config_update(self._payload())})
                except Exception as exc: self._json({"ok": False, "error": str(exc)}, 409)

            def do_POST(self) -> None:
                path = urlparse(self.path).path
                if path.startswith("/api/wrist-calibration/"):
                    if owner.wrist_calibration is None:
                        return self._json({"error": "腕部采集未启用"}, 404)
                    if not self._local_robot_request():
                        return self._json({"error": "设备连接与采样仅允许本机同源网页操作"}, 403)
                    try:
                        return self._json(owner.wrist_calibration.command(path.rsplit("/", 1)[-1], self._payload()))
                    except (ValueError, RuntimeError, OSError) as exc:
                        return self._json({"error": str(exc)}, 409)
                if path.startswith("/api/robot-replay/"):
                    if owner.robot_replay is None:
                        return self._json({"error": "真机回放服务未启用"}, 404)
                    if not self._local_robot_request():
                        return self._json({"error": "真机回放仅允许本机同源网页操作"}, 403)
                    try:
                        payload = self._payload()
                        if path == "/api/robot-replay/preview":
                            result = owner.robot_replay.preview(payload)
                        elif path == "/api/robot-replay/start":
                            result = owner.robot_replay.start(payload)
                        elif path == "/api/robot-replay/restore":
                            result = owner.robot_replay.restore(payload)
                        elif path == "/api/robot-replay/confirm":
                            result = owner.robot_replay.confirm(payload)
                        elif path == "/api/robot-replay/cancel":
                            result = owner.robot_replay.cancel()
                        else:
                            return self._json({"error": "bad request"}, 400)
                        return self._json(result)
                    except (ValueError, RuntimeError, FileNotFoundError,
                            OSError) as exc:
                        return self._json({"error": str(exc)}, 409)
                action = path.rsplit("/", 1)[-1]
                if path.startswith("/api/calibration/") and owner.calibration_control:
                    if action not in ("start", "cancel"):
                        return self._json({"error": "bad request"}, 400)
                    try: owner.calibration_control(action); return self._json({"ok": True})
                    except Exception as exc: return self._json({"ok": False, "error": str(exc)}, 409)
                if path.startswith("/api/system/") and owner.system_control:
                    if action not in ("start", "stop"): return self._json({"error": "bad request"}, 400)
                    try: owner.system_control(action); return self._json({"ok": True})
                    except Exception as exc: return self._json({"ok": False, "error": str(exc)}, 409)
                if action not in ("start", "stop"): return self._json({"error": "bad request"}, 400)
                try: owner.control(action); self._json({"ok": True})
                except Exception as exc: self._json({"ok": False, "error": str(exc)}, 409)

        self.httpd = ThreadingHTTPServer((self.host, self.port), Handler)
        threading.Thread(target=self.httpd.serve_forever, name="dashboard", daemon=True).start()

    def stop(self) -> None:
        if self.wrist_calibration is not None:
            self.wrist_calibration.shutdown()
        if self.robot_replay is not None:
            self.robot_replay.shutdown()
        if self.httpd: self.httpd.shutdown(); self.httpd.server_close()
