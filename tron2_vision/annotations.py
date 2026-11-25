"""Lossless, source-indexed mask annotations and a local brush editor."""
import base64
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import io
import json
from pathlib import Path
from urllib.parse import urlparse

import h5py
import numpy as np
from PIL import Image, ImageDraw

CLASSES = {0: "background", 1: "left_o6", 2: "task_objects"}
IMAGE_KEY = "observations/images/d435i_rgb"


def fingerprint(path):
    path = Path(path).resolve()
    s = path.stat()
    return {"path": str(path), "size": s.st_size, "mtime_ns": s.st_mtime_ns}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, obj):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def export_project(data_root, cache_root, episodes, output, samples=5, extra_frames=()):
    if samples < 1 or not episodes or len(set(episodes)) != len(episodes):
        raise ValueError("Choose positive samples and unique episode IDs")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "images").mkdir()
    (output / "labels").mkdir()
    (output / "initial_labels").mkdir()
    extra = {}
    for item in extra_frames:
        ep, row = map(int, item.split(":"))
        if ep not in episodes:
            raise ValueError("Extra frame episode is not selected")
        extra.setdefault(ep, []).append(row)
    project = {"contract": "top-mask-annotations-v1", "classes": CLASSES, "frames": [],
               "note": "SAM3 initial masks are drafts. Reviewer identity is recorded; assistant review is not human ground truth."}
    for episode in episodes:
        source_path = Path(data_root) / f"episode_{episode:06d}.hdf5"
        cache_path = Path(cache_root) / source_path.stem / "foreground.hdf5"
        with h5py.File(source_path, "r") as source, h5py.File(cache_path, "r") as cache:
            source_stamp = fingerprint(source_path)
            if json.loads(cache.attrs["source_stamp"]) != source_stamp:
                raise ValueError("Annotation seed cache does not match original source")
            if not np.array_equal(cache["source/frame_index"][:], np.arange(len(source[IMAGE_KEY]))) or cache["masks/hand"].shape != source[IMAGE_KEY].shape[:3]:
                raise ValueError("Seed masks must preserve all original frame rows")
            for cached, original in [("unified_timestamp_ns", "unified_timestamp_ns"), ("frame_number", "frame_number")]:
                if not np.array_equal(cache[f"source/{cached}"][:], source[f"camera/d435i/{original}"][:]):
                    raise ValueError("Seed cache is not aligned with source")
            valid = np.flatnonzero(source["camera/d435i/valid"][:])
            if not len(valid):
                raise ValueError("No valid camera frames")
            chosen = set(valid[np.linspace(0, len(valid) - 1, min(samples, len(valid))).round().astype(int)].tolist())
            for row in extra.get(episode, []):
                if row not in valid:
                    raise ValueError(f"Invalid source frame {episode}:{row}")
                chosen.add(row)
            for row in sorted(chosen):
                frame_id = f"{episode:06d}_{row:06d}"
                rgb = source[IMAGE_KEY][row]
                label = np.zeros(rgb.shape[:2], np.uint8)
                label[cache["masks/objects"][row].astype(bool)] = 2
                label[cache["masks/hand"][row].astype(bool)] = 1
                Image.fromarray(rgb).save(output / "images" / f"{frame_id}.png")
                for folder in ("labels", "initial_labels"):
                    Image.fromarray(label).save(output / folder / f"{frame_id}.png")
                project["frames"].append({"id": frame_id, "episode": episode, "row": row,
                                          "frame_number": int(source["camera/d435i/frame_number"][row]),
                                          "timestamp_ns": str(int(source["camera/d435i/unified_timestamp_ns"][row])),
                                          "source": source_stamp, "rgb_sha256": sha256(output / "images" / f"{frame_id}.png"),
                                          "status": "draft", "revision": 0, "reviewer": None, "history": [],
                                          "label_sha256": sha256(output / "labels" / f"{frame_id}.png")})
    write_json(output / "project.json", project)
    return project


def load_project(root, check_sources=True):
    root = Path(root)
    project = json.loads((root / "project.json").read_text())
    if project["contract"] != "top-mask-annotations-v1":
        raise ValueError("Unsupported annotation project")
    if check_sources:
        for source in {f["source"]["path"]: f["source"] for f in project["frames"]}.values():
            if fingerprint(source["path"]) != source:
                raise ValueError("Original source has changed")
    return project


def save_label(root, frame_id, label, revision, reviewer, reviewed=False, operations=None):
    root = Path(root)
    project = load_project(root)
    frame = next((f for f in project["frames"] if f["id"] == frame_id), None)
    if frame is None:
        raise ValueError("Unknown frame")
    if frame["revision"] != revision:
        raise ValueError("Revision changed: reload this frame before saving")
    if not reviewer.strip():
        raise ValueError("A reviewer name is required")
    with Image.open(root / "images" / f"{frame_id}.png") as rgb:
        expected_shape = (rgb.height, rgb.width)
    label = np.asarray(label)
    if label.shape != expected_shape or label.dtype != np.uint8 or not np.isin(label, [0, 1, 2]).all():
        raise ValueError("Labels must be uint8 at the original resolution with values 0,1,2")
    path = root / "labels" / f"{frame_id}.png"
    temporary = path.with_suffix(".tmp.png")
    Image.fromarray(label).save(temporary)
    temporary.replace(path)
    frame["revision"] += 1
    frame["status"] = "reviewed" if reviewed else "draft"
    frame["reviewer"] = reviewer
    frame["label_sha256"] = sha256(path)
    frame["history"].append({"revision": frame["revision"], "reviewer": reviewer, "reviewed": reviewed, "operations": operations or ["brush editor save"]})
    write_json(root / "project.json", project)
    return frame


def apply_polygons(root, frame_id, operations, reviewer="assistant", reviewed=False):
    root = Path(root)
    project = load_project(root)
    frame = next(f for f in project["frames"] if f["id"] == frame_id)
    image = Image.open(root / "labels" / f"{frame_id}.png").convert("L")
    draw = ImageDraw.Draw(image)
    for op in operations:
        if op["label"] not in (0, 1, 2) or len(op["points"]) < 3:
            raise ValueError("Invalid polygon")
        if any(x < 0 or x >= image.width or y < 0 or y >= image.height for x, y in op["points"]):
            raise ValueError("Polygon outside original image")
        draw.polygon([tuple(p) for p in op["points"]], fill=op["label"])
    return save_label(root, frame_id, np.asarray(image), frame["revision"], reviewer, reviewed, operations)


def serve(root, port=8091):
    root = Path(root).resolve()
    load_project(root)
    page = Path(__file__).with_name("editor.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def send(self, content, kind="application/json", code=200):
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/":
                return self.send(page, "text/html; charset=utf-8")
            if path == "/api/project":
                return self.send(json.dumps(load_project(root)).encode())
            parts = path.strip("/").split("/")
            if len(parts) == 2 and parts[0] in ("images", "labels", "initial_labels"):
                filename = parts[1]
                ids = {f["id"] + ".png" for f in load_project(root)["frames"]}
                if filename in ids:
                    return self.send((root / parts[0] / filename).read_bytes(), "image/png")
            self.send(b"Not found", "text/plain", 404)

        def do_POST(self):
            try:
                if urlparse(self.path).path != "/api/save":
                    raise ValueError("Unknown action")
                origin = self.headers.get("Origin")
                if origin and urlparse(origin).netloc != self.headers.get("Host"):
                    raise ValueError("Origin mismatch")
                length = int(self.headers.get("Content-Length", 0))
                if not 0 < length <= 4 * 1024 * 1024:
                    raise ValueError("Invalid request size")
                data = json.loads(self.rfile.read(length))
                encoded = data["png"].split(",", 1)[1]
                image = Image.open(io.BytesIO(base64.b64decode(encoded, validate=True)))
                # The browser saves RGBA with R=G=B=class ID.
                label = np.asarray(image)
                if label.ndim == 3:
                    if not np.array_equal(label[..., 0], label[..., 1]) or not np.array_equal(label[..., 0], label[..., 2]):
                        raise ValueError("Invalid label encoding")
                    label = label[..., 0]
                frame = save_label(root, data["id"], label, data["revision"], data["reviewer"], bool(data["reviewed"]))
                self.send(json.dumps(frame).encode())
            except (ValueError, KeyError, OSError) as exc:
                self.send(json.dumps({"error": str(exc)}).encode(), code=400)

    server = HTTPServer(("127.0.0.1", port), Handler)
    print(f"Annotation editor: http://127.0.0.1:{server.server_port}", flush=True)
    server.serve_forever()
