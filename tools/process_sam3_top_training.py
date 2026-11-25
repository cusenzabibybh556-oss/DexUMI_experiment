#!/usr/bin/env python3
"""Export causal top foreground caches; original episodes remain read-only.

Run with the SAM3 virtualenv. Exported caches require review before training.
"""
import argparse
import json
from pathlib import Path
import time

import cv2
import h5py
import numpy as np
from PIL import Image, ImageDraw
import torch

from sam3_top_stream import TopForegroundStream
from try_sam3_top_foreground import summarize

CONTRACT = "sam3-top-causal-tracking-fusion-v2"

def stamp(path):
    s = path.stat()
    return {"path": str(path.resolve()), "size": s.st_size, "mtime_ns": s.st_mtime_ns}


def process(stream, source_path, root):
    destination = root / source_path.stem
    destination.mkdir()
    stream.reset()
    rows = []
    writer = None
    source_stamp = stamp(source_path)
    with h5py.File(source_path, "r") as source, h5py.File(destination / "foreground.hdf5", "w") as out:
        frames = source["observations/images/d435i_rgb"]
        n, height, width, _ = frames.shape
        valid = source["camera/d435i/valid"][:].astype(bool)
        frame_numbers = source["camera/d435i/frame_number"][:]
        timestamps = source["camera/d435i/unified_timestamp_ns"][:]
        out.attrs["contract"] = CONTRACT
        out.attrs["source_stamp"] = json.dumps(source_stamp)
        out.attrs["causal"] = True
        out.attrs["review_status"] = "pending"
        out.attrs["redetect_every"] = stream.redetect_every
        out.create_dataset("source/frame_index", data=np.arange(n))
        out.create_dataset("source/frame_number", data=frame_numbers)
        out.create_dataset("source/unified_timestamp_ns", data=timestamps)
        out.create_dataset("source/camera_valid", data=valid)
        rgb_out = out.create_dataset("foreground/rgb", frames.shape, dtype="u1", chunks=(1, height, width, 3), compression="gzip", compression_opts=1)
        masks_out = {name: out.create_dataset(f"masks/{name}", (n, height, width), dtype="u1", chunks=(1, height, width), compression="gzip", compression_opts=1) for name in ("hand", "objects", "union")}
        instances_out = out.create_dataset("masks/instances", (n, 3, height, width), dtype="u1", chunks=(1, 1, height, width), compression="gzip", compression_opts=1)
        good = np.flatnonzero(valid)
        if not len(good):
            raise ValueError("No valid top camera frames")
        fps = max(1, len(good) - 1) * 1e9 / max(1, int(timestamps[good[-1]]) - int(timestamps[good[0]]))
        writer = cv2.VideoWriter(str(destination / "comparison.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width * 2, height))
        if not writer.isOpened():
            raise RuntimeError("Cannot create preview")
        preview_indices = set(good[np.unique(np.linspace(0, len(good) - 1, 7).round().astype(int))])
        previews = []
        try:
            for index in good:
                rgb = frames[int(index)]
                result = stream.step(rgb, int(frame_numbers[index]))
                rgb_out[index] = result["rgb"]
                for name, dataset in masks_out.items():
                    dataset[index] = result[name]
                instances_out[index] = result["instances"]
                assert np.array_equal(result["rgb"], np.where(result["union"][..., None], rgb, 0))
                rows.append({"frame": int(index), "timestamp_ns": int(timestamps[index]), "hand_pixels": int(result["hand"].sum()), "object_pixels": int(result["objects"].sum()), "instance_pixels": result["instances"].sum(axis=(1, 2)).tolist(), "object_scores": result["object_scores"], "corrections": result["corrections"], "total_ms": result["total_ms"], "duplicate": result["duplicate"], "initialized": result["initialized"]})
                writer.write(cv2.cvtColor(np.concatenate([rgb, result["rgb"]], axis=1), cv2.COLOR_RGB2BGR))
                if index in preview_indices:
                    pair = Image.fromarray(np.concatenate([rgb, result["rgb"]], axis=1))
                    ImageDraw.Draw(pair).text((10, 10), f"{source_path.stem} frame {index} | raw / causal tracking", fill="yellow", stroke_width=1, stroke_fill="black")
                    previews.append(pair)
                    pair.save(destination / f"frame_{index:06d}.png")
                if len(rows) % 40 == 0 or len(rows) == len(good):
                    print(f"{source_path.stem} {len(rows)}/{len(good)} hand={rows[-1]['hand_pixels']} total={rows[-1]['total_ms']:.1f}ms", flush=True)
        finally:
            writer.release()
        sheet = Image.new("RGB", (width * 2, len(previews) * height))
        for i, preview in enumerate(previews):
            sheet.paste(preview, (0, height * i))
        sheet.save(destination / "comparison.png")
    if stamp(source_path) != source_stamp:
        raise RuntimeError("Source changed during preprocessing")
    metrics = {"source": source_stamp, "records": n, "valid_frames": len(rows), "empty_hand_masks": sum(row["hand_pixels"] == 0 for row in rows), "empty_objects_masks": sum(row["object_pixels"] == 0 for row in rows), "timing_ms": summarize([row["total_ms"] for row in rows if not row["duplicate"] and not row["initialized"]]), "frames": rows}
    (destination / "report.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/home/zmy/workplace/insert-data/transhandr_demo/o6"))
    parser.add_argument("--episodes", nargs="+", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sam3-root", type=Path, default=Path("/home/zmy/Projects/sam3"))
    parser.add_argument("--redetect-every", type=int, default=30)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    stream = TopForegroundStream(args.sam3_root, redetect_every=args.redetect_every)
    episodes, errors = [], []
    begin = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for episode in args.episodes:
        try:
            metrics = process(stream, args.data_root / f"episode_{episode:06d}.hdf5", args.output)
            episodes.append(metrics)
        except (ValueError, RuntimeError, OSError) as exc:
            errors.append({"episode": episode, "error": str(exc)})
            print(f"FAILED episode {episode}: {exc}", flush=True)
        (args.output / "progress.json").write_text(json.dumps({"episodes": episodes, "errors": errors}, indent=2))
    summary = {"contract": "sam3-top-causal-tracking-v1", "causal": True, "inpainting": False, "review_status": "pending", "gpu": torch.cuda.get_device_name(), "redetect_every": args.redetect_every, "wall_s": time.perf_counter() - begin, "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3, "timing_scope": "RGB ready to CPU black RGB; excludes capture, I/O, ACT and robot", "episodes": episodes, "errors": errors}
    summary["contract"] = CONTRACT
    (args.output / "report.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"episodes_completed": len(episodes), "errors": errors, "wall_s": summary["wall_s"], "empty_hand_masks": [e["empty_hand_masks"] for e in episodes]}, indent=2), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
