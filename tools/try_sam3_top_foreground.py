#!/usr/bin/env python3
"""Causal SAM3 trial on recorded top RGB; originals are opened read-only.

Each image is segmented independently, with no future frames or inpainting.
HDF5/PNG outputs are lossless; MP4 files are previews only. This is a trial,
not a robot controller or an ACT dataset converter.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import h5py
import numpy as np
from PIL import Image, ImageDraw
import torch


def select_hand(rgb, masks, refine):
    """Select by white-pixel overlap; optionally reproduce old white-anchor rule."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    white = (hsv[..., 1] <= 65) & (hsv[..., 2] >= 135)
    overlap = np.asarray([(m & white).sum() for m in masks])
    if not len(overlap) or overlap.max() == 0:
        return np.zeros(rgb.shape[:2], bool), "missing", 0.0
    selected = masks[int(overlap.argmax())]
    fraction = float((selected & white).sum()) / max(1, int(selected.sum()))
    if not refine or fraction >= 0.6:
        return selected, "semantic", fraction
    seed = cv2.morphologyEx((selected & white).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(seed)
    if count <= 1:
        return np.zeros_like(selected), "missing", fraction
    seed = (labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
    contours, _ = cv2.findContours(seed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(seed, contours, -1, 1, cv2.FILLED)
    refined = cv2.dilate(seed, np.ones((5, 5), np.uint8)).astype(bool) & selected
    return refined, "white-anchor", fraction


def infer(processor, rgb, prompts):
    torch.cuda.synchronize()
    start = time.perf_counter()
    state = processor.set_image(Image.fromarray(rgb))
    torch.cuda.synchronize()
    image_done = time.perf_counter()
    results = []
    prompt_ms = []
    for prompt in prompts:
        processor.reset_all_prompts(state)
        begin = time.perf_counter()
        state = processor.set_text_prompt(prompt, state)
        masks = state["masks"][:, 0].cpu().numpy().astype(bool)
        scores = state["scores"].float().cpu().numpy()
        torch.cuda.synchronize()
        prompt_ms.append((time.perf_counter() - begin) * 1000)
        results.append((masks, scores))
    return results, {"backbone_ms": (image_done - start) * 1000, "prompt_ms": prompt_ms, "model_ms": (time.perf_counter() - start) * 1000}


def summarize(values):
    x = np.asarray(values, float)
    return {"mean": float(x.mean()), "p50": float(np.median(x)), "p95": float(np.percentile(x, 95)), "min": float(x.min()), "max": float(x.max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/home/zmy/workplace/insert-data/transhandr_demo/o6"))
    parser.add_argument("--episodes", type=int, nargs="+", default=[66, 73, 91])
    parser.add_argument("--sam3-root", type=Path, default=Path("/home/zmy/Projects/sam3"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=5, help="Uniformly spaced valid frames per episode; 0 means all valid frames.")
    parser.add_argument("--hand-prompt", default="white robot hand")
    parser.add_argument("--object-prompt", default="cup")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--resolution", type=int, default=1008)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()
    if args.samples < 0 or args.warmup < 0:
        parser.error("samples and warmup must be nonnegative")
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.sam3_root))
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    torch.set_num_threads(4)
    np.random.seed(0)
    torch.manual_seed(0)
    prompts = [args.hand_prompt, args.object_prompt]
    checkpoint = args.checkpoint or args.sam3_root / "checkpoints/modelscope_sam3/sam3.pt"
    report = {"config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, "prompts": prompts,
              "causal": True, "inpainting": False, "precision": "bfloat16 autocast", "gpu": torch.cuda.get_device_name(),
              "torch": torch.__version__, "timing_scope": "host RGB ready through GPU inference, CPU masks and black compositing; excludes capture, disk I/O, ACT and execution", "episodes": []}
    start = time.perf_counter()
    model = build_sam3_image_model(checkpoint_path=str(checkpoint), load_from_HF=False, device="cuda", compile=False)
    processor = Sam3Processor(model, resolution=args.resolution, device="cuda", confidence_threshold=args.threshold)
    torch.cuda.synchronize()
    report["model_load_s"] = time.perf_counter() - start
    print(f"Model loaded in {report['model_load_s']:.2f}s", flush=True)
    with h5py.File(args.data_root / f"episode_{args.episodes[0]:06d}.hdf5", "r") as source:
        first = int(np.flatnonzero(source["camera/d435i/valid"][:])[0])
        warm_rgb = source["observations/images/d435i_rgb"][first]
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(args.warmup):
            infer(processor, warm_rgb, prompts)
        torch.cuda.reset_peak_memory_stats()
        for episode in args.episodes:
            process_episode(processor, episode, prompts, args, report)
    rows = [row for episode in report["episodes"] for row in episode["frames"]]
    report["steady_total_ms"] = summarize([r["total_ms"] for r in rows])
    report["steady_model_ms"] = summarize([r["model_ms"] for r in rows])
    report["steady_backbone_ms"] = summarize([r["backbone_ms"] for r in rows])
    report["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 1024**3
    report["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 1024**3
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k not in ("episodes", "config")}, indent=2), flush=True)


def process_episode(processor, episode, prompts, args, report):
    source_path = args.data_root / f"episode_{episode:06d}.hdf5"
    output_dir = args.output / source_path.stem
    output_dir.mkdir()
    metadata = {"episode": episode, "source": str(source_path), "frames": []}
    with h5py.File(source_path, "r") as source, h5py.File(output_dir / "foreground.hdf5", "w") as out:
        frames = source["observations/images/d435i_rgb"]
        indices = np.flatnonzero(source["camera/d435i/valid"][:])
        if args.samples:
            indices = indices[np.unique(np.linspace(0, len(indices) - 1, min(args.samples, len(indices))).round().astype(int))]
        n, height, width = len(indices), frames.shape[1], frames.shape[2]
        out.attrs["source_path"] = str(source_path)
        out.attrs["causal"] = True
        out.attrs["prompts_json"] = json.dumps(prompts)
        out.attrs["note"] = "Trial sidecar, indexed by source/frame_index; not a complete ACT dataset."
        out.create_dataset("source/frame_index", data=indices)
        for field in ("unified_timestamp_ns", "frame_number"):
            out.create_dataset(f"source/{field}", data=source[f"camera/d435i/{field}"][:][indices])
        rgb_out = out.create_dataset("foreground/rgb", (n, height, width, 3), dtype="u1", chunks=(1, height, width, 3), compression="gzip", compression_opts=1)
        mask_out = {name: out.create_dataset(f"masks/{name}", (n, height, width), dtype="u1", chunks=(1, height, width), compression="gzip", compression_opts=1) for name in ("hand", "hand_refined", "objects", "union")}
        previews = []
        preview_positions = set(np.unique(np.linspace(0, n - 1, min(5, n)).round().astype(int)))
        writer = None
        if not args.samples:
            # Preserve elapsed camera time in the preview, rather than assuming capture ran at 30 Hz.
            stamps = source["camera/d435i/unified_timestamp_ns"][:][indices]
            fps = (n - 1) * 1e9 / max(1, int(stamps[-1]) - int(stamps[0])) if n > 1 else 30.0
            writer = cv2.VideoWriter(str(output_dir / "raw_vs_foreground.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width * 2, height))
            if not writer.isOpened():
                raise RuntimeError("Cannot create preview video")
        try:
            for local, index in enumerate(indices):
                rgb = frames[int(index)]
                start = time.perf_counter()
                results, timing = infer(processor, rgb, prompts)
                hand, mode, fraction = select_hand(rgb, results[0][0], False)
                refined, refined_mode, _ = select_hand(rgb, results[0][0], True)
                objects = results[1][0].any(axis=0) if len(results[1][0]) else np.zeros((height, width), bool)
                union = hand | objects
                foreground = np.where(union[..., None], rgb, 0).astype(np.uint8)
                timing["total_ms"] = (time.perf_counter() - start) * 1000
                # Validate the compositing contract on every frame; no RGB synthesis or rescaling.
                assert np.array_equal(foreground[union], rgb[union]) and not foreground[~union].any()
                rgb_out[local] = foreground
                for name, mask in (("hand", hand), ("hand_refined", refined), ("objects", objects), ("union", union)):
                    mask_out[name][local] = mask
                row = {"index": int(index), "hand_count": len(results[0][0]), "object_count": len(results[1][0]), "hand_scores": results[0][1].tolist(), "object_scores": results[1][1].tolist(), "hand_pixels": int(hand.sum()), "object_pixels": int(objects.sum()), "hand_white_fraction": fraction, "hand_mode": mode, "refined_mode": refined_mode, "refined_hand_pixels": int(refined.sum()), **timing}
                row["source_frame_index"] = int(index)
                metadata["frames"].append(row)
                if writer is not None:
                    writer.write(cv2.cvtColor(np.concatenate([rgb, foreground], axis=1), cv2.COLOR_RGB2BGR))
                if local in preview_positions:
                    refined_rgb = np.where((refined | objects)[..., None], rgb, 0).astype(np.uint8)
                    tiles = []
                    for label, array in (("RAW", rgb), ("SAM3 semantic", foreground), ("Old white-anchor rule", refined_rgb)):
                        tile = Image.fromarray(array)
                        ImageDraw.Draw(tile).text((10, 10), f"ep {episode} frame {index} | {label}", fill=(255, 255, 0), stroke_width=1, stroke_fill=(0, 0, 0))
                        tiles.append(tile)
                    previews.append(tiles)
                    Image.fromarray(foreground).save(output_dir / f"frame_{index:06d}_foreground.png")
                if local % 20 == 0 or local + 1 == n:
                    print(f"episode={episode} {local + 1}/{n} frame={index} hand={row['hand_pixels']} objects={row['object_count']} total={timing['total_ms']:.1f}ms", flush=True)
        finally:
            if writer is not None:
                writer.release()
        sheet = Image.new("RGB", (width * 3, height * len(previews)))
        for row_id, tiles in enumerate(previews):
            for column, tile in enumerate(tiles):
                sheet.paste(tile, (column * width, row_id * height))
        sheet.save(output_dir / "comparison.png")
    report["episodes"].append(metadata)
    (args.output / "progress.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
