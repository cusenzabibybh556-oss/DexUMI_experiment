"""Train on reviewed masks, then export autonomous per-frame foreground caches."""
import json
from pathlib import Path
import random
import time

import cv2
import h5py
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from .annotations import CLASSES, IMAGE_KEY, fingerprint, load_project, sha256, write_json
from .model import CACHE_CONTRACT, MODEL_CONTRACT, ForegroundNet, ForegroundPredictor


def tensors(project, frames, input_size, device):
    images, labels = [], []
    for frame in frames:
        image_path = Path(project) / "images" / f"{frame['id']}.png"
        label_path = Path(project) / "labels" / f"{frame['id']}.png"
        if sha256(image_path) != frame["rgb_sha256"] or sha256(label_path) != frame["label_sha256"]:
            raise ValueError("Annotation files changed without updating their revision")
        rgb = np.asarray(Image.open(image_path).convert("RGB"))
        label = np.asarray(Image.open(label_path))
        if label.shape != rgb.shape[:2] or not np.isin(label, [0, 1, 2]).all():
            raise ValueError("Invalid annotation labels")
        images.append(torch.from_numpy(rgb.copy()).permute(2, 0, 1).float()[None] / 255)
        labels.append(torch.from_numpy(label.astype(np.int64))[None, None])
    x = F.interpolate(torch.cat(images), size=input_size, mode="bilinear", align_corners=False).to(device)
    y = F.interpolate(torch.cat(labels).float(), size=input_size, mode="nearest")[:, 0].long().to(device)
    return x, y


def iou_metrics(confusion):
    union = confusion.sum(0) + confusion.sum(1) - confusion.diag()
    iou = confusion.diag().float() / union.clamp_min(1)
    return {CLASSES[i]: float(iou[i]) for i in range(3)}


def train(project, output, validation_episodes, steps=300, batch_size=4, input_size=(240, 320), device="cuda", pretrained=True):
    if steps < 1 or batch_size < 1 or min(input_size) < 32:
        raise ValueError("Invalid training dimensions or steps")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    data = load_project(project)
    project_sha256 = sha256(Path(project) / "project.json")
    reviewed = [f for f in data["frames"] if f["status"] == "reviewed"]
    training = [f for f in reviewed if f["episode"] not in validation_episodes]
    validation = [f for f in reviewed if f["episode"] in validation_episodes]
    if not training or not validation:
        raise ValueError("Need reviewed train frames and independent validation episodes")
    # Validate held-out label revisions too; their RGB and labels never enter training.
    for frame in validation:
        for folder, field in [("images", "rgb_sha256"), ("labels", "label_sha256")]:
            if sha256(Path(project) / folder / f"{frame['id']}.png") != frame[field]:
                raise ValueError("Validation annotation changed without updating its revision")
    torch.set_num_threads(4)
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    x, y = tensors(project, training, input_size, device)
    model = ForegroundNet(pretrained).to(device).train()
    encoder_ids = {id(p) for p in model.encoder.parameters()}
    optimizer = torch.optim.AdamW([{"params": model.encoder.parameters(), "lr": 3e-5},
                                  {"params": [p for p in model.parameters() if id(p) not in encoder_ids], "lr": 1e-3}], weight_decay=1e-4)
    weights = torch.tensor([.2, 1., 1.], device=device)
    history = []
    begin = time.perf_counter()
    for step in range(steps):
        selected = torch.randint(len(training), (min(batch_size, len(training)),), device=device)
        batch = x[selected]
        # Paired geometry stays fixed; mild RGB brightness is sufficient for the chain test.
        batch = (batch * (0.85 + 0.3 * torch.rand(len(selected), 1, 1, 1, device=device))).clamp(0, 1)
        logits = model(batch)
        ce = F.cross_entropy(logits, y[selected], weight=weights)
        probs = logits.softmax(1)
        target = F.one_hot(y[selected], 3).permute(0, 3, 1, 2).float()
        dimensions = (0, 2, 3)
        dice = 1 - ((2 * (probs * target).sum(dimensions) + 1) / (probs.sum(dimensions) + target.sum(dimensions) + 1))[1:].mean()
        loss = ce + dice
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite segmentation loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 0 or (step + 1) % 25 == 0 or step + 1 == steps:
            row = {"step": step + 1, "loss": float(loss.detach()), "cross_entropy": float(ce.detach()), "dice_loss": float(dice.detach())}
            history.append(row)
            print(json.dumps(row), flush=True)
    metadata = {"input_size": list(input_size), "classes": CLASSES, "normalize": "ImageNet RGB mean/std", "resize": "RGB bilinear align_corners=False; original-resolution logits bilinear then argmax", "train_episodes": sorted({f["episode"] for f in training}), "validation_episodes": sorted({f["episode"] for f in validation}), "train_frames": [f["id"] for f in training], "validation_frames": [f["id"] for f in validation], "annotation_manifest_sha256": sha256(Path(project) / "project.json"), "annotation_reviewers": sorted({f["reviewer"] for f in reviewed}), "steps": steps, "purpose": "experimental data pipeline; no deployment success claim"}
    checkpoint = output / "foreground.pt"
    if sha256(Path(project) / "project.json") != project_sha256:
        raise ValueError("Annotations changed during training; rerun with a fixed revision")
    torch.save({"contract": MODEL_CONTRACT, "model": model.cpu().state_dict(), "metadata": metadata}, checkpoint)
    del model, optimizer, x, y
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    predictor = ForegroundPredictor(checkpoint, device)
    confusion = torch.zeros(3, 3, dtype=torch.int64)
    comparisons = []
    for frame in validation:
        rgb = np.asarray(Image.open(Path(project) / "images" / f"{frame['id']}.png").convert("RGB"))
        target = np.asarray(Image.open(Path(project) / "labels" / f"{frame['id']}.png"))
        result = predictor.predict(rgb)
        counts = np.bincount((3 * target.astype(int) + result["labels"]).ravel(), minlength=9).reshape(3, 3)
        confusion += torch.from_numpy(counts)
        annotated = np.where((target != 0)[..., None], rgb, 0)
        comparisons.append(Image.fromarray(np.concatenate([rgb, annotated, result["rgb"]], axis=1)))
    sheet = Image.new("RGB", (comparisons[0].width, sum(im.height for im in comparisons)))
    at = 0
    for im in comparisons:
        sheet.paste(im, (0, at)); at += im.height
    sheet.save(output / "validation_raw_label_prediction.png")
    report = {"contract": MODEL_CONTRACT, "metadata": metadata, "history": history, "wall_s": time.perf_counter() - begin, "parameters": sum(p.numel() for p in predictor.model.parameters()), "validation_iou": iou_metrics(confusion), "validation_confusion": confusion.tolist(), "checkpoint_sha256": predictor.checkpoint_sha256, "note": "Metrics use assistant-reviewed seed labels when that reviewer is present; not independent human ground truth."}
    write_json(output / "report.json", report)
    return report


def process(checkpoint, data_root, episodes, output, device="cuda"):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    predictor = ForegroundPredictor(checkpoint, device)
    report = {"contract": CACHE_CONTRACT, "model_sha256": predictor.checkpoint_sha256, "model_metadata": predictor.metadata, "review_status": "experimental", "causal": True, "episodes": [], "timing_scope": "RGB ready through preprocessing, inference, CPU mask and original-pixel composition; excludes capture, disk I/O and ACT"}
    for episode in episodes:
        source_path = Path(data_root) / f"episode_{episode:06d}.hdf5"
        destination = output / source_path.stem
        destination.mkdir()
        source_stamp = fingerprint(source_path)
        rows = []
        with h5py.File(source_path, "r") as source, h5py.File(destination / "foreground.hdf5", "w") as cache:
            frames = source[IMAGE_KEY]
            n, height, width, _ = frames.shape
            valid = source["camera/d435i/valid"][:].astype(bool)
            good = np.flatnonzero(valid)
            if not len(good):
                raise ValueError("No valid camera frames")
            cache.attrs.update(contract=CACHE_CONTRACT, source_stamp=json.dumps(source_stamp), causal=True, review_status="experimental", model_sha256=predictor.checkpoint_sha256, model_metadata=json.dumps(predictor.metadata))
            cache["source/frame_index"] = np.arange(n)
            for name, original in [("frame_number", "frame_number"), ("unified_timestamp_ns", "unified_timestamp_ns"), ("camera_valid", "valid")]:
                cache[f"source/{name}"] = source[f"camera/d435i/{original}"][:]
            images = cache.create_dataset("foreground/rgb", frames.shape, dtype="u1", chunks=(1, height, width, 3), compression="gzip", compression_opts=1)
            masks = {name: cache.create_dataset(f"masks/{name}", (n, height, width), dtype="u1", chunks=(1, height, width), compression="gzip", compression_opts=1) for name in ("hand", "objects", "union", "labels")}
            warm = frames[int(good[0])]
            for _ in range(3):
                predictor.predict(warm)
            stamps = source["camera/d435i/unified_timestamp_ns"][:]
            fps = max(1, len(good) - 1) * 1e9 / max(1, int(stamps[good[-1]]) - int(stamps[good[0]]))
            writer = cv2.VideoWriter(str(destination / "comparison.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width * 2, height))
            if not writer.isOpened():
                raise RuntimeError("Cannot create preview video")
            previews = []
            picks = set(good[np.linspace(0, len(good) - 1, min(7, len(good))).round().astype(int)])
            try:
                for index in good:
                    rgb = frames[int(index)]
                    if device.startswith("cuda"):
                        torch.cuda.synchronize()
                    start = time.perf_counter()
                    result = predictor.predict(rgb)
                    if device.startswith("cuda"):
                        torch.cuda.synchronize()
                    elapsed = (time.perf_counter() - start) * 1000
                    images[index] = result["rgb"]
                    for name, dataset in masks.items():
                        dataset[index] = result[name]
                    assert np.array_equal(result["rgb"], np.where(result["union"][..., None], rgb, 0))
                    row = {"row": int(index), "total_ms": elapsed, "hand_pixels": int(result["hand"].sum()), "objects_pixels": int(result["objects"].sum()), "mean_confidence": result["mean_confidence"]}
                    rows.append(row)
                    pair = np.concatenate([rgb, result["rgb"]], axis=1)
                    writer.write(cv2.cvtColor(pair, cv2.COLOR_RGB2BGR))
                    if index in picks:
                        im = Image.fromarray(pair)
                        previews.append(im)
                        im.save(destination / f"frame_{index:06d}.png")
            finally:
                writer.release()
            sheet = Image.new("RGB", (width * 2, height * len(previews)))
            for i, im in enumerate(previews):
                sheet.paste(im, (0, height * i))
            sheet.save(destination / "comparison.png")
        if fingerprint(source_path) != source_stamp:
            raise ValueError("Source changed during processing")
        metrics = {"episode": episode, "source": source_stamp, "records": n, "valid_frames": len(rows), "empty_hand_masks": sum(r["hand_pixels"] == 0 for r in rows), "empty_objects_masks": sum(r["objects_pixels"] == 0 for r in rows), "mean_ms": float(np.mean([r["total_ms"] for r in rows])), "p95_ms": float(np.percentile([r["total_ms"] for r in rows], 95)), "frames": rows}
        report["episodes"].append(metrics)
        write_json(destination / "report.json", metrics)
        write_json(output / "report.json", report)
        print(json.dumps({k: v for k, v in metrics.items() if k not in ("frames", "source")}), flush=True)
    return report
