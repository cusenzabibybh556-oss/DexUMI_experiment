"""Offline training, deterministic prior evaluation and portable model bundles."""
from dataclasses import fields
from pathlib import Path
import json
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import CONTRACT, Config
from .data import ACTDataset, write_json
from .geometry import rotation6d_to_matrix
from .model import ACT, loss


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(name):
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; explicitly choose --device cpu")
    return torch.device(name)


def loader(dataset, config, shuffle=False):
    generator = torch.Generator().manual_seed(config.seed)
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=shuffle,
                      num_workers=config.workers, pin_memory=config.device.startswith("cuda"),
                      generator=generator)


def batch_to(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


@torch.no_grad()
def measure(model, batches, stats, device, max_batches=None, collect=False):
    model.eval()
    mean, std = np.asarray(stats["action_mean"]), np.asarray(stats["action_std"])
    positions, angles, hands, first_positions = [], [], [], []
    predictions, targets, masks = [], [], []
    absolute_sum = count = invalid_rotations = 0
    for i, batch in enumerate(batches):
        if max_batches is not None and i >= max_batches:
            break
        b = batch_to(batch, device)
        pred, _, _ = model(b["image"], b["state"])
        pred, target = pred.cpu().numpy(), batch["action"].numpy()
        valid = ~batch["is_pad"].numpy()
        absolute_sum += np.abs(pred-target)[valid].sum()
        count += int(valid.sum()) * target.shape[-1]
        decoded, truth = pred * std + mean, target * std + mean
        positions.extend(np.linalg.norm(decoded[valid, :3]-truth[valid, :3], axis=-1).tolist())
        hands.extend(np.abs(decoded[valid, 9:]-truth[valid, 9:]).tolist())
        first_positions.extend(np.linalg.norm(decoded[:, 0, :3]-truth[:, 0, :3], axis=-1).tolist())
        for a, t in zip(decoded[valid, 3:9], truth[valid, 3:9]):
            try:
                r = rotation6d_to_matrix(a) @ rotation6d_to_matrix(t).T
                angles.append(float(np.rad2deg(np.arccos(np.clip((np.trace(r)-1)/2, -1, 1)))))
            except ValueError:
                invalid_rotations += 1
        if collect:
            predictions.append(decoded)
            targets.append(truth)
            masks.append(~valid)
    if not count:
        raise ValueError("No valid evaluation targets")
    hand = np.asarray(hands)
    report = {"normalized_l1": float(absolute_sum/count), "valid_action_steps": len(positions),
              "o6_error_target": "action/command (glove-derived O6 commands), not measured q_pos",
              "position_rmse_mm": float(np.sqrt(np.mean(np.square(positions)))*1000),
              "position_p95_mm": float(np.percentile(positions, 95)*1000),
              "first_step_position_p95_mm": float(np.percentile(first_positions, 95)*1000),
              "rotation_mean_deg": float(np.mean(angles)) if angles else None,
              "rotation_p95_deg": float(np.percentile(angles, 95)) if angles else None,
              "invalid_predicted_rotations": invalid_rotations,
              "o6_mae_per_channel": hand.mean(0).tolist(), "o6_mae": float(hand.mean()),
              "inference_latent": "zero prior, no future action encoder"}
    arrays = {"predicted": np.concatenate(predictions), "target": np.concatenate(targets),
              "is_pad": np.concatenate(masks)} if collect else None
    return report, arrays


def checkpoint_payload(model, optimizer, config, step, best, manifest, stats):
    return {"contract": CONTRACT, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "config": config.to_dict(), "step": step, "best_validation_l1": best,
            "manifest_sha256": manifest["manifest_sha256"], "manifest": manifest, "stats": stats,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy_rng": np.random.get_state(), "python_rng": random.getstate()}


def save_checkpoint(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def train(prepared, output, overrides=None, resume=None, eval_batches=None):
    dataset = ACTDataset(prepared, "train")
    validation = ACTDataset(prepared, "val")
    config = Config(**{**dataset.config.to_dict(), **(overrides or {})}).validate()
    # Runtime/model overrides are allowed; changing data semantics requires prepare again.
    protected = ["data_root", "foreground_root", "allow_experimental_foreground", "deduplicate_identical_timestamps", "task", "episodes", "calibration", "time_scale", "action_hz", "chunk_size",
                 "history_steps", "image_size", "quaternion_format", "max_source_gap_s", "min_future_steps",
                 "normalization_floor"]
    if any(getattr(config, k) != getattr(dataset.config, k) for k in protected):
        raise ValueError("Data semantics changed: run prepare with the new config")
    seed_all(config.seed)
    device = select_device(config.device)
    out = Path(output)
    if out.exists() and resume is None:
        raise FileExistsError("Training output exists; choose a new output or explicitly --resume")
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(resume, map_location="cpu", weights_only=False) if resume else None
    if checkpoint:
        if checkpoint["contract"] != CONTRACT or checkpoint["manifest_sha256"] != dataset.manifest["manifest_sha256"]:
            raise ValueError("Resume checkpoint belongs to a different dataset or action contract")
        for name in [f.name for f in fields(Config) if f.name not in {"max_steps", "device", "workers", "eval_every", "checkpoint_every"}]:
            if checkpoint["config"][name] != getattr(config, name):
                raise ValueError(f"Resume config changed: {name}")
    model = ACT(config, load_pretrained=False if checkpoint else None).to(device)
    backbone = list(model.backbone.parameters())
    ids = {id(p) for p in backbone}
    optimizer = torch.optim.AdamW([{"params": [p for p in model.parameters() if id(p) not in ids]},
                                   {"params": backbone, "lr": config.backbone_learning_rate}],
                                  lr=config.learning_rate, weight_decay=1e-4)
    start, best = 0, float("inf")
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start, best = checkpoint["step"], checkpoint["best_validation_l1"]
        torch.set_rng_state(checkpoint["torch_rng"])
        if checkpoint.get("cuda_rng") and device.type == "cuda":
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        random.setstate(checkpoint["python_rng"])
    if start >= config.max_steps:
        raise ValueError("max_steps must exceed the resumed step")
    write_json(out / "config.json", config.to_dict())
    write_json(out / "manifest.json", dataset.manifest)
    write_json(out / "stats.json", dataset.stats)
    write_json(out / "runtime.json", {"torch": torch.__version__, "device": str(device),
               "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
               "pretrained_backbone": config.pretrained_backbone,
               "resume_shuffle": "new seeded sampler pass; not bitwise continuation of prior minibatch order"})
    batches = loader(dataset, config, True)
    val_batches = loader(validation, config)
    iterator = iter(batches)
    started = time.monotonic()
    for step in range(start + 1, config.max_steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(batches)
            batch = next(iterator)
        model.train()
        b = batch_to(batch, device)
        optimizer.zero_grad(set_to_none=True)
        predicted, mu, logvar = model(b["image"], b["state"], b["action"], b["is_pad"])
        losses = loss(predicted, b["action"], b["is_pad"], mu, logvar, config.kl_weight)
        if not torch.isfinite(losses["loss"]):
            raise RuntimeError(f"Non-finite loss at step {step}")
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip, error_if_nonfinite=True)
        optimizer.step()
        record = {"step": step, **{k: float(v.detach().cpu()) for k, v in losses.items()}}
        if step % config.eval_every == 0 or step == config.max_steps:
            report, _ = measure(model, val_batches, dataset.stats, device, eval_batches)
            record["validation"] = report
            improved = report["normalized_l1"] < best
            best = min(best, report["normalized_l1"])
            if improved:
                save_checkpoint(out / "policy_best.ckpt", checkpoint_payload(model, optimizer, config, step, best, dataset.manifest, dataset.stats))
        if step % config.checkpoint_every == 0 or step == config.max_steps:
            save_checkpoint(out / "policy_last.ckpt", checkpoint_payload(model, optimizer, config, step, best, dataset.manifest, dataset.stats))
        with (out / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(record, allow_nan=False) + "\n")
        if step == start+1 or step % 10 == 0 or "validation" in record:
            print(json.dumps(record, ensure_ascii=False), flush=True)
    summary = {"completed_steps": config.max_steps, "best_validation_l1": best,
               "elapsed_s": time.monotonic()-started, "train_samples": len(dataset),
               "validation_samples": len(validation), "hardware_connections": 0}
    write_json(out / "summary.json", summary)
    return summary


def evaluate(prepared, checkpoint, output, split="test", device="cpu", batch_size=8, max_batches=None):
    out = Path(output)
    if out.exists():
        raise FileExistsError(out)
    bundle = torch.load(checkpoint, map_location="cpu", weights_only=False)
    dataset = ACTDataset(prepared, split)
    if bundle["contract"] != CONTRACT or bundle["manifest_sha256"] != dataset.manifest["manifest_sha256"]:
        raise ValueError("Checkpoint and prepared dataset differ")
    config = Config(**{**bundle["config"], "device": device, "batch_size": batch_size}).validate()
    model = ACT(config, load_pretrained=False).to(select_device(device))
    model.load_state_dict(bundle["model"])
    report, arrays = measure(model, loader(dataset, config), bundle["stats"], select_device(device), max_batches, collect=True)
    report.update(split=split, checkpoint=str(Path(checkpoint).resolve()), samples=len(arrays["is_pad"]),
                  sample_limit_batches=max_batches, hardware_connections=0)
    out.mkdir(parents=True)
    write_json(out / "report.json", report)
    np.savez_compressed(out / "predictions.npz", **arrays)
    # Baseline uses zero nominal wrist motion and the current measured hand state.
    means, stds = np.asarray(bundle["stats"]["state_mean"]), np.asarray(bundle["stats"]["state_std"])
    values = []
    # Baseline needs numeric states only; do not decompress each image a second time.
    dataset.images = False
    for i in range(report["samples"]):
        sample = dataset[i]
        current_hand = sample["state"].numpy()[-6:] * stds[-6:] + means[-6:]
        valid = ~arrays["is_pad"][i]
        truth = arrays["target"][i][valid]
        values.append((np.linalg.norm(truth[:, :3], axis=-1), np.abs(truth[:, 9:]-current_hand)))
    report["hold_baseline"] = {"position_p95_mm": float(np.percentile(np.concatenate([x[0] for x in values]), 95)*1000),
                               "o6_mae": float(np.concatenate([x[1] for x in values]).mean())}
    evaluated_items = dataset.items[:report["samples"]]
    write_json(out / "sample_index.json", [{"source": path, "image_row": row} for path, row in evaluated_items])
    plot_predictions(out / "trajectory.png", arrays, config.action_hz)
    write_json(out / "report.json", report)
    return report


def plot_predictions(path, arrays, hz):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    first = int(np.argmax((~arrays["is_pad"]).sum(1)))
    valid = ~arrays["is_pad"][first]
    times = (np.arange(len(valid))[valid] + 1) / hz
    truth = arrays["target"][first][valid]
    prediction = arrays["predicted"][first][valid]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for j, name in enumerate(["X", "Y", "Z"]):
        line, = axes[0].plot(times, truth[:, j]*1000, label=name+" target")
        axes[0].plot(times, prediction[:, j]*1000, "--", color=line.get_color(), label=name+" predicted")
    for j, name in enumerate(["thumb flex", "thumb abd", "index", "middle", "ring", "pinky"]):
        line, = axes[1].plot(times, truth[:, 9+j], label=name)
        axes[1].plot(times, prediction[:, 9+j], "--", color=line.get_color())
    axes[0].set(xlabel="Physical time after observation (s)", ylabel="Spatial position delta (mm)",
                title="Nominal wrist trajectory: target / dashed prediction")
    axes[1].set(xlabel="Physical time after observation (s)", ylabel="O6 SDK units",
                title="O6 trajectory: target / dashed prediction")
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(fontsize=7, ncol=2)
    fig.savefig(path, dpi=150)
    plt.close(fig)
