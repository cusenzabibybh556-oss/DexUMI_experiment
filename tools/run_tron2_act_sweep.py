#!/usr/bin/env python3
"""Sequential, validation-selected ACT hyperparameter sweep on one GPU.

The prepared dataset and its train/validation/test episode split are frozen.
Candidates are ranked only by full validation normalized L1. The held-out test
split is evaluated once, for the selected candidate, after all training ends.
No hardware modules are imported or connected.
"""
import argparse
from datetime import datetime
import gc
import json
import os
from pathlib import Path
import shutil
import sys
import traceback
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tron2_act.data import write_json
from tron2_act.engine import evaluate, train


CANDIDATES = [
    {
        "name": "lr_5e6_kl10",
        "reason": "Conservative optimizer step; tests whether the 1e-5 baseline is noisy on the small dataset.",
        "overrides": {"learning_rate": 5e-6, "backbone_learning_rate": 5e-6, "kl_weight": 10.0},
    },
    {
        "name": "lr_2e5_kl10",
        "reason": "Faster optimizer step; tests under-training at the baseline learning rate.",
        "overrides": {"learning_rate": 2e-5, "backbone_learning_rate": 2e-5, "kl_weight": 10.0},
    },
    {
        "name": "backbone_1e6_kl10",
        "reason": "Protect pretrained visual features while training the ACT heads at the baseline rate.",
        "overrides": {"learning_rate": 1e-5, "backbone_learning_rate": 1e-6, "kl_weight": 10.0},
    },
    {
        "name": "lr_1e5_kl1",
        "reason": "Weaker posterior regularization; tests whether KL=10 suppresses useful latent structure.",
        "overrides": {"learning_rate": 1e-5, "backbone_learning_rate": 1e-5, "kl_weight": 1.0},
    },
    {
        "name": "lr_1e5_kl50",
        "reason": "Stronger prior matching for zero-latent deployment inference.",
        "overrides": {"learning_rate": 1e-5, "backbone_learning_rate": 1e-5, "kl_weight": 50.0},
    },
    {
        "name": "compact_384_dropout02",
        "reason": "Lower-capacity, more regularized model for 20 training episodes.",
        "overrides": {
            "hidden_dim": 384,
            "heads": 8,
            "encoder_layers": 3,
            "decoder_layers": 5,
            "feedforward_dim": 2048,
            "dropout": 0.2,
            "learning_rate": 1e-5,
            "backbone_learning_rate": 1e-6,
            "kl_weight": 10.0,
        },
    },
]


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def validation_best(metrics_path):
    best = None
    with Path(metrics_path).open() as stream:
        for line in stream:
            record = json.loads(line)
            if "validation" not in record:
                continue
            candidate = {"step": record["step"], **record["validation"]}
            if best is None or candidate["normalized_l1"] < best["normalized_l1"]:
                best = candidate
    if best is None:
        raise RuntimeError(f"No full-validation record in {metrics_path}")
    return best


def reference_result(path, expected_manifest):
    if path is None:
        return None
    path = path.resolve()
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest["manifest_sha256"] != expected_manifest:
        raise ValueError("Reference run uses a different prepared dataset")
    return {
        "name": "existing_baseline",
        "trained_in_sweep": False,
        "train_dir": str(path),
        "checkpoint": str(path / "policy_best.ckpt"),
        "validation": validation_best(path / "metrics.jsonl"),
        "config": json.loads((path / "config.json").read_text()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-run", type=Path)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--minimum-free-gib", type=float, default=8.0)
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.workers < 0:
        parser.error("steps/batch-size must be positive and workers nonnegative")

    prepared = args.prepared.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    prepared_manifest = json.loads((prepared / "manifest.json").read_text())
    common = {
        "max_steps": args.steps,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "eval_every": args.eval_every,
        "checkpoint_every": args.checkpoint_every,
        "device": "cuda",
        "seed": 0,
    }
    manifest = {
        "created_at": now(),
        "prepared": str(prepared),
        "prepared_manifest_sha256": prepared_manifest["manifest_sha256"],
        "selection": "lowest full-validation normalized L1; held-out test evaluated once after selection",
        "common": common,
        "candidates": CANDIDATES,
        "hardware_connections": 0,
    }
    manifest_path = output / "sweep_manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        for key in ("prepared", "prepared_manifest_sha256", "common", "candidates"):
            if old[key] != manifest[key]:
                raise ValueError(f"Existing sweep has a different {key}")
    else:
        write_json(manifest_path, manifest)

    status_path = output / "status.json"
    status = {
        "state": "running",
        "phase": "training",
        "pid": os.getpid(),
        "started_at": now(),
        "hardware_connections": 0,
    }
    write_json(status_path, status)
    results = []
    reference = reference_result(args.reference_run, prepared_manifest["manifest_sha256"])
    if reference:
        results.append(reference)

    try:
        for index, candidate in enumerate(CANDIDATES, 1):
            free_gib = shutil.disk_usage(output).free / 1024**3
            if free_gib < args.minimum_free_gib:
                raise RuntimeError(
                    f"Only {free_gib:.2f} GiB free before {candidate['name']}; "
                    f"minimum is {args.minimum_free_gib:.2f} GiB"
                )
            candidate_root = output / "candidates" / candidate["name"]
            train_dir = candidate_root / "train"
            candidate_root.mkdir(parents=True, exist_ok=True)
            overrides = {**common, **candidate["overrides"]}
            write_json(candidate_root / "overrides.json", overrides)
            status.update(
                current_candidate=candidate["name"],
                candidate_index=index,
                candidate_count=len(CANDIDATES),
                free_gib_before=free_gib,
                updated_at=now(),
            )
            write_json(status_path, status)

            summary_path = train_dir / "summary.json"
            if not summary_path.exists():
                resume = train_dir / "policy_last.ckpt"
                if train_dir.exists() and not resume.exists():
                    raise RuntimeError(
                        f"Incomplete non-resumable output at {train_dir}; preserve it for inspection "
                        "and choose a new sweep output"
                    )
                train(prepared, train_dir, overrides,
                      str(resume) if resume.exists() else None)
            result = {
                "name": candidate["name"],
                "trained_in_sweep": True,
                "train_dir": str(train_dir),
                "checkpoint": str(train_dir / "policy_best.ckpt"),
                "validation": validation_best(train_dir / "metrics.jsonl"),
                "config": json.loads((train_dir / "config.json").read_text()),
                "training": json.loads(summary_path.read_text()),
            }
            results = [x for x in results if x["name"] != candidate["name"]]
            results.append(result)
            write_json(output / "ranking_partial.json", sorted(
                results, key=lambda x: x["validation"]["normalized_l1"]))
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except (ImportError, RuntimeError):
                pass

        ranking = sorted(results, key=lambda x: x["validation"]["normalized_l1"])
        winner = ranking[0]
        write_json(output / "ranking.json", ranking)
        status.update(phase="final_test", winner=winner["name"], updated_at=now())
        write_json(status_path, status)
        test_dir = output / "selected_test"
        if not (test_dir / "report.json").exists():
            test_report = evaluate(prepared, winner["checkpoint"], test_dir,
                                   split="test", device="cuda", batch_size=args.batch_size)
        else:
            test_report = json.loads((test_dir / "report.json").read_text())
        final = {
            "finished_at": now(),
            "winner": winner,
            "test": test_report,
            "ranking": ranking,
            "selection": manifest["selection"],
            "hardware_connections": 0,
        }
        write_json(output / "sweep_summary.json", final)
        status.update(state="completed", phase="completed", finished_at=final["finished_at"])
        write_json(status_path, status)
        print(json.dumps(final, ensure_ascii=False, indent=2), flush=True)
    except BaseException as exc:
        status.update(
            state="failed",
            phase="failed",
            finished_at=now(),
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )
        write_json(status_path, status)
        raise


if __name__ == "__main__":
    main()
