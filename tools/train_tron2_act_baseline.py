#!/usr/bin/env python3
"""Run ACT baseline training and a single final held-out evaluation.

The job writes status.json throughout its lifetime and can run independently of
the launching terminal. No hardware APIs are used.
"""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tron2_act.data import write_json
from tron2_act.engine import evaluate, train


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "train").exists() and args.resume is None:
        raise FileExistsError("Training directory already exists; use a fresh output or an explicit checkpoint resume")
    started = now()
    status = {"state": "running", "phase": "training", "pid": os.getpid(), "started_at": started,
              "prepared": str(args.prepared.resolve()), "output": str(out), "hardware_connections": 0}
    write_json(out / "status.json", status)
    code = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (ROOT / "tron2_act").rglob("*.py")}
    write_json(out / "source_sha256.json", code)
    overrides = {"max_steps": args.steps, "batch_size": args.batch_size, "workers": args.workers,
                 "eval_every": args.eval_every, "checkpoint_every": args.checkpoint_every, "device": "cuda"}
    write_json(out / "training_overrides.json", overrides)
    try:
        training = train(args.prepared, out / "train", overrides,
                         str(args.resume) if args.resume else None)
        status.update(phase="final_test", training_finished_at=now())
        write_json(out / "status.json", status)
        # The test split is evaluated once, after selection by validation L1.
        test = evaluate(args.prepared, out / "train/policy_best.ckpt", out / "test",
                        split="test", device="cuda", batch_size=args.batch_size)
        result = {"started_at": started, "finished_at": now(), "training": training, "test": test,
                  "selection": "lowest full-validation normalized L1; test used once after training",
                  "hardware_connections": 0}
        write_json(out / "baseline_summary.json", result)
        status.update(state="completed", phase="completed", finished_at=result["finished_at"])
        write_json(out / "status.json", status)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    except BaseException as exc:
        status.update(state="failed", phase="failed", finished_at=now(),
                      error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        write_json(out / "status.json", status)
        raise


if __name__ == "__main__":
    main()
