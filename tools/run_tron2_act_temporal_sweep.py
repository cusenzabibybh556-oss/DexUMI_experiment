#!/usr/bin/env python3
"""Train ACT chunk-length variants and compare a common validation horizon.

Different chunk lengths are not ranked by their full-chunk loss because the
short model would receive an easier objective. Every model is instead compared
on the same first 25 future actions from the same validation episodes. Only the
selected model is evaluated on the held-out test split. No hardware is used.
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

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tron2_act.data import write_json
from tron2_act.engine import evaluate, train
from tron2_act.geometry import rotation6d_to_matrix


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def best_validation(metrics_path):
    best = None
    with Path(metrics_path).open() as stream:
        for line in stream:
            row = json.loads(line)
            if "validation" in row:
                value = {"step": row["step"], **row["validation"]}
                if best is None or value["normalized_l1"] < best["normalized_l1"]:
                    best = value
    if best is None:
        raise RuntimeError(f"No validation result in {metrics_path}")
    return best


def common_prefix_report(evaluation, checkpoint, steps=25):
    arrays = np.load(Path(evaluation) / "predictions.npz")
    prediction = arrays["predicted"][:, :steps]
    target = arrays["target"][:, :steps]
    valid = ~arrays["is_pad"][:, :steps]
    bundle = torch.load(checkpoint, map_location="cpu", weights_only=False)
    std = np.asarray(bundle["stats"]["action_std"], dtype=np.float64)
    difference = prediction - target
    normalized_l1 = float(np.abs(difference / std)[valid].mean())
    position = np.linalg.norm(difference[valid, :3], axis=-1)
    hand = np.abs(difference[valid, 9:])
    angles = []
    invalid = 0
    for predicted, truth in zip(prediction[valid, 3:9], target[valid, 3:9]):
        try:
            relative = rotation6d_to_matrix(predicted) @ rotation6d_to_matrix(truth).T
            angles.append(float(np.rad2deg(np.arccos(np.clip((np.trace(relative)-1)/2,-1,1)))))
        except ValueError:
            invalid += 1
    return {
        "common_future_steps": steps,
        "common_horizon_s": steps / 20.0,
        "valid_action_steps": int(valid.sum()),
        "normalized_l1": normalized_l1,
        "position_rmse_mm": float(np.sqrt(np.mean(np.square(position))) * 1000),
        "position_p95_mm": float(np.percentile(position, 95) * 1000),
        "rotation_mean_deg": float(np.mean(angles)) if angles else None,
        "rotation_p95_deg": float(np.percentile(angles, 95)) if angles else None,
        "invalid_predicted_rotations": invalid,
        "o6_mae": float(hand.mean()),
        "o6_mae_per_channel": hand.mean(0).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepared-25", type=Path, required=True)
    parser.add_argument("--prepared-50", type=Path, required=True)
    parser.add_argument("--prepared-75", type=Path, required=True)
    parser.add_argument("--prepared-100", type=Path, required=True)
    parser.add_argument("--prepared-125", type=Path, required=True)
    parser.add_argument("--prepared-150", type=Path, required=True)
    parser.add_argument("--prepared-175", type=Path, required=True)
    parser.add_argument("--prepared-200", type=Path, required=True)
    parser.add_argument("--initial-25", type=Path,
                        help="Checkpoint used to continue an already-started chunk25 run")
    parser.add_argument("--initial-50", type=Path,
                        help="Existing best chunk50 checkpoint to continue to the common step budget")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--minimum-free-gib", type=float, default=8.0)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "status.json"
    status = {"state":"running","phase":"training","pid":os.getpid(),
              "started_at":now(),"hardware_connections":0}
    write_json(status_path,status)
    common = {"max_steps":args.steps,"batch_size":args.batch_size,"workers":args.workers,
              "eval_every":500,"checkpoint_every":1000,"device":"cuda","seed":0}
    models = [
        {"name":"chunk25","chunk_size":25,"prepared":args.prepared_25.resolve(),
         "initial":args.initial_25.resolve() if args.initial_25 else None},
        {"name":"chunk50","chunk_size":50,"prepared":args.prepared_50.resolve(),
         "initial":args.initial_50.resolve() if args.initial_50 else None},
        {"name":"chunk75","chunk_size":75,"prepared":args.prepared_75.resolve(),"initial":None},
        {"name":"chunk100","chunk_size":100,"prepared":args.prepared_100.resolve(),"initial":None},
        {"name":"chunk125","chunk_size":125,"prepared":args.prepared_125.resolve(),"initial":None},
        {"name":"chunk150","chunk_size":150,"prepared":args.prepared_150.resolve(),"initial":None},
        {"name":"chunk175","chunk_size":175,"prepared":args.prepared_175.resolve(),"initial":None},
        {"name":"chunk200","chunk_size":200,"prepared":args.prepared_200.resolve(),"initial":None},
    ]
    write_json(output/"sweep_manifest.json",{
        "created_at":now(),"models":[{k:str(v) if isinstance(v,Path) else v for k,v in x.items()} for x in models],
        "common_training":common,"selection":"minimum normalized L1 over the common first 25 validation steps",
        "test_policy":"held-out test evaluated once for the selected chunk","hardware_connections":0})
    results=[];failures=[]
    try:
        for index,model in enumerate(models,1):
            free_gib=shutil.disk_usage(output).free/1024**3
            if free_gib<args.minimum_free_gib:
                raise RuntimeError(f"Only {free_gib:.2f} GiB free; refusing another checkpoint")
            status.update(current_candidate=model["name"],candidate_index=index,
                          candidate_count=len(models),free_gib_before=free_gib,updated_at=now())
            write_json(status_path,status)
            train_dir=output/"candidates"/model["name"]/"train"
            try:
                if not (train_dir/"summary.json").exists():
                    local_resume=train_dir/"policy_last.ckpt"
                    resume=local_resume if local_resume.exists() else model["initial"]
                    if train_dir.exists() and not resume:
                        raise RuntimeError(f"Non-resumable partial output: {train_dir}")
                    if model["initial"] and not (train_dir/"policy_best.ckpt").exists():
                        train_dir.mkdir(parents=True,exist_ok=True)
                        shutil.copy2(model["initial"],train_dir/"policy_best.ckpt")
                    train(model["prepared"],train_dir,common,str(resume) if resume else None)
                checkpoint=train_dir/"policy_best.ckpt"
                validation_dir=output/"validation"/model["name"]
                if not (validation_dir/"report.json").exists():
                    evaluate(model["prepared"],checkpoint,validation_dir,split="val",
                             device="cuda",batch_size=args.batch_size)
                result={"name":model["name"],"chunk_size":model["chunk_size"],
                        "horizon_s":model["chunk_size"]/20.,"prepared":str(model["prepared"]),
                        "train_dir":str(train_dir),"checkpoint":str(checkpoint),
                        "full_chunk_best_validation":best_validation(train_dir/"metrics.jsonl"),
                        "common_25_step_validation":common_prefix_report(validation_dir,checkpoint,25),
                        "continued_from":str(model["initial"]) if model["initial"] else None}
                results.append(result)
                write_json(output/"ranking_partial.json",sorted(results,key=lambda x:x["common_25_step_validation"]["normalized_l1"]))
            except Exception as exc:
                failures.append({"name":model["name"],"error":f"{type(exc).__name__}: {exc}",
                                 "traceback":traceback.format_exc()})
                write_json(output/"failures.json",failures)
            finally:
                gc.collect();torch.cuda.empty_cache()
        if not results:
            raise RuntimeError("Every temporal candidate failed")
        ranking=sorted(results,key=lambda x:x["common_25_step_validation"]["normalized_l1"])
        write_json(output/"ranking.json",ranking)
        winner=ranking[0]
        status.update(phase="final_test",winner=winner["name"],updated_at=now())
        write_json(status_path,status)
        test_dir=output/"selected_test"
        if not (test_dir/"report.json").exists():
            test=evaluate(winner["prepared"],winner["checkpoint"],test_dir,
                          split="test",device="cuda",batch_size=args.batch_size)
        else:
            test=json.loads((test_dir/"report.json").read_text())
        summary={"finished_at":now(),"winner":winner,"test":test,
                 "ranking":ranking,"failures":failures,"hardware_connections":0}
        write_json(output/"sweep_summary.json",summary)
        status.update(state="completed",phase="completed",finished_at=summary["finished_at"])
        write_json(status_path,status)
        print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    except BaseException as exc:
        status.update(state="failed",phase="failed",finished_at=now(),
                      error=f"{type(exc).__name__}: {exc}",traceback=traceback.format_exc())
        write_json(status_path,status)
        raise


if __name__=="__main__":
    main()
