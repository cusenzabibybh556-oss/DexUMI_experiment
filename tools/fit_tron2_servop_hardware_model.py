#!/usr/bin/env python3
"""Fit and cross-validate a ServoP residual model from real ACT hardware logs.

This is deliberately offline: no robot SDK, CAN interface, ROS process, or
parameter write is imported.  Validation splits whole hardware runs so adjacent
frames from one experiment can never appear on both sides of a fold.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import glob
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tron2_hardware_residual import FEATURE_NAMES, OUTPUT_NAMES, SCHEMA, feature_vector


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metric(values):
    values = np.asarray(values, float)
    return dict(count=int(len(values)), mean=float(np.mean(values)),
                median=float(np.median(values)), rms=float(np.sqrt(np.mean(values**2))),
                p95=float(np.percentile(values, 95)), maximum=float(np.max(values)))


def quaternion_rotation(values):
    values = np.asarray(values, float)
    return Rotation.from_quat(values[..., [1, 2, 3, 0]])


def load_run(path):
    commands = {"act_control": [], "settle_pose": []}
    feedback = {"act_control": [], "settle_pose": []}
    event_counts = Counter()
    with Path(path).open(errors="replace") as source:
        for line in source:
            try:
                row = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                event_counts["invalid_json"] += 1
                continue
            event = row.get("event")
            event_counts[event] += 1
            phase = row.get("phase")
            if event == "pose_replay_command" and phase in commands and row.get("ramp", 0) > .99:
                model = row.get("model", {})
                if not all(key in model for key in ("model_shift_m", "model_rotation_vector_rad",
                                                     "filtered_reference_velocity_rad_s")):
                    continue
                issued_p = np.asarray(row["issued_left_pos"][:3], float)
                issued_r = quaternion_rotation(row["issued_left_pos"][3:]).as_matrix()
                correction_p = np.asarray(row["correction_xyz_m"], float)
                correction_r = Rotation.from_rotvec(row["correction_rotation_vector_rad"]).as_matrix()
                preview_p = issued_p-correction_p
                preview_r = correction_r.T@issued_r
                commands[phase].append(dict(
                    time=float(row["offset_applied_monotonic_ns"])/1e9,
                    preview_p=preview_p, preview_r=preview_r,
                    correction_p=correction_p,
                    correction_r=correction_r,
                    residual_p=np.asarray(row["residual_xyz_m"], float),
                    residual_r=np.asarray(row["residual_rotation_vector_rad"], float),
                    model_p=np.asarray(model["model_shift_m"], float),
                    model_r=np.asarray(model["model_rotation_vector_rad"], float),
                    velocity=np.asarray(model["filtered_reference_velocity_rad_s"], float)))
            elif (event == "pose_replay_feedback" and phase in feedback
                  and row.get("integrator_accepted")):
                feedback[phase].append(dict(
                    time=float(row["feedback_pair_ns"])/1e9,
                    nominal_p=np.asarray(row["nominal_left_pos"][:3], float),
                    nominal_r=quaternion_rotation(row["nominal_left_pos"][3:]).as_matrix(),
                    actual_p=np.asarray(row["actual_left_pos"][:3], float),
                    actual_r=quaternion_rotation(row["actual_left_pos"][3:]).as_matrix()))
    return commands, feedback, event_counts


def interpolation_arrays(rows):
    rows = sorted(rows, key=lambda row: row["time"])
    times = np.asarray([row["time"] for row in rows])
    if len(times) < 2 or np.any(np.diff(times) <= 0):
        raise ValueError("Command timestamps are insufficient or not strictly increasing")
    arrays = {key: np.asarray([row[key] for row in rows]) for key in
              ("preview_p", "correction_p", "residual_p", "residual_r",
               "model_p", "model_r", "velocity")}
    arrays["preview_r"] = Rotation.from_matrix([row["preview_r"] for row in rows])
    arrays["correction_r"] = Rotation.from_matrix([row["correction_r"] for row in rows])
    return times, arrays


def at(times, values, query):
    values = np.asarray(values)
    return np.array([np.interp(query, times, values[:, column]) for column in range(values.shape[1])])


def estimate_pairing_delay(commands, feedback, configured_preview_s=.124):
    """Pair a command's preview state with the same nominal state in feedback."""
    times, arrays = interpolation_arrays(commands)
    feedback = sorted(feedback, key=lambda row: row["time"])
    ft = np.asarray([row["time"] for row in feedback])
    nominal = np.asarray([row["nominal_p"] for row in feedback])
    if len(ft) < 10 or np.linalg.norm(np.ptp(nominal, axis=0)) < .005:
        return configured_preview_s, None
    candidates = np.arange(.080, .141, .001)
    scores = []
    for delay in candidates:
        query = ft-delay
        valid = (query >= times[0]) & (query <= times[-1])
        if valid.sum() < 10:
            scores.append(np.inf)
            continue
        preview = np.column_stack([np.interp(query[valid], times, arrays["preview_p"][:, j]) for j in range(3)])
        scores.append(float(np.sqrt(np.mean(np.sum((preview-nominal[valid])**2, axis=1)))))
    best = int(np.argmin(scores))
    if not np.isfinite(scores[best]):
        return configured_preview_s, None
    return float(candidates[best]), float(scores[best])


def samples_for_phase(run, phase, commands, feedback, delay, settle_limit=20):
    if len(commands) < 2 or not feedback:
        return []
    times, arrays = interpolation_arrays(commands)
    feedback = sorted(feedback, key=lambda row: row["time"])
    if phase == "settle_pose":
        feedback = feedback[-settle_limit:]
        delay = 0.
    samples = []
    slerp_preview = Slerp(times, arrays["preview_r"])
    slerp_correction = Slerp(times, arrays["correction_r"])
    for row in feedback:
        query = row["time"]-delay
        if not times[0] <= query <= times[-1]:
            continue
        preview_p = at(times, arrays["preview_p"], query)
        preview_r = slerp_preview(query).as_matrix()
        current_p = at(times, arrays["correction_p"], query)
        current_r = slerp_correction(query).as_matrix()
        residual_p = at(times, arrays["residual_p"], query)
        residual_r = at(times, arrays["residual_r"], query)
        model_p = at(times, arrays["model_p"], query)
        model_r_vector = at(times, arrays["model_r"], query)
        velocity = at(times, arrays["velocity"], query)
        error_p = row["nominal_p"]-row["actual_p"]
        error_r = row["nominal_r"]@row["actual_r"].T
        target_p = current_p+error_p
        target_r = error_r@current_r
        target_r_vector = Rotation.from_matrix(target_r).as_rotvec()
        residual = np.r_[target_p-model_p,
                         Rotation.from_matrix(target_r@Rotation.from_rotvec(model_r_vector).as_matrix().T).as_rotvec()]
        current_error_r = Rotation.from_matrix(error_r).as_rotvec()
        if (not np.isfinite(residual).all() or np.linalg.norm(error_p) >= .05
                or np.linalg.norm(current_error_r) >= np.deg2rad(15)
                or np.linalg.norm(target_p) >= .06 or np.linalg.norm(target_r_vector) >= np.deg2rad(8)):
            continue
        samples.append(dict(run=run, phase=phase,
            x=feature_vector(model_p, model_r_vector, velocity,
                             current_p, Rotation.from_matrix(current_r).as_rotvec(),
                             residual_p, residual_r),
            y=np.r_[error_p, current_error_r], target=np.r_[target_p, target_r_vector],
            physical=np.r_[model_p, model_r_vector],
            current=np.r_[current_p, Rotation.from_matrix(current_r).as_rotvec()],
            current_error=np.r_[error_p, current_error_r]))
    return samples


def group_folds(groups, folds):
    counts = Counter(groups)
    bins = [[] for _ in range(folds)]
    sizes = np.zeros(folds, int)
    for group, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        index = int(np.argmin(sizes))
        bins[index].append(group)
        sizes[index] += count
    return bins


def robust_ridge(x, y, groups, strength, iterations=8):
    mean = np.mean(x, axis=0)
    scale = np.std(x, axis=0)
    scale[scale < 1e-9] = 1.
    design = np.column_stack([np.ones(len(x)), (x-mean)/scale])
    counts = Counter(groups)
    base = np.asarray([1/counts[group] for group in groups], float)
    base /= np.mean(base)
    coefficients = []
    penalty = np.eye(design.shape[1]);penalty[0, 0] = 0
    for column in range(y.shape[1]):
        weights = base.copy()
        beta = np.zeros(design.shape[1])
        for _ in range(iterations):
            root = np.sqrt(weights)
            lhs = (design*root[:, None]).T@(design*root[:, None])+strength*len(x)*penalty
            rhs = (design*root[:, None]).T@(y[:, column]*root)
            beta = np.linalg.solve(lhs, rhs)
            error = y[:, column]-design@beta
            robust_scale = max(1.4826*np.median(np.abs(error-np.median(error))), 1e-7)
            huber = np.minimum(1., 1.5*robust_scale/np.maximum(np.abs(error), 1e-12))
            weights = base*huber
        coefficients.append(beta)
    coefficients = np.asarray(coefficients)
    return dict(mean=mean, scale=scale, intercept=coefficients[:, 0], coefficients=coefficients[:, 1:])


def predict(model, x):
    prediction = (x-model["mean"])/model["scale"]@model["coefficients"].T+model["intercept"]
    gate = np.maximum(np.linalg.norm(x[:, 19:22], axis=1)/.001,
                      np.linalg.norm(x[:, 22:25], axis=1)/np.deg2rad(.2))
    return prediction*np.clip(gate, 0, 1)[:, None]


def compose(base, adjustment):
    result = np.empty_like(base)
    result[:, :3] = base[:, :3]+adjustment[:, :3]
    for index, (existing, extra) in enumerate(zip(base[:, 3:], adjustment[:, 3:])):
        result[index, 3:] = (Rotation.from_rotvec(extra)*Rotation.from_rotvec(existing)).as_rotvec()
    return result


def errors(target, prediction):
    translation = np.linalg.norm(target[:, :3]-prediction[:, :3], axis=1)*1000
    rotation = []
    for wanted, actual in zip(target[:, 3:], prediction[:, 3:]):
        rotation.append(np.rad2deg(Rotation.from_rotvec(wanted).inv().__mul__(Rotation.from_rotvec(actual)).magnitude()))
    return translation, np.asarray(rotation)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-glob", default=str(ROOT/"artifacts/tron2_act/hardware*/trace.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    if not 3 <= args.folds <= 10:
        parser.error("--folds must be within 3..10")
    paths = sorted(Path(path) for path in glob.glob(args.trace_glob))
    if not paths:
        parser.error("No trace files matched")
    args.output.mkdir(parents=True, exist_ok=False)

    all_samples, runs, rejected = [], [], []
    for path in paths:
        try:
            commands, feedback, counts = load_run(path)
            if len(commands["act_control"]) >= 2 and feedback["act_control"]:
                delay, score = estimate_pairing_delay(commands["act_control"], feedback["act_control"])
            else:
                delay, score = .124, None
            run_samples = samples_for_phase(path.parent.name, "act_control", commands["act_control"],
                                            feedback["act_control"], delay)
            run_samples += samples_for_phase(path.parent.name, "settle_pose", commands["settle_pose"],
                                             feedback["settle_pose"], delay)
            if not run_samples:
                rejected.append(dict(path=str(path), reason="no paired pose command/feedback samples",
                                     event_counts=dict(counts)))
                continue
            all_samples.extend(run_samples)
            runs.append(dict(run=path.parent.name, path=str(path.resolve()), sha256=digest(path),
                             pairing_delay_ms=delay*1000,
                             pairing_preview_nominal_rms_mm=None if score is None else score*1000,
                             samples=len(run_samples), phases=dict(Counter(row["phase"] for row in run_samples))))
        except Exception as exc:
            rejected.append(dict(path=str(path), reason=f"{type(exc).__name__}: {exc}"))
    if len(runs) < args.folds:
        raise RuntimeError("Too few complete hardware runs for grouped validation")

    x = np.asarray([row["x"] for row in all_samples])
    y = np.asarray([row["y"] for row in all_samples])
    target = np.asarray([row["target"] for row in all_samples])
    physical = np.asarray([row["physical"] for row in all_samples])
    current = np.asarray([row["current"] for row in all_samples])
    groups = np.asarray([row["run"] for row in all_samples])
    phases = np.asarray([row["phase"] for row in all_samples])
    folds = group_folds(groups, args.folds)
    strengths = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    candidates = []
    predictions_by_strength = {}
    for strength in strengths:
        prediction = np.empty_like(target)
        for held_out in folds:
            test = np.isin(groups, held_out);train = ~test
            model = robust_ridge(x[train], y[train], groups[train], strength)
            predicted_adjustment = predict(model, x[test])
            prediction[test] = compose(current[test], predicted_adjustment)
        translation, rotation = errors(target, prediction)
        act = phases == "act_control"
        score = float(np.median(translation[act])+np.median(rotation[act]))
        candidates.append(dict(strength=strength, selection_score=score,
                               translation_mm=metric(translation[act]), rotation_deg=metric(rotation[act])))
        predictions_by_strength[strength] = prediction
    selected = min(candidates, key=lambda row: row["selection_score"])["strength"]
    oof = predictions_by_strength[selected]
    final_model = robust_ridge(x, y, groups, selected)

    current_translation, current_rotation = errors(target, current)
    physical_translation, physical_rotation = errors(target, physical)
    oof_translation, oof_rotation = errors(target, oof)
    act = phases == "act_control"
    static = phases == "settle_pose"
    validation = {}
    for name, mask in (("act_control", act), ("settle_pose", static), ("all", np.ones(len(x), bool))):
        validation[name] = {
            "samples": int(mask.sum()),
            "existing_controller": {"translation_mm": metric(current_translation[mask]),
                                    "rotation_deg": metric(current_rotation[mask])},
            "physical_feedforward_only": {"translation_mm": metric(physical_translation[mask]),
                                           "rotation_deg": metric(physical_rotation[mask])},
            "hardware_model_grouped_oof": {"translation_mm": metric(oof_translation[mask]),
                                           "rotation_deg": metric(oof_rotation[mask])},
        }

    payload = dict(schema=SCHEMA, model_version="servop-real-residual-20261124-v1",
        created_utc=datetime.now(timezone.utc).isoformat(),
        purpose="Candidate bounded correction on top of the existing closed loop; not enabled for hardware deployment",
        feature_names=FEATURE_NAMES, output_names=OUTPUT_NAMES,
        feature_mean=final_model["mean"].tolist(), feature_scale=final_model["scale"].tolist(),
        adjustment_intercept=final_model["intercept"].tolist(),
        adjustment_coefficients=final_model["coefficients"].tolist(),
        selected_ridge_strength=selected,
        prediction_limits={"maximum_total_translation_m": .04, "maximum_total_rotation_deg": 5.},
        feedback_gate={"full_scale_translation_m": .001, "full_scale_rotation_deg": .2,
                       "zero_residual_zero_adjustment": True},
        training_contract={
            "target": "simultaneous nominal-minus-measured SDK wrist error remaining after the existing closed loop",
            "base": "existing MuJoCo pose-reference feed-forward plus bounded online pose feedback",
            "validation": "whole-run grouped out-of-fold counterfactual replay",
            "settle_samples_per_run_maximum": 20,
            "hardware_activation": False,
        },
        source_summary={"trace_files_scanned": len(paths), "trace_files_used": len(runs),
                        "paired_samples": len(all_samples), "run_names": [row["run"] for row in runs]},
    )
    model_path = args.output/"model.json"
    model_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2)+"\n")
    report = dict(schema="tron2-servop-hardware-fit-report-v1",
        model=str(model_path.resolve()), model_sha256=digest(model_path),
        source=dict(trace_glob=args.trace_glob, files_scanned=len(paths), runs=runs, rejected=rejected),
        data=dict(samples=len(all_samples), feature_count=x.shape[1],
                  phase_counts=dict(Counter(phases)), run_count=len(runs)),
        pairing_delay_ms=metric([row["pairing_delay_ms"] for row in runs]),
        cross_validation=dict(folds=folds, candidates=candidates, selected_strength=selected,
                              metrics=validation),
        interpretation={
            "translation": "Counterfactual SDK-wrist tracking error after adding the learned bounded adjustment; local ServoP response is assumed unity.",
            "rotation": "Same counterfactual calculation on SO(3).",
            "limitations": [
                "SDK wrist pose is joint-derived feedback, not an external optical ground truth.",
                "All useful hardware runs were collected in a narrow task/workspace and on one robot.",
                "No cup contact, force, tactile, task-success label, or failed-start trace is used as a correction target.",
                "The final model uses all usable runs; accuracy estimates are exclusively whole-run out-of-fold predictions.",
            ],
        })
    (args.output/"report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    np.savez_compressed(args.output/"paired_samples.npz", features=x, residual_targets=y,
                        total_targets=target, physical_predictions=physical,
                        existing_predictions=current, grouped_oof_predictions=oof,
                        groups=groups, phases=phases)
    print(json.dumps({"output": str(args.output.resolve()), "runs": len(runs), "samples": len(all_samples),
                      "selected_strength": selected, "act_control": validation["act_control"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

