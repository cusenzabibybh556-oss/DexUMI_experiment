#!/usr/bin/env python3
"""Validate a fitted ServoP hardware model without opening robot/CAN/ROS links.

Two independent checks are reported:
1. whole-run out-of-fold counterfactual replay against measured SDK wrist poses;
2. MuJoCo kinematic replay of every available real Tracker demonstration, using
   the same IK, correction magnitude, and correction slew limits as deployment.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import replay_tron2_fixed_start as fixed
from tools.validate_tron2_episode_batch import EpisodeReference
from tracker_replay_mapping import load_calibration
from tron2_hardware_residual import HardwareResidualModel, feature_vector
from tron2_pose_reference import PoseReferenceModel, limit_correction, load_profile


DEFAULT_DATA = Path('/home/zmy/workplace/insert-data/transhandr_demo/o6')
DEFAULT_MAPPING = ROOT/'tracker_calibrations/left_mapping_final_20261027/calibration.json'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats(values):
    values = np.asarray(values, float)
    return dict(count=int(len(values)), median=float(np.median(values)),
                rms=float(np.sqrt(np.mean(values**2))), p95=float(np.percentile(values, 95)),
                maximum=float(np.max(values)))


def pose_errors(target, prediction):
    translation = np.linalg.norm(target[:, :3]-prediction[:, :3], axis=1)*1000
    rotation = np.array([np.rad2deg((Rotation.from_rotvec(wanted)*
        Rotation.from_rotvec(actual).inv()).magnitude())
        for wanted, actual in zip(target[:, 3:], prediction[:, 3:])])
    return translation, rotation


def counterfactual(npz_path):
    arrays = np.load(npz_path)
    target = arrays['total_targets']
    existing = arrays['existing_predictions']
    learned = arrays['grouped_oof_predictions']
    groups = arrays['groups'].astype(str)
    phases = arrays['phases'].astype(str)
    active = phases == 'act_control'
    current_t, current_r = pose_errors(target[active], existing[active])
    learned_t, learned_r = pose_errors(target[active], learned[active])
    by_run = []
    for run in sorted(set(groups[active])):
        selected = active & (groups == run)
        c_t, c_r = pose_errors(target[selected], existing[selected])
        l_t, l_r = pose_errors(target[selected], learned[selected])
        by_run.append(dict(run=run, samples=int(selected.sum()),
            existing_translation_median_mm=float(np.median(c_t)),
            learned_translation_median_mm=float(np.median(l_t)),
            existing_translation_p95_mm=float(np.percentile(c_t, 95)),
            learned_translation_p95_mm=float(np.percentile(l_t, 95)),
            existing_rotation_p95_deg=float(np.percentile(c_r, 95)),
            learned_rotation_p95_deg=float(np.percentile(l_r, 95))))
    return dict(
        samples=int(active.sum()),
        existing=dict(translation_mm=stats(current_t), rotation_deg=stats(current_r)),
        learned_grouped_oof=dict(translation_mm=stats(learned_t), rotation_deg=stats(learned_r)),
        relative_improvement_percent=dict(
            translation_median=float(100*(1-np.median(learned_t)/np.median(current_t))),
            translation_p95=float(100*(1-np.percentile(learned_t,95)/np.percentile(current_t,95))),
            rotation_median=float(100*(1-np.median(learned_r)/np.median(current_r))),
            rotation_p95=float(100*(1-np.percentile(learned_r,95)/np.percentile(current_r,95)))),
        runs_improved=dict(
            translation_median=sum(row['learned_translation_median_mm'] < row['existing_translation_median_mm'] for row in by_run),
            translation_p95=sum(row['learned_translation_p95_mm'] < row['existing_translation_p95_mm'] for row in by_run),
            rotation_p95=sum(row['learned_rotation_p95_deg'] < row['existing_rotation_p95_deg'] for row in by_run),
            total=len(by_run)),
        by_run=by_run,
        _plot=(current_t, learned_t, current_r, learned_r))


def smooth_ramp(t, duration=5.):
    value = float(np.clip(t/duration, 0, 1))
    return value*value*(3-2*value)


def simulate_episode(path, mapping, hardware, rate=20., speed=.5):
    reference = EpisodeReference(path, mapping, speed=speed)
    q0 = np.r_[fixed.FIXED_LEFT_Q, np.zeros(9)]
    engine = PoseReferenceModel(load_profile(), q0)
    anchor, initial_rotation, _ = engine.forward(q0)
    previous_p = np.zeros(3);previous_r = np.eye(3)
    maximum = Counter()
    correction_norms = [];rotation_norms = [];adjustment_norms = [];feature_z = []
    rate_limited = ik_limited = 0
    steps = max(2, int(np.ceil(reference.duration*rate))+1)
    times = np.linspace(0, reference.duration, steps)
    for index, t in enumerate(times):
        dt = 1/rate if index else 1/rate
        target_p, target_r = reference.target(t, anchor, initial_rotation)
        ramp = smooth_ramp(t)
        issued_p, issued_r, info = engine.step(target_p, target_r, dt, weight=ramp)
        current_p = issued_p-target_p
        current_r_matrix = issued_r@target_r.T
        current_r = Rotation.from_matrix(current_r_matrix).as_rotvec()
        model_p = np.asarray(info['model_shift_m'])
        model_r = np.asarray(info['model_rotation_vector_rad'])
        velocity = np.asarray(info['filtered_reference_velocity_rad_s'])
        zeros = np.zeros(3)
        x = feature_vector(model_p, model_r, velocity, current_p, current_r, zeros, zeros)
        adjustment = hardware.predict_adjustment(x)*ramp
        requested_p = current_p+adjustment[:3]
        requested_r = Rotation.from_rotvec(adjustment[3:]).as_matrix()@current_r_matrix
        if np.linalg.norm(requested_p) > hardware.max_translation:
            requested_p *= hardware.max_translation/np.linalg.norm(requested_p)
        vector = Rotation.from_matrix(requested_r).as_rotvec()
        if np.linalg.norm(vector) > hardware.max_rotation:
            requested_r = Rotation.from_rotvec(vector*hardware.max_rotation/np.linalg.norm(vector)).as_matrix()
        correction_p, correction_r, limited = limit_correction(previous_p, previous_r,
            requested_p, requested_r, dt, .04, 5.)
        rate_limited += int(limited)
        candidate_p, candidate_r, fraction = engine.constrain_issued(
            target_p+correction_p, correction_r@target_r)
        ik_limited += int(fraction < 1-1e-9)
        correction_p = candidate_p-target_p
        correction_r = candidate_r@target_r.T
        correction_norms.append(np.linalg.norm(correction_p)*1000)
        rotation_norms.append(np.rad2deg(Rotation.from_matrix(correction_r).magnitude()))
        adjustment_norms.append(np.linalg.norm(adjustment[:3])*1000)
        feature_z.append(np.max(np.abs((x-hardware.mean)/hardware.scale)))
        previous_p, previous_r = correction_p, correction_r
        maximum['reference_velocity_limited'] += int(info.get('reference_velocity_limited', False))
        maximum['reference_ik_limited'] += int(info.get('reference_ik_limited', False))
    return dict(episode_id=int(path.stem.split('_')[-1]), path=str(path.resolve()),
        source_sha256=digest(path), samples=steps, duration_s=reference.duration,
        maximum_correction_translation_mm=float(max(correction_norms)),
        maximum_correction_rotation_deg=float(max(rotation_norms)),
        maximum_learned_adjustment_translation_mm=float(max(adjustment_norms)),
        correction_rate_limited_samples=rate_limited,
        correction_ik_limited_samples=ik_limited,
        model_reference_velocity_limited_samples=maximum['reference_velocity_limited'],
        model_reference_ik_limited_samples=maximum['reference_ik_limited'],
        maximum_standardized_feature_distance=float(max(feature_z)),
        strict_source_replay_eligible=reference.audit['strict_replay_eligible'])


@lru_cache(maxsize=4)
def cached_inputs(mapping_path, model_path):
    return load_calibration(mapping_path), HardwareResidualModel.load(model_path)


def simulation_job(job):
    path, mapping_path, model_path, rate, speed = job
    mapping, hardware = cached_inputs(mapping_path, model_path)
    return simulate_episode(Path(path), mapping, hardware, rate, speed)


def plot_validation(path, current_t, learned_t, current_r, learned_r):
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for axis, old, new, label in ((axes[0], current_t, learned_t, 'Position error (mm)'),
                                   (axes[1], current_r, learned_r, 'Rotation error (deg)')):
        for values, name in ((old, 'Existing controller'), (new, 'Hardware model (held-out runs)')):
            ordered = np.sort(values)
            axis.plot(ordered, np.linspace(0, 100, len(ordered)), label=name)
        axis.set_xlabel(label);axis.set_ylabel('Samples below error (%)');axis.grid(alpha=.25)
        axis.legend()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit-output', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--mapping', type=Path, default=DEFAULT_MAPPING)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rate', type=float, default=20.)
    parser.add_argument('--speed', type=float, default=.5)
    parser.add_argument('--episodes', nargs='*', type=int)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if not 10 <= args.rate <= 50 or not 0 < args.speed <= 1 or not 1 <= args.workers <= 8:
        parser.error('Invalid simulation rate or speed')
    args.output.mkdir(parents=True, exist_ok=False)
    model_path = args.fit_output/'model.json'
    npz_path = args.fit_output/'paired_samples.npz'
    hardware = HardwareResidualModel.load(model_path)
    replay = counterfactual(npz_path)
    plot_values = replay.pop('_plot')
    plot_validation(args.output/'held_out_error_cdf.png', *plot_values)

    paths = sorted(args.data_root.glob('episode_*.hdf5'))
    if args.episodes is not None:
        requested = set(args.episodes)
        paths = [path for path in paths if int(path.stem.split('_')[-1]) in requested]
    episodes, rejected = [], []
    jobs = [(str(path), str(args.mapping), str(model_path), args.rate, args.speed) for path in paths]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(simulation_job, job): Path(job[0]) for job in jobs}
        for future in as_completed(futures):
            path = futures[future]
            try:
                episodes.append(future.result())
            except Exception as exc:
                rejected.append(dict(path=str(path.resolve()), error_type=type(exc).__name__, error=str(exc)))
    episodes.sort(key=lambda row: row['episode_id'])
    rejected.sort(key=lambda row: row['path'])
    if not episodes:
        raise RuntimeError('No demonstration trajectory completed MuJoCo validation')
    maximum_translation = max(row['maximum_correction_translation_mm'] for row in episodes)
    maximum_rotation = max(row['maximum_correction_rotation_deg'] for row in episodes)
    ik_limited = sum(row['correction_ik_limited_samples'] for row in episodes)
    total_steps = sum(row['samples'] for row in episodes)
    improvements = replay['relative_improvement_percent']
    acceptance = dict(
        grouped_oof_translation_median_improved_over_5_percent=improvements['translation_median'] > 5,
        grouped_oof_translation_p95_improved_over_5_percent=improvements['translation_p95'] > 5,
        grouped_oof_rotation_p95_not_worse=improvements['rotation_p95'] >= 0,
        all_requested_demonstrations_simulated=len(rejected) == 0,
        correction_within_40mm=maximum_translation <= 40+1e-9,
        rotation_within_5deg=maximum_rotation <= 5+1e-9,
        candidate_ik_never_projected=ik_limited == 0)
    report = dict(schema='tron2-servop-hardware-validation-v1',
        candidate=dict(path=str(model_path.resolve()), sha256=hardware.sha256,
                       hardware_activation=False),
        real_hardware_counterfactual=replay,
        mujoco_demonstration_replay=dict(
            data_root=str(args.data_root.resolve()), mapping=str(args.mapping.resolve()),
            mapping_sha256=digest(args.mapping), speed=args.speed, rate_hz=args.rate,
            episodes_requested=len(paths), episodes_completed=len(episodes), rejected=rejected,
            total_steps=total_steps, maximum_correction_translation_mm=maximum_translation,
            maximum_correction_rotation_deg=maximum_rotation,
            correction_rate_limited_samples=sum(row['correction_rate_limited_samples'] for row in episodes),
            correction_ik_limited_samples=ik_limited,
            maximum_standardized_feature_distance=max(row['maximum_standardized_feature_distance'] for row in episodes),
            episodes=episodes),
        acceptance=acceptance, accepted=all(acceptance.values()),
        interpretation=[
            'Accuracy is estimated only from whole-run held-out real hardware logs.',
            'MuJoCo replay checks bounds, slew, and IK across real demonstration trajectories; it is not evidence of cup-task success.',
            'The candidate remains disabled in hardware deployment code pending a deliberate on-robot shadow test.',
        ])
    (args.output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    (args.output/'summary.json').write_text(json.dumps({
        'accepted':report['accepted'], 'acceptance':acceptance,
        'relative_improvement_percent':improvements,
        'runs_improved':replay['runs_improved'],
        'episodes_completed':len(episodes), 'episodes_rejected':len(rejected),
        'maximum_correction_translation_mm':maximum_translation,
        'maximum_correction_rotation_deg':maximum_rotation,
        'correction_ik_limited_samples':ik_limited}, ensure_ascii=False, indent=2)+'\n')
    print((args.output/'summary.json').read_text())


if __name__ == '__main__':
    main()
