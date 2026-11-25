#!/usr/bin/env python3
"""Offline acquisition audit and frozen-controller MuJoCo episode comparisons.

Never imports a robot client or collector. Original HDF5 files are read-only.
Exact duplicate motion samples may be removed in a diagnostic view. Gaps remain
in time and are explicitly marked as unsuitable for strict real replay.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
from dataclasses import asdict, replace
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import h5py
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tracker_replay_mapping import load_calibration, calibrated_relative_motion
from tools.simulate_tron2_z_feedback import MODEL, PhysicsSettings, experiment
from tron2_pose_compensation import PoseConfig
from tron2_z_compensation import Config

DEFAULT_DATA = Path('/home/zmy/workplace/insert-data/transhandr_demo/o6')
DEFAULT_PROFILE = ROOT/'validation_results/pose_timing_pipeline_20261103/candidate_simulation_profile.json'
DEFAULT_MAPPING = ROOT/'tracker_calibrations/left_mapping_final_20261027/calibration.json'


def stats(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return dict(count=int(len(x)), median=float(np.median(x)) if len(x) else None,
                p95=float(np.percentile(x, 95)) if len(x) else None,
                maximum=float(np.max(x)) if len(x) else None)


def motion_digest(arrays):
    h = hashlib.sha256()
    for key in sorted(arrays):
        a = np.ascontiguousarray(arrays[key])
        h.update(key.encode()); h.update(str(a.shape).encode()); h.update(a.dtype.str.encode()); h.update(a.tobytes())
    return h.hexdigest()


def read_episode(path):
    with h5py.File(path, 'r') as f:
        a = {key: np.asarray(f[key][:]) for key in [
            'timestamps/master_timestamp_ns', 'vive_tracker/pos', 'vive_tracker/quat', 'vive_tracker/valid']}
        optional = ['vive_tracker/source_timestamp', 'vive_tracker/unified_timestamp_ns',
            'vive_tracker/host_receive_timestamp_ns', 'sync/tracker_interpolation_gap_ms',
            'camera/d405/frame_number', 'camera/d405/device_timestamp']
        a.update({key: np.asarray(f[key][:]) for key in optional if key in f})
        attrs = dict(position_unit=str(f['vive_tracker'].attrs.get('position_unit', '')),
            quaternion_order=str(f['vive_tracker'].attrs.get('quat_order', '')),
            hand_side=str(f.attrs.get('hand_side', 'unknown')),
            schema=str(f.attrs.get('schema_version', 'unknown')),
            tracker_max_gap_ms=float(f.attrs.get('tracker_max_gap_ms', 30)),
            master_source=str(f.attrs.get('master_source', 'unknown')),
            map_identity_present=any('map_uuid' in k.lower() or 'map_id' in k.lower() for k in f.attrs))
    return a, attrs


def inspect_arrays(arrays, attrs, max_gap_s=.15):
    t = np.asarray(arrays['timestamps/master_timestamp_ns'], np.int64)
    p = np.asarray(arrays['vive_tracker/pos'][:, 0], float)
    q = np.asarray(arrays['vive_tracker/quat'][:, 0], float)
    v = np.asarray(arrays['vive_tracker/valid'][:, 0], bool)
    if not len(t) or p.shape != (len(t), 3) or q.shape != (len(t), 4) or v.shape != t.shape:
        raise ValueError('Inconsistent or empty source arrays')
    norm = np.linalg.norm(q, axis=1)
    finite = np.isfinite(p).all(axis=1) & np.isfinite(q).all(axis=1) & (norm > 1e-6)
    good = np.flatnonzero(v & finite)
    if len(good) < 2:
        raise ValueError('Fewer than two valid source poses')
    dt = np.diff(t[good])/1e9
    duplicate = np.flatnonzero(dt == 0)
    identical = [j for j in duplicate if np.array_equal(p[good[j]], p[good[j+1]])
        and (np.array_equal(q[good[j]], q[good[j+1]]) or np.array_equal(q[good[j]], -q[good[j+1]]))]
    if (dt < 0).any() or len(identical) != len(duplicate):
        diagnostic_indices = None
    else:
        keep = np.ones(len(good), bool); keep[np.array(identical, int)+1] = False
        diagnostic_indices = good[keep]
    proper = attrs['position_unit'] in ('meter', 'meters', 'm') and attrs['hand_side'] == 'left'
    proper &= ''.join(c for c in attrs['quaternion_order'].lower() if c.isalpha()) == 'wxyz'
    strict_reasons = []
    if not proper: strict_reasons.append('Unexpected units, quaternion metadata, or hand side')
    if np.any(dt <= 0): strict_reasons.append('Non-increasing valid master timestamps')
    if dt.max() > max_gap_s: strict_reasons.append(f'Valid source gap {dt.max():.6f}s exceeds {max_gap_s}s')
    positive = dt > 0
    dp = np.diff(p[good], axis=0)
    # Any consistent permutation of quaternion components preserves the
    # inter-frame angle magnitude; convention semantics audited separately.
    rotation = Rotation.from_quat(q[good][:, [1, 2, 3, 0]])
    angles = np.rad2deg((rotation[1:]*rotation[:-1].inv()).magnitude())
    speed = np.linalg.norm(dp[positive], axis=1)/dt[positive]
    angular_speed = angles[positive]/dt[positive]
    report = dict(frames=len(t), valid_pose_frames=len(good), invalid_pose_frames=int(len(t)-len(good)),
        motion_input_sha256=motion_digest(arrays), attributes=attrs,
        source_duration_s=float((t[good[-1]]-t[good[0]])/1e9),
        duplicate_timestamp_pairs=int(len(duplicate)), identical_duplicate_pairs=len(identical),
        duplicate_source_rows=good[duplicate+1].tolist(), backward_timestamp_pairs=int((dt < 0).sum()),
        source_gap_ms=stats(dt[positive]*1000), gap_over_150ms_count=int((dt > max_gap_s).sum()),
        quaternion_norm_error=stats(abs(norm[good]-1)),
        raw_quaternion_sign_flip_pairs=int((np.sum(q[good][1:]*q[good][:-1], axis=1) < 0).sum()),
        position_step_mm=stats(np.linalg.norm(dp, axis=1)*1000), rotation_step_deg=stats(angles),
        source_linear_speed_m_s=stats(speed), source_angular_speed_deg_s=stats(angular_speed),
        heuristic_speed_over_1m_s=int((speed > 1).sum()),
        heuristic_rotation_over_360deg_s=int((angular_speed > 360).sum()),
        strict_replay_eligible=not strict_reasons, strict_replay_reasons=strict_reasons,
        diagnostic_view_available=bool(proper and diagnostic_indices is not None),
        source_validity_caveat='HDF5 contains master-aligned interpolated poses, not the original raw pose stream or independent motion ground truth')
    for key in ['sync/tracker_interpolation_gap_ms']:
        if key in arrays: report['recorded_tracker_bracket_gap_ms'] = stats(arrays[key])
    key = 'vive_tracker/unified_timestamp_ns'
    if key in arrays: report['tracker_unified_minus_master_ms'] = stats((arrays[key][:, 0]-t)/1e6)
    key = 'vive_tracker/host_receive_timestamp_ns'
    if key in arrays: report['tracker_bracket_host_minus_master_ms'] = stats((arrays[key][:, 0]-t)/1e6)
    key = 'camera/d405/frame_number'
    if key in arrays:
        df = np.diff(arrays[key][good])
        report['camera_duplicate_frame_pairs'] = int((df == 0).sum())
        report['camera_skipped_frame_numbers'] = int(np.maximum(df-1, 0).sum())
        report['timestamp_duplicate_with_camera_duplicate'] = int((df[duplicate] == 0).sum())
    key = 'vive_tracker/source_timestamp'
    if key in arrays:
        src = arrays[key][:, 0][good]
        report['tracker_interpolated_source_step_ms'] = stats(np.diff(src))
    return report, diagnostic_indices


class EpisodeReference:
    def __init__(self, path, mapping, speed=.5, max_gap_s=.15):
        if not np.isfinite(speed) or speed <= 0:
            raise ValueError('Invalid replay speed')
        arrays, attrs = read_episode(path)
        self.audit, indices = inspect_arrays(arrays, attrs, max_gap_s)
        if not self.audit['diagnostic_view_available']:
            raise ValueError('Source cannot form a consistent diagnostic trajectory')
        t = arrays['timestamps/master_timestamp_ns'][indices]
        self.times = (t-t[0])/1e9/speed
        p = np.asarray(arrays['vive_tracker/pos'][indices, 0], float)
        q = np.asarray(arrays['vive_tracker/quat'][indices, 0], float)
        trajectory = SimpleNamespace(positions=p, quaternions=q)
        self.delta, dq, _ = calibrated_relative_motion(trajectory, mapping, 1., 1.,
            mode='tracker-relative', quaternion_format='pyvut-legacy')
        self.rotations = Slerp(self.times, Rotation.from_quat(dq[:, [1, 2, 3, 0]]))
        self.duration = float(self.times[-1])
        declared = Rotation.from_quat(q[:, [1, 2, 3, 0]])
        corrected = Rotation.from_quat(q[:, [0, 3, 2, 1]])
        rd = declared*declared[0].inv(); rc = corrected*corrected[0].inv()
        self.audit['relative_rotation_semantic_difference_deg'] = stats(np.rad2deg((rd*rc.inv()).magnitude()))
        self.audit['diagnostic_removed_duplicate_frames'] = self.audit['identical_duplicate_pairs']
        self.audit['mapped_maximum_rise_mm'] = float(self.delta[:, 2].max()*1000)
        self.audit['mapped_vertical_range_mm'] = float(np.ptp(self.delta[:, 2])*1000)
        self.audit['mapped_maximum_displacement_mm'] = float(np.linalg.norm(self.delta, axis=1).max()*1000)
        self.metadata = dict(episode=str(path), episode_id=int(path.stem.split('_')[-1]), speed=speed,
            scale=1., orientation_scale=1., trajectory_duration_s=self.duration,
            source_motion_sha256=self.audit['motion_input_sha256'], calibration_sha256=mapping['sha256'],
            mapping='Existing calibrated tracker-relative mapping, pyvut-legacy quaternion correction; reanchored to simulation neutral',
            strict_replay_eligible=self.audit['strict_replay_eligible'],
            diagnostic_removed_exact_duplicate_frames=self.audit['identical_duplicate_pairs'],
            preserved_source_gaps_over_150ms=self.audit['gap_over_150ms_count'],
            interpolation='Linear position and shortest-path SO(3); original duration preserved; no trajectory scale fitting')

    def target(self, t, anchor, initial_rotation):
        t = float(np.clip(t, 0, self.duration))
        p = np.asarray(anchor)+np.array([np.interp(t, self.times, self.delta[:, j]) for j in range(3)])
        return p, self.rotations(t).as_matrix()@initial_rotation


def settings_from_profile(profile):
    d = dict(profile['physics_settings'])
    d['z'] = Config(**d['z']); d['pose_feedback'] = PoseConfig(**d['pose_feedback'])
    return PhysicsSettings(**d)


MODES = {'none': ('none', 'capture'), 'continuous_Z': ('continuous', 'capture'),
         'full_pose': ('pose_reference_ff_feedback', 'capture'),
         'full_pose_receipt_clock': ('pose_reference_ff_feedback', 'host-midpoint')}


def worker(job):
    source, mapping_path, profile_path, output, speed, label = job
    folder = Path(output); folder.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    record = dict(case=label, episode_id=int(Path(source).stem.split('_')[-1]))
    try:
        mapping = load_calibration(mapping_path)
        profile = json.loads(Path(profile_path).read_text())
        if not profile['simulation_only'] or profile['hardware_executable']:
            raise ValueError('Expected frozen simulation-only profile')
        reference = EpisodeReference(Path(source), mapping, speed)
        mode, pairing = MODES[label]
        settings = replace(settings_from_profile(profile), feedback_pairing=pairing,
                           reference_preview_s=profile['physics_settings']['reference_preview_s'] if label.startswith('full_pose') else 0.)
        record.update(reference=reference.metadata, physics_settings=asdict(settings))
        payload = profile['payload_model']
        with (folder/'console.txt').open('w') as log, redirect_stdout(log):
            metrics, _ = experiment(MODEL, payload['mass_kg'], payload['COM_wrist_m'], mode, .02, 3,
                folder, reference, settings, payload)
        assert not metrics['explicit_torque_feedforward']
        record.update(status='completed', metrics=metrics)
    except Exception as exc:
        record.update(status='rejected', error_type=type(exc).__name__, error=str(exc))
    record['wall_seconds'] = time.monotonic()-start
    (folder/'result.json').write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--profile', type=Path, default=DEFAULT_PROFILE)
    parser.add_argument('--calibration', type=Path, default=DEFAULT_MAPPING)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--episodes', type=int, nargs='+', default=list(range(101)))
    parser.add_argument('--speed', type=float, default=.5)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--modes', choices=list(MODES), nargs='+', default=list(MODES))
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.workers <= 16 or len(set(args.episodes)) != len(args.episodes):
        parser.error('Invalid worker count or repeated episodes')
    args.output.mkdir(parents=True, exist_ok=False)
    mapping = load_calibration(args.calibration)
    audit, available = [], []
    for i in args.episodes:
        path = args.data_root/f'episode_{i:06d}.hdf5'
        row = dict(episode_id=i, source=str(path))
        if not path.exists():
            row['status'] = 'missing'
        else:
            try:
                ref = EpisodeReference(path, mapping, args.speed)
                row.update(status='audited', audit=ref.audit, reference=ref.metadata)
                available.append(path)
            except Exception as exc:
                row.update(status='invalid', error=str(exc))
        audit.append(row)
    (args.output/'acquisition_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2)+'\n')
    sources = [Path(__file__), ROOT/'tools/simulate_tron2_z_feedback.py', ROOT/'tron2_pose_compensation.py',
        ROOT/'tron2_timing_calibration.py', ROOT/'tron2_z_compensation.py', ROOT/'tracker_replay_mapping.py',
        ROOT/'tools/tron2_payload_model.py', MODEL, args.profile, args.calibration,
        ROOT/'TransHandR0725/TransHandR/scripts/multisource_core.py',
        ROOT/'TransHandR0725/TransHandR/scripts/collect_transhandr_multisource.py']
    profile = json.loads(args.profile.read_text())
    urdf = Path(profile['payload_model']['source'])
    sources.append(urdf)
    manifest = dict(hardware_connections=0, hardware_parameter_writes=0,
        requested_episodes=args.episodes, speed=args.speed, modes=args.modes, frozen_profile=profile,
        source_sha256={str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        acquisition_thresholds=dict(strict_max_valid_gap_s=.15, diagnostic_duplicate_policy='Remove only equal timestamp AND identical motion poses; preserve all other timing gaps'),
        control_parameters_fit_on_batch=False, geometry_rescaled_on_batch=False,
        test_set_caveat='82 and 91 were already inspected in development; other source episodes newly evaluated',
        partial_run_policy='Only full-trajectory completed cases enter paired accuracy metrics; rejected cases remain explicit')
    (args.output/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    print('AUDIT', len(audit), 'requested,',len(available),'diagnostic trajectories,',
        sum(r['status']=='audited' and r['audit']['strict_replay_eligible'] for r in audit),'strict', flush=True)
    if args.audit_only: return
    jobs = [(str(p), str(args.calibration), str(args.profile),
        str(args.output/f'episode_{int(p.stem.split("_")[-1]):03d}'/label), args.speed, label)
        for p in available for label in args.modes]
    results = []
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = [pool.submit(worker, job) for job in jobs]
        for future in as_completed(futures):
            r = future.result(); results.append(r)
            print('CASE',r['episode_id'],r['case'],r['status'],r.get('error',''),flush=True)
            (args.output/'progress.json').write_text(json.dumps(dict(done=len(results), total=len(jobs),
                completed=sum(x['status']=='completed' for x in results), rejected=sum(x['status']=='rejected' for x in results)),indent=2)+'\n')
    results.sort(key=lambda r:(r['episode_id'], r['case']))
    (args.output/'cases.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
    print('FINISHED',len(results),flush=True)


if __name__ == '__main__':
    main()
