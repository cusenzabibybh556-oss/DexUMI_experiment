#!/usr/bin/env python3
"""Frozen-controller speed/translation-scale sweep in offline MuJoCo only.

The strict source cohort and profile are frozen before execution. No robot SDK
or network calls; original recordings and prior results remain read-only.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
from dataclasses import asdict, replace
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.validate_tron2_episode_batch import (DEFAULT_PROFILE, DEFAULT_MAPPING, EpisodeReference,
    settings_from_profile, read_episode, motion_digest)
from tools.simulate_tron2_z_feedback import MODEL, experiment
from tracker_replay_mapping import load_calibration

BASELINE = ROOT/'validation_results/episode_0_100_pose_sim_20261103'
LABELS = {'continuous_Z': 'continuous', 'full_pose': 'pose_reference_ff_feedback'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cell_key(speed, scale):
    def number(x): return format(x, '.8g').replace('.', 'p')
    return f'speed_{number(speed)}_scale_{number(scale)}'


class ScaledEpisodeReference(EpisodeReference):
    def __init__(self, path, mapping, speed, scale):
        if isinstance(scale, bool) or not np.isfinite(scale) or not 0 < scale <= 3:
            raise ValueError('Translation scale must be positive, finite, and at most 3')
        super().__init__(path, mapping, speed)
        if not self.audit['strict_replay_eligible']:
            raise ValueError('Sweep requires strictly replay-eligible source data')
        self.delta = self.delta*scale
        self.metadata.update(scale=float(scale), orientation_scale=1.,
            scale_contract='Translation-only scaling about first pose; rotations unchanged',
            speed_contract='Source timestamps divided by speed; physical delay/preview not rescaled')


def cached_baseline(job, reference, settings, folder):
    if not job['cache_verified'] or job['speed'] != .5 or job['scale'] != 1.:
        return None
    old_folder = Path(job['baseline'])/f"episode_{job['episode_id']:03d}"/job['case']
    old_path = old_folder/'result.json'
    if not old_path.exists(): return None
    old = json.loads(old_path.read_text())
    if (old['status'] != 'completed' or old['physics_settings'] != asdict(settings)
            or old['reference']['source_motion_sha256'] != reference.metadata['source_motion_sha256']
            or old['reference']['speed'] != .5 or old['reference']['scale'] != 1.
            or old['reference']['orientation_scale'] != 1.):
        return None
    evidence = {}
    for source in old_folder.iterdir():
        if source.suffix in ('.npz', '.xml') or source.name == 'console.txt':
            os.link(source, folder/source.name)
            evidence[source.name] = digest(source)
    return dict(metrics=old['metrics'], source_result=str(old_path),
        source_result_sha256=digest(old_path), artifact_sha256=evidence)


def run_job(job):
    folder = Path(job['folder']); folder.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    record = {k: job[k] for k in ['episode_id', 'speed', 'scale', 'case']}
    record['cell'] = cell_key(job['speed'], job['scale'])
    record['artifact_directory'] = str(folder)
    try:
        profile = json.loads(Path(job['profile']).read_text())
        if not profile['simulation_only'] or profile['hardware_executable']:
            raise ValueError('Expected a frozen simulation-only profile')
        mapping = load_calibration(job['calibration'])
        reference = ScaledEpisodeReference(Path(job['source']), mapping, job['speed'], job['scale'])
        if reference.metadata['source_motion_sha256'] != job['expected_motion_sha256']:
            raise ValueError('Source motion changed after cohort freeze')
        settings = replace(settings_from_profile(profile), feedback_pairing='capture',
            reference_preview_s=profile['physics_settings']['reference_preview_s'] if job['case']=='full_pose' else 0.)
        record.update(reference=reference.metadata, physics_settings=asdict(settings))
        cached = cached_baseline(job, reference, settings, folder)
        if cached is not None:
            metrics = cached.pop('metrics'); record['reused_baseline'] = cached
        else:
            payload = profile['payload_model']
            with (folder/'console.txt').open('w') as log, redirect_stdout(log):
                metrics, _ = experiment(MODEL, payload['mass_kg'], payload['COM_wrist_m'], LABELS[job['case']],
                    .02, 3, folder, reference, settings, payload)
        assert not metrics['explicit_torque_feedforward']
        record.update(status='completed', metrics=metrics)
    except Exception as exc:
        record.update(status='rejected', error_type=type(exc).__name__, error=str(exc))
    record['wall_seconds'] = time.monotonic()-started
    (folder/'result.json').write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
    return record


def plan_jobs(args, cohort, cache_verified):
    return [dict(episode_id=row['episode_id'], source=row['source'],
        expected_motion_sha256=row['audit']['motion_input_sha256'], speed=speed, scale=scale, case=case,
        folder=str((args.output/cell_key(speed, scale)/f"episode_{row['episode_id']:03d}"/case).resolve()),
        profile=str(args.profile.resolve()), calibration=str(args.calibration.resolve()),
        baseline=str(args.baseline.resolve()), cache_verified=cache_verified)
        for speed in args.speeds for scale in args.scales for row in cohort for case in LABELS]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--baseline', type=Path, default=BASELINE)
    parser.add_argument('--profile', type=Path, default=DEFAULT_PROFILE)
    parser.add_argument('--calibration', type=Path, default=DEFAULT_MAPPING)
    parser.add_argument('--speeds', nargs='+', type=float, default=[.2, .5, 1., 1.5])
    parser.add_argument('--scales', nargs='+', type=float, default=[.5, 1., 1.5])
    parser.add_argument('--episodes', nargs='+', type=int)
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--no-cache', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if (not 1 <= args.workers <= 16 or any(not np.isfinite(x) or not 0 < x <= 3 for x in args.speeds+args.scales)
            or len(set(args.speeds)) != len(args.speeds) or len(set(args.scales)) != len(args.scales)):
        parser.error('Speeds/scales must be unique, finite, in (0,3]; workers in [1,16]')
    audit = json.loads((args.baseline/'acquisition_audit.json').read_text())
    cohort = [r for r in audit if r['status']=='audited' and r['audit']['strict_replay_eligible']]
    if args.episodes is not None:
        if len(set(args.episodes)) != len(args.episodes) or not set(args.episodes) <= {r['episode_id'] for r in cohort}:
            parser.error('Requested episodes must belong to the frozen strict cohort')
        cohort = [r for r in cohort if r['episode_id'] in args.episodes]
    for row in cohort:
        if motion_digest(read_episode(Path(row['source']))[0]) != row['audit']['motion_input_sha256']:
            raise ValueError(f"Source changed since baseline: episode{row['episode_id']}")
    profile = json.loads(args.profile.read_text())
    sources = [Path(__file__), ROOT/'tools/validate_tron2_episode_batch.py', ROOT/'tools/simulate_tron2_z_feedback.py',
        ROOT/'tron2_pose_compensation.py', ROOT/'tron2_z_compensation.py', ROOT/'tracker_replay_mapping.py',
        ROOT/'tools/tron2_payload_model.py', MODEL, args.profile, args.calibration, Path(profile['payload_model']['source'])]
    source_hashes = {str(p.resolve()): digest(p) for p in sources}
    old_manifest_path = args.baseline/'manifest.json'
    old_manifest = json.loads(old_manifest_path.read_text())
    cache_verified = not args.no_cache and all(Path(p).exists() and digest(p)==v
        for p,v in old_manifest['source_sha256'].items()) and old_manifest['frozen_profile']==profile
    manifest = dict(schema='tron2-frozen-speed-scale-sweep-v1', hardware_connections=0,
        hardware_parameter_writes=0, speeds=args.speeds, translation_scales=args.scales, orientation_scale=1.,
        cohort=cohort, frozen_profile=profile, source_sha256=source_hashes,
        baseline_manifest_sha256=digest(old_manifest_path), cache_verified=cache_verified,
        baseline_manifest=str(old_manifest_path.resolve()), control_parameters_refit=False,
        preview_rescaled_with_speed=False, joint_or_pose_limits_relaxed=False,
        metric_contract='Original nominal time; whole-trajectory completed cases only; rejected cases remain in denominators')
    if args.resume:
        previous = json.loads((args.output/'manifest.json').read_text())
        if previous != manifest: raise ValueError('Resume requires unchanged code, sources, cohort, and sweep settings')
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    jobs = plan_jobs(args, cohort, cache_verified)
    results, todo = [], []
    for job in jobs:
        target = Path(job['folder'])/'result.json'
        if args.resume and target.exists(): results.append(json.loads(target.read_text()))
        else:
            if Path(job['folder']).exists():
                raise RuntimeError(f"Incomplete output directory needs review before resuming: {job['folder']}")
            todo.append(job)
    print('PLAN',len(cohort),'episodes',len(args.speeds)*len(args.scales),'cells',len(jobs),'comparisons',
          'cache',cache_verified,'resume_results',len(results),flush=True)
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = [pool.submit(run_job,job) for job in todo]
        for future in as_completed(futures):
            record=future.result();results.append(record)
            print('CASE',record['cell'],record['episode_id'],record['case'],record['status'],
                  'cached' if 'reused_baseline' in record else record.get('error',''),flush=True)
            (args.output/'progress.json').write_text(json.dumps(dict(done=len(results), total=len(jobs),
                completed=sum(r['status']=='completed' for r in results), rejected=sum(r['status']=='rejected' for r in results),
                reused=sum('reused_baseline' in r for r in results)),indent=2)+'\n')
    results.sort(key=lambda r:(r['speed'],r['scale'],r['episode_id'],r['case']))
    (args.output/'cases.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
    print('FINISHED',len(results),flush=True)


if __name__=='__main__':
    main()
