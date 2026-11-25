#!/usr/bin/env python3
"""Audit and summarize a completed frozen-parameter offline episode batch."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.validate_tron2_episode_batch import read_episode, motion_digest, stats


def describe(x):
    x = np.asarray(x, float)
    return dict(n=len(x), mean=float(x.mean()), median=float(np.median(x)),
                p05=float(np.percentile(x, 5)), p95=float(np.percentile(x, 95)),
                minimum=float(x.min()), maximum=float(x.max())) if len(x) else dict(n=0)


def comparison(index, ids, base, candidate):
    pairs = [(i, index[i, base]['metrics'], index[i, candidate]['metrics']) for i in ids
             if (i, base) in index and (i, candidate) in index
             and index[i, base]['status'] == index[i, candidate]['status'] == 'completed']
    b = np.array([a['position_norm_P95_mm'] for _, a, _ in pairs])
    c = np.array([a['position_norm_P95_mm'] for _, _, a in pairs])
    reduction = (1-c/b)*100 if len(b) else []
    r = dict(baseline=base, candidate=candidate, requested_episode_count=len(ids),
        paired_complete_count=len(pairs), paired_episode_ids=[i for i, _, _ in pairs],
        unpaired_episode_ids=[i for i in ids if i not in [j for j, _, _ in pairs]],
        baseline_position_P95_mm=describe(b), candidate_position_P95_mm=describe(c),
        paired_position_reduction_percent=describe(reduction),
        baseline_Z_RMSE_mm=describe([a['Z_trajectory_RMSE_mm'] for _, a, _ in pairs]),
        candidate_Z_RMSE_mm=describe([a['Z_trajectory_RMSE_mm'] for _, _, a in pairs]),
        baseline_orientation_P95_deg=describe([a['orientation_P95_deg'] for _, a, _ in pairs]),
        candidate_orientation_P95_deg=describe([a['orientation_P95_deg'] for _, _, a in pairs]),
        position_improved=sum(int(z < y) for y, z in zip(b, c)), position_worsened=sum(int(z > y) for y, z in zip(b, c)),
        orientation_improved=sum(z['orientation_P95_deg'] < y['orientation_P95_deg'] for _, y, z in pairs),
        both_improved=sum(z['orientation_P95_deg'] < y['orientation_P95_deg'] and
            z['position_norm_P95_mm'] < y['position_norm_P95_mm'] for _, y, z in pairs),
        candidate_pass_5mm_3deg=sum(z['orientation_P95_deg'] <= 3 and z['position_norm_P95_mm'] <= 5 for _, _, z in pairs))
    if len(pairs):
        rng = np.random.default_rng(20261103)
        bootstrap = np.median(np.asarray(reduction)[rng.integers(0, len(pairs), (5000, len(pairs)))], axis=1)
        r['median_reduction_bootstrap_95CI_percent'] = np.percentile(bootstrap, [2.5, 97.5]).tolist()
    return r


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args(); out = args.directory
    manifest = json.loads((out/'manifest.json').read_text())
    for path, digest in manifest['source_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
    acquisition = json.loads((out/'acquisition_audit.json').read_text())
    cases = json.loads((out/'cases.json').read_text())
    index = {(c['episode_id'], c['case']): c for c in cases}
    available = [a['episode_id'] for a in acquisition if a['status'] == 'audited']
    strict = [a['episode_id'] for a in acquisition if a['status'] == 'audited' and a['audit']['strict_replay_eligible']]
    newly_checked = [i for i in strict if i not in [82, 91]]
    for a in acquisition:
        if a['status'] == 'audited':
            assert motion_digest(read_episode(Path(a['source']))[0]) == a['audit']['motion_input_sha256'], a['source']
    summary = dict(requested=len(acquisition), available=len(available),
        missing=[a['episode_id'] for a in acquisition if a['status'] == 'missing'],
        invalid=[a for a in acquisition if a['status'] == 'invalid'], strict_count=len(strict),
        diagnostic_only=[i for i in available if i not in strict], newly_checked_strict_count=len(newly_checked),
        planned_simulations=len(available)*len(manifest['modes']), finished_cases=len(cases),
        completed=sum(c['status']=='completed' for c in cases), rejected=[c for c in cases if c['status']!='completed'],
        comparisons={})
    for label, ids in [('strict', strict), ('diagnostic_inclusive', available), ('new_strict_excluding82_91', newly_checked)]:
        summary['comparisons'][label] = [comparison(index, ids, b, c) for b, c in [
            ('none', 'continuous_Z'), ('none', 'full_pose'), ('continuous_Z', 'full_pose'),
            ('full_pose', 'full_pose_receipt_clock')]]
    summary['quality'] = dict(
        duplicate_episodes=[a['episode_id'] for a in acquisition if a.get('audit', {}).get('duplicate_timestamp_pairs', 0)],
        duplicate_pairs=sum(a.get('audit', {}).get('duplicate_timestamp_pairs', 0) for a in acquisition),
        gap_episodes=[a['episode_id'] for a in acquisition if a.get('audit', {}).get('gap_over_150ms_count', 0)],
        gap_count=sum(a.get('audit', {}).get('gap_over_150ms_count', 0) for a in acquisition),
        maximum_gap_ms=max(a['audit']['source_gap_ms']['maximum'] for a in acquisition if a['status']=='audited'),
        maximum_quaternion_norm_error=max(a['audit']['quaternion_norm_error']['maximum'] for a in acquisition if a['status']=='audited'),
        map_identity_missing_count=sum(not a['audit']['attributes']['map_identity_present'] for a in acquisition if a['status']=='audited'))
    summary['limitations'] = [
        'All errors are relative to mapped recorded Tracker targets, not independently measured human wrist motion.',
        'Shared physics and feedforward model, approximate PD gains, no contact, static aggregate payload; no validated digital twin.',
        'Diagnostic inclusive trajectories drop only identical duplicate poses and bridge original time gaps; original strict replay would reject these records.',
        'Recorded Tracker source time is interpolated; current pyvut timestamps host parsing time, not verified sensor exposure/capture time.',
        'Previously calibrated world-axis mapping reused; historical recording map identity and human wrist-to-Tracker transform unverified.',
        'Uniform speed .5; not a cross-speed or real-robot precision guarantee.',
        'New episodes evaluated with frozen settings; no per-episode refit. Episode82 and91 had been seen previously.']
    audits, flat = [], []
    for c in cases:
        row = dict(episode=c['episode_id'], case=c['case'], status=c['status'],
            strict_replay_eligible=c.get('reference', {}).get('strict_replay_eligible'), error=c.get('error', ''))
        if c['status'] == 'completed':
            m = c['metrics']; mode=m['mode']
            folder = out/f"episode_{c['episode_id']:03d}"/c['case']
            d = np.load(folder/f'mass_0.396_{mode}.npz')
            p = np.load(folder/f'mass_0.396_{mode}_pose.npz')
            active = (d['time_s'] >= 30) & (d['time_s'] <= m['trajectory_end_s'])
            e = (d['actual_xyz_m']-d['nominal_xyz_m'])*1000
            p95 = np.percentile(np.linalg.norm(e[active], axis=1), 95)
            zrmse = np.sqrt(np.mean(e[active, 2]**2))
            rotation = np.rad2deg(Rotation.from_matrix(p['actual_rotation']@np.swapaxes(p['nominal_rotation'], 1, 2)).magnitude())
            assert abs(p95-m['position_norm_P95_mm']) < 1e-8
            assert abs(zrmse-m['Z_trajectory_RMSE_mm']) < 1e-8
            assert abs(np.percentile(rotation[active], 95)-m['orientation_P95_deg']) < 1e-8
            dt = np.diff(np.rint(d['time_s']*1e9).astype(np.int64))/1e9
            fp = p['residual_translation_m']; fr = Rotation.from_rotvec(p['residual_rotation_vector_rad'])
            pc = np.linalg.norm(fp, axis=1).max(); rc=np.rad2deg(fr.magnitude()).max()
            pv=(np.linalg.norm(np.diff(fp, axis=0), axis=1)/dt).max()
            rv=(np.rad2deg((fr[1:]*fr[:-1].inv()).magnitude())/dt).max()
            assert pc <= .01+1e-10 and rc <= 2+1e-8 and pv <= .003+1e-10 and rv <= 1+1e-8
            generated=d['generated_joint_reference_rad']; delivered=d['delivered_joint_reference_rad']; t=d['time_s']
            expected=np.column_stack([np.interp(t-c['physics_settings']['reference_delay_s'], t, generated[:,j]) for j in range(16)])
            assert c['physics_settings']['reference_filter_tau_s'] == 0
            assert np.max(abs(expected-delivered)) < 1e-9
            row.update(position_P95_mm=m['position_norm_P95_mm'], Z_RMSE_mm=m['Z_trajectory_RMSE_mm'],
                orientation_P95_deg=m['orientation_P95_deg'], position_max_mm=float(np.linalg.norm(e[active],axis=1).max()),
                residual_max_mm=pc*1000, residual_max_deg=rc, screening_pass=p95 <= 5 and m['orientation_P95_deg'] <= 3)
            audits.append(dict(episode=c['episode_id'],case=c['case'],samples=int(active.sum()),
                metrics_recomputed=True, residual_bounds_and_rates_verified=True, reference_delay_reconstructed=True))
        flat.append(row)
    fields=['episode','case','status','strict_replay_eligible','position_P95_mm','Z_RMSE_mm','orientation_P95_deg',
        'position_max_mm','residual_max_mm','residual_max_deg','screening_pass','error']
    with (out/'metrics.csv').open('w') as f:
        writer=csv.DictWriter(f,fields); writer.writeheader();writer.writerows(flat)
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    (out/'independent_audit.json').write_text(json.dumps(dict(cases=audits, passed=True,
        original_motion_inputs_unchanged=True, manifest_sources_unchanged=True),indent=2)+'\n')

    fig, ax=plt.subplots(3,1,figsize=(14,10),sharex=True)
    labels={'none':'No correction','continuous_Z':'Continuous Z, capture pairing',
        'full_pose':'Frozen full pose, capture pairing','full_pose_receipt_clock':'Full pose, receipt pairing'}
    for mode in manifest['modes']:
        done=[index[i,mode] for i in available if index[i,mode]['status']=='completed']
        ids=[c['episode_id'] for c in done]
        for a,key in zip(ax,['position_norm_P95_mm','Z_trajectory_RMSE_mm','orientation_P95_deg']):
            a.plot(ids,[c['metrics'][key] for c in done],'.-',linewidth=.8,markersize=4,label=labels[mode])
    ax[0].set_ylabel('3D position P95 (mm)');ax[0].axhline(5,color='grey',linestyle='--');ax[0].legend(fontsize=8)
    ax[1].set_ylabel('Z RMSE (mm)');ax[2].set_ylabel('Orientation P95 (deg)');ax[2].set_xlabel('Episode index')
    for a in ax:
        a.grid(alpha=.2)
        for i in summary['diagnostic_only']:a.axvspan(i-.35,i+.35,color='orange',alpha=.12)
    fig.suptitle('Episode 0-100 simulation: frozen parameters, speed 0.5; orange = acquisition diagnostic only')
    fig.tight_layout();fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)

    fig,ax=plt.subplots(figsize=(7,6))
    for name,ids,color in [('Strict source',strict,'tab:blue'),('Diagnostic source',summary['diagnostic_only'],'tab:orange')]:
        pairs=[(index[i,'continuous_Z'],index[i,'full_pose']) for i in ids
            if index[i,'continuous_Z']['status']==index[i,'full_pose']['status']=='completed']
        b=[a['metrics']['position_norm_P95_mm'] for a,_ in pairs]
        c=[a['metrics']['position_norm_P95_mm'] for _,a in pairs]
        ax.scatter(b,c,label=name,color=color,alpha=.7)
    hi=max(c['metrics']['position_norm_P95_mm'] for c in cases if c['status']=='completed')
    ax.plot([0,hi],[0,hi],'--',color='grey',label='Equal error')
    ax.set_xlabel('Continuous Z: position P95 (mm)');ax.set_ylabel('Full pose: position P95 (mm)')
    ax.set_title('Paired whole-trajectory comparison; lower is better');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'paired_comparison.png',dpi=160);plt.close(fig)
    print(json.dumps({k:summary[k] for k in ['requested','available','strict_count','completed']},ensure_ascii=False))
    for scope,comparisons in summary['comparisons'].items():
        for c in comparisons:
            print(scope,c['baseline'],'->',c['candidate'],'pairs',c['paired_complete_count'],
                'median P95',c['baseline_position_P95_mm'].get('median'),c['candidate_position_P95_mm'].get('median'),
                'median reduction%',c['paired_position_reduction_percent'].get('median'),
                'worse',c['position_worsened'])


if __name__=='__main__':
    main()
