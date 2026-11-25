#!/usr/bin/env python3
"""离线审计 DexUMI 标定覆盖范围、删组和换参考帧敏感性；不连接硬件。

验证组始终不进入拟合；敏感性不是标定真值误差或统计置信区间。
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from dexumi_wrist_solver import fit_dexumi, tracker_pose_for_calibration
from solve_tron2_tracker_calibration import load_samples, PoseGroup, group_errors, statistics, transform


def matrix_difference(first, second):
    a, b = np.asarray(first), np.asarray(second)
    return {'translation_mm': float(np.linalg.norm(a[:3, 3]-b[:3, 3])*1000),
            'rotation_deg': float(np.rad2deg(Rotation.from_matrix(a[:3, :3] @ b[:3, :3].T).magnitude()))}


def coverage(poses):
    p = np.array([t[:3, 3] for t in poses])
    pairs = [Rotation.from_matrix(a[:3, :3].T @ b[:3, :3]).magnitude()
             for i, a in enumerate(poses) for b in poses[i+1:]]
    return {'xyz_span_mm': (np.ptp(p, axis=0)*1000).tolist(),
            'max_pair_position_mm': float(np.linalg.norm(p[:, None]-p, axis=2).max()*1000),
            'max_pair_rotation_deg': float(np.rad2deg(max(pairs)))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--calibration', type=Path, required=True)
    p.add_argument('--compare', type=Path)
    p.add_argument('--output', type=Path, help='审计输出 JSON；默认放在标定结果旁')
    args = p.parse_args()
    baseline = json.loads(args.calibration.read_text())
    meta, groups, rejected, digest = load_samples(baseline['input_file'])
    if digest != baseline['input_sha256']:
        raise ValueError('原始样本已改变，不能与此结果混用')
    selected=set(baseline['fit_group_ids']) | set(baseline['validation_group_ids'])
    groups=[g for g in groups if g.id in selected]
    groups = [PoseGroup(g.id, g.label, g.robot,
                       tracker_pose_for_calibration(g.tracker, baseline['tracker_quaternion_convention']),
                       g.statuses, g.pair_count) for g in groups]
    held = set(baseline['validation_group_ids'])
    training = [g for g in groups if g.id not in held]
    offset = meta['robot_kinematics']['tcp_offset_m']
    C = transform(np.eye(3), -np.asarray(offset))
    report = {'source': str(args.calibration.resolve()), 'input_sha256': digest,
              'fit_group_ids': [g.id for g in training], 'validation_group_ids': sorted(held),
              'coverage': {'tracker': coverage([g.tracker for g in groups]),
                           'sdk_tcp': coverage([g.robot for g in groups]),
                           'zero_tcp_flange': coverage([g.robot @ C for g in groups])},
              'baseline_metrics': baseline['metrics'], 'rejected_groups': rejected,
              'observability': baseline['observability'],
              'leave_one_out': [], 'reference_sensitivity': [],
              'note': '以下是数据选择敏感性，不是真值误差或置信区间；未重新拟合验证组，未应用于回放。'}
    output = args.output or args.calibration.parent/'sampling_sensitivity_audit.json'

    def save():
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))

    def compare_fit(result, group_id):
        return {'group_id': group_id,
                'T_ET_delta': matrix_difference(result['T_ET'], baseline['T_ET']),
                'T_BW_delta': matrix_difference(result['T_BW'], baseline['T_BW']),
                'validation': result['metrics']['validation'],
                'translation_bound_hit': result['translation_bound_hit']}

    if args.compare:
        old = json.loads(args.compare.read_text())
        if old['tracker_quaternion_convention'] != baseline['tracker_quaternion_convention']:
            raise ValueError('对照结果的输入四元数约定不同')
        old_rows = [group_errors(g, np.array(old['T_ET']), np.array(old['T_WB'])) for g in groups]
        report['previous_mapping_on_new_data'] = {
            'source': str(args.compare.resolve()),
            'all': statistics(old_rows),
            'validation': statistics([r for r in old_rows if r['id'] in held]),
            'T_ET_delta': matrix_difference(baseline['T_ET'], old['T_ET']),
            'T_BW_delta': matrix_difference(baseline['T_BW'], old['T_BW']),
        }
    save()
    for g in training:
        try:
            fit = fit_dexumi([h for h in groups if h.id != g.id], offset, held)
            report['leave_one_out'].append(compare_fit(fit, g.id))
        except ValueError as exc:
            report['leave_one_out'].append({'group_id': g.id, 'error': str(exc)})
        print(f'删组审计 {g.id}', flush=True)
        save()
    for g in training:
        try:
            fit = fit_dexumi([g]+[h for h in groups if h.id != g.id], offset, held)
            report['reference_sensitivity'].append(compare_fit(fit, g.id))
        except ValueError as exc:
            report['reference_sensitivity'].append({'group_id': g.id, 'error': str(exc)})
        print(f'参考帧审计 {g.id}', flush=True)
        save()
    for method in ('leave_one_out', 'reference_sensitivity'):
        rows = [r for r in report[method] if 'error' not in r]
        summary = {'completed': len(rows), 'failed': len(report[method])-len(rows)}
        if rows:
            summary.update({frame: {key: max(r[frame][key] for r in rows)
                                    for key in ('translation_mm', 'rotation_deg')}
                            for frame in ('T_ET_delta', 'T_BW_delta')})
        report[method+'_summary'] = summary
    report['completed'] = True
    save()
    print(json.dumps({k: report[k] for k in ('coverage', 'leave_one_out_summary', 'reference_sensitivity_summary')}, indent=2))
    print(output)


if __name__ == '__main__':
    main()
