#!/usr/bin/env python3
"""只读检验冻结标定矩阵；不拟合、不调整矩阵或阈值、不连接任何硬件。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from compare_tron2_calibration_scales import check_compatible, load_diagnostic, matrix_set
from solve_tron2_tracker_calibration import group_errors, statistics


def validate(input_path, calibration_path, continuity_evidence, group_ids=None):
    if not continuity_evidence.strip():
        raise ValueError('须声明Tracker地图和安装连续性，不能仅凭MAC推定')
    raw_calibration = Path(calibration_path).read_bytes()
    cal = json.loads(raw_calibration)
    meta, groups, rejected, digest, event_ids = load_diagnostic(input_path, cal['tracker_quaternion_convention'])
    check_compatible(meta, cal['session'])
    _, _, X, Y = matrix_set(cal, meta['robot_kinematics']['tcp_offset_m'])
    fit_hashes = {s['input_sha256'] for s in cal.get('fit_sources', [])}
    if cal.get('input_sha256'):
        fit_hashes.add(cal['input_sha256'])
    if digest in fit_hashes:
        raise ValueError('新验收文件与拟合来源相同，不得作为全新独立验收')
    if group_ids is not None:
        if len(set(group_ids)) != len(group_ids) or not set(group_ids) <= set(event_ids):
            raise ValueError('指定验收组ID重复或不存在，不能静默改变验收集合')
        groups = [g for g in groups if g.id in group_ids]
    rows = [group_errors(g, X, Y) for g in groups]
    normal = [r for r in rows if r['tracking_statuses'] == [2]]
    position = statistics(normal).get('wrist_position_error_mm', {})
    rotation = statistics(normal).get('rotation_error_deg', {})
    # 沿用原主求解器门槛，只评估，不以新数据重新拟合或放宽门槛。
    gate = {'minimum_normal_groups': 3, 'position_rmse_mm': 5.0, 'position_max_mm': 15.0,
            'rotation_rmse_deg': 2.0, 'rotation_max_deg': 6.0}
    passed = (len(normal) >= gate['minimum_normal_groups']
        and position.get('rmse', float('inf')) <= gate['position_rmse_mm']
        and position.get('max', float('inf')) <= gate['position_max_mm']
        and rotation.get('rmse', float('inf')) <= gate['rotation_rmse_deg']
        and rotation.get('max', float('inf')) <= gate['rotation_max_deg'])
    return {'calibration_file': str(Path(calibration_path).resolve()),
        'calibration_sha256': hashlib.sha256(raw_calibration).hexdigest(),
        'input_file': str(Path(input_path).resolve()), 'input_sha256': digest,
        'same_map_mount_evidence': continuity_evidence, 'tracker_quaternion_convention': cal['tracker_quaternion_convention'],
        'hardware_connected': False, 'refitted': False, 'scale_fitted': False,
        'raw_sample_event_count': len(event_ids), 'normal_group_ids': [r['id'] for r in normal],
        'requested_accuracy_group_ids': group_ids,
        'rejected_groups': rejected, 'metrics_status2_only': statistics(normal),
        'metrics_including_status18_diagnostic': statistics(rows), 'per_group': rows,
        'unchanged_original_acceptance_gate': gate, 'gate_scope': 'new validation data only; original FIT must also pass for full acceptance',
        'passes_gate': bool(passed),
        'warnings': ['status18原始标记保留，只列辅助诊断，未计入正常验收。',
            '不同于拟合文件不自动证明全新独立：需由实验流程确认这些姿态未用于选模型。',
            '机器人腕部参考来自关节角+模型FK，残差并非外部物理真值精度。']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--calibration', type=Path, action='append', required=True)
    parser.add_argument('--continuity-evidence', required=True)
    parser.add_argument('--group-ids', type=int, nargs='+', help='只评价预定验收ID，不将起点、过渡、返回计入精度')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        rows = [validate(args.input, c, args.continuity_evidence, args.group_ids) for c in args.calibration]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as f:
            json.dump({'frozen_comparisons': rows, 'no_selection_or_refit': True}, f,
                      indent=2, ensure_ascii=False, allow_nan=False)
            f.write('\n')
        for r in rows:
            print(r['calibration_file'], json.dumps(r['metrics_status2_only'], ensure_ascii=False),
                  'passes_gate=' + str(r['passes_gate']))
    except (ValueError, KeyError, TypeError, OSError, np.linalg.LinAlgError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
