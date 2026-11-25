#!/usr/bin/env python3
"""离线合并预先声明的拟合组，冻结无尺度、多参考 DexUMI 刚体候选。

任何留出组都不参加拟合、种子选择或权重选择。会话间地图/安装连续性需要
明确的过程声明，元数据身份检查不能替代真实地图 UUID 或外部长度真值。
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from audit_tron2_tracker_calibration import pose_coverage, transform_difference
from compare_tron2_calibration_scales import check_compatible, load_diagnostic, matrix_set
from review_dexumi_rigid_methods import multi_reference_fit
from solve_tron2_tracker_calibration import PoseGroup, group_errors, inverse, statistics


def load_selected(source, main_meta, convention, fit_ids, held_ids):
    """严格匹配已有角色，只接受整组 status2；禁止静默换组或重复加权。"""
    if len(set(fit_ids)) != len(fit_ids) or set(fit_ids) & set(held_ids):
        raise ValueError('拟合组重复或混入留出组')
    meta, groups, rejected, digest, _ = load_diagnostic(Path(source), convention)
    check_compatible(meta, main_meta)
    available = {g.id: g for g in groups if g.statuses == [2]}
    if not set(fit_ids) <= available.keys():
        raise ValueError('声明的拟合组缺失或非严格status2：' + str(source))
    return meta, [available[i] for i in fit_ids], [available[i] for i in held_ids
        if i in available], rejected, digest


def summarize(groups, X, Y):
    rows = [group_errors(g, X, Y) for g in groups]
    return {'summary': statistics(rows), 'per_group': rows}


def build(args):
    plan = json.loads(args.merge_plan.read_text())
    continuity = json.loads(args.continuity_manifest.read_text())
    main = json.loads(args.calibration.read_text())
    path = Path(plan['new_session']['input']).resolve()
    if Path(continuity['same_map_mount_as']).resolve() != path or not continuity.get('evidence'):
        raise ValueError('缺少本轮对应的地图/安装连续性明确声明')
    convention = main['tracker_quaternion_convention']
    meta, _, _, main_digest, event_ids = load_diagnostic(path, convention)
    if main_digest != main['input_sha256']:
        raise ValueError('主标定源文件SHA不匹配')
    declared_fit = {event_ids[i] for i in plan['new_session']['fit_sample_indices']}
    declared_held = {event_ids[i] for i in plan['new_session']['held_out_sample_indices']}
    if not set(main['fit_group_ids']) <= declared_fit or not set(main['validation_group_ids']) <= declared_held:
        raise ValueError('本轮求解器角色与预先声明采样角色不一致')
    sessions = list(plan['old_sessions_fit_candidate']) + [{
        'input': str(path), 'calibration': str(args.calibration.resolve()),
        'fit_group_ids': main['fit_group_ids'],
        'preserved_held_out_group_ids': main['validation_group_ids']}]
    C, seed, _, _ = matrix_set(main, meta['robot_kinematics']['tcp_offset_m'])
    fit, held, records, provenance = [], [], [], []
    seen = set()
    for session_number, item in enumerate(sessions):
        source = Path(item['input']).resolve()
        if source in seen:
            raise ValueError('同一源会话不能重复进入拟合')
        seen.add(source)
        cal = json.loads(Path(item['calibration']).read_text())
        if (cal['tracker_quaternion_convention'] != convention
                or set(cal['fit_group_ids']) != set(item['fit_group_ids'])
                or set(cal['validation_group_ids']) != set(item['preserved_held_out_group_ids'])):
            raise ValueError('历史原始FIT/held角色或四元数定义被改变')
        source_meta, selected, check, rejected, digest = load_selected(source, meta, convention,
            item['fit_group_ids'], item['preserved_held_out_group_ids'])
        if digest != cal['input_sha256']:
            raise ValueError('历史标定源文件SHA不匹配')
        provenance.append({'session_index': session_number, 'input_file': str(source),
            'input_sha256': digest, 'original_calibration': item['calibration'],
            'fit_group_ids': item['fit_group_ids'],
            'preserved_held_out_group_ids': item['preserved_held_out_group_ids'],
            'source_map_label': source_meta.get('map_label'),
            'source_mount_label': source_meta.get('mount_label'),
            'rejected_group_count': len(rejected)})
        for role, selected_groups, target in [('fit', selected, fit), ('cross_development', check, held)]:
            for g in selected_groups:
                unique_id = session_number * 10000 + g.id
                records.append({'id': unique_id, 'source_group_id': g.id,
                    'session_index': session_number, 'input_file': str(source), 'role': role})
                target.append(PoseGroup(unique_id, g.label, g.robot, g.tracker, g.statuses, g.pair_count))
    flange = [PoseGroup(g.id, g.label, g.robot @ C, g.tracker, g.statuses, g.pair_count) for g in fit]
    X_F, Y, optimization = multi_reference_fit(flange, seed)
    X_E = C @ X_F
    # 先冻结矩阵。旧留出结果只在此后计算，不得用于调目标、权重或重新拟合。
    result = {'schema': 'tracker-tron2-calibration-global-multireference-v1',
        'created_at': datetime.now(timezone.utc).isoformat(), 'status': 'frozen_candidate_pending_new_validation',
        'method': 'DexUMI fixed installation X; all ordered FIT-reference relative motions; mean world transform',
        'objective': 'mean ||[p,rotvec](F_r X inv(Q_r) Q_i inv(X)) - [p,rotvec](F_i)||, r!=i',
        'changed_from_original': 'single reference replaced by every FIT reference; world transform averaged over FIT only',
        'objective_units': 'original mixed metres/radians, unweighted group pairs; no fitted scale',
        'transform_convention': 'T_WT = T_WB @ T_BE @ T_ET; T_BE = T_BW @ T_WT @ inverse(T_ET)',
        'tracker_quaternion_convention': convention, 'position_unit': 'm', 'rotation_matrix_convention': 'active',
        'T_ET': X_E.tolist(), 'T_FT': X_F.tolist(), 'T_WB': Y.tolist(), 'T_BW': inverse(Y).tolist(),
        'session': meta, 'fit_group_ids': [g.id for g in fit], 'validation_group_ids': [],
        'fit_sources': provenance, 'group_provenance': records,
        'same_map_mount_evidence': continuity['evidence'],
        'merge_plan': str(args.merge_plan.resolve()),
        'merge_plan_sha256': hashlib.sha256(args.merge_plan.read_bytes()).hexdigest(),
        'optimization': optimization, 'fit_metrics': summarize(fit, X_E, Y),
        'flange_coverage': pose_coverage([g.robot for g in flange], [g.id for g in flange]),
        'scale_fitted': False, 'hardware_connected': False, 'applied_to_replay': False,
        'validation_policy': 'all old held sets are cross-development; final new poses must not influence this frozen fit',
        'warnings': ['世界地图/安装一致性基于持续连接与未拆装声明；没有硬件地图UUID证明。',
            'SDK腕部数据来自实测关节角+模型FK，不是独立笛卡尔真值。',
            '不得将当前裸腕Tracker安装矩阵直接用于历史真人腕部episode。',
            '此候选尚需全新姿态验收；未修改主标定、控制参数或回放配置。']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        json.dump(result, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print('FROZEN', str(args.output.resolve()), digest, flush=True)
    diagnostics = {'frozen_calibration': str(args.output.resolve()), 'frozen_sha256': digest,
        'cross_development_metrics': summarize(held, X_E, Y),
        'per_session': [], 'prescheduled_sensitivity': [], 'new_validation_used': False}
    for i, record in enumerate(provenance):
        fit_local = [g for g in fit if g.id // 10000 == i]
        held_local = [g for g in held if g.id // 10000 == i]
        diagnostics['per_session'].append({'input': record['input_file'],
            'fit': summarize(fit_local, X_E, Y), 'cross_development': summarize(held_local, X_E, Y)})
    # 不依照误差挑选：四个预定会话各删一轮，以及六个等间隔顺序的组LOO。
    selections = [('leave_session_out', i, [g for g in flange if g.id // 10000 != i]) for i in range(len(provenance))]
    selections += [('leave_group_out', flange[i].id, [g for j, g in enumerate(flange) if j != i])
        for i in np.linspace(0, len(flange)-1, 6, dtype=int)]
    if args.sensitivity:
        for kind, omitted, subset in selections:
            try:
                x, y, detail = multi_reference_fit(subset, X_F)
                diagnostics['prescheduled_sensitivity'].append({'kind': kind, 'omitted': int(omitted),
                    'T_ET_change': transform_difference(X_E, C @ x),
                    'T_BW_change': transform_difference(inverse(Y), inverse(y)),
                    'optimization': detail})
            except ValueError as exc:
                diagnostics['prescheduled_sensitivity'].append({'kind': kind, 'omitted': int(omitted), 'error': str(exc)})
    with args.output.with_name(args.output.stem + '_diagnostics.json').open('x') as f:
        json.dump(diagnostics, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')
    print('FIT', statistics([group_errors(g, X_E, Y) for g in fit]), flush=True)
    print('CROSS_DEVELOPMENT', diagnostics['cross_development_metrics']['summary'], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--merge-plan', type=Path, required=True)
    parser.add_argument('--continuity-manifest', type=Path, required=True)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sensitivity', action='store_true')
    args = parser.parse_args()
    try:
        build(args)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
