#!/usr/bin/env python3
"""离线比较固定刚性安装的参考帧策略，不更改正式 DexUMI 求解器或回放配置。

原方法保留；额外输出：固定X只平均世界变换、多个参考帧的相对运动拟合、
联合SE(3)辅助复核。额外方法均为诊断，无尺度自由度，不按验证误差选模型。
"""
from __future__ import annotations
import argparse
from datetime import datetime
import json
from pathlib import Path
import uuid

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from audit_tron2_tracker_calibration import pose_coverage, transform_difference
from compare_tron2_calibration_scales import check_compatible, cross_validate, load_diagnostic, matrix_set
from solve_tron2_tracker_calibration import PoseGroup, fit_transforms, group_errors, inverse, statistics, transform


def mean_world(groups, X):
    """仅使用拟合组：先平均世界旋转，再以其最小二乘平移估计共同世界变换。"""
    rotations = [g.tracker[:3, :3] @ X[:3, :3].T @ g.robot[:3, :3].T for g in groups]
    R = Rotation.from_matrix(np.asarray(rotations)).mean().as_matrix()
    p = np.mean([g.tracker[:3, 3] - R @ (g.robot[:3, 3] + g.robot[:3, :3] @ X[:3, 3]) for g in groups], axis=0)
    return transform(R, p)


def multi_reference_fit(groups, seed):
    """将原版单参考相对目标拓展为所有拟合参考组；明确作为辅助诊断。"""
    poses = np.asarray([g.robot for g in groups])
    tracker = np.asarray([g.tracker for g in groups])
    relative = np.asarray([inverse(q) for q in tracker])[:, None] @ tracker[None, :]
    actual = np.concatenate([poses[:, :3, 3], Rotation.from_matrix(poses[:, :3, :3]).as_rotvec()], axis=1)
    mask = ~np.eye(len(groups), dtype=bool)
    def unpack(values):
        return transform(Rotation.from_rotvec(values[3:]).as_matrix(), values[:3])
    def objective(values):
        X = unpack(values)
        predicted = poses[:, None] @ X @ relative @ inverse(X)
        sixd = np.concatenate([predicted[:, :, :3, 3], Rotation.from_matrix(
            predicted[:, :, :3, :3].reshape(-1, 3, 3)).as_rotvec().reshape(len(groups), len(groups), 3)], axis=2)
        return float(np.mean(np.linalg.norm(sixd - actual[None, :], axis=2)[mask]))
    x0 = np.r_[np.clip(seed[:3, 3], -.199, .199), Rotation.from_matrix(seed[:3, :3]).as_rotvec()]
    result = minimize(objective, x0, method='L-BFGS-B', bounds=[(-.2, .2)] * 3 + [(-np.pi, np.pi)] * 3,
                      options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 40})
    if not result.success or not np.isfinite(result.fun):
        raise ValueError('多参考相对拟合未收敛：' + str(result.message))
    X = unpack(result.x)
    return X, mean_world(groups, X), {'objective_mixed_m_rad': float(result.fun),
                                    'translation_bound_hit': bool(np.any(abs(result.x[:3]) > .1999))}


def evaluate(groups, fit_ids, held_ids, X, Y):
    rows = [group_errors(g, X, Y) for g in groups]
    return {'fit': statistics([r for r in rows if r['id'] in fit_ids]),
            'held_out': statistics([r for r in rows if r['id'] in held_ids]), 'per_group': rows}


def review(groups, calibration, offset):
    fit_ids, held_ids = set(calibration['fit_group_ids']), set(calibration['validation_group_ids'])
    normal = [g for g in groups if g.statuses == [2]]
    available = {g.id for g in normal}
    if fit_ids & held_ids or not fit_ids | held_ids <= available:
        raise ValueError('原标定的拟合/留出组缺失、不独立或非status2，不能静默更换分组')
    train = [g for g in normal if g.id in fit_ids]
    selected = [g for g in normal if g.id in fit_ids | held_ids]
    C, X_F, X_E, Y = matrix_set(calibration, offset)
    flange = [PoseGroup(g.id, g.label, g.robot @ C, g.tracker, g.statuses, g.pair_count) for g in train]
    methods = {'original_single_reference': (X_E, Y, {'changed_definition': False}),
               'original_X_mean_world_fit_only': (X_E, mean_world(train, X_E), {'changed_definition': True})}
    try:
        X_multi, Y_multi, details = multi_reference_fit(flange, X_F)
        methods['multi_reference_relative_fit_only'] = (C @ X_multi, Y_multi,
            {**details, 'changed_definition': True, 'objective': 'all fit-reference E_r X inv(T_r) T_i inv(X), original XYZ+rotvec norm'})
    except ValueError as exc:
        methods['multi_reference_relative_fit_only'] = {'error': str(exc)}
    try:
        X_joint, Y_joint, details = fit_transforms(flange)
        methods['joint_SE3_fit_only'] = (C @ X_joint, Y_joint,
            {'changed_definition': True, 'objective': 'auxiliary robust SE3 residuals, fixed X and world Y, no scale', 'diagnostics': details})
    except ValueError as exc:
        methods['joint_SE3_fit_only'] = {'error': str(exc)}
    result = {'diagnostic_only': True, 'scale_fitted': False, 'applied_to_mapping': False,
              'fit_group_ids': sorted(fit_ids), 'validation_group_ids': sorted(held_ids), 'methods': {},
              'flange_rotvec_branch_check': pose_coverage([g.robot for g in flange], [g.id for g in flange])}
    for name, values in methods.items():
        if isinstance(values, dict):
            result['methods'][name] = values
            continue
        X, world, details = values
        result['methods'][name] = {
            **details, 'T_ET': X.tolist(), 'T_WB': world.tolist(), 'T_BW': inverse(world).tolist(),
            'metrics': evaluate(selected, fit_ids, held_ids, X, world),
            'T_ET_change_from_original': transform_difference(X_E, X),
            'T_BW_change_from_original': transform_difference(inverse(Y), inverse(world))}
    result['warnings'] = ['所有诊断方法仅使用原拟合组，不按留出误差选方法或参数。',
                          '多参考与联合SE3明确改变优化定义，仅作主DexUMI结果复核，未替换主矩阵。',
                          '多个相对位姿共享同一原始姿态，不是相互独立的统计样本。',
                          '求解残差来自两套测量与机器人FK一致性，不是外部真值精度。']
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--calibration', type=Path, required=True)
    p.add_argument('--cross-manifest', type=Path)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    try:
        cal = json.loads(args.calibration.read_text())
        meta, groups, rejected, digest, _ = load_diagnostic(args.input, cal['tracker_quaternion_convention'])
        check_compatible(meta, cal['session'])
        if digest != cal['input_sha256']:
            raise ValueError('输入样本不是候选标定的原始数据')
        report = review(groups, cal, meta['robot_kinematics']['tcp_offset_m'])
        if args.cross_manifest:
            for method in report['methods'].values():
                if 'T_ET' in method:
                    variant = {**cal, 'T_ET': method['T_ET'], 'T_WB': method['T_WB'], 'T_BW': method['T_BW']}
                    method['cross_validation'] = cross_validate(args.cross_manifest, args.input, meta, variant)
        report.update(input_file=str(args.input.resolve()), input_sha256=digest,
                      calibration_file=str(args.calibration.resolve()), rejected_groups=rejected, hardware_connected=False)
        output = args.output or args.input.parent / 'diagnostics' / (
            'rigid_method_review_' + datetime.now().strftime('%Y%m%d_%H%M%S_') + uuid.uuid4().hex[:8] + '.json')
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('x') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, allow_nan=False)
            f.write('\n')
        print('离线方法复核：' + str(output.resolve()))
        for name, method in report['methods'].items():
            print(name, json.dumps({k: v for k, v in method.get('metrics', {}).items() if k != 'per_group'}, ensure_ascii=False))
    except (KeyError, ValueError, TypeError, OSError, np.linalg.LinAlgError) as exc:
        p.error(str(exc))


if __name__ == '__main__':
    main()
