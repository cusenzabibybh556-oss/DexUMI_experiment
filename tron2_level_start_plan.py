"""离线规划大臂竖直、小臂上仰5°、手腕中立的左臂准备姿态。

物理大臂轴为 proximal_yaw_L_Link 的局部 -Z（原始 STL 长轴 Z 283.29 mm），
小臂轴为 wrist_yaw_L_Link 局部 -Z（原始 STL 长轴 Z 205.58 mm）。
关节原点之间存在横向结构偏置，不能把 SDK 虚拟 TCP 或肩肘原点连线当成长轴。
中立位要求手的延伸方向（法兰 -Z）与小臂平行，掌面随小臂上仰，
掌心法向（法兰 -Y）指向小臂法向平面内的下方，不额外侧翻或弯腕。
本模块没有设备连接或运动入口。
"""
from __future__ import annotations

import math
import numpy as np
from scipy.spatial.transform import Rotation

from tron2_large_calibration_plan import _check_path, _disable_removed_graspers


JOINT_MARGIN_RAD = math.radians(5.)
MAX_STEP_RAD = math.radians(5.)
# 已明确的准备目标：保留上一个正式准备点的小臂水平朝向约10.498°。
# 它是固定目标，不使用某次编码器小幅漂移来重新定义朝向。
NEUTRAL_FOREARM_HEADING_RAD = 0.18322281083347552
NEUTRAL_FOREARM_ELEVATION_DEG = 5.
NEUTRAL_LEFT_Q_RAD = (
    0., 0., NEUTRAL_FOREARM_HEADING_RAD,
    -math.pi/2-math.radians(NEUTRAL_FOREARM_ELEVATION_DEG), -math.pi/2, 0., 0.,
)


def geometry(model, q):
    """返回基座系的真实连杆长轴和法兰，不需要已标定的 Tracker。"""
    model._forward(q)
    d = model.data
    origin = d.xpos[model.base]
    base_rotation = d.xmat[model.base].reshape(3, 3)
    body = model.model.body
    upper = body('proximal_yaw_L_Link').id
    fore = body('wrist_yaw_L_Link').id
    elbow = body('elbow_L_Link').id
    rotation = base_rotation.T @ d.xmat[model.body].reshape(3, 3)
    return {
        'flange_m': base_rotation.T @ (d.xpos[model.body]-origin),
        'rotation': rotation,
        'upper_arm_axis': base_rotation.T @ (-d.xmat[upper].reshape(3, 3)[:, 2]),
        'forearm_axis': base_rotation.T @ (-d.xmat[fore].reshape(3, 3)[:, 2]),
        'hand_extension_axis': -rotation[:, 2],
        'palm_normal': -rotation[:, 1],
        'elbow_m': base_rotation.T @ (d.xpos[elbow]-origin),
    }


def geometry_angles(g):
    """从同一套连杆/手部轴向计算角度，供规划报告和到位验收共同使用。"""
    upper, fore = np.asarray(g['upper_arm_axis']), np.asarray(g['forearm_axis'])
    hand, palm = np.asarray(g['hand_extension_axis']), np.asarray(g['palm_normal'])
    upper_error = float(np.rad2deg(np.arccos(np.clip(-upper[2], -1., 1.))))
    # 把重力向下与掌心法向投影到小臂法向平面，单独度量绕小臂的翻腕。
    # 小臂上仰5°时，全局掌心法向也会偏离向下5°，这不算额外弯腕。
    down = np.array([0., 0., -1.])
    down_projection = down-np.dot(down, fore)*fore
    palm_projection = palm-np.dot(palm, fore)*fore
    down_norm, palm_norm = np.linalg.norm(down_projection), np.linalg.norm(palm_projection)
    roll_reference_valid = bool(down_norm > 1e-9 and palm_norm > 1e-9)
    palm_roll_error = (float(np.rad2deg(np.arccos(np.clip(
        np.dot(down_projection/down_norm, palm_projection/palm_norm), -1., 1.))))
        if roll_reference_valid else None)
    return {
        'upper_arm_vertical_error_deg': upper_error,
        'upper_arm_from_down_vertical_deg': upper_error,  # 保留已有报告字段名。
        'forearm_elevation_deg': float(np.rad2deg(np.arcsin(np.clip(fore[2], -1., 1.)))),
        'forearm_yaw_deg': float(np.rad2deg(np.arctan2(fore[1], fore[0]))),
        'hand_forearm_alignment_deg': float(np.rad2deg(np.arccos(np.clip(np.dot(hand, fore), -1., 1.)))),
        'palm_down_error_deg': float(np.rad2deg(np.arccos(np.clip(-palm[2], -1., 1.)))),
        'palm_roll_error_deg': palm_roll_error,
        'palm_roll_reference_valid': roll_reference_valid,
        'hand_elevation_deg': float(np.rad2deg(np.arcsin(np.clip(hand[2], -1., 1.)))),
    }


def build_level_start_plan(model, start):
    """只改变左7关节，前往大臂竖直、小臂上仰5°、手腕中立的固定目标。

    返回 target_q16, stages_q16, report。不保留此前带侧弯的整体法兰朝向。
    右臂/头部沿用这次起点快照，所有实际关节角和完整路径均检查后才返回。
    """
    start = np.asarray(start, dtype=float).copy()
    model.check_joints(start)
    removed = _disable_removed_graspers(model)
    lower, upper = model.lower.copy(), model.upper.copy()
    lower[[5, 6, 12, 13]] = np.maximum(lower[[5, 6, 12, 13]], -.7505)
    upper[[5, 6, 12, 13]] = np.minimum(upper[[5, 6, 12, 13]], .7505)
    if np.any(start < lower) or np.any(start > upper):
        raise ValueError('起点不满足模型/SDK/MoveJ 限位交集')
    if np.min(np.minimum(start[:7]-lower[:7], upper[:7]-start[:7])) < JOINT_MARGIN_RAD-1e-8:
        raise ValueError('左臂起点没有保留5°关节余量')
    g0 = geometry(model, start)
    target = start.copy()
    target[:7] = NEUTRAL_LEFT_Q_RAD
    model.check_joints(target)
    g1 = geometry(model, target)
    final_angles = geometry_angles(g1)
    if (final_angles['upper_arm_vertical_error_deg'] > 1e-5
            or abs(final_angles['forearm_elevation_deg']-NEUTRAL_FOREARM_ELEVATION_DEG) > 1e-5
            or final_angles['hand_forearm_alignment_deg'] > 1e-5
            or not final_angles['palm_roll_reference_valid']
            or final_angles['palm_roll_error_deg'] > 1e-5
            or abs(final_angles['palm_down_error_deg']-NEUTRAL_FOREARM_ELEVATION_DEG) > 1e-5
            or abs(math.radians(final_angles['forearm_yaw_deg'])-NEUTRAL_FOREARM_HEADING_RAD) > 1e-7):
        raise ValueError('固定中立目标与模型连杆轴或法兰定义不一致，未生成动作')
    error = _check_path(model, start, target, lower, upper)
    if error:
        raise ValueError('中立准备姿态完整路径检查失败：'+str(error))
    dense_count = max(2, 1+int(math.ceil(np.max(np.abs(target-start))/math.radians(.5))))
    dense_q = np.array([start+(target-start)*a for a in np.linspace(0., 1., dense_count)])
    margins = np.minimum(dense_q[:, :7]-lower[:7], upper[:7]-dense_q[:, :7])
    if margins.min() < JOINT_MARGIN_RAD-1e-8:
        raise ValueError('路径未保留至少5°关节限位余量')
    geometries = [geometry(model, q) for q in dense_q]
    flange_path = np.array([g['flange_m'] for g in geometries])
    elbow_path = np.array([g['elbow_m'] for g in geometries])
    g1 = geometry(model, target)
    if flange_path[:, 2].min() < min(g0['flange_m'][2], g1['flange_m'][2])-.005:
        raise ValueError('关节路径使腕部下探超过起止最低位置5 mm')
    n = max(1, int(math.ceil(np.max(np.abs(target-start))/MAX_STEP_RAD)))
    stages = [start+(target-start)*(i+1)/n for i in range(n)]
    stages[-1] = target.copy()
    start_tcp = model.pose(start)[0]
    target_tcp, target_quat = model.pose(target)

    report = {
        'purpose': '左臂准备目标：大臂竖直、小臂上仰5°、手腕中立、掌面随小臂倾斜',
        'geometry_definition': {
            'upper_arm_axis': 'proximal_yaw_L_Link local [0,0,-1] in base_Link',
            'forearm_axis': 'wrist_yaw_L_Link local [0,0,-1] in base_Link',
            'hand_extension_axis': 'wrist_roll_L_Link local [0,0,-1] in base_Link',
            'palm_normal': 'wrist_roll_L_Link local [0,-1,0] in base_Link',
            'upper_arm_original_mesh_span_m': [.10350636, .10457091, .28329375],
            'forearm_original_mesh_span_m': [.08054593, .09393781, .20558254],
            'note': '物理壳体长轴；结构偏置使关节原点连线与长轴略有不同',
        },
        'start_q_rad': start.tolist(), 'target_q_rad': target.tolist(),
        'stages_q_rad': [q.tolist() for q in stages], 'segment_count': n,
        'stage_duration_seconds': 3., 'maximum_step_deg': 5.,
        'path_collision_resolution_deg': .5,
        'start_angles_deg': geometry_angles(g0), 'target_angles_deg': final_angles,
        'start_upper_arm_axis': g0['upper_arm_axis'].tolist(),
        'target_upper_arm_axis': g1['upper_arm_axis'].tolist(),
        'start_forearm_axis': g0['forearm_axis'].tolist(),
        'target_forearm_axis': g1['forearm_axis'].tolist(),
        'start_hand_extension_axis': g0['hand_extension_axis'].tolist(),
        'target_hand_extension_axis': g1['hand_extension_axis'].tolist(),
        'start_palm_normal': g0['palm_normal'].tolist(),
        'target_palm_normal': g1['palm_normal'].tolist(),
        'start_flange_m': g0['flange_m'].tolist(), 'target_flange_m': g1['flange_m'].tolist(),
        'start_elbow_m': g0['elbow_m'].tolist(), 'target_elbow_m': g1['elbow_m'].tolist(),
        'start_sdk_position_m': start_tcp.tolist(), 'target_sdk_position_m': target_tcp.tolist(),
        'target_sdk_quaternion_wxyz': target_quat.tolist(),
        'sdk_position_change_m': (target_tcp-start_tcp).tolist(),
        'flange_orientation_change_deg': float(np.rad2deg(Rotation.from_matrix(g1['rotation']@g0['rotation'].T).magnitude())),
        'left_joint_change_deg': np.rad2deg(target[:7]-start[:7]).tolist(),
        'minimum_left_joint_margin_deg': np.rad2deg(margins.min(axis=0)).tolist(),
        'path_flange_min_m': flange_path.min(axis=0).tolist(),
        'path_flange_max_m': flange_path.max(axis=0).tolist(),
        'path_elbow_min_m': elbow_path.min(axis=0).tolist(),
        'path_elbow_max_m': elbow_path.max(axis=0).tolist(),
        'right_and_head_fixed': bool(np.array_equal(target[7:], start[7:])),
        'model_self_collision_passed': True, 'removed_model_geometry': removed,
        'missing_geometry': ['实际Tracker/O6/转接支架', '线缆', '桌面及外部环境'],
        'target_definition': '固定关节解；大臂竖直、小臂保留10.498°水平朝向并上仰5°；手腕pitch/roll为0，掌面随小臂倾斜且无额外翻腕',
        'forearm_heading_rad': NEUTRAL_FOREARM_HEADING_RAD,
        'forearm_elevation_deg': NEUTRAL_FOREARM_ELEVATION_DEG,
        'previous_full_flange_orientation_preserved': False,
        'already_at_target': bool(np.max(np.abs(target[:7]-start[:7])) <= 1e-8),
        'height_tradeoff': '中立位改变完整末端朝向与SDK TCP位置；不保持旧起点XYZ',
        'hardware_connected': False, 'motion_sent': False,
    }
    return target, stages, report


# 简短兼容名，供一次性离线执行器使用。
buildplan = build_level_start_plan
