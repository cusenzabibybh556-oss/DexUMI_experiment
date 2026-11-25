"""将回放位姿送入离线限位余量规划，保存可复查结果；没有真机发送入口。"""

from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
from uuid import uuid4

import numpy as np


def write_ik_plan(args, source, times, delta_positions, delta_quaternions,
                  anchor_q, anchor_position, anchor_quaternion):
    # 普通 ServoP 回放和只读预检仍不要求安装本地 IK 依赖。
    try:
        from tron2_left_ik import LeftArmIK, JOINT_NAMES
        from tron2_ik_planner import MarginPlanOptions, plan_with_joint_margin
        from replay_tron2_tracker import compose_pose
        import scipy  # noqa: F401，提前报告可选依赖缺失。
    except ImportError as exc:
        raise RuntimeError("离线 IK 规划需要 scipy 和 mujoco，请使用已安装这些依赖的 dexumi 环境") from exc

    if args.ik_report is None:
        path = Path(__file__).parent / "validation_results" / (
            f"ik_margin_{Path(source).stem}_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:8]}.json"
        )
    else:
        path = args.ik_report.expanduser().resolve()
    if path.suffix.lower() != ".json":
        raise ValueError("--ik-report 必须使用 .json 后缀")
    artifact = path.with_suffix(".npz")
    if path.exists() or artifact.exists():
        raise ValueError(f"IK 报告或轨迹已存在，不覆盖：{path} / {artifact}")

    model = LeftArmIK()
    # 这里只优化左臂运动学；未读真机的右臂和头部用占位零值，不能当成保持目标。
    start = np.r_[anchor_q, np.zeros(9)]
    poses = [compose_pose(anchor_position, anchor_quaternion, p, u)
             for p, u in zip(delta_positions, delta_quaternions)]
    positions = np.array([p for p, _ in poses])
    quaternions = np.array([u for _, u in poses])
    options = MarginPlanOptions(max_joint_speed=args.ik_max_joint_speed,
                                min_margin_deg=args.ik_min_margin_deg)
    plan, stats = plan_with_joint_margin(model, start, positions, quaternions, times,
                                         options, progress=lambda s: print(s, flush=True))
    report = {
        "source": str(source), "model": str(model.model_path),
        "parameters": {"axis_map": args.axis_map, "scale": args.scale,
                       "orientation_scale": args.orientation_scale, "speed": args.speed,
                       "rate_hz": args.rate,
                       "tracker_interpolation": getattr(args, "interpolation", "linear"),
                       "tracker_mapping": getattr(args, "tracker_mapping_info", None)},
        "planner_options": asdict(options),
        "anchor_position_m": np.asarray(anchor_position).tolist(),
        "anchor_quaternion_wxyz": np.asarray(anchor_quaternion).tolist(),
        "anchor_left_q_rad": np.asarray(anchor_q).tolist(),
        "left_joint_names": JOINT_NAMES[:7],
        "joint_lower_deg": np.rad2deg(model.lower[:7]).tolist(),
        "joint_upper_deg": np.rad2deg(model.upper[:7]).tolist(),
        "joint_min_deg": np.rad2deg(plan[:, :7].min(axis=0)).tolist(),
        "joint_max_deg": np.rad2deg(plan[:, :7].max(axis=0)).tolist(),
        "trajectory_artifact": str(artifact),
        "trajectory_use": "offline left-arm estimate only; not a ServoP or ServoJ executable command file",
        "right_and_head": "not measured; excluded from saved joint trajectory",
        "missing_validation": ["actual O6 and mount", "table and wires", "collision",
                               "dynamics", "robot controller IK branch"],
        **stats,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    # 排他创建，避免把之前已审阅的轨迹悄悄覆盖成另一组参数。
    with path.open("x", encoding="utf-8") as report_file:
        with artifact.open("xb") as trajectory_file:
            np.savez_compressed(trajectory_file, times_s=times,
                                offline_left_q_rad=plan[:, :7], positions_m=positions,
                                quaternions_wxyz=quaternions)
        json.dump(report, report_file, ensure_ascii=False, indent=2, allow_nan=False)
        report_file.write("\n")
    print(f"已选择离线候选 {stats['selected_candidate']+1}：整段最小关节余量 "
          f"{stats['minimum_margin_deg']:.3f}°，最接近限位的是 {stats['critical_joint']}，"
          f"回放时间 {stats['critical_time_s']:.3f}s。")
    print(f"最大关节速度 {stats['max_joint_speed_rad_s']:.4f} rad/s；"
          f"位姿误差 {stats['max_position_error_mm']:.5f} mm / {stats['max_angle_error_deg']:.5f}°。")
    print(f"离线报告：{path}\n完整离线轨迹：{artifact}")
    print("这不是 ServoP 固件的分支选择：未改变准备姿态，未连接机器人或 O6，"
          "未验证碰撞，也未发送任何运动指令。")
    return report
