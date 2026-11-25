#!/usr/bin/env python3
"""离线求解 T_WT = T_WB @ T_BE @ T_ET，保存变换、逐组误差和独立验证报告。

约定 T_AB 将 B 系坐标变换至 A 系；位置单位为米，输入四元数为 WXYZ。
不导入机器人 SDK、不连接硬件、不修改回放配置。
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import uuid

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from capture_tron2_tracker_calibration import pose_values, pose_spread, JOINT_FK_SOURCE


class CalibrationError(ValueError):
    def __init__(self, message, diagnostics=None):
        super().__init__(message)
        self.diagnostics = diagnostics or {}


@dataclass
class PoseGroup:
    id: int
    label: str
    robot: np.ndarray      # T_BE
    tracker: np.ndarray    # T_WT
    statuses: list[int]
    pair_count: int


def transform(rotation, translation):
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = translation
    return result


def inverse(T):
    return transform(T[:3, :3].T, -T[:3, :3].T @ T[:3, 3])


def mean_pose(frames):
    stats = pose_spread(frames)
    q = stats["mean_quaternion_wxyz"]
    return transform(Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix(), stats["mean_position_m"])


def integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} 必须为整数")
    return value


def load_samples(path, include_status20=False, diagnostic_statuses=()):
    """重新核对原始帧；不依赖缓存统计，也不把一组里的多帧当成多种姿态。"""
    raw = Path(path).read_bytes()
    events = []
    for index, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except ValueError as exc:
            raise CalibrationError(f"第 {index} 行不是完整 JSON；请先结束采集") from exc
        if not isinstance(event, dict):
            raise CalibrationError(f"第 {index} 行必须是 JSON 对象")
        events.append(event)
    if not events or events[0].get("event") != "session":
        raise CalibrationError("缺少会话头；需要采集入口生成的 samples.jsonl")
    metadata = events[0]
    if sum(e.get("event") == "session" for e in events) != 1:
        raise CalibrationError("一个文件只能包含一个会话，不能混合不同地图/安装的样本")
    if events[-1].get("event") != "session_end":
        raise CalibrationError("请先结束采集并断开，再求解完整会话")
    joint_fk = (metadata.get("schema") == "tracker-tron2-pairs-v2"
                and metadata.get("robot_pose_source") == JOINT_FK_SOURCE)
    legacy = (metadata.get("schema") == "tracker-tron2-pairs-v1"
              and metadata.get("robot_pose_source") == "response_get_servo_pose")
    if (not (joint_fk or legacy) or metadata.get("position_unit") != "m"
            or metadata.get("quaternion_order") != "wxyz"
            or metadata.get("transform_convention") != "T_WT = T_WB @ T_BE @ T_ET"
            or metadata.get("arm") not in {"left", "right"}):
        raise CalibrationError("文件版本、单位、手侧或坐标约定不匹配，不能猜测转换")
    if joint_fk:
        model = metadata.get("robot_kinematics", {})
        digest = model.get("model_sha256", "")
        if (metadata.get("arm") != "left" or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)
                or model.get("base_frame") != "base_Link"
                or model.get("wrist_frame") != "wrist_roll_L_Link"
                or len(model.get("joint_names", [])) != 16
                or np.asarray(model.get("tcp_offset_m", [])).shape != (3,)
                or not np.isfinite(model["tcp_offset_m"]).all()):
            raise CalibrationError("关节正解样本缺少明确的模型、TCP 或关节顺序")
    excluded = {integer(e.get("id"), "排除组号") for e in events if e.get("event") == "exclude"}
    groups, rejected, ids, identities = [], [], set(), set()
    if not set(diagnostic_statuses) <= {18, 20}:
        raise CalibrationError("诊断状态仅支持明确记录的18或20，不能纳入已知丢失状态")
    accepted_statuses = ({2, 20} if include_status20 else {2}) | set(diagnostic_statuses)
    for event in events[1:]:
        if event.get("event") not in {"sample", "exclude", "session_end"}:
            raise CalibrationError("不支持的采集事件：" + str(event.get("event")))
        if event.get("event") != "sample":
            continue
        group_id = integer(event.get("id"), "样本组号")
        if group_id <= 0 or group_id in ids:
            raise CalibrationError("样本组号重复或不是正整数")
        ids.add(group_id)
        reason = None
        try:
            if group_id in excluded or event.get("excluded"):
                raise ValueError("用户已排除此组")
            start = integer(event.get("started_monotonic_ns"), "采样开始时间")
            end = integer(event.get("ended_monotonic_ns"), "采样结束时间")
            if end <= start:
                raise ValueError("采样窗口为空")
            trackers, robots = event.get("tracker", []), event.get("robot", [])
            if not trackers or not robots:
                raise ValueError("缺少 Tracker 或机器人新帧")
            for frames in (trackers, robots):
                last = -1
                for frame in frames:
                    pose_values(frame["position_m"], frame["quaternion_wxyz"])
                    ts = integer(frame["received_monotonic_ns"], "接收时间")
                    if not start <= ts <= end or ts <= last:
                        raise ValueError("帧时间不递增或超出采样窗口")
                    last = ts
                times = [start] + [f["received_monotonic_ns"] for f in frames] + [end]
                if max(np.diff(times)) > 500_000_000:
                    raise ValueError("采样窗口有超过 0.5 秒的数据空缺")
                stats = pose_spread(frames)
                if stats["position_radius_mm"] > 3 or stats["rotation_radius_deg"] > 1:
                    raise ValueError("本组未静止：位置波动半径 >3 mm 或旋转 >1°")
            statuses = sorted({integer(f["tracking_status"], "跟踪状态") for f in trackers})
            if not set(statuses) <= accepted_statuses:
                raise ValueError(f"跟踪状态 {statuses} 未纳入；status=20 可显式选择诊断求解")
            macs = {f["mac"] for f in trackers}
            indices = {integer(f["tracker_index"], "Tracker 编号") for f in trackers}
            if len(macs) != 1 or not all(isinstance(m, str) and m for m in macs) or indices != {metadata.get("tracker_index")}:
                raise ValueError("Tracker 身份或编号不一致")
            last_robot_stamp = -1
            for frame in robots:
                rs = integer(frame["request_started_monotonic_ns"], "机器人请求时间")
                re = frame["received_monotonic_ns"]
                if not start <= rs <= re or re - rs > 250_000_000:
                    raise ValueError("机器人请求超出窗口或耗时 >250 ms")
                if joint_fk:
                    q = np.asarray(frame.get("joint_position_rad", []), dtype=float)
                    stamp = float(frame.get("robot_timestamp_ms", float("nan")))
                    if (q.shape != (16,) or not np.isfinite(q).all()
                            or frame.get("pose_source") != JOINT_FK_SOURCE
                            or not np.isfinite(stamp) or stamp <= last_robot_stamp):
                        raise ValueError("关节正解帧缺少实测关节或新鲜时间戳")
                    last_robot_stamp = stamp
            # 使用原始接收时间重新匹配，每个 Tracker 帧至多计入一次。
            paired_t, paired_r, used = [], [], set()
            for robot in robots:
                ti = min(range(len(trackers)), key=lambda k: abs(trackers[k]["received_monotonic_ns"] - robot["received_monotonic_ns"]))
                gap = abs(trackers[ti]["received_monotonic_ns"] - robot["received_monotonic_ns"])
                if gap <= 100_000_000 and ti not in used:
                    used.add(ti)
                    paired_t.append(trackers[ti])
                    paired_r.append(robot)
            if len(used) < 3:
                raise ValueError("至少需要 3 对不同的新帧，接收时间差须 ≤100 ms")
            groups.append(PoseGroup(group_id, str(event.get("label", "")), mean_pose(paired_r),
                                    mean_pose(paired_t), statuses, len(used)))
            identities.update(macs)
        except (KeyError, TypeError, ValueError) as exc:
            reason = str(exc)
        if reason:
            rejected.append({"id": group_id, "reason": reason})
    if excluded - ids:
        raise CalibrationError("排除记录引用了不存在的样本")
    if len(identities) > 1:
        raise CalibrationError("同一会话内 Tracker 身份变化，不能求一个安装变换")
    return metadata, groups, rejected, hashlib.sha256(raw).hexdigest()


def initial_estimate(groups):
    """由 A X = X B 初始化旋转，再线性求安装/世界平移。"""
    if len(groups) < 3:
        raise CalibrationError("至少需要 3 个合格且旋转轴不平行的姿态组；建议采 12–20 组")
    a, b = [], []
    for i, j in combinations(range(len(groups)), 2):
        A = inverse(groups[i].robot) @ groups[j].robot
        B = inverse(groups[i].tracker) @ groups[j].tracker
        a.append(Rotation.from_matrix(A[:3, :3]).as_rotvec())
        b.append(Rotation.from_matrix(B[:3, :3]).as_rotvec())
    a, b = np.asarray(a), np.asarray(b)
    sa, sb = np.linalg.svd(a, compute_uv=False), np.linalg.svd(b, compute_uv=False)
    max_angle = float(np.degrees(np.linalg.norm(a, axis=1).max()))
    ratios = [float(s[1] / s[0]) if s[0] > 1e-12 else 0. for s in (sa, sb)]
    diagnostics = {"max_robot_rotation_span_deg": max_angle,
                   "robot_rotation_singular_values": sa.tolist(), "tracker_rotation_singular_values": sb.tolist(),
                   "second_axis_ratio": min(ratios)}
    if max_angle < 5 or min(ratios) < .05:
        raise CalibrationError("姿态变化不足：需要明显绕至少两个不平行轴旋转；重复姿态、纯平移或单轴旋转不能可靠确定 T_ET", diagnostics)
    U, _, Vt = np.linalg.svd(a.T @ b)
    Rx = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
    Ry = Rotation.from_matrix(np.stack([g.tracker[:3, :3] @ Rx.T @ g.robot[:3, :3].T for g in groups])).mean().as_matrix()
    M = np.vstack([np.hstack((Ry @ g.robot[:3, :3], np.eye(3))) for g in groups])
    rhs = np.concatenate([g.tracker[:3, 3] - Ry @ g.robot[:3, 3] for g in groups])
    translations, _, rank, singular = np.linalg.lstsq(M, rhs, rcond=None)
    diagnostics.update(translation_rank=int(rank), translation_singular_values=singular.tolist(),
                       translation_condition_number=float(singular[0] / max(singular[-1], 1e-15)))
    if rank < 6 or singular[-1] / singular[0] < .005:
        raise CalibrationError("安装平移不可充分观测，请增加不同轴的姿态变化", diagnostics)
    return transform(Rx, translations[:3]), transform(Ry, translations[3:]), diagnostics


def fit_transforms(groups, position_scale_m=.005, rotation_scale_rad=np.deg2rad(2)):
    X0, Y0, diagnostics = initial_estimate(groups)
    def unpack(v):
        X = transform(Rotation.from_rotvec(v[:3]).as_matrix() @ X0[:3, :3], X0[:3, 3] + v[3:6])
        Y = transform(Rotation.from_rotvec(v[6:9]).as_matrix() @ Y0[:3, :3], Y0[:3, 3] + v[9:12])
        return X, Y

    def residual(v):
        X, Y = unpack(v)
        values = []
        for g in groups:
            prediction = Y @ g.robot @ X
            values.extend((prediction[:3, 3] - g.tracker[:3, 3]) / position_scale_m)
            # SO(3) 上的旋转差；绝不直接相减欧拉角/旋转向量/四元数。
            values.extend(Rotation.from_matrix(g.tracker[:3, :3].T @ prediction[:3, :3]).as_rotvec() / rotation_scale_rad)
        return np.asarray(values)

    result = least_squares(residual, np.zeros(12), loss="soft_l1", f_scale=1.,
                           x_scale="jac", max_nfev=1200, ftol=1e-11, xtol=1e-11, gtol=1e-11)
    X, Y = unpack(result.x)
    diagnostics.update(optimizer_success=bool(result.success), optimizer_message=str(result.message),
                       optimizer_evaluations=int(result.nfev), robust_cost=float(result.cost),
                       position_residual_scale_m=float(position_scale_m), rotation_residual_scale_deg=float(np.degrees(rotation_scale_rad)))
    if not result.success or not np.isfinite(result.x).all():
        raise CalibrationError("优化未收敛，未生成可用变换", diagnostics)
    # 以局部 1 rad / 0.1 m 扰动检查 12 个参数的数值可辨识性；不是统计置信区间。
    step = 1e-6
    J = np.column_stack([(residual(result.x + np.eye(12)[j] * step) - residual(result.x - np.eye(12)[j] * step)) / (2 * step) for j in range(12)])
    singular = np.linalg.svd(J * np.array([1, 1, 1, .1, .1, .1] * 2), compute_uv=False)
    rank = int(np.count_nonzero(singular > singular[0] * 1e-6))
    diagnostics.update(jacobian_rank=rank, jacobian_scaled_singular_values=singular.tolist(),
                       jacobian_scaled_condition_number=float(singular[0] / max(singular[-1], 1e-15)))
    if rank < 12:
        raise CalibrationError("联合解存在不可辨识自由度，请增加不同方向的姿态组", diagnostics)
    return X, Y, diagnostics


def group_errors(group, X, Y):
    tracker_prediction = Y @ group.robot @ X
    wrist_prediction = inverse(Y) @ group.tracker @ inverse(X)
    delta_wrist = wrist_prediction[:3, 3] - group.robot[:3, 3]
    return {"id": group.id, "label": group.label, "pair_count": group.pair_count,
            "tracking_statuses": group.statuses,
            "tracker_position_error_mm": float(np.linalg.norm(tracker_prediction[:3, 3] - group.tracker[:3, 3]) * 1000),
            "wrist_position_error_mm": float(np.linalg.norm(delta_wrist) * 1000),
            "wrist_position_delta_mm": (delta_wrist * 1000).tolist(),
            "rotation_error_deg": float(np.degrees(Rotation.from_matrix(group.tracker[:3, :3].T @ tracker_prediction[:3, :3]).magnitude()))}


def statistics(rows):
    if not rows:
        return None
    result = {"count": len(rows)}
    for key in ("tracker_position_error_mm", "wrist_position_error_mm", "rotation_error_deg"):
        x = np.array([r[key] for r in rows])
        result[key] = {"mean": float(x.mean()), "median": float(np.median(x)),
                       "rmse": float(np.sqrt(np.mean(x ** 2))), "p95": float(np.percentile(x, 95)), "max": float(x.max())}
    return result


def solve_groups(groups, validation_ids=None, position_tolerance_mm=5., rotation_tolerance_deg=2.):
    if len(groups) < 3:
        raise CalibrationError(f"只有 {len(groups)} 个可用姿态组，至少需要 3 组；建议 12–20 组")
    ids = {g.id for g in groups}
    if validation_ids is None:
        # 固定的整组划分，与拟合误差无关，不用验证集选参数或重新拟合。
        count = max(3, math.ceil(len(groups) * .2)) if len(groups) >= 9 else 0
        validation_ids = {groups[i].id for i in np.linspace(0, len(groups) - 1, count, dtype=int)} if count else set()
    else:
        validation_ids = set(validation_ids)
        if not validation_ids <= ids:
            raise CalibrationError("指定验证组不存在、已排除或未通过采样检查")
    train = [g for g in groups if g.id not in validation_ids]
    validation = [g for g in groups if g.id in validation_ids]
    X, Y, diagnostics = fit_transforms(train)
    rows = [{**group_errors(g, X, Y), "split": "validation" if g.id in validation_ids else "fit"} for g in groups]
    fit_stats = statistics([r for r in rows if r["split"] == "fit"])
    val_stats = statistics([r for r in rows if r["split"] == "validation"])
    def passes(stats):
        if not stats:
            return False
        return all(stats[key]["rmse"] <= tol and stats[key]["max"] <= 3 * tol for key, tol in (
            ("wrist_position_error_mm", position_tolerance_mm), ("rotation_error_deg", rotation_tolerance_deg)))
    checks = {"fit_residuals_passed": passes(fit_stats), "held_out_residuals_passed": passes(val_stats),
              "independent_validation_groups": len(validation),
              "thresholds": {"position_rmse_mm": position_tolerance_mm, "position_max_mm": 3 * position_tolerance_mm,
                             "rotation_rmse_deg": rotation_tolerance_deg, "rotation_max_deg": 3 * rotation_tolerance_deg,
                             "source": "project_review_defaults_not_robot_limits"}}
    warnings = []
    if len(validation) < 3:
        warnings.append("独立验证不足 3 组；拟合误差不能代替独立验证")
    if not checks["fit_residuals_passed"]:
        warnings.append("拟合残差超过参考阈值，请检查异常组、安装松动、时间对应或坐标约定")
    if validation and not checks["held_out_residuals_passed"]:
        warnings.append("独立验证误差超过参考阈值；保留所有逐组误差，不自动删掉异常组")
    numerical_passed = checks["fit_residuals_passed"] and checks["held_out_residuals_passed"] and len(validation) >= 3
    return {"T_ET": X.tolist(), "T_WB": Y.tolist(), "T_BW": inverse(Y).tolist(),
            "fit_group_ids": [g.id for g in train], "validation_group_ids": [g.id for g in validation],
            "metrics": {"fit": fit_stats, "validation": val_stats, "all": statistics(rows)},
            "per_group": rows, "observability": diagnostics, "checks": checks,
            "numerical_checks_passed": bool(numerical_passed), "warnings": warnings}


def format_report(report):
    lines = ["TRON2 / Tracker 腕部标定求解报告", f"状态：{report['status']}",
             f"输入：{report['input_file']}", "坐标约定：T_WT = T_WB @ T_BE @ T_ET；位置 m，四元数 WXYZ"]
    if report.get("error"):
        lines.append("原因：" + report["error"])
    for warning in report.get("warnings", []):
        lines.append("提示：" + warning)
    if "T_ET" in report:
        lines += ["T_ET（Tracker → 腕部）：", np.array2string(np.array(report["T_ET"]), precision=9),
                  "T_WB（机器人基座 → Tracker 地图）：", np.array2string(np.array(report["T_WB"]), precision=9),
                  "T_BW 为 T_WB 的逆，用于从地图变换到基座。"]
        for split, name in (("fit", "拟合组"), ("validation", "独立验证组")):
            stats = report["metrics"][split]
            if not stats:
                lines.append(f"{name}：无")
                continue
            lines.append(f"{name}：{stats['count']} 组")
            for key, label, unit in (("wrist_position_error_mm", "腕部位置", "mm"),
                                     ("tracker_position_error_mm", "Tracker 位置", "mm"),
                                     ("rotation_error_deg", "旋转", "deg")):
                s = stats[key]
                lines.append(f"  {label}：均值 {s['mean']:.4f}，RMSE {s['rmse']:.4f}，中位数 {s['median']:.4f}，P95 {s['p95']:.4f}，最大 {s['max']:.4f} {unit}")
        lines += ["逐组误差（位置为变换后左/右腕原点与实测腕原点之差）：", "组号  用途        腕位置 mm   旋转 deg   Tracker位置 mm"]
        for row in report["per_group"]:
            lines.append(f"{row['id']:4d}  {row['split']:10s}  {row['wrist_position_error_mm']:10.4f}  {row['rotation_error_deg']:9.4f}  {row['tracker_position_error_mm']:14.4f}")
        th = report["checks"]["thresholds"]
        lines.append(f"参考阈值（非厂家限位）：位置 RMSE≤{th['position_rmse_mm']:g} mm、最大≤{th['position_max_mm']:g} mm；旋转 RMSE≤{th['rotation_rmse_deg']:g}°、最大≤{th['rotation_max_deg']:g}°。")
        obs = report["observability"]
        lines.append(f"可观测性：拟合组最大相对转角 {obs['max_robot_rotation_span_deg']:.2f}°，第二轴比值 {obs['second_axis_ratio']:.4f}，Jacobian 秩 {obs['jacobian_rank']}/12，缩放条件数 {obs['jacobian_scaled_condition_number']:.2f}。")
    for rejected in report.get("rejected_groups", []):
        lines.append(f"未使用组 {rejected['id']}：{rejected['reason']}")
    lines.append("本程序只做离线估计，未应用至回放。误差反映两套测量的一致性，不等同于外部真值精度；安装/地图/TCP 定义改变后需复核。")
    return "\n".join(lines) + "\n"


def solve_file(path, output_dir=None, include_status20=False, validation_ids=None,
               position_tolerance_mm=5., rotation_tolerance_deg=2.):
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        path = path / "samples.jsonl"
    output = Path(output_dir).expanduser().resolve() if output_dir else path.parent / "solutions" / (
        datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema": "tracker-tron2-calibration-v1", "status": "failed", "input_file": str(path),
              "created_at": datetime.now().astimezone().isoformat(), "output_dir": str(output),
              "applied_to_replay": False, "numerical_checks_passed": False}
    try:
        for value in (position_tolerance_mm, rotation_tolerance_deg):
            if not math.isfinite(value) or value <= 0:
                raise CalibrationError("误差参考阈值必须是有限正数")
        metadata, groups, rejected, digest = load_samples(path, include_status20)
        report.update(session=metadata, input_sha256=digest, rejected_groups=rejected,
                      used_group_count=len(groups), include_status20=bool(include_status20))
        report.update(solve_groups(groups, validation_ids, position_tolerance_mm, rotation_tolerance_deg))
        if metadata.get("robot_pose_source") == JOINT_FK_SOURCE:
            report["warnings"].append("机器人腕位姿由实测关节与记录的运动学模型计算；误差包含模型/零位偏差，非直接笛卡尔真值")
        diagnostic = bool(metadata.get("mock")) or any(20 in g.statuses for g in groups)
        if metadata.get("mock"):
            report["warnings"].append("输入为模拟数据，仅用于软件验证")
        if any(20 in g.statuses for g in groups):
            report["warnings"].append("使用了 status=20 未确认状态，仅为诊断估计，不能认定标定有效")
        report["diagnostic_only"] = diagnostic
        report["status"] = "passed" if report["numerical_checks_passed"] and not diagnostic else "review_required"
        calibration = {key: report[key] for key in ("schema", "status", "T_ET", "T_WB", "T_BW", "input_file", "input_sha256", "session", "diagnostic_only", "numerical_checks_passed", "applied_to_replay")}
        calibration.update(transform_convention="T_WT = T_WB @ T_BE @ T_ET", position_unit="m",
                           report_file=str(output / "report.json"), warnings=report["warnings"])
        (output / "calibration.json").write_text(json.dumps(calibration, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        with (output / "errors.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            fields = ["id", "split", "label", "pair_count", "tracker_position_error_mm", "wrist_position_error_mm", "rotation_error_deg"]
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(report["per_group"])
    except (ValueError, OSError, KeyError, TypeError, np.linalg.LinAlgError) as exc:
        report.update(status="failed", error=str(exc))
        # 本次新建目录内的失败结果不能留下看似成功的变换文件。
        (output / "calibration.json").unlink(missing_ok=True)
        if isinstance(exc, CalibrationError):
            report["failure_diagnostics"] = exc.diagnostics
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (output / "report.txt").write_text(format_report(report), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="samples.jsonl 或会话目录")
    parser.add_argument("--output-dir", type=Path, help="新的结果目录；省略则在会话下自动新建 solutions/时间戳")
    parser.add_argument("--include-status-20", action="store_true", help="仅诊断：允许使用 status=20，结果始终标记待确认")
    parser.add_argument("--validation-ids", help="不参与拟合的整组编号，如 3,7,12；省略则 ≥9 组时自动留出至少 3 组")
    parser.add_argument("--position-tolerance-mm", type=float, default=5., help="位置 RMSE 参考阈值，最大误差参考值为其 3 倍")
    parser.add_argument("--rotation-tolerance-deg", type=float, default=2., help="旋转 RMSE 参考阈值，最大误差参考值为其 3 倍")
    args = parser.parse_args()
    try:
        validation_ids = None if args.validation_ids is None else {int(s) for s in args.validation_ids.split(",") if s.strip()}
        report = solve_file(args.input, args.output_dir, args.include_status_20, validation_ids,
                            args.position_tolerance_mm, args.rotation_tolerance_deg)
    except (ValueError, OSError) as exc:
        print(f"错误：{exc}")
        return 1
    print(format_report(report))
    print(f"结果目录：{report['output_dir']}")
    return 1 if report["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
