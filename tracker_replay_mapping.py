"""Tracker 标定矩阵的纯几何回放转换；不连接设备、不发送运动指令。

tracker-relative 只把 Tracker 原点的相对运动换到基座轴向，不假定旧数据与
机器人上的 Tracker 安装相同。wrist-relative 先用完整安装变换求腕位姿，
仅适用于源数据具有对应安装关系的情况。两种模式都以首帧为零增量，
由调用方把增量叠加到准备位姿；本模块不把历史地图原点当作运动目标。
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


_CONVENTIONS = {"raw-xyzw", "legacy-wzyx"}
_FORMATS = {"pyvut-legacy", "wxyz"}


def _se3(value, name: str) -> np.ndarray:
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是数值 4×4 刚体变换") from exc
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{name} 必须是有限的 4×4 刚体变换")
    if not np.allclose(matrix[3], [0., 0., 0., 1.], rtol=0., atol=1e-7):
        raise ValueError(f"{name} 齐次矩阵末行必须为 [0,0,0,1]")
    rotation = matrix[:3, :3]
    if (not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0., atol=1e-6)
            or not math.isclose(float(np.linalg.det(rotation)), 1., abs_tol=1e-6)):
        raise ValueError(f"{name} 必须包含正交且 det=+1 的旋转；不允许反射或缩放")
    return matrix.copy()


def _read_object(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as source:
        result = json.load(source)
    if not isinstance(result, dict):
        raise ValueError(f"JSON 顶层必须是对象：{path}")
    return result


def load_calibration(path: Path | str, arm: str = "left") -> dict:
    """校验 SE(3)、单位、手侧及验收哈希，并保留可写入日志的来源信息。

    用户验收记录只用于追溯，不会启用运动或修改冻结文件中的验收状态。
    """
    if arm not in {"left", "right"}:
        raise ValueError("arm 必须是 left 或 right")
    source = Path(path).expanduser().resolve()
    # 同一份字节同时用于解析与哈希，避免读取期间文件变化造成来源不一致。
    raw = source.read_bytes()
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("标定 JSON 顶层必须是对象")
    if data.get("position_unit") != "m":
        raise ValueError("标定 position_unit 必须为 m，不能猜测或自动转换单位")
    convention = data.get("tracker_quaternion_convention")
    if convention not in _CONVENTIONS:
        raise ValueError("标定 tracker_quaternion_convention 必须为 raw-xyzw 或 legacy-wzyx")
    session = data.get("session", {})
    if not isinstance(session, dict):
        raise ValueError("标定 session 必须是对象")
    declared_arms = [value for value in (data.get("arm"), session.get("arm"))
                     if value is not None]
    if not declared_arms or any(value != arm for value in declared_arms):
        raise ValueError(f"标定手侧 {declared_arms!r} 与回放 {arm} 不匹配")
    result = dict(data)
    result["T_BW"] = _se3(data.get("T_BW"), "T_BW").tolist()
    result["T_ET"] = _se3(data.get("T_ET"), "T_ET").tolist()
    result["source"] = str(source)
    result["sha256"] = hashlib.sha256(raw).hexdigest()
    result["arm"] = arm
    result["robot_id"] = session.get("robot_id", data.get("robot_id"))
    result["acceptance"] = None
    acceptance_path = source.parent / "acceptance.json"
    if acceptance_path.is_file():
        acceptance = _read_object(acceptance_path)
        named_file = acceptance.get("calibration_file")
        if named_file is not None and not isinstance(named_file, str):
            raise ValueError("acceptance.json calibration_file 必须是文件路径字符串")
        claims_this_file = (named_file is None
                            or (source.parent / named_file).resolve() == source)
        if claims_this_file:
            if acceptance.get("calibration_sha256") != result["sha256"]:
                raise ValueError("标定文件 SHA256 与 acceptance.json 不一致，不能沿用该验收记录")
            result["acceptance"] = dict(acceptance, source=str(acceptance_path))
    return result


def resolve_quaternion_format(calibration: dict, quaternion_format: str | None) -> str:
    """输入默认仍是本项目保存的 WXYZ；raw-xyzw 标定显式纠正旧 pyvut 排列。"""
    convention = calibration.get("tracker_quaternion_convention")
    if convention not in _CONVENTIONS:
        raise ValueError("不支持的标定四元数约定")
    if quaternion_format is None:
        return "pyvut-legacy" if convention == "raw-xyzw" else "wxyz"
    if quaternion_format not in _FORMATS:
        raise ValueError("quaternion_format 必须为 pyvut-legacy 或 wxyz")
    return quaternion_format


def calibrated_relative_motion(
    trajectory,
    calibration: dict,
    translation_scale: float,
    orientation_scale: float,
    mode: str = "tracker-relative",
    quaternion_format: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回基座系位置增量、WXYZ 旋转增量及相对转角（度）。

    wrist-relative: S_i = T_BW Q_i inv(T_ET)，然后分别取 p_i-p_0 与
    R_i R_0^T；不是 inv(S_0) S_i，因此不会把平移误转到首帧腕部轴向。
    相对旋转由调用方左乘准备朝向；不把首帧朝向当成已标定的绝对朝向。
    """
    if (not math.isfinite(translation_scale) or translation_scale < 0
            or not math.isfinite(orientation_scale) or not 0 <= orientation_scale <= 1):
        raise ValueError("平移倍率须为有限非负数，旋转倍率须在 0 至 1 之间")
    if mode not in {"tracker-relative", "wrist-relative"}:
        raise ValueError("mode 必须为 tracker-relative 或 wrist-relative")
    world = _se3(calibration.get("T_BW"), "T_BW")
    mount = _se3(calibration.get("T_ET"), "T_ET")
    source_format = resolve_quaternion_format(calibration, quaternion_format)
    positions = np.asarray(trajectory.positions, dtype=np.float64)
    quaternions = np.asarray(trajectory.quaternions, dtype=np.float64)
    if (positions.ndim != 2 or positions.shape[1:] != (3,) or len(positions) == 0
            or quaternions.shape != (len(positions), 4)):
        raise ValueError("Tracker positions/quaternions 必须分别为非空 N×3、N×4")
    if not np.isfinite(positions).all() or not np.isfinite(quaternions).all():
        raise ValueError("Tracker 位姿含非有限数值")
    norms = np.linalg.norm(quaternions, axis=1)
    if not np.isfinite(norms).all() or np.any(norms < 1e-9):
        raise ValueError("Tracker 四元数为零或无效")
    quaternions = quaternions / norms[:, None]
    if (source_format == "pyvut-legacy"
            and calibration["tracker_quaternion_convention"] == "raw-xyzw"):
        quaternions = quaternions[:, [1, 0, 3, 2]]
    rotations = Rotation.from_quat(quaternions[:, [1, 2, 3, 0]]).as_matrix()

    mapped_rotations = world[:3, :3] @ rotations
    if mode == "wrist-relative":
        mapped_rotations = mapped_rotations @ mount[:3, :3].T
        # T_BE 的平移：R_BW p_WT + t_BW - R_BE t_ET。
        mapped_positions = (positions @ world[:3, :3].T + world[:3, 3]
                            - mapped_rotations @ mount[:3, 3])
        delta_positions = mapped_positions - mapped_positions[0]
    else:
        # 安装未校验的历史数据只转换 Tracker 原点增量，不套用机器人的安装偏移。
        delta_positions = (positions - positions[0]) @ world[:3, :3].T
    delta_positions *= translation_scale
    delta_rotations = mapped_rotations @ mapped_rotations[0].T
    rotvec = Rotation.from_matrix(delta_rotations).as_rotvec() * orientation_scale
    delta_quaternions = Rotation.from_rotvec(rotvec).as_quat()[:, [3, 0, 1, 2]]
    angles = np.rad2deg(np.linalg.norm(rotvec, axis=1))
    # 首帧不能因矩阵数值误差造成启动目标跳变。q 与 -q 的符号不应影响插值。
    delta_positions[0] = 0.
    delta_quaternions[0] = [1., 0., 0., 0.]
    angles[0] = 0.
    for index in range(1, len(delta_quaternions)):
        if np.dot(delta_quaternions[index - 1], delta_quaternions[index]) < 0:
            delta_quaternions[index] *= -1
    return delta_positions, delta_quaternions, angles
