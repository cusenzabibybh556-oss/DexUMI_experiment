"""TRON2 左臂离线 IK；不连接机器人，不发送任何运动指令。

SDK TCP 来自官方控制器 robot.urdf 的固定末端变换，并非 O6 掌心标定值。
MuJoCo 模型用于运动学和关节限位核对；不据此宣称包含桌面/O6 的完整碰撞验证。
默认规划比较多条连续关节轨迹，优先选择整段最小限位余量最大的候选。
结果仅用于离线预检；ServoP 不接收本模块的关节解，不代表固件采用同一分支。
"""

from pathlib import Path

import numpy as np

from replay_tron2_tracker import quaternion_to_matrix, quaternion_distance_deg


DEFAULT_MODEL = Path(
    "/home/zmy/Dexverse_workspace/DexVerse/model/tron2-robot-description"
    "/tron2a/DACH_TRON2A/xml/robot.xml"
)
SDK_TCP_OFFSET = np.array([-.0338, 0., -.2125])
JOINT_NAMES = [
    *(f"{part}_{side}_Joint" for side in ("L", "R") for part in (
        "proximal_pitch", "proximal_roll", "proximal_yaw", "elbow",
        "wrist_yaw", "wrist_pitch", "wrist_roll")),
    "head_pitch_Joint", "head_yaw_Joint",
]
# 官方 SDK 与模型限位取交集；不扩大固件文档限位。
# 使用手册角度列的精确值，避免四舍五入后的弧度把 ±90° 扩成 ±90.0002°。
# https://www.limxdynamics.com/zh/documents/844648486841487360，1.5 节。
SDK_LOWER = np.deg2rad([
    -180, -15, -210, -150, -100, -45, -90,
    -180, -183, -85, -150, -80, -45, -90, -45, -90,
])
SDK_UPPER = np.deg2rad([
    149, 183, 85, 15, 80, 45, 90,
    149, 15, 210, 15, 100, 45, 90, 60, 90,
])


class LeftArmIK:
    def __init__(self, model_path=DEFAULT_MODEL):
        try:
            import mujoco
        except ImportError as exc:
            raise RuntimeError("离线左臂 IK 需要当前 Python 环境安装 mujoco") from exc
        self.mj = mujoco
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise RuntimeError(f"找不到 TRON2 IK 模型：{self.model_path}")
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        self.qadr = np.array([self.model.joint(n).qposadr[0] for n in JOINT_NAMES])
        self.dadr = np.array([self.model.joint(n).dofadr[0] for n in JOINT_NAMES])
        model_limits = np.array([self.model.joint(n).range for n in JOINT_NAMES])
        self.lower = np.maximum(model_limits[:, 0], SDK_LOWER)
        self.upper = np.minimum(model_limits[:, 1], SDK_UPPER)
        self.body = self.model.body("wrist_roll_L_Link").id
        self.base = self.model.body("base_Link").id
        self.jacp = np.zeros((3, self.model.nv))
        self.jacr = np.zeros((3, self.model.nv))

    def check_joints(self, q):
        q = np.asarray(q, dtype=float)
        if q.shape != (16,) or not np.isfinite(q).all():
            raise ValueError("离线 IK 需要 16 个有限关节值，顺序为左7、右7、头部 pitch/yaw")
        invalid = np.flatnonzero((q < self.lower) | (q > self.upper))
        if invalid.size:
            i = int(invalid[0])
            raise RuntimeError(f"模型关节越限：{JOINT_NAMES[i]}={q[i]:.5f} rad")

    def _forward(self, q):
        self.data.qpos[self.qadr] = q
        self.mj.mj_kinematics(self.model, self.data)
        self.mj.mj_comPos(self.model, self.data)
        rotation = self.data.xmat[self.body].reshape(3, 3).copy()
        point = self.data.xpos[self.body] + rotation @ SDK_TCP_OFFSET
        return point - self.data.xpos[self.base], rotation, point

    def pose(self, q):
        self.check_joints(q)
        p, _, _ = self._forward(q)
        return p, self.data.xquat[self.body].copy()

    def check_anchor(self, q, position, quaternion):
        p, u = self.pose(q)
        error = np.linalg.norm(p - position)
        angle = quaternion_distance_deg(u, quaternion)
        if error > .002 or angle > .5:
            raise RuntimeError(
                f"本机 IK 模型与实机反馈不一致：{error*1000:.2f} mm / {angle:.2f}°；"
                "请核对模型、TCP 和关节零位；离线 IK 不发送运动指令"
            )
        return float(error), float(angle)

    def plan(self, q_start, positions, quaternions, times):
        """默认使用整段限位余量择优；不发送关节目标、不改变起点或笛卡尔轨迹。"""
        from tron2_ik_planner import plan_with_joint_margin
        return plan_with_joint_margin(self, q_start, positions, quaternions, times)

    def plan_tracking(self, q_start, positions, quaternions, times):
        """旧逐帧跟踪解，仅保留作离线对照；贴限不等于其它 IK 分支不可行。"""
        """完整预计算；失败即拒绝整个计划，不边计算边发送，也不静默缩放轨迹。"""
        q_start = np.asarray(q_start, dtype=float).copy()
        self.check_joints(q_start)
        positions = np.asarray(positions, dtype=float)
        quaternions = np.asarray(quaternions, dtype=float)
        times = np.asarray(times, dtype=float)
        n = len(times)
        if (positions.shape != (n, 3) or quaternions.shape != (n, 4) or n < 2
                or not np.isfinite(positions).all() or not np.isfinite(quaternions).all()
                or not np.isfinite(times).all() or np.any(np.diff(times) <= 0)):
            raise ValueError("IK 目标、四元数或严格递增的时间轴无效")
        self.check_anchor(q_start, positions[0], quaternions[0])
        plan = np.tile(q_start, (n, 1))
        q = q_start.copy()
        max_error = 0.
        for i, (target, quat) in enumerate(zip(positions, quaternions)):
            desired_rotation = quaternion_to_matrix(quat)
            for _ in range(70):
                p, rotation, point = self._forward(q)
                position_error = target - p
                # 目标与当前 SO(3) 的世界坐标旋转误差；每次步长受限。
                angular_error = .5 * sum(np.cross(rotation[:, k], desired_rotation[:, k])
                                         for k in range(3))
                if np.linalg.norm(position_error) <= .00001 and np.linalg.norm(angular_error) <= .0002:
                    break
                self.mj.mj_jac(self.model, self.data, self.jacp, self.jacr, point, self.body)
                jac = np.vstack((self.jacp[:, self.dadr[:7]], .25*self.jacr[:, self.dadr[:7]]))
                error = np.concatenate((position_error, .25*angular_error))
                step = jac.T @ np.linalg.solve(jac@jac.T + .00001*np.eye(6), error)
                step *= min(1., .025/max(float(np.max(np.abs(step))), 1e-12))
                q[:7] = np.clip(q[:7] + step, self.lower[:7]+1e-5, self.upper[:7]-1e-5)
            p, actual_quat = self.pose(q)
            error = float(np.linalg.norm(target-p))
            angle = quaternion_distance_deg(actual_quat, quat)
            if error > .001 or angle > .5:
                raise RuntimeError(
                    f"左臂 IK 在第 {i}/{n} 帧未找到满足限位及误差的解："
                    f"{error*1000:.2f} mm / {angle:.2f}°；"
                    "整段未发送。请检查姿态/轨迹，或降低 --scale 后重新预检"
                )
            max_error = max(max_error, error)
            plan[i, :7] = q[:7]
        # 第一条始终是启动时真实关节角，禁止 IK 预热改变接管首帧。
        plan[0] = q_start
        velocities = np.diff(plan[:, :7], axis=0)/np.diff(times)[:, None]
        peak_velocity = float(np.max(np.abs(velocities)))
        if peak_velocity > .5:
            raise RuntimeError(
                f"IK 轨迹最大关节速度 {peak_velocity:.3f} rad/s 超过首轮测试 0.5 rad/s；"
                "整段未发送，请降低 --speed 或 --scale"
            )
        if not np.array_equal(plan[:, 7:], np.tile(q_start[7:], (n, 1))):
            raise RuntimeError("内部错误：右臂或头部关节目标发生变化")
        return plan, {"max_position_error_mm": max_error*1000,
                      "max_joint_speed_rad_s": peak_velocity, "frames": n}
