#!/usr/bin/env python3
"""Offline MuJoCo Z-feedback experiments. No robot SDK or networking imports.

Uses a local damped-least-squares IK and PD motor approximation, NOT the vendor
ServoP runtime. Fixtures: fixed base, contacts disabled, factory grippers removed,
synthetic payload at an explicitly assumed COM. Never a real-robot validation.
"""
import argparse
from collections import deque
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tron2_z_compensation import Config, ZCompensationController
from tools.tron2_payload_model import assembly
from tron2_pose_compensation import PoseConfig, PoseCorrection

MODEL = Path('/home/zmy/Dexverse_workspace/DexVerse/model/tron2-robot-description/tron2a/DACH_TRON2A/xml/robot.xml')
NAMES = [f'{part}_{side}_Joint' for side in ('L', 'R') for part in
         ('proximal_pitch', 'proximal_roll', 'proximal_yaw', 'elbow', 'wrist_yaw', 'wrist_pitch', 'wrist_roll')]
NAMES += ['head_pitch_Joint', 'head_yaw_Joint']
TCP = np.array([-.0338, 0., -.2125])
KP = np.array([420, 420, 400, 400, 300, 200, 150]*2+[20, 20], float)
KD = np.array([12, 12, 15, 15, 10, 10, 10]*2+[2, 2], float)


@dataclass(frozen=True)
class PhysicsSettings:
    reference_delay_s: float = 0.
    reference_filter_tau_s: float = 0.
    reference_preview_s: float = 0.
    feedback_delay_s: float = .05
    feedback_jitter_s: float = 0.
    feedback_pairing: str = 'capture'
    joint_friction_nm: float = 0.
    pose_ff_scale: float = 1.
    pose_velocity_ff: bool = False
    pose_velocity_filter_tau_s: float = .03
    seed: int = 7
    z: Config = field(default_factory=Config)
    pose_feedback: PoseConfig = field(default_factory=PoseConfig)

    def __post_init__(self):
        for name, limit in [('reference_delay_s', .3), ('reference_filter_tau_s', .2),
                            ('reference_preview_s', .3),
                            ('feedback_delay_s', .3), ('feedback_jitter_s', .05), ('joint_friction_nm', 2.)]:
            value = getattr(self, name)
            if not np.isfinite(value) or not 0 <= value <= limit:
                raise ValueError(f'Invalid simulated {name}')
        if not np.isfinite(self.pose_ff_scale) or not 0 <= self.pose_ff_scale <= 1.5:
            raise ValueError('Invalid model pose feedforward scale')
        if (not isinstance(self.pose_velocity_ff, bool) or not np.isfinite(self.pose_velocity_filter_tau_s)
                or not 0 < self.pose_velocity_filter_tau_s <= .2):
            raise ValueError('Invalid model velocity reference feedforward')
        if self.feedback_pairing not in ('capture', 'host-midpoint'):
            raise ValueError('Unknown simulated feedback pairing')
        if (self.feedback_jitter_s > self.feedback_delay_s or not isinstance(self.z, Config)
                or not isinstance(self.pose_feedback, PoseConfig)):
            raise ValueError('Invalid feedback jitter or Z controller configuration')


def lagged_reference(history, due):
    """Interpolate generated joint references at an earlier simulation time."""
    while len(history) > 1 and history[1][0] <= due:
        history.popleft()
    if len(history) == 1 or due <= history[0][0]:
        return history[0][1].copy()
    alpha = np.clip((due-history[0][0])/(history[1][0]-history[0][0]), 0, 1)
    return (1-alpha)*history[0][1]+alpha*history[1][1]


def modified_xml(source, mass, com, payload_spec=None):
    tree = ET.parse(source).getroot()
    tree.find('compiler').set('meshdir', str((source.parent/'../meshes').resolve()))
    tree.find('option').set('timestep', str(1/900))
    base = tree.find(".//body[@name='base_Link']")
    for j in list(base.findall('joint')):
        if j.get('type') == 'free':
            base.remove(j)
    for parent in tree.iter():
        for child in list(parent):
            if child.tag == 'body' and child.get('name', '').startswith('grasper_'):
                parent.remove(child)
    for geom in tree.iter('geom'):
        geom.set('contype', '0'); geom.set('conaffinity', '0')
    if payload_spec is not None:
        wrist = tree.find(".//body[@name='wrist_roll_L_Link']")
        body = ET.SubElement(wrist, 'body', name='vendor_O6_rigid_aggregate')
        inertia = np.asarray(payload_spec['inertia_wrist_kg_m2'])
        ET.SubElement(body, 'inertial', pos=' '.join(map(str, payload_spec['COM_wrist_m'])),
                      mass=str(payload_spec['mass_kg']),
                      fullinertia=' '.join(map(str, inertia[(0, 1, 2, 0, 0, 1), (0, 1, 2, 1, 2, 2)])))
    elif mass > 0:
        wrist = tree.find(".//body[@name='wrist_roll_L_Link']")
        body = ET.SubElement(wrist, 'body', name='synthetic_payload', pos=' '.join(map(str, com)))
        inertia = .4*mass*.03**2
        ET.SubElement(body, 'inertial', pos='0 0 0', mass=str(mass),
                      diaginertia=' '.join([str(inertia)]*3))
    return ET.tostring(tree, encoding='unicode')


def lift(t, amplitude, ramp):
    """30 s static learning, then hold/up/hold/down/hold; all in simulation time."""
    s = t-33
    if s < 0:
        return 0.
    if s < ramp:
        a = s/ramp
        return amplitude*a*a*(3-2*a)
    if s < ramp+3:
        return amplitude
    if s < 2*ramp+3:
        a = (s-ramp-3)/ramp
        return amplitude*(1-a*a*(3-2*a))
    return 0.


class TraceReference:
    """Read only completed linear nominal trajectories; no hardware imports."""
    def __init__(self, path):
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if any(r['event'] in ('control_abort', 'error') for r in rows):
            raise ValueError('Reference trace contains an abort/error')
        if not any(r['event'] == 'z_replay_closed' for r in rows):
            raise ValueError('Reference trace has not closed')
        plan = next(r for r in rows if r['event'] == 'fixed_start_plan')
        if plan.get('tracker_interpolation', 'linear') != 'linear':
            raise ValueError('This reference replay supports linear input only')
        start = next(r for r in rows if r['event'] == 'replay_start')['start_monotonic_ns']
        commands = [r for r in rows if r['event'] == 'z_replay_command']
        sends = [r for r in rows if r['event'] == 'servop_command']
        if len(commands) != len(sends):
            raise ValueError('Unpaired nominal/send trace records')
        active = np.array([r['phase'] == 'trajectory' for r in commands])
        self.times = np.array([(r['monotonic_ns']-start)/1e9 for r in sends])[active]
        self.poses = np.array([r['nominal_left_pos'] for r in commands])[active]
        if len(self.times) < 2 or not np.all(np.diff(self.times) > 0) or not np.isfinite(self.poses).all():
            raise ValueError('Invalid reference times/poses')
        self.delta = self.poses[:, :3]-self.poses[0, :3]
        rotations = Rotation.from_quat(self.poses[:, [4, 5, 6, 3]])
        self.rotations = Slerp(self.times, rotations)
        self.initial_inverse = rotations[0].inv().as_matrix()
        self.duration = float(self.times[-1])
        self.metadata = dict(trace=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            episode=plan['episode'], speed=plan['speed'], scale=plan['scale'],
            orientation_scale=plan['orientation_scale'], trajectory_duration_s=self.duration,
            nominal_target_count=int(active.sum()),
            mapping='Identical base-frame translation deltas; identical spatial rotation deltas; reanchored to simulation neutral pose')

    def target(self, t, anchor, initial_rotation):
        t = float(np.clip(t, self.times[0], self.times[-1]))
        p = anchor+np.array([np.interp(t, self.times, self.delta[:, k]) for k in range(3)])
        r = self.rotations(t).as_matrix()@self.initial_inverse@initial_rotation
        return p, r


def experiment(source, mass, com, mode, amplitude, ramp, output, reference=None, settings=None, payload_spec=None):
    settings = settings or PhysicsSettings()
    use_z = mode in ('continuous', 'gravity_ff_continuous', 'pose_reference_ff_z')
    use_ff = mode in ('gravity_ff', 'gravity_ff_continuous')
    use_pose_feedback = mode == 'pose_reference_ff_feedback'
    use_pose_ff = mode in ('pose_reference_ff', 'pose_reference_ff_z', 'pose_reference_ff_feedback')
    if mode not in ('none', 'continuous', 'gravity_ff', 'gravity_ff_continuous', 'pose_reference_ff', 'pose_reference_ff_z', 'pose_reference_ff_feedback'):
        raise ValueError('Unknown simulation control mode')
    xml = modified_xml(source, mass, com, payload_spec)
    model = mujoco.MjModel.from_xml_string(xml)
    data, ik, gravity = [mujoco.MjData(model) for _ in range(3)]
    ids = np.array([model.joint(n).id for n in NAMES])
    qa, da = model.jnt_qposadr[ids], model.jnt_dofadr[ids]
    model.dof_frictionloss[da[:14]] += settings.joint_friction_nm
    actuator_ids = np.array([np.flatnonzero(model.actuator_trnid[:, 0] == j)[0] for j in ids])
    assert model.nq == model.nv == 16
    assert np.all(model.actuator_gear[actuator_ids, 0] == 1)
    q0 = np.zeros(16)
    q0[:7] = [0, 0, .18322281083347552, -np.pi/2, -np.pi/2, 0, 0]
    q0[9] = -1.4
    for d in (data, ik, gravity):
        d.qpos[qa] = q0
        mujoco.mj_forward(model, d)
    body = model.body('wrist_roll_L_Link').id
    def pose(d):
        r = d.xmat[body].reshape(3, 3)
        return d.xpos[body]+r@TCP, r
    anchor, orientation = pose(data)
    anchor, orientation = anchor.copy(), orientation.copy()
    jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    qref = q0.copy()
    qissued = q0.copy()
    controller = ZCompensationController(settings.z, active_limit_m=settings.z.max_offset_m)
    pose_controller = PoseCorrection(settings.pose_feedback)
    pending = deque()
    reference_history = deque()
    motor_q = q0.copy()
    rng = np.random.default_rng(settings.seed)
    records = []
    actual_positions, nominal_positions, orientation_errors = [], [], []
    generated_references, delivered_references, issued_positions = [], [], []
    actual_rotations, nominal_rotations, pose_offsets_p, pose_offsets_r = [], [], [], []
    trajectory_end = 30+reference.duration if reference is not None else 39+2*ramp
    duration = trajectory_end+2 if reference is not None else trajectory_end
    correction = 0.
    max_ik_error = 0.
    max_ik_angle_deg = 0.
    max_pose_ff_shift_m = max_pose_ff_angle_deg = 0.
    previous_nominal_q = q0.copy()
    filtered_nominal_velocity = np.zeros(16)
    feedforward = np.zeros(16)
    def target(t):
        if reference is not None:
            return reference.target(t-30, anchor, orientation)
        position = anchor.copy()
        position[2] += lift(t, amplitude, ramp)
        return position, orientation
    for tick in range(int(round(duration*900))+1):
        t = tick/900
        if tick % 3 == 0:
            mujoco.mj_forward(model, data)
            now = 1_000_000_000+round(t*1e9)
            p, actual_orientation = pose(data)
            nominal_position, desired_orientation = target(t)
            nominal_z = nominal_position[2]
            # Capture pairing retains the original idealized experiment. Host
            # midpoint pairing represents unknown internal pose age, separately
            # from a 2 ms synthetic network query RTT; these are not conflated.
            if tick % 45 == 0:
                delay = settings.feedback_delay_s+rng.uniform(-settings.feedback_jitter_s, settings.feedback_jitter_s)
                pending.append((tick+round(delay*900), tick, now, nominal_z, float(p[2]),
                                nominal_position.copy(), desired_orientation.copy(), p.copy(), actual_orientation.copy()))
            while pending and pending[0][0] <= tick:
                _, sequence, sent, nz, az, measured_nominal_p, measured_nominal_r, measured_p, measured_r = pending.popleft()
                if settings.feedback_pairing == 'host-midpoint':
                    sent = max(0, now-2_000_000)
                    measured_nominal_p, measured_nominal_r = target(t-.001)
                    nz = float(measured_nominal_p[2])
                if use_z:
                    controller.observe(sequence, sent, now, now, nz, az, True)
                if use_pose_feedback:
                    pose_controller.observe(sequence, sent, now, now, measured_nominal_p, measured_nominal_r, measured_p, measured_r)
            correction = controller.advance(now) if use_z else 0.
            feedback_p, feedback_r = pose_controller.advance(now) if use_pose_feedback else (np.zeros(3), np.eye(3))
            desired = nominal_position.copy()
            command_orientation = desired_orientation
            if settings.reference_preview_s:
                desired, command_orientation = target(t+settings.reference_preview_s)
                desired = desired.copy()
            preview_position, preview_rotation = desired.copy(), command_orientation.copy()
            desired[2] += correction
            desired += feedback_p
            command_orientation = feedback_r@command_orientation
            for _ in range(12):
                ik.qpos[qa] = qref
                mujoco.mj_forward(model, ik)
                pi, ri = pose(ik)
                position_error = desired-pi
                rotation_error = .5*sum(np.cross(ri[:, k], command_orientation[:, k]) for k in range(3))
                if np.linalg.norm(position_error) < 1e-6 and np.linalg.norm(rotation_error) < 1e-5:
                    break
                mujoco.mj_jac(model, ik, jp, jr, pi, body)
                jac = np.vstack((jp[:, da[:7]], .25*jr[:, da[:7]]))
                step = jac.T@np.linalg.solve(jac@jac.T+1e-6*np.eye(6), np.r_[position_error, .25*rotation_error])
                step *= min(1., .02/max(np.max(abs(step)), 1e-12))
                qref[:7] = np.clip(qref[:7]+step, model.jnt_range[ids[:7], 0]+1e-4,
                                  model.jnt_range[ids[:7], 1]-1e-4)
            ik.qpos[qa] = qref
            mujoco.mj_forward(model, ik)
            residual = float(np.linalg.norm(desired-pose(ik)[0]))
            angle_residual = float(np.rad2deg(Rotation.from_matrix(pose(ik)[1]@command_orientation.T).magnitude()))
            max_ik_angle_deg = max(max_ik_angle_deg, angle_residual)
            max_ik_error = max(max_ik_error, residual)
            if residual > .001:
                raise RuntimeError(f'Simulation IK residual {residual*1000:.3f} mm; rejecting this scenario')
            if angle_residual > .5:
                raise RuntimeError(f'Simulation IK angle residual {angle_residual:.3f} deg at {t:.3f} s; rejecting this scenario')
            motor_reference = qref.copy()
            issued_position = desired.copy()
            issued_rotation = command_orientation.copy()
            if use_pose_ff:
                # Estimate the PD position offset required for static support.
                # Map its small-angle effect to a CARTESIAN pose using the
                # Jacobian, then independently solve that
                # pose from the last issued reference; the extra nullspace DOF
                # is not assumed to be directly controllable through ServoP.
                gravity.qpos[qa] = qref
                gravity.qvel[:] = 0
                mujoco.mj_forward(model, gravity)
                raw_velocity = (qref-previous_nominal_q)*300
                velocity_alpha = -np.expm1(-1/(300*settings.pose_velocity_filter_tau_s))
                filtered_nominal_velocity += velocity_alpha*(raw_velocity-filtered_nominal_velocity)
                previous_nominal_q = qref.copy()
                if settings.pose_velocity_ff and np.max(abs(filtered_nominal_velocity[:7])) > 2:
                    raise RuntimeError('Model velocity reference exceeds experimental 2 rad/s bound')
                desired_support = gravity.qfrc_bias[da[:7]].copy()
                if settings.pose_velocity_ff:
                    # Under the modeled PD law, Kd*dq_nominal is the reference
                    # offset needed to avoid producing damping tracking lag.
                    # This is converted to a pose reference, not published as torque.
                    desired_support += KD[:7]*filtered_nominal_velocity[:7]
                delta_q = settings.pose_ff_scale*desired_support/KP[:7]
                ik.qpos[qa] = qref
                mujoco.mj_forward(model, ik)
                mujoco.mj_jac(model, ik, jp, jr, pose(ik)[0], body)
                ff_position = desired+jp[:, da[:7]]@delta_q
                ff_rotation = Rotation.from_rotvec(jr[:, da[:7]]@delta_q).as_matrix()@command_orientation
                shift_m = float(np.linalg.norm(ff_position-desired))
                angle_deg = float(np.rad2deg(Rotation.from_matrix(ff_rotation@command_orientation.T).magnitude()))
                max_pose_ff_shift_m = max(max_pose_ff_shift_m, shift_m)
                max_pose_ff_angle_deg = max(max_pose_ff_angle_deg, angle_deg)
                if shift_m > .04 or angle_deg > 5:
                    raise RuntimeError('Model pose feedforward exceeds 40 mm / 5 degree experimental bounds')
                for _ in range(12):
                    ik.qpos[qa] = qissued
                    mujoco.mj_forward(model, ik)
                    pi, ri = pose(ik)
                    pe = ff_position-pi
                    re = .5*sum(np.cross(ri[:, k], ff_rotation[:, k]) for k in range(3))
                    if np.linalg.norm(pe) < 1e-6 and np.linalg.norm(re) < 1e-5:
                        break
                    mujoco.mj_jac(model, ik, jp, jr, pi, body)
                    jac = np.vstack((jp[:, da[:7]], .25*jr[:, da[:7]]))
                    step = jac.T@np.linalg.solve(jac@jac.T+1e-6*np.eye(6), np.r_[pe, .25*re])
                    step *= min(1., .02/max(np.max(abs(step)), 1e-12))
                    qissued[:7] = np.clip(qissued[:7]+step, model.jnt_range[ids[:7], 0]+1e-4,
                                          model.jnt_range[ids[:7], 1]-1e-4)
                ik.qpos[qa] = qissued
                mujoco.mj_forward(model, ik)
                if np.linalg.norm(ff_position-pose(ik)[0]) > .001:
                    raise RuntimeError('Model pose feedforward Cartesian re-solve failed')
                ff_angle = float(np.rad2deg(Rotation.from_matrix(pose(ik)[1]@ff_rotation.T).magnitude()))
                if ff_angle > .5:
                    raise RuntimeError(f'Model pose feedforward angular re-solve failed at {t:.3f} s: {ff_angle:.3f} deg')
                motor_reference = qissued.copy()
                issued_position = ff_position.copy()
                issued_rotation = ff_rotation.copy()
            if use_pose_feedback and (np.linalg.norm(issued_position-preview_position) > .04
                    or np.rad2deg(Rotation.from_matrix(issued_rotation@preview_rotation.T).magnitude()) > 5):
                raise RuntimeError('Combined 6D reference correction exceeds 40 mm / 5 degree bounds')
            reference_history.append((t, motor_reference))
            delayed = lagged_reference(reference_history, t-settings.reference_delay_s)
            alpha = 1. if settings.reference_filter_tau_s == 0 else -np.expm1(-1/(300*settings.reference_filter_tau_s))
            motor_q += alpha*(delayed-motor_q)
            if use_ff:
                gravity.qpos[:] = data.qpos
                gravity.qvel[:] = 0
                mujoco.mj_forward(model, gravity)
                feedforward = gravity.qfrc_bias[da].copy()
            # Issued reference includes preview and model pose feedforward;
            # nominal and the integral Z-only component remain separate.
            records.append([t, nominal_z, float(issued_position[2]), float(p[2]), correction])
            actual_positions.append(p.copy());nominal_positions.append(nominal_position.copy())
            actual_rotations.append(actual_orientation.copy());nominal_rotations.append(desired_orientation.copy())
            pose_offsets_p.append(feedback_p.copy());pose_offsets_r.append(Rotation.from_matrix(feedback_r).as_rotvec())
            issued_positions.append(issued_position)
            generated_references.append(motor_reference.copy());delivered_references.append(motor_q.copy())
            orientation_errors.append(np.rad2deg(Rotation.from_matrix(actual_orientation@desired_orientation.T).magnitude()))
        torque = KP*(motor_q-data.qpos[qa])-KD*data.qvel[da]+feedforward
        data.ctrl[actuator_ids] = np.clip(torque, model.actuator_ctrlrange[actuator_ids, 0], model.actuator_ctrlrange[actuator_ids, 1])
        mujoco.mj_step(model, data)
        if not np.isfinite(data.qpos).all() or np.max(abs(data.qvel)) > 50:
            raise RuntimeError('Simulation diverged')
    a = np.array(records)
    error = (a[:, 3]-a[:, 1])*1000
    active = (a[:, 0] >= 30) & (a[:, 0] <= trajectory_end)
    tail = a[:, 0] >= duration-2
    xyz = (np.array(actual_positions)-np.array(nominal_positions))*1000
    angular = np.array(orientation_errors)
    result = dict(mode=mode, assumed_payload_kg=mass,
        assumed_payload_COM_in_wrist_frame_m=payload_spec['COM_wrist_m'] if payload_spec is not None else list(com),
        Z_trajectory_RMSE_mm=float(np.sqrt(np.mean(error[active]**2))),
        Z_trajectory_abs_P95_mm=float(np.percentile(abs(error[active]), 95)),
        Z_final_two_seconds_mean_mm=float(error[tail].mean()),
        first_half_second_max_drop_mm=float(-error[a[:, 0] <= .5].min()),
        maximum_correction_mm=float(a[:, 4].max()*1000), maximum_IK_residual_mm=max_ik_error*1000,
        maximum_IK_angle_residual_deg=max_ik_angle_deg,
        trajectory_end_s=trajectory_end, trajectory_command_samples=int(active.sum()),
        planned_maximum_rise_mm=float((a[active, 1].max()-anchor[2])*1000),
        observed_maximum_rise_mm=float((a[active, 3].max()-anchor[2])*1000),
        XYZ_RMSE_mm=np.sqrt(np.mean(xyz[active]**2, axis=0)).tolist(),
        position_norm_P95_mm=float(np.percentile(np.linalg.norm(xyz[active], axis=1), 95)),
        orientation_RMSE_deg=float(np.sqrt(np.mean(angular[active]**2))),
        orientation_P95_deg=float(np.percentile(angular[active], 95)),
        maximum_downward_error_mm=float(max(0., -error[active].min())),
        maximum_upward_error_mm=float(max(0., error[active].max())),
        correction_at_cap_percent=float(np.mean(a[active, 4] >= settings.z.max_offset_m-1e-9)*100) if use_z else 0.,
        explicit_torque_feedforward=use_ff, model_pose_reference_feedforward=use_pose_ff,
        residual_pose_feedback=use_pose_feedback,
        maximum_pose_feedback_translation_mm=float(np.linalg.norm(pose_offsets_p, axis=1).max()*1000),
        maximum_pose_feedback_rotation_deg=float(np.rad2deg(np.linalg.norm(pose_offsets_r, axis=1).max())),
        maximum_model_pose_shift_mm=max_pose_ff_shift_m*1000,
        maximum_model_pose_rotation_deg=max_pose_ff_angle_deg,
        physics_settings=asdict(settings), payload_model=payload_spec)
    stem = f'mass_{mass:g}_{mode}'
    np.savez_compressed(output/(stem+'.npz'), time_s=a[:, 0], nominal_z_m=a[:, 1],
        issued_z_m=a[:, 2], actual_z_m=a[:, 3], correction_m=a[:, 4],
        actual_xyz_m=np.array(actual_positions), nominal_xyz_m=np.array(nominal_positions),
        issued_xyz_m=np.array(issued_positions), generated_joint_reference_rad=np.array(generated_references),
        delivered_joint_reference_rad=np.array(delivered_references), orientation_error_deg=angular)
    # Rotations and residual offsets are kept for independent clock/pose audits.
    np.savez_compressed(output/(stem+'_pose.npz'), time_s=a[:, 0], actual_rotation=np.array(actual_rotations),
        nominal_rotation=np.array(nominal_rotations), residual_translation_m=np.array(pose_offsets_p),
        residual_rotation_vector_rad=np.array(pose_offsets_r))
    (output/(stem+'.xml')).write_text(xml)
    print(json.dumps(result), flush=True)
    return result, a


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=MODEL)
    parser.add_argument('--masses-kg', nargs='+', type=float, default=[.396])
    parser.add_argument('--payload-com-m', nargs=3, type=float, default=[-.0338, 0, -.08])
    parser.add_argument('--lift-mm', type=float, default=20)
    parser.add_argument('--ramp-s', type=float, default=3)
    parser.add_argument('--reference-trace', type=Path,
                        help='Replay a completed linear nominal 6D trajectory from a JSONL log; never reuses issued correction')
    parser.add_argument('--modes', nargs='+', choices=('none', 'continuous', 'gravity_ff', 'gravity_ff_continuous',
                                                     'pose_reference_ff', 'pose_reference_ff_z', 'pose_reference_ff_feedback'),
                        default=['none', 'continuous', 'gravity_ff'])
    parser.add_argument('--reference-delay-ms', type=float, default=0.)
    parser.add_argument('--reference-filter-ms', type=float, default=0.)
    parser.add_argument('--reference-preview-ms', type=float, default=0.,
                        help='Known-trajectory preview in simulation only; not a causal prediction for an unknown task')
    parser.add_argument('--feedback-delay-ms', type=float, default=50.)
    parser.add_argument('--feedback-jitter-ms', type=float, default=0.)
    parser.add_argument('--feedback-pairing', choices=('capture', 'host-midpoint'), default='capture')
    parser.add_argument('--joint-friction-nm', type=float, default=0.)
    parser.add_argument('--pose-ff-scale', type=float, default=1.)
    parser.add_argument('--pose-velocity-ff', action='store_true')
    parser.add_argument('--z-gain-per-s', type=float, default=.5)
    parser.add_argument('--z-rate-mm-s', type=float, default=1.)
    parser.add_argument('--z-max-mm', type=float, default=20.)
    parser.add_argument('--o6-urdf', type=Path)
    parser.add_argument('--hand-closure', type=float, default=0.)
    parser.add_argument('--o6-mount-pos-m', nargs=3, type=float, default=[-.0317, 0, -.0812])
    parser.add_argument('--o6-mount-rpy-deg', nargs=3, type=float, default=[180, 0, 0])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if (not all(np.isfinite(args.masses_kg)) or any(m < 0 or m > 2 for m in args.masses_kg)
            or not np.isfinite(args.payload_com_m).all() or np.linalg.norm(args.payload_com_m) > .3
            or not np.isfinite(args.lift_mm) or not 0 < args.lift_mm <= 50
            or not np.isfinite(args.ramp_s) or not 1 <= args.ramp_s <= 10):
        parser.error('Invalid synthetic experiment bounds')
    if not 0 < args.z_max_mm <= 50 or not 0 < args.z_rate_mm_s <= 10 or not 0 <= args.z_gain_per_s <= 3:
        parser.error('Invalid simulated controller bounds')
    settings = PhysicsSettings(reference_delay_s=args.reference_delay_ms/1000,
        reference_preview_s=args.reference_preview_ms/1000,
        reference_filter_tau_s=args.reference_filter_ms/1000, feedback_delay_s=args.feedback_delay_ms/1000,
        feedback_jitter_s=args.feedback_jitter_ms/1000, feedback_pairing=args.feedback_pairing,
        joint_friction_nm=args.joint_friction_nm,
        pose_ff_scale=args.pose_ff_scale,
        pose_velocity_ff=args.pose_velocity_ff,
        z=Config(gain_per_s=args.z_gain_per_s, max_rate_m_s=args.z_rate_mm_s/1000, max_offset_m=args.z_max_mm/1000))
    args.output.mkdir(parents=True, exist_ok=False)
    reference = TraceReference(args.reference_trace) if args.reference_trace is not None else None
    results, data = [], []
    for mass in args.masses_kg:
        payload = assembly(args.o6_urdf, mass, args.hand_closure, args.o6_mount_pos_m, args.o6_mount_rpy_deg) if args.o6_urdf else None
        for mode in args.modes:
            result, a = experiment(args.model, mass, args.payload_com_m, mode, args.lift_mm/1000, args.ramp_s, args.output, reference, settings, payload)
            results.append(result);data.append((result, a))
    report = dict(engine='MuJoCo '+mujoco.__version__, robot_connections=0,
        algorithm_config=asdict(settings.z), feedback_rate_hz=20, synthetic_feedback_delay_ms=args.feedback_delay_ms,
        physics_settings=asdict(settings),
        motor_target_rate_hz=300, physics_rate_hz=900,
        reference=reference.metadata if reference is not None else None,
        limitations=['Local DLS IK and PD approximation, NOT vendor ServoP firmware.',
                     'Base fixed, contacts disabled; no table, cable, contact or grasp validation.',
                     'Payload may use vendor inertias; mount/adapter distribution is still assumed, NOT identified O6 assembly COM.',
                     'Ideal model-gravity feedforward is an experimental control, NOT a deployed patch.'],
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                       [args.model, Path(__file__), ROOT/'tron2_z_compensation.py']}, results=results)
    (args.output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(args.masses_kg), 1, figsize=(10, 3.5*len(args.masses_kg)), squeeze=False)
    for i, mass in enumerate(args.masses_kg):
        ax = axes[i, 0]
        for result, a in data:
            if result['assumed_payload_kg'] == mass:
                ax.plot(a[:, 0], (a[:, 3]-a[:, 1])*1000, label=result['mode'])
        ax.axvline(30, color='k', ls='--', lw=.7)
        ax.set_title(f'Synthetic payload {mass:g} kg; trajectory starts at 30 s')
        ax.set_ylabel('Actual - nominal Z (mm)');ax.grid(alpha=.3);ax.legend()
    axes[-1, 0].set_xlabel('Simulation time (s)')
    fig.tight_layout();fig.savefig(args.output/'comparison.png', dpi=150);plt.close(fig)


if __name__ == '__main__':
    main()
