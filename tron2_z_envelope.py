"""Offline sampled IK/collision envelope for bounded Z position correction.

No hardware clients are created. Returned joint paths are model estimates, never
robot commands. Five layers and sampled joint-space chords are not a proof of
the entire Cartesian envelope or of the controller's redundant IK branch.
"""

from dataclasses import asdict
from itertools import product
import json
import math
from numbers import Real
from pathlib import Path

import numpy as np

from tools.diagnose_tron2_servop import strict_model
from tron2_large_calibration_plan import _check_path, _disable_removed_graspers
from tron2_ik_planner import MarginPlanOptions, plan_with_joint_margin
from replay_tron2_tracker import quaternion_distance_deg


class EnvelopeValidationError(RuntimeError):
    """Rejected envelope; ``report`` contains the offline failure evidence."""

    def __init__(self, message, report):
        super().__init__(message)
        self.report = report


def _sample_indices(count, stride):
    indices = list(range(0, count, stride))
    if indices[-1] != count-1:
        indices.append(count-1)
    return indices


def _check_sampled_path(model, plan, stride=10):
    indices = _sample_indices(len(plan), stride)
    result = {'method': 'sampled joint-space chords via _check_path',
              'frame_stride': stride, 'last_frame_included': True,
              'segments_checked': 0, 'passed': True}
    for left, right in zip(indices[:-1], indices[1:]):
        error = _check_path(model, plan[left], plan[right], model.lower, model.upper)
        result['segments_checked'] += 1
        if error:
            result.update(passed=False, failure_from_frame=left,
                          failure_to_frame=right, error=str(error))
            break
    return result


def _reserve(report_path):
    if report_path is None:
        return None, None, None
    path = Path(report_path).expanduser().resolve()
    if path.suffix.lower() != '.json':
        raise ValueError('envelope report_path must end in .json')
    artifact = path.with_suffix('.npz')
    if path.exists() or artifact.exists():
        raise FileExistsError(f'envelope evidence already exists: {path} / {artifact}')
    path.parent.mkdir(parents=True, exist_ok=True)
    report_file = path.open('x', encoding='utf-8')
    try:
        artifact_file = artifact.open('xb')
    except Exception:
        report_file.close()
        path.unlink()  # only the empty reservation just created by this call
        raise
    return report_file, artifact_file, artifact


def validate_replay_envelope(q_start, anchor, times, positions, quaternions,
                             max_offset_m, report_path=None):
    """Return ``(strict_model, report)`` or raise EnvelopeValidationError.

    Inputs: actual 16-axis start q, DualArmPose-like anchor, absolute nominal
    left XYZ/WXYZ and strictly increasing seconds. Timing/scale/rotation are
    never changed. The start must match both anchor and nominal first pose.
    Optional exclusive JSON/NPZ evidence is reserved before model construction;
    failed reviews also leave valid reports and any available offline paths.
    """
    report_file, artifact_file, artifact = _reserve(report_path)
    report = {'status': 'running', 'hardware_connected': False,
              'hardware_commands_sent': 0, 'joint_paths_are_commands': False,
              'layers': [], 'errors': [], 'collision_method': 'sampled',
              'max_joint_speed_rad_s': .5, 'minimum_margin_required_deg': 1.,
              'correction_rate_bound_m_s': .001,
              'trajectory_collision_frame_stride': 10,
              'cross_layer_collision_frame_stride': 30,
              'limitations': [
                  'Five sampled offsets do not prove every intermediate offset.',
                  'Joint-space chords do not certify all Cartesian interpolation paths.',
                  'Layer IK branches are not enforced by ServoP firmware.',
                  'Finite-difference offset sensitivity is an estimate, not an analytic bound.',
                  'Actual O6, mount, table, wires and dynamics are not fully modeled.',
              ]}
    arrays = {}
    try:
        q = np.asarray(q_start, dtype=float).copy()
        ts = np.asarray(times, dtype=float).copy()
        ps = np.asarray(positions, dtype=float).copy()
        us = np.asarray(quaternions, dtype=float).copy()
        if (q.shape != (16,) or ts.ndim != 1 or len(ts) < 2
                or ps.shape != (len(ts), 3) or us.shape != (len(ts), 4)
                or not all(np.isfinite(x).all() for x in (q, ts, ps, us))
                or np.any(np.diff(ts) <= 0) or np.any(np.linalg.norm(us, axis=1) < 1e-12)):
            raise ValueError('invalid finite q/pose arrays or strictly increasing times')
        if (isinstance(max_offset_m, bool) or not isinstance(max_offset_m, Real)
                or not math.isfinite(max_offset_m) or max_offset_m < 0):
            raise ValueError('max_offset_m must be finite and nonnegative')
        ap = np.asarray(anchor.left_position, dtype=float)
        au = np.asarray(anchor.left_quaternion, dtype=float)
        if (ap.shape != (3,) or au.shape != (4,) or not np.isfinite(ap).all()
                or not np.isfinite(au).all() or np.linalg.norm(au) < 1e-12):
            raise ValueError('invalid left anchor pose')
        model = strict_model()
        removed = _disable_removed_graspers(model)
        model.check_joints(q)
        model.check_anchor(q, ap, au)
        model.check_anchor(q, ps[0], us[0])
        if np.linalg.norm(ps[0]-ap) > 1e-8 or quaternion_distance_deg(us[0], au) > 1e-5:
            raise ValueError('nominal first pose must equal the supplied anchor')
        offsets = np.unique(np.linspace(0., float(max_offset_m), 5))
        high = ps.max(0).copy()
        high[2] += max_offset_m
        options = MarginPlanOptions(max_joint_speed=.5, min_margin_deg=1.)
        report.update(frames=len(ts), duration_s=float(ts[-1]-ts[0]),
                      offsets_m=offsets.tolist(), removed_model_geometry=removed,
                      effective_lower_rad=model.lower.tolist(), effective_upper_rad=model.upper.tolist(),
                      absolute_position_min_m=ps.min(0).tolist(),
                      absolute_position_max_m=high.tolist(), planner_options=asdict(options))
        arrays.update(times_s=ts, nominal_positions_m=ps, nominal_quaternions_wxyz=us,
                      start_q_rad=q, offsets_m=offsets)
        paths = {}
        for index, offset in enumerate(offsets):
            layer = {'index': index, 'offset_m': float(offset), 'status': 'running'}
            report['layers'].append(layer)
            layer_ps = ps.copy()
            layer_ps[:, 2] += offset
            try:
                seed = q.copy()
                if offset > 0:
                    duration = float(offset/.001)
                    lift_t = np.linspace(0., duration, max(2, int(math.ceil(duration*300))+1))
                    lift_ps = np.tile(ps[0], (len(lift_t), 1))
                    lift_ps[:, 2] += np.linspace(0., offset, len(lift_t))
                    lift_us = np.tile(us[0], (len(lift_t), 1))
                    lift_q, lift_metrics = model.plan_tracking(q, lift_ps, lift_us, lift_t)
                    layer['seed_lift_metrics'] = lift_metrics
                    layer['seed_lift_collision'] = _check_sampled_path(model, lift_q)
                    if not layer['seed_lift_collision']['passed']:
                        raise RuntimeError('seed lift collision: '+layer['seed_lift_collision']['error'])
                    seed = lift_q[-1].copy()
                    arrays[f'offline_layer_{index}_seed_lift_q_rad'] = lift_q
                    arrays[f'offline_layer_{index}_seed_lift_times_s'] = lift_t
                paths[index] = {}
                layer['candidates'] = []
                for candidate_id, weight in enumerate(options.motion_weights):
                    candidate = {'id': candidate_id, 'motion_weight': weight,
                                 'valid': False, 'progress': []}
                    layer['candidates'].append(candidate)
                    single_options = MarginPlanOptions(max_joint_speed=.5, min_margin_deg=1.,
                                                       motion_weights=(weight,))
                    try:
                        plan, stats = plan_with_joint_margin(model, seed, layer_ps, us, ts,
                            single_options, progress=candidate['progress'].append)
                        # Single-weight planning returns local ID 0; preserve
                        # the original four-weight ID for family selection.
                        stats = {**stats, 'id': candidate_id, 'motion_weight': weight,
                                 'selected_candidate': candidate_id}
                        candidate['planner'] = stats
                        if not np.array_equal(plan[:, 7:], np.tile(q[7:], (len(ts), 1))):
                            raise RuntimeError('offline plan changed the held arm or head')
                        candidate['trajectory_collision'] = _check_sampled_path(model, plan)
                        if not candidate['trajectory_collision']['passed']:
                            raise RuntimeError('trajectory collision: '+candidate['trajectory_collision']['error'])
                        candidate['valid'] = True
                        paths[index][candidate_id] = plan
                        arrays[f'offline_layer_{index}_candidate_{candidate_id}_left_q_rad'] = plan[:, :7]
                    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                        candidate['error'] = str(exc)
                if not paths[index]:
                    raise RuntimeError('no valid candidate: '+'; '.join(c.get('error','') for c in layer['candidates']))
                layer['status'] = 'passed'
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                layer.update(status='failed', error=str(exc))
                report['errors'].append(f'layer {index} Z={offset*1000:g}mm: {exc}')
        if report['errors']:
            raise RuntimeError('; '.join(report['errors']))
        # Build velocity-feasible edges before doing the more expensive sampled
        # cross-layer collision queries. Keep every complete valid candidate.
        edges = {}
        report['candidate_pair_edges'] = []
        for low in range(len(offsets)-1):
            dz = float(offsets[low+1]-offsets[low])
            for left_id, a in paths[low].items():
                for right_id, b in paths[low+1].items():
                    extra = np.abs(b[:, :7]-a[:, :7])/dz*.001
                    nominal = np.maximum(np.abs(np.diff(a[:, :7], axis=0)/np.diff(ts)[:, None]),
                                         np.abs(np.diff(b[:, :7], axis=0)/np.diff(ts)[:, None]))
                    combined = nominal+np.maximum(extra[:-1], extra[1:])
                    frame, joint = np.unravel_index(np.argmax(combined), combined.shape)
                    peak = float(combined[frame, joint])
                    item = {'lower_layer': low, 'upper_layer': low+1,
                            'lower_candidate': left_id, 'upper_candidate': right_id,
                            'sampled': True, 'last_frame_included': True, 'samples_checked': 0,
                            'estimated_added_peak_joint_speed_rad_s': float(extra.max()),
                            'estimated_combined_peak_joint_speed_rad_s': peak,
                            'estimated_speed_critical_frame': int(frame+1),
                            'estimated_speed_critical_joint_index': int(joint),
                            'estimated_speed_passed': peak <= .5+1e-6}
                    edges[low, left_id, right_id] = item
                    report['candidate_pair_edges'].append(item)
        def candidate_stats(layer_index, candidate_id):
            return report['layers'][layer_index]['candidates'][candidate_id]['planner']
        families = []
        for ids in product(*(tuple(paths[i]) for i in range(len(offsets)))):
            selected_edges = [edges[i, ids[i], ids[i+1]] for i in range(len(ids)-1)]
            if not all(e['estimated_speed_passed'] for e in selected_edges):
                continue
            stats = [candidate_stats(i, cid) for i, cid in enumerate(ids)]
            margin = min(s['minimum_margin_deg'] for s in stats)
            speed = max([s['max_joint_speed_rad_s'] for s in stats]+
                        [e['estimated_combined_peak_joint_speed_rad_s'] for e in selected_edges])
            acceleration = sum(s.get('acceleration_energy', 0.) for s in stats)
            velocity = sum(s.get('velocity_energy', 0.) for s in stats)
            families.append(((-margin, speed, acceleration, velocity, ids), ids))
        families.sort()
        selection = {'method': 'whole-family max-min margin, then combined peak speed, acceleration, velocity, IDs',
                     'generated_family_limit': 4**len(offsets),
                     'speed_feasible_families': len(families), 'collision_tested_families': 0,
                     'scope': 'sampled model existence; not a firmware branch guarantee'}
        report['family_selection'] = selection
        report['cross_layer_checks'] = []
        selected = None
        for score, ids in families:
            selection['collision_tested_families'] += 1
            feasible = True
            for low in range(len(ids)-1):
                item = edges[low, ids[low], ids[low+1]]
                if 'collision_passed' not in item:
                    a, b = paths[low][ids[low]], paths[low+1][ids[low+1]]
                    for frame in _sample_indices(len(ts), 30):
                        error = _check_path(model, a[frame], b[frame], model.lower, model.upper)
                        item['samples_checked'] += 1
                        if error:
                            item.update(collision_passed=False, failure_frame=frame, error=str(error))
                            break
                    else:
                        item['collision_passed'] = True
                if not item['collision_passed']:
                    feasible = False
                    break
            if feasible:
                selected = ids
                selection.update(selected_candidate_ids=list(ids), minimum_margin_deg=-score[0],
                                 estimated_combined_peak_joint_speed_rad_s=score[1])
                report['cross_layer_checks'] = [edges[i, ids[i], ids[i+1]] for i in range(len(ids)-1)]
                break
        if selected is None:
            report['cross_layer_checks'] = [e for e in edges.values() if 'collision_passed' in e]
            message = ('no candidate family satisfies estimated joint speed <= .5rad/s'
                       if not families else 'no candidate family satisfies cross layer sampled collision constraints')
            report['errors'].append(message)
            raise RuntimeError(message)
        for index, cid in enumerate(selected):
            layer = report['layers'][index]
            candidate = layer['candidates'][cid]
            layer.update(selected_candidate=cid, planner=candidate['planner'],
                         trajectory_collision=candidate['trajectory_collision'])
            arrays[f'offline_layer_{index}_joint_q_rad'] = paths[index][cid]
        report['minimum_margin_deg'] = selection['minimum_margin_deg']
        report['max_estimated_combined_joint_speed_rad_s'] = selection['estimated_combined_peak_joint_speed_rad_s']
        report['status'] = 'passed'
        return model, report
    except Exception as exc:
        report['status'] = 'failed'
        report['error'] = str(exc)
        if isinstance(exc, (FileExistsError, KeyboardInterrupt, SystemExit)):
            raise
        raise EnvelopeValidationError(str(exc), report) from exc
    finally:
        if report_file is not None:
            report['trajectory_artifact'] = str(artifact)
            try:
                json.dump(report, report_file, ensure_ascii=False, indent=2, allow_nan=False)
                report_file.write('\n')
                np.savez_compressed(artifact_file, **arrays)
            finally:
                report_file.close()
                artifact_file.close()
