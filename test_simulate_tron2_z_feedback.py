"""Offline reference-log contracts for physics experiments; no robot runtime."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from collections import deque

import numpy as np
from scipy.spatial.transform import Rotation

from tools.simulate_tron2_z_feedback import TraceReference, PhysicsSettings, lagged_reference
from tools.tron2_payload_model import aggregate_hand, assembly


class ReferenceTraceTests(unittest.TestCase):
    def rows(self, mode='linear'):
        q = Rotation.from_euler('z', 30, degrees=True).as_quat()[[3, 0, 1, 2]].tolist()
        rows = [dict(event='fixed_start_plan', episode='example.hdf5', speed=.5,
                     scale=1, orientation_scale=1, tracker_interpolation=mode),
                dict(event='replay_start', start_monotonic_ns=1_000_000_000)]
        for t, phase, pose in [(0, 'trajectory', [.4, .2, -.3, 1, 0, 0, 0]),
                                (1, 'trajectory', [.5, .2, -.28]+q),
                                (2, 'final_hold', [.5, .2, -.28]+q)]:
            issued = pose.copy();issued[2] += .015
            rows += [dict(event='z_replay_command', phase=phase, nominal_left_pos=pose,
                          issued_left_pos=issued, monotonic_ns=1_000_000_000+int(t*1e9)),
                     dict(event='servop_command', left_pos=issued,
                          monotonic_ns=1_000_000_000+int(t*1e9))]
        rows.append(dict(event='z_replay_closed'))
        return rows

    def load(self, folder, rows):
        p = Path(folder)/'trace.jsonl'
        p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        return TraceReference(p)

    def test_uses_nominal_not_issued_and_excludes_post_trajectory_hold(self):
        with TemporaryDirectory() as folder:
            ref = self.load(folder, self.rows())
            self.assertEqual(ref.duration, 1.)
            self.assertEqual(ref.metadata['nominal_target_count'], 2)
            anchor = np.array([.45, .25, .9])
            initial = Rotation.from_euler('x', 20, degrees=True).as_matrix()
            p, r = ref.target(.5, anchor, initial)
            np.testing.assert_allclose(p, anchor+[.05, 0, .01])
            np.testing.assert_allclose(r, Rotation.from_euler('z', 15, degrees=True).as_matrix()@initial)
            np.testing.assert_allclose(ref.target(-30, anchor, initial)[0], anchor)

    def test_rejects_unfinished_failed_or_step_input_instead_of_faking_linear(self):
        with TemporaryDirectory() as folder:
            for rows in [self.rows()[:-1], self.rows()+[dict(event='control_abort')], self.rows('none')]:
                with self.subTest(rows=rows), self.assertRaises(ValueError):
                    self.load(folder, rows)


class PhysicsSettingsTests(unittest.TestCase):
    def test_delayed_reference_interpolates_without_future_reference_leak(self):
        history = deque([(0., np.zeros(2)), (1., np.ones(2)), (2., np.ones(2)*2)])
        np.testing.assert_allclose(lagged_reference(history, -.1), [0, 0])
        np.testing.assert_allclose(lagged_reference(history, .5), [.5, .5])
        np.testing.assert_allclose(lagged_reference(history, 1.5), [1.5, 1.5])
        self.assertEqual(history[0][0], 1.)
        np.testing.assert_allclose(lagged_reference(history, 3.), [2, 2])

    def test_invalid_delay_jitter_or_pairing_are_rejected(self):
        for values in [dict(reference_delay_s=-.1), dict(reference_delay_s=float('nan')),
                       dict(feedback_jitter_s=.03, feedback_delay_s=.01),
                       dict(reference_preview_s=.31), dict(feedback_pairing='unknown')]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                PhysicsSettings(**values)

    def test_vendor_mass_and_parallel_axis_inertia_are_preserved(self):
        xml = '''<robot name="toy"><link name="root"><inertial>
        <mass value="1"/><inertia ixx="0.1" iyy="0.1" izz="0.1" ixy="0" ixz="0" iyz="0"/>
        </inertial></link><link name="child"><inertial><mass value="1"/>
        <inertia ixx="0.1" iyy="0.1" izz="0.1" ixy="0" ixz="0" iyz="0"/></inertial></link>
        <joint name="fixed" type="fixed"><parent link="root"/><child link="child"/>
        <origin xyz="0.2 0 0"/></joint></robot>'''
        with TemporaryDirectory() as folder:
            p = Path(folder)/'hand.urdf';p.write_text(xml)
            mass, com, inertia = aggregate_hand(p)
            self.assertEqual(mass, 2)
            np.testing.assert_allclose(com, [.1, 0, 0])
            np.testing.assert_allclose(inertia, np.diag([.2, .22, .22]))
            model = assembly(p, 2, mount_pos=[0, 0, 0], mount_rpy_deg=[0, 0, 90])
            np.testing.assert_allclose(model['COM_wrist_m'], [0, .1, 0], atol=1e-15)
            with self.assertRaisesRegex(ValueError, 'below vendor'):
                assembly(p, 1.)


if __name__ == '__main__':
    unittest.main()
