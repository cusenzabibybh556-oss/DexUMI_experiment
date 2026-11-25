"""Acquisition contract tests; all fixtures are offline synthetic files."""
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from tools.validate_tron2_episode_batch import EpisodeReference, inspect_arrays
from tools.analyze_tron2_episode_batch import comparison


def arrays(t, positions=None):
    n = len(t)
    return {'timestamps/master_timestamp_ns': np.rint(np.asarray(t)*1e9).astype(np.int64),
        'vive_tracker/pos': np.zeros((n, 1, 3)) if positions is None else np.asarray(positions)[:, None, :],
        'vive_tracker/quat': np.tile([1., 0, 0, 0], (n, 1, 1)),
        'vive_tracker/valid': np.ones((n, 1), bool)}


ATTRS = dict(position_unit='meter', quaternion_order='[w, x, y, z]', hand_side='left')


class AcquisitionContractTests(unittest.TestCase):
    def test_identical_duplicate_is_diagnostic_only_and_first_frame_is_preserved(self):
        a = arrays([0, 0, .033, .066])
        report, indices = inspect_arrays(a, ATTRS)
        self.assertFalse(report['strict_replay_eligible'])
        self.assertTrue(report['diagnostic_view_available'])
        self.assertEqual(report['identical_duplicate_pairs'], 1)
        np.testing.assert_array_equal(indices, [0, 2, 3])

    def test_conflicting_duplicate_or_backward_time_is_not_silently_reordered(self):
        for a in [arrays([0, 0, .033], [[0, 0, 0], [.001, 0, 0], [.002, 0, 0]]),
                  arrays([0, .04, .03])]:
            report, indices = inspect_arrays(a, ATTRS)
            self.assertFalse(report['diagnostic_view_available'])
            self.assertIsNone(indices)

    def test_invalid_frames_are_excluded_but_resulting_gap_is_retained(self):
        a = arrays([0, .05, .1, .2])
        a['vive_tracker/valid'][1:3] = False
        report, indices = inspect_arrays(a, ATTRS)
        self.assertEqual(report['invalid_pose_frames'], 2)
        self.assertFalse(report['strict_replay_eligible'])
        np.testing.assert_array_equal(indices, [0, 3])
        self.assertEqual(report['source_gap_ms']['maximum'], 200)

    def test_zero_quaternion_cannot_become_valid_by_normalization(self):
        a = arrays([0, .03, .06]); a['vive_tracker/quat'][1] = 0
        report, indices = inspect_arrays(a, ATTRS)
        self.assertEqual(report['invalid_pose_frames'], 1)
        np.testing.assert_array_equal(indices, [0, 2])

    def test_diagnostic_reference_keeps_duration_scale_and_legacy_rotation_semantics(self):
        a = arrays([0, .1, .1, .4], [[0, 0, 0], [.01, 0, 0], [.01, 0, 0], [.04, 0, 0]])
        desired = Rotation.from_euler('z', [0, 10, 10, 40], degrees=True)
        corrected_wxyz = desired.as_quat()[:, [3, 0, 1, 2]]
        # Invert the production legacy component correction [1,0,3,2].
        a['vive_tracker/quat'][:, 0] = corrected_wxyz[:, [1, 0, 3, 2]]
        mapping = dict(T_BW=np.eye(4).tolist(), T_ET=np.eye(4).tolist(),
            tracker_quaternion_convention='raw-xyzw', sha256='fixture')
        with TemporaryDirectory() as folder:
            p = Path(folder)/'episode_000001.hdf5'
            with h5py.File(p, 'w') as f:
                for key, value in a.items(): f.create_dataset(key, data=value)
                f['vive_tracker'].attrs.update(position_unit='meter', quat_order='[w, x, y, z]')
                f.attrs['hand_side'] = 'left'
            original = p.read_bytes()
            ref = EpisodeReference(p, mapping, .5)
            self.assertEqual(ref.duration, .8)
            self.assertFalse(ref.audit['strict_replay_eligible'])
            pos, rot = ref.target(.4, np.array([1, 2, 3]), np.eye(3))
            np.testing.assert_allclose(pos, [1.02, 2, 3])
            np.testing.assert_allclose(rot, Rotation.from_euler('z', 20, degrees=True).as_matrix(), atol=1e-12)
            self.assertEqual(p.read_bytes(), original)


class PairedSummaryTests(unittest.TestCase):
    def completed(self, position, orientation=.5):
        return dict(status='completed', metrics=dict(position_norm_P95_mm=position,
            orientation_P95_deg=orientation, Z_trajectory_RMSE_mm=position/2))

    def test_rejected_cases_remain_in_requested_denominator_without_partial_metrics(self):
        records={(0,'base'):self.completed(10), (0,'candidate'):self.completed(2),
            (1,'base'):self.completed(10), (1,'candidate'):dict(status='rejected'),
            (2,'base'):dict(status='rejected'), (2,'candidate'):self.completed(2)}
        r=comparison(records,[0,1,2],'base','candidate')
        self.assertEqual(r['requested_episode_count'],3)
        self.assertEqual(r['paired_complete_count'],1)
        self.assertEqual(r['unpaired_episode_ids'],[1,2])

    def test_improvement_is_paired_per_episode_and_regressions_are_counted(self):
        records={(0,'base'):self.completed(10), (0,'candidate'):self.completed(5),
            (1,'base'):self.completed(100), (1,'candidate'):self.completed(90)}
        r=comparison(records,[0,1],'base','candidate')
        self.assertAlmostEqual(r['paired_position_reduction_percent']['median'],30)
        records[1,'candidate']=self.completed(110)
        r=comparison(records,[0,1],'base','candidate')
        self.assertEqual(r['position_improved'],1)
        self.assertEqual(r['position_worsened'],1)
        json.dumps(r)  # Summaries must not leak NumPy scalar counters.


if __name__ == '__main__':
    unittest.main()
