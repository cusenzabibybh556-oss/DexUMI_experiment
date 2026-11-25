"""离线验证 Tracker 报文与 DexUMI 双坐标变换，不访问设备。"""
import struct
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from dexumi_wrist_solver import fit_dexumi, tracker_pose_for_calibration, wrist_from_tracker
from solve_tron2_tracker_calibration import PoseGroup, transform
from tracker_passive_reader import decode_report


class CalibrationTests(unittest.TestCase):
    def test_explicit_raw_xyzw_conversion_and_wrist_mapping(self):
        correct = np.array([.2,.3,-.4,.7])  # 原始 XYZW
        correct /= np.linalg.norm(correct)
        legacy_wxyz = correct[[0,3,2,1]]
        stored = transform(Rotation.from_quat(legacy_wxyz[[1,2,3,0]]).as_matrix(), [.1,.2,.3])
        expected = transform(Rotation.from_quat(correct).as_matrix(), [.1,.2,.3])
        np.testing.assert_allclose(tracker_pose_for_calibration(stored,'raw-xyzw'),expected,atol=1e-12)
        X=transform(Rotation.from_rotvec([.1,.2,.3]).as_matrix(),[.01,.02,.03])
        Y=transform(Rotation.from_rotvec([-.3,.2,.1]).as_matrix(),[1.,2.,3.])
        from solve_tron2_tracker_calibration import inverse
        profile={'T_ET':X,'T_BW':inverse(Y),'tracker_quaternion_convention':'raw-xyzw'}
        np.testing.assert_allclose(wrist_from_tracker(stored,profile),inverse(Y)@expected@inverse(X),atol=1e-12)

    def test_passive_report_matches_position_and_wxyz_order(self):
        data = bytearray(39)
        struct.pack_into('<3f', data, 2, .1, -.2, .3)
        struct.pack_into('<4e', data, 14, .5, .25, -.5, np.sqrt(.4375))
        data[36] = 2
        mac = '23:32:ae:d4:f6:a1'
        header = struct.pack('<BH6sHB', 0x28, 42, bytes.fromhex(mac.replace(':', '')), 0x110, 39)
        report = header + data
        frame = decode_report(report, mac)
        np.testing.assert_allclose(frame['position_m'], [.1, -.2, .3], atol=1e-7)
        np.testing.assert_allclose(frame['quaternion_wxyz'], [.5, np.sqrt(.4375), -.5, .25], atol=5e-4)
        self.assertEqual(frame['tracking_status'], 2)
        self.assertIsNone(decode_report(report, '00:00:00:00:00:00'))
        self.assertIsNone(decode_report(report[:20], mac))
        data[36] = 18
        self.assertEqual(decode_report(header+data, mac)['tracking_status'], 18)

    def test_recovers_both_transforms_with_tcp_offset_and_held_out_poses(self):
        rng = np.random.default_rng(12)
        offset = np.array([-.0338, 0, -.2125])
        C = transform(np.eye(3), -offset)
        X = C @ transform(Rotation.from_rotvec([.4, -.3, .7]).as_matrix(), [.04, -.02, .07])
        Y = transform(Rotation.from_rotvec([.2, .5, -.8]).as_matrix(), [1., -.4, .3])
        groups = []
        for i in range(16):
            P = transform(Rotation.from_rotvec(rng.normal(0, .35, 3)).as_matrix(), rng.normal(0, .15, 3))
            groups.append(PoseGroup(i+1, str(i), P, Y@P@X, [2], 20))
        result = fit_dexumi(groups, offset, {4, 8, 12, 16})
        self.assertEqual(result['status'], 'passed')
        np.testing.assert_allclose(result['T_ET'], X, atol=1e-5)
        np.testing.assert_allclose(result['T_WB'], Y, atol=1e-5)
        self.assertEqual(result['metrics']['validation']['count'], 4)
        groups = [PoseGroup(g.id,g.label,g.robot,g.tracker,[18],g.pair_count) for g in groups]
        diagnostic = fit_dexumi(groups, offset, {4,8,12,16})
        self.assertEqual(diagnostic['status'], 'review_required')
        self.assertTrue(diagnostic['geometric_validation_passed'])
        self.assertEqual(diagnostic['unconfirmed_tracking_statuses'], [18])


if __name__ == '__main__':
    unittest.main()
