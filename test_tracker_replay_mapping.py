import copy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from tracker_replay_mapping import calibrated_relative_motion, load_calibration


def pose(rotvec=(0., 0., 0.), translation=(0., 0., 0.)):
    out = np.eye(4)
    out[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    out[:3, 3] = translation
    return out


def trajectory(poses):
    return SimpleNamespace(
        positions=np.asarray([p[:3, 3] for p in poses]),
        quaternions=Rotation.from_matrix(np.asarray([p[:3, :3] for p in poses])).as_quat()[:, [3, 0, 1, 2]],
    )


def calibration(world=None, mount=None, convention="raw-xyzw"):
    return {
        "T_BW": (np.eye(4) if world is None else world).tolist(),
        "T_ET": (np.eye(4) if mount is None else mount).tolist(),
        "position_unit": "m",
        "tracker_quaternion_convention": convention,
        "session": {"arm": "left", "robot_id": "test-robot"},
    }


def as_rotations(wxyz):
    return Rotation.from_quat(wxyz[:, [1, 2, 3, 0]]).as_matrix()


class CalibratedMotionTests(unittest.TestCase):
    def test_stationary_wrist_rotating_tracker_lever_arm(self):
        world = pose((.2, -.3, .5), (3., 5., -2.))
        mount = pose((-.1, .4, .2), (.13, -.04, .08))
        wrists = [pose((.2, .1, 0.), (.4, .2, -.3)),
                  pose((.5, -.3, .6), (.4, .2, -.3))]
        raw = trajectory([np.linalg.inv(world) @ w @ mount for w in wrists])
        dp, dq, _ = calibrated_relative_motion(raw, calibration(world, mount), 1., 1.,
                                               "wrist-relative", "wxyz")
        np.testing.assert_allclose(dp, 0., atol=1e-12)
        np.testing.assert_allclose(as_rotations(dq)[1], wrists[1][:3, :3] @ wrists[0][:3, :3].T,
                                   atol=1e-12)
        tracker_dp, _, _ = calibrated_relative_motion(raw, calibration(world, mount), 1., 1.,
                                                      "tracker-relative", "wxyz")
        self.assertGreater(np.linalg.norm(tracker_dp[1]), .02)

    def test_full_matrix_golden_with_position_and_rotation_scaling(self):
        world = pose((.3, -.6, .1), (10., -20., 3.))
        mount = pose((-.2, .1, .4), (.05, -.07, .12))
        samples = [pose((.2, .3, -.2), (1., 2., 3.)),
                   pose((.4, -.2, .6), (1.1, 2.3, 2.9)),
                   pose((-.1, .5, .4), (.9, 2.1, 3.2))]
        golden = [world @ q @ np.linalg.inv(mount) for q in samples]
        dp, dq, angles = calibrated_relative_motion(trajectory(samples), calibration(world, mount),
                                                    .4, .5, "wrist-relative", "wxyz")
        for i, target in enumerate(golden):
            expected = Rotation.from_matrix(target[:3, :3] @ golden[0][:3, :3].T).as_rotvec() * .5
            np.testing.assert_allclose(dp[i], (target[:3, 3] - golden[0][:3, 3]) * .4, atol=1e-12)
            np.testing.assert_allclose(as_rotations(dq)[i], Rotation.from_rotvec(expected).as_matrix(),
                                       atol=1e-12)
            self.assertAlmostEqual(angles[i], np.rad2deg(np.linalg.norm(expected)), places=10)

    def test_nontrivial_base_axes_and_world_origin_cancel(self):
        world = pose((0., 0., np.pi / 2), (1e3, -2e3, 3e3))
        samples = trajectory([pose(translation=(5., 7., 8.)), pose(translation=(6., 7., 8.))])
        dp, dq, angles = calibrated_relative_motion(samples, calibration(world), 2., 1.,
                                                    quaternion_format="wxyz")
        np.testing.assert_allclose(dp, [[0., 0., 0.], [0., 2., 0.]], atol=1e-12)
        np.testing.assert_allclose(dq, [[1., 0., 0., 0.]] * 2, atol=1e-12)
        np.testing.assert_allclose(angles, 0., atol=1e-12)

    def test_legacy_correction_and_standard_bypass_are_equivalent(self):
        raw = trajectory([pose((.1, .2, -.3)), pose((-.4, .6, .2), (.1, .2, .3))])
        legacy = copy.deepcopy(raw)
        legacy.quaternions = legacy.quaternions[:, [1, 0, 3, 2]]
        expected = calibrated_relative_motion(raw, calibration(), 1., 1., quaternion_format="wxyz")
        corrected = calibrated_relative_motion(legacy, calibration(), 1., 1.,
                                                quaternion_format="pyvut-legacy")
        default = calibrated_relative_motion(legacy, calibration(), 1., 1.)
        for a, b, c in zip(expected, corrected, default):
            np.testing.assert_allclose(a, b, atol=1e-12)
            np.testing.assert_allclose(a, c, atol=1e-12)

    def test_legacy_solver_convention_keeps_original_values(self):
        raw = trajectory([pose((.1, .2, -.3)), pose((-.4, .6, .2))])
        cal = calibration(convention="legacy-wzyx")
        a = calibrated_relative_motion(raw, cal, 1., 1.)
        b = calibrated_relative_motion(raw, cal, 1., 1., quaternion_format="pyvut-legacy")
        for x, y in zip(a, b):
            np.testing.assert_allclose(x, y, atol=1e-12)

    def test_sign_and_norm_do_not_change_rotation_and_first_delta_exact(self):
        raw = trajectory([pose((.1, .3, -.2)), pose((.2, -.1, .3)), pose((.3, .5, -.1))])
        negated = copy.deepcopy(raw)
        negated.quaternions *= np.array([2., -3., 4.])[:, None]
        a = calibrated_relative_motion(raw, calibration(), 1., 1., quaternion_format="wxyz")
        b = calibrated_relative_motion(negated, calibration(), 1., 1., quaternion_format="wxyz")
        for x, y in zip(a, b):
            np.testing.assert_allclose(x, y, atol=1e-12)
        np.testing.assert_array_equal(a[0][0], [0., 0., 0.])
        np.testing.assert_array_equal(a[1][0], [1., 0., 0., 0.])
        self.assertTrue(np.all(np.sum(a[1][1:] * a[1][:-1], axis=1) >= 0.))

    def test_zero_scales_hold_start(self):
        raw = trajectory([pose(), pose((.3, .5, .1), (.4, .2, .3))])
        dp, dq, angles = calibrated_relative_motion(raw, calibration(), 0., 0., quaternion_format="wxyz")
        np.testing.assert_array_equal(dp, np.zeros((2, 3)))
        np.testing.assert_array_equal(dq, [[1., 0., 0., 0.]] * 2)
        np.testing.assert_array_equal(angles, [0., 0.])

    def test_invalid_scales_modes_quaternions_and_shapes(self):
        raw = trajectory([pose(), pose()])
        for scale, orient in [(-1., 0.), (np.nan, 0.), (1., np.inf), (1., 1.1), (1., -.1)]:
            with self.subTest(scale=scale, orient=orient), self.assertRaises(ValueError):
                calibrated_relative_motion(raw, calibration(), scale, orient)
        for mode, fmt in [("bad", "wxyz"), ("tracker-relative", "xyzw")]:
            with self.subTest(mode=mode, fmt=fmt), self.assertRaises(ValueError):
                calibrated_relative_motion(raw, calibration(), 1., 1., mode, fmt)
        for positions, quats in [(np.empty((0, 3)), np.empty((0, 4))),
                                 (np.zeros((2, 3)), np.zeros((2, 4))),
                                 (np.zeros((2, 2)), raw.quaternions),
                                 (np.full((2, 3), np.nan), raw.quaternions)]:
            with self.subTest(shape=positions.shape), self.assertRaises(ValueError):
                calibrated_relative_motion(SimpleNamespace(positions=positions, quaternions=quats),
                                           calibration(), 1., 1.)


class CalibrationLoadingTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "calibration.json"

    def write(self, data):
        self.path.write_text(json.dumps(data))

    def test_valid_metadata_and_frozen_acceptance_preserved(self):
        self.write(calibration())
        frozen = self.path.read_bytes()
        sha = hashlib.sha256(frozen).hexdigest()
        sidecar = {"calibration_file": "calibration.json", "calibration_sha256": sha,
                   "status": "user_accepted_measured_accuracy"}
        (self.path.parent / "acceptance.json").write_text(json.dumps(sidecar))
        loaded = load_calibration(self.path)
        self.assertEqual(loaded["sha256"], sha)
        self.assertEqual(loaded["robot_id"], "test-robot")
        self.assertEqual(loaded["source"], str(self.path.resolve()))
        self.assertEqual(loaded["acceptance"]["status"], sidecar["status"])
        self.assertEqual(self.path.read_bytes(), frozen)

    def test_bad_or_missing_matching_sidecar_hash_rejected(self):
        self.write(calibration())
        for digest in ["wrong", None]:
            acceptance = {"calibration_file": "calibration.json", "calibration_sha256": digest}
            (self.path.parent / "acceptance.json").write_text(json.dumps(acceptance))
            with self.subTest(digest=digest), self.assertRaisesRegex(ValueError, "SHA256"):
                load_calibration(self.path)

    def test_unrelated_sidecar_not_attributed_to_candidate(self):
        self.write(calibration())
        (self.path.parent / "acceptance.json").write_text(json.dumps({
            "calibration_file": "different.json", "calibration_sha256": "unrelated"}))
        self.assertIsNone(load_calibration(self.path)["acceptance"])

    def test_wrong_arm_unit_convention_or_missing_metadata_rejected(self):
        variants = []
        for key, value in [("position_unit", "mm"), ("position_unit", None),
                           ("tracker_quaternion_convention", "unknown")]:
            data = calibration(); data[key] = value; variants.append(data)
        data = calibration(); data["session"]["arm"] = "right"; variants.append(data)
        data = calibration(); del data["session"]["arm"]; variants.append(data)
        data = calibration(); data["arm"] = "right"; variants.append(data)
        for index, data in enumerate(variants):
            self.write(data)
            with self.subTest(index=index), self.assertRaises(ValueError):
                load_calibration(self.path)

    def test_reflection_scaling_bad_homogeneous_row_and_nonfinite_rejected(self):
        bad = []
        for value in [-1., 2., float("nan")]:
            matrix = np.eye(4); matrix[0, 0] = value; bad.append(matrix.tolist())
        matrix = np.eye(4); matrix[3, 0] = 1.; bad.append(matrix.tolist())
        bad.append(np.eye(3).tolist())
        for name in ["T_BW", "T_ET"]:
            for index, matrix in enumerate(bad):
                data = calibration(); data[name] = matrix; self.write(data)
                with self.subTest(name=name, index=index), self.assertRaises(ValueError):
                    load_calibration(self.path)


if __name__ == "__main__":
    unittest.main()
