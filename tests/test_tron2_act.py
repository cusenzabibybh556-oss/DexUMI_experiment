"""Meaningful offline checks: spatial semantics, time warping, masks and split isolation."""
import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
from scipy.spatial.transform import Rotation
import torch

from tron2_act.config import Config
from tron2_act.data import ACTDataset, Episode, prepare, source_stamp
from tron2_act.foreground import validate_foreground
from tron2_act.geometry import decode_quaternions, hold_commands, interpolate, matrix_to_rotation6d, rotation6d_to_matrix
from tron2_act.model import ACT, loss


def fixture(path, episode_id=0):
    n = 35
    seconds = np.arange(n) / 30
    timestamps = 1_000_000_000 + np.rint(seconds * 1e9).astype(np.int64)
    p = np.column_stack([.1*seconds + episode_id*.01, seconds*0, seconds*0])
    r = Rotation.from_rotvec(np.column_stack([seconds*0, seconds*0, .2*seconds]))
    with h5py.File(path, "w") as f:
        f.attrs.update(hand_type="o6", hand_side="left", task_name="fixture")
        f["timestamps/master_timestamp_ns"] = timestamps
        f["vive_tracker/pos"] = p[:, None]
        f["vive_tracker/quat"] = r.as_quat()[:, [3, 0, 1, 2]][:, None]
        f["vive_tracker/valid"] = np.ones((n, 1), bool)
        f["vive_tracker/unified_timestamp_ns"] = timestamps[:, None]
        f["vive_tracker"].attrs.update(position_unit="meter", quat_order="[w, x, y, z]")
        f["dexterous_hand/joint_positions"] = np.tile([200, 110, 100, 120, 130, 140], (n, 1)) + seconds[:, None]
        f["dexterous_hand/state_valid"] = np.ones(n, bool)
        f["dexterous_hand/unified_timestamp_ns"] = timestamps
        f["action/command"] = np.tile([210, 80, 160, 170, 180, 190], (n, 1)) + np.arange(n)[:, None]
        f["action/valid"] = np.ones(n, bool)
        f["action/command_timestamp_ns"] = timestamps - 1_000_000
        f["camera/d435i/unified_timestamp_ns"] = timestamps
        f["camera/d435i/frame_number"] = np.arange(n)
        f["camera/d435i/valid"] = np.ones(n, bool)
        # Tiny raw camera image is enough for model IO tests; no synthetic real-data claims.
        f["observations/images/d435i_rgb"] = np.full((n, 32, 32, 3), 80+episode_id, np.uint8)


def calibration(path):
    r = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    t = np.eye(4);t[:3, :3] = r
    path.write_text(json.dumps({"T_BW": t.tolist(), "T_ET": np.eye(4).tolist(),
                               "position_unit": "m", "arm": "left", "tracker_quaternion_convention": "raw-xyzw"}))
    return {"T_BW": t.tolist()}


def foreground_fixture(source_path, root):
    path = root / source_path.stem / "foreground.hdf5"
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(source_path, "r") as source, h5py.File(path, "w") as out:
        rgb = source["observations/images/d435i_rgb"][:]
        rgb[:, :8] = 0
        out["foreground/rgb"] = rgb
        out.attrs.update(contract="sam3-top-causal-tracking-fusion-v2", causal=True, review_status="accepted", source_stamp=json.dumps(source_stamp(source_path)))
        out["source/frame_index"] = np.arange(len(rgb))
        for key in ["frame_number", "unified_timestamp_ns"]:
            out[f"source/{key}"] = source[f"camera/d435i/{key}"][:]
        out["source/camera_valid"] = source["camera/d435i/valid"][:]
    return path


def duplicate_fixture_row(path, row=10):
    with h5py.File(path, "a") as f:
        names = []
        f.visititems(lambda name, obj: names.append(name) if isinstance(obj, h5py.Dataset) else None)
        for name in names:
            values = f[name][:]
            attributes = dict(f[name].attrs)
            del f[name]
            f[name] = np.concatenate([values[:row + 1], values[row:row + 1], values[row + 1:]], axis=0)
            f[name].attrs.update(attributes)


class DuplicateTimestampTests(unittest.TestCase):
    def test_identical_duplicates_preserve_labels_and_original_camera_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "episode_000000.hdf5"; fixture(path)
            mapping = calibration(root / "cal.json")
            reference = Episode(path, Config(quaternion_format="wxyz"), mapping)
            duplicate_fixture_row(path)
            before = source_stamp(path)
            with self.assertRaisesRegex(ValueError, "strictly increasing"):
                Episode(path, Config(quaternion_format="wxyz"), mapping)
            foreground_fixture(path, root / "foreground")
            config = Config(quaternion_format="wxyz", foreground_root=str(root / "foreground"), deduplicate_identical_timestamps=True)
            episode = Episode(path, config, mapping)
            self.assertEqual(episode.timestamp_audit["duplicate_master_rows"], [11])
            self.assertEqual(episode.timestamp_audit["tracker_samples_removed"], 1)
            self.assertEqual(episode.timestamp_audit["o6_samples_removed"], 1)
            self.assertNotIn(11, episode.rows)
            self.assertEqual(len(reference.rows), len(episode.rows))
            np.testing.assert_array_equal(reference.pose_t, episode.pose_t)
            np.testing.assert_array_equal(reference.hand_t, episode.hand_t)
            for raw_row in reference.rows:
                mapped_row = raw_row + int(raw_row > 10)
                expected, actual = reference.sample(raw_row), episode.sample(mapped_row)
                if expected is None:
                    self.assertIsNone(actual)
                else:
                    for a, b in zip(expected, actual):
                        np.testing.assert_array_equal(a, b)
            self.assertEqual(source_stamp(path), before)

    def test_conflicting_duplicate_command_or_image_is_rejected(self):
        for key in ["action/command", "observations/images/d435i_rgb", "vive_tracker/pos"]:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary); path = root / "episode_000000.hdf5"; fixture(path)
                mapping = calibration(root / "cal.json")
                duplicate_fixture_row(path)
                with h5py.File(path, "a") as f:
                    values = f[key][11]; values.flat[0] += 1; f[key][11] = values
                with self.assertRaisesRegex(ValueError, "Conflicting duplicate master"):
                    Episode(path, Config(quaternion_format="wxyz", deduplicate_identical_timestamps=True), mapping)

    def test_time_reversal_is_never_sorted_or_repaired(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); path = root / "episode_000000.hdf5"; fixture(path)
            mapping = calibration(root / "cal.json")
            with h5py.File(path, "a") as f:
                f["timestamps/master_timestamp_ns"][11] = f["timestamps/master_timestamp_ns"][10] - 1
            with self.assertRaisesRegex(ValueError, "strictly increasing"):
                Episode(path, Config(quaternion_format="wxyz", deduplicate_identical_timestamps=True), mapping)

    def test_conflicting_tracker_time_is_not_silently_deduplicated(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); path = root / "episode_000000.hdf5"; fixture(path)
            mapping = calibration(root / "cal.json")
            with h5py.File(path, "a") as f:
                f["vive_tracker/unified_timestamp_ns"][11] = f["vive_tracker/unified_timestamp_ns"][10]
            with self.assertRaisesRegex(ValueError, "conflicting states"):
                Episode(path, Config(quaternion_format="wxyz", deduplicate_identical_timestamps=True), mapping)


class ForegroundTests(unittest.TestCase):
    def test_foreground_pixel_input_keeps_actual_state_and_command_targets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cal = root / "cal.json"; mapping = calibration(cal)
            for i in range(3):
                path = root / f"episode_{i:06d}.hdf5"
                fixture(path, i)
                foreground_fixture(path, root / "foreground")
            config = Config(data_root=str(root), foreground_root=str(root / "foreground"), calibration=str(cal), quaternion_format="wxyz", image_size=[32, 32])
            prepare(config, root / "prepared")
            dataset = ACTDataset(root / "prepared", "train", normalized=False)
            sample = dataset[0]
            path, row = dataset.items[0]
            expected = Episode(path, Config(quaternion_format="wxyz"), mapping).sample(row)
            np.testing.assert_array_equal(sample["state"], expected[0])
            np.testing.assert_array_equal(sample["action"], expected[1])
            self.assertEqual(float(sample["image"][:, 0, :].sum()), 0)
            self.assertGreater(float(sample["image"][:, 16, :].sum()), 0)
            cache = dataset.foreground_files[path]
            with h5py.File(cache, "a") as f:
                f["foreground/rgb"][row, 16, 16] = 0
            with self.assertRaisesRegex(ValueError, "Foreground cache changed"):
                ACTDataset(root / "prepared", "train")

    def test_misaligned_or_pending_foreground_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "episode_000000.hdf5"; fixture(source)
            cache = foreground_fixture(source, root / "cache")
            with h5py.File(cache, "a") as f:
                f["source/unified_timestamp_ns"][5] += 1
            with h5py.File(source, "r") as f:
                with self.assertRaisesRegex(ValueError, "timestamp"):
                    validate_foreground(source, f, root / "cache", source_stamp(source))
            foreground_fixture(source, root / "cache")
            with h5py.File(cache, "a") as f:
                f.attrs["review_status"] = "pending"
            with h5py.File(source, "r") as f:
                with self.assertRaisesRegex(ValueError, "review"):
                    validate_foreground(source, f, root / "cache", source_stamp(source))


class GeometryTests(unittest.TestCase):
    def test_commands_are_causal_and_not_linearly_interpolated(self):
        values, valid = hold_commands(np.array([0., .1]), np.array([[10]*6, [90]*6]),
                                     np.array([-.01, .05, .1, .2, .21]), .2)
        np.testing.assert_array_equal(valid, [False, True, True, True, False])
        np.testing.assert_array_equal(values[1], [10]*6)
        np.testing.assert_array_equal(values[2], [90]*6)
        _, valid = hold_commands(np.array([0., .1]), np.array([[10]*6, [90]*6]), np.array([.2]), .2, .05)
        self.assertFalse(valid[0])

    def test_qpos_feedback_and_action_commands_are_separate(self):
        with tempfile.TemporaryDirectory() as root:
            p = Path(root)/"episode_0.hdf5";fixture(p)
            mapping = calibration(Path(root)/"cal.json")
            ep = Episode(p, Config(quaternion_format="wxyz", image_size=[32, 32]), mapping)
            state, action, pad = ep.sample(15)
            # The camera observes t=0.5s. Input uses true feedback, output uses sent glove target.
            np.testing.assert_allclose(state[-6:], np.array([200, 110, 100, 120, 130, 140]) + .5)
            np.testing.assert_allclose(action[0, -6:], np.array([210, 80, 160, 170, 180, 190]) + 15)
            self.assertFalse(np.allclose(action[0, -6:], state[-6:]))

    def test_future_feedback_is_not_required_as_action_label(self):
        with tempfile.TemporaryDirectory() as root:
            p = Path(root)/"episode_0.hdf5";fixture(p)
            mapping = calibration(Path(root)/"cal.json")
            with h5py.File(p, "a") as f:
                f["dexterous_hand/state_valid"][16:] = False
            ep = Episode(p, Config(quaternion_format="wxyz", image_size=[32, 32]), mapping)
            sample = ep.sample(15)
            self.assertIsNotNone(sample)
            self.assertGreater((~sample[2]).sum(), 10)

    def test_missing_commands_never_fall_back_to_feedback(self):
        with tempfile.TemporaryDirectory() as root:
            p = Path(root)/"episode_0.hdf5";fixture(p)
            mapping = calibration(Path(root)/"cal.json")
            with h5py.File(p, "a") as f:
                del f["action/command"]
            with self.assertRaises(KeyError):
                Episode(p, Config(quaternion_format="wxyz", image_size=[32, 32]), mapping)

    def test_legacy_quaternion_decoder_is_explicit(self):
        r = Rotation.from_euler("xyz", [32, -20, 41], degrees=True)
        correct = r.as_quat()[[3, 0, 1, 2]]
        legacy = correct[[1, 0, 3, 2]]
        np.testing.assert_allclose(decode_quaternions(legacy[None], "pyvut-legacy")[0], r.as_matrix())
        self.assertGreater(np.linalg.norm(decode_quaternions(legacy[None], "wxyz")[0]-r.as_matrix()), .1)

    def test_rotation_columns_roundtrip(self):
        r = Rotation.from_euler("xyz", [[10, 20, 30], [-60, 5, 170]], degrees=True).as_matrix()
        np.testing.assert_allclose(rotation6d_to_matrix(matrix_to_rotation6d(r)), r, atol=1e-12)
        with self.assertRaises(ValueError):
            rotation6d_to_matrix(np.zeros(6))

    def test_gap_is_invalid_but_exact_endpoint_is_valid(self):
        _, good = interpolate(np.array([0., 1., 2.]), np.zeros((3, 3)), np.array([0., .5, 1., 2.1]), .1)
        np.testing.assert_array_equal(good, [True, False, True, False])

    def test_retiming_spatial_history_and_future(self):
        with tempfile.TemporaryDirectory() as root:
            p = Path(root)/"episode_0.hdf5";fixture(p)
            mapping = calibration(Path(root)/"cal.json")
            c = Config(quaternion_format="wxyz", time_scale=.4, image_size=[32, 32])
            ep = Episode(p, c, mapping)
            state, action, pad = ep.sample(15)
            # source motion is +X; R_BW turns it into +base-Y; not wrist-local X.
            np.testing.assert_allclose(action[0, :3], [0, .002, 0], atol=1e-7)
            np.testing.assert_allclose(state[:3], [0, -.004, 0], atol=1e-7)
            np.testing.assert_allclose(rotation6d_to_matrix(action[0, 3:9]),
                                       Rotation.from_rotvec([0, 0, .004]).as_matrix(), atol=1e-7)
            self.assertEqual(state.shape, (24,))
            self.assertTrue(pad.any())
            self.assertTrue(np.all(np.diff(pad.astype(int)) >= 0))

    def test_missing_history_and_gap_cut_future(self):
        with tempfile.TemporaryDirectory() as root:
            p = Path(root)/"episode_0.hdf5";fixture(p)
            mapping = calibration(Path(root)/"cal.json")
            with h5py.File(p, "a") as f:
                f["vive_tracker/valid"][10:22] = False
            c = Config(quaternion_format="wxyz", image_size=[32, 32])
            ep = Episode(p, c, mapping)
            self.assertIsNone(ep.sample(0))
            sample = ep.sample(8)
            self.assertIsNotNone(sample)
            self.assertLess((~sample[2]).sum(), c.chunk_size)
            c.history_steps = 0
            self.assertEqual(Episode(p, c, mapping).sample(0)[0].shape, (6,))


class DatasetAndModelTests(unittest.TestCase):
    def test_split_statistics_and_model_training(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            cal = root/"calibration.json";calibration(cal)
            for i in range(3):fixture(root/f"episode_{i:06d}.hdf5", i)
            c = Config(data_root=str(root), calibration=str(cal), quaternion_format="wxyz", image_size=[32, 32],
                       hidden_dim=32, heads=4, encoder_layers=1, decoder_layers=1, feedforward_dim=64,
                       pretrained_backbone=False, device="cpu", chunk_size=8)
            prepare(c, root/"prepared")
            train = ACTDataset(root/"prepared", "train")
            expected = np.stack([train.episode(p).sample(r)[0] for p, r in train.items]).mean(0)
            np.testing.assert_allclose(train.stats["state_mean"], expected, atol=1e-5)
            splits = train.manifest["splits"]
            ids = [{r["episode_id"] for r in splits[s]} for s in ["train", "val", "test"]]
            self.assertFalse(ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
            self.assertEqual(train.stats["source_split"], "train")
            sample = train[0]
            model = ACT(c).train()
            batch = {k: v[None] for k, v in sample.items()}
            prediction, mu, logvar = model(batch["image"], batch["state"], batch["action"], batch["is_pad"])
            losses = loss(prediction, batch["action"], batch["is_pad"], mu, logvar, 10)
            losses["loss"].backward()
            self.assertTrue(torch.isfinite(model.action_head.weight.grad).all())
            self.assertEqual(prediction.shape, (1, 8, 15))
            model.eval()
            with torch.no_grad():
                prediction, mu, logvar = model(batch["image"], batch["state"])
            self.assertIsNone(mu)
            self.assertTrue(torch.isfinite(prediction).all())
            # Source mutation must invalidate the manifest instead of silently changing a split.
            with h5py.File(root/f"episode_{next(iter(ids[0])):06d}.hdf5", "a") as f:
                f.attrs["changed"] = True
            with self.assertRaises(ValueError):ACTDataset(root/"prepared", "train")

    def test_padding_does_not_dilute_loss(self):
        prediction = torch.ones(1, 4, 15)
        target = torch.zeros_like(prediction)
        mu = logvar = torch.zeros(1, 3)
        a = loss(prediction, target, torch.tensor([[False, True, True, True]]), mu, logvar, 10)["l1"]
        b = loss(prediction, target, torch.zeros(1, 4, dtype=torch.bool), mu, logvar, 10)["l1"]
        self.assertEqual(float(a), float(b))


if __name__ == "__main__":
    unittest.main()
