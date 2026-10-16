"""O6 recorded-state timeline tests; no ROS, CAN or robot connections."""

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from replay_o6_hand_ros import load_states


def make_episode(path: Path) -> None:
    with h5py.File(path, "w") as handle:
        handle.attrs["hand_type"] = "o6"
        handle.attrs["hand_side"] = "left"
        handle.attrs["control_source"] = "linkerforce"
        handle.create_dataset("timestamps/master_timestamp_ns",
                              data=[1_000_000_000, 1_033_000_000, 1_066_000_000])
        handle.create_dataset("action/command_timestamp_ns",
                              data=[999_000_000, 1_030_000_000, 1_063_000_000])
        handle.create_dataset("action/valid", data=[True, False, True])
        handle.create_dataset("action/command", data=np.array([
            [255, 100, 200, 201, 202, 203],
            [255, 90, 190, 191, 192, 193],
            [250, 80, 180, 181, 182, 183],
        ], dtype=np.float32))
        handle.create_dataset("dexterous_hand/unified_timestamp_ns",
                              data=[1_000_000_000, -1, 1_066_000_000])
        handle.create_dataset("dexterous_hand/state_valid",
                              data=[True, False, True])
        handle.create_dataset("dexterous_hand/joint_positions", data=np.array([
            [240, 120, 150, 151, 152, 153],
            [np.nan] * 6,
            [241, 100, 140, 141, 142, 143],
        ], dtype=np.float32))


class O6StateTests(unittest.TestCase):
    def test_state_timestamps_share_master_origin_and_speed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode.hdf5"
            make_episode(path)
            due_ns, states = load_states(path, 0.2, 1_000_000_000)
        np.testing.assert_array_equal(due_ns, [0, 330_000_000])
        self.assertEqual(states.shape, (2, 6))
        self.assertEqual(states[0].tolist(), [240, 120, 150, 151, 152, 153])

    def test_rejects_wrong_hand_and_out_of_range_states(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode.hdf5"
            make_episode(path)
            with h5py.File(path, "r+") as handle:
                handle.attrs["hand_side"] = "right"
            with self.assertRaisesRegex(ValueError, "O6 left-hand"):
                load_states(path, 0.2, 1_000_000_000)
            with h5py.File(path, "r+") as handle:
                handle.attrs["hand_side"] = "left"
                handle["dexterous_hand/joint_positions"][0, 0] = 70
            with self.assertRaisesRegex(ValueError, "hardware range"):
                load_states(path, 0.2, 1_000_000_000)

    def test_state_origin_can_follow_an_invalid_first_tracker_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode.hdf5"
            make_episode(path)
            due_ns, states = load_states(path, 0.2, 1_033_000_000)
        np.testing.assert_array_equal(due_ns, [0, 165_000_000])
        self.assertEqual(states.shape, (2, 6))

    def test_valid_state_must_be_aligned_to_master_timestamp(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode.hdf5"
            make_episode(path)
            with h5py.File(path, "r+") as handle:
                handle["dexterous_hand/unified_timestamp_ns"][2] += 1_000_000
            with self.assertRaisesRegex(ValueError, "not aligned"):
                load_states(path, 0.2, 1_000_000_000)

    def test_rejects_missing_state_instead_of_falling_back_to_action(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode.hdf5"
            make_episode(path)
            with h5py.File(path, "r+") as handle:
                del handle["dexterous_hand/joint_positions"]
            with self.assertRaisesRegex(ValueError, "recorded state datasets missing"):
                load_states(path, 0.2, 1_000_000_000)

    def test_does_not_need_recorded_action_command(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode.hdf5"
            make_episode(path)
            with h5py.File(path, "r+") as handle:
                del handle["action"]
            due_ns, states = load_states(path, 0.2, 1_000_000_000)
        np.testing.assert_array_equal(due_ns, [0, 330_000_000])
        self.assertEqual(states[0].tolist(), [240, 120, 150, 151, 152, 153])


if __name__ == "__main__":
    unittest.main()
