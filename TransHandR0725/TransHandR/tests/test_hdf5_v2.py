from pathlib import Path
import sys
import time
import json
import socket
import threading

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "config"))

from multisource_collection import CollectionConfig
from collect_transhandr_multisource import RecorderApp
from multisource_core import EpisodeWriter, palm_at_master, tactile_at_master
from telemetry_protocol import make_packet
from validate_multisource_episode import validate


def test_v3_schema_uses_an_independent_tactile_axis(tmp_path):
    config = CollectionConfig(
        control_source="linkerforce", target_hand_model="o6", hand_type="o6", hand_side="left",
        image_width=8, image_height=6, enable_right_hand=False,
        enable_exoskeleton_operator=True, enable_tactile=True,
        enable_head_camera=False,
        selected_channels=["d405", "operator_exoskeleton", "hand_state", "action", "tactile"],
    )
    path = tmp_path / "episode_000000.hdf5"
    writer = EpisodeWriter(path, config, {}, 0)
    expected_shapes = {
        "action/command": (0, 6),
        "dexterous_hand/joint_positions": (0, 6),
        "observations/right_hand/world_keypoints": (0, 25, 3),
        "operator/exoskeleton/joint_positions": (0, 21),
        "dexterous_hand/tactile/index": (0,),
        "dexterous_hand/tactile_stream/matrix": (0, 5, 12, 6),
        "dexterous_hand/palm/index": (0,),
        "dexterous_hand/palm_stream/matrix": (0, 14, 15),
        "dexterous_hand/force_stream/value": (0, 6),
    }
    for name, shape in expected_shapes.items():
        assert writer.f[name].shape == shape
    assert writer.f.attrs["schema_version"] == "quality-first-v3.2"
    assert writer.f.attrs["control_source"] == "linkerforce"
    assert writer.f.attrs["target_hand_model"] == "o6"
    matrix = np.full((5, 12, 6), -1, np.int16)
    matrix[:, :10, :4] = 0
    matrix[4, 9, 3] = 123
    assert writer.append_tactile(matrix, 900_000_000, 7) == 0
    assert writer.append_tactile(matrix, 990_000_000, 8) == 1
    for name, value in {
        "timestamps/master_timestamp_ns": 1_000_000_000,
        "dexterous_hand/tactile/index": 1,
        "dexterous_hand/tactile/age_ms": 10.0,
        "dexterous_hand/tactile/valid": True,
    }.items():
        dataset = writer.f[name]
        dataset.resize((1,) + dataset.shape[1:])
        dataset[0] = value
    resolved, age_ms, index = tactile_at_master(writer.f, 0)
    assert index == 1 and age_ms == 10.0
    assert np.array_equal(resolved, matrix)
    writer.close()
    with h5py.File(path, "r") as episode:
        assert episode["dexterous_hand/tactile_stream/matrix"].dtype == np.dtype("int16")
        assert episode["dexterous_hand/tactile_stream/matrix"].shape[0] == 2
        assert "dexterous_hand/tactile/matrix" not in episode


def test_v3_keeps_100_tactile_events_for_300_master_frames(tmp_path):
    config = CollectionConfig(
        control_source="linkerforce", target_hand_model="o6", hand_type="o6",
        hand_side="left", image_width=8, image_height=6,
        enable_right_hand=False, enable_exoskeleton_operator=True,
        enable_tactile=True, selected_channels=["d405", "tactile"],
    )
    path = tmp_path / "episode_000001.hdf5"
    writer = EpisodeWriter(path, config, {}, 1)
    base = 1_000_000_000
    tactile_times = base + np.arange(100, dtype=np.int64) * 100_000_000
    matrix = np.full((5, 12, 6), -1, np.int16)
    matrix[:, :10, :4] = 0
    for sequence, timestamp in enumerate(tactile_times):
        matrix[sequence % 5, sequence % 10, sequence % 4] = sequence
        writer.append_tactile(matrix, int(timestamp), sequence)

    master_times = base + np.arange(300, dtype=np.int64) * (1_000_000_000 // 30)
    datasets = []
    writer.f.visititems(lambda name, obj: datasets.append((name, obj))
                        if isinstance(obj, h5py.Dataset) else None)
    for name, dataset in datasets:
        if (not dataset.ndim or dataset.chunks is None
                or name.startswith("dexterous_hand/tactile_stream/")
                or name.startswith("dexterous_hand/palm_stream/")
                or name.startswith("dexterous_hand/force_stream/")):
            continue
        dataset.resize((300,) + dataset.shape[1:])
        if dataset.dtype.kind == "f":
            dataset[:] = np.nan
        elif dataset.dtype.kind == "b":
            dataset[:] = False
        elif dataset.dtype.kind == "u":
            dataset[:] = 0
        else:
            dataset[:] = -1
    writer.f["timestamps/master_timestamp_ns"][:] = master_times
    writer.f["camera/d405/valid"][:] = True
    writer.f["camera/d405/frame_number"][:] = np.arange(300)
    indices = np.searchsorted(tactile_times, master_times, side="right") - 1
    ages = (master_times - tactile_times[indices]) / 1e6
    writer.f["dexterous_hand/tactile/index"][:] = indices
    writer.f["dexterous_hand/tactile/age_ms"][:] = ages
    writer.f["dexterous_hand/tactile/valid"][:] = True
    writer.count = 300
    writer.close()

    with h5py.File(path, "r") as episode:
        assert len(episode["timestamps/master_timestamp_ns"]) == 300
        assert len(episode["dexterous_hand/tactile_stream/matrix"]) == 100
        assert episode["dexterous_hand/tactile_stream/matrix"][0, 0, 10, 5] == -1
    assert validate(path)["ok"]
    with h5py.File(path, "r+") as episode:
        del episode["dexterous_hand/force_stream"]
        episode.attrs["schema_version"] = "quality-first-v3.0"
    assert validate(path)["ok"]


def test_palm_stream_is_async_and_master_refs_are_causal(tmp_path):
    config = CollectionConfig(
        control_source="linkerforce", target_hand_model="o6", hand_type="o6",
        hand_side="left", image_width=8, image_height=6,
        enable_right_hand=False, enable_tactile=True,
        selected_channels=["d405", "tactile"],
    )
    path = tmp_path / "palm.hdf5"
    writer = EpisodeWriter(path, config, {}, 0)
    matrix = np.zeros((14, 15), np.int16)
    matrix[13, 14] = 240
    assert writer.append_palm(matrix, 1_000_000_000, 0) == 0
    assert writer.append_palm(matrix, 1_100_000_000, 1) == 1
    for name, values in {
        "timestamps/master_timestamp_ns": [1_020_000_000, 1_050_000_000, 1_250_000_000],
        "dexterous_hand/palm/index": [0, 0, -1],
        "dexterous_hand/palm/age_ms": [20.0, 50.0, np.nan],
        "dexterous_hand/palm/valid": [True, True, False],
    }.items():
        ds = writer.f[name]
        ds.resize((3,) + ds.shape[1:])
        ds[:] = values
    first, age, index = palm_at_master(writer.f, 0)
    assert index == 0 and age == 20.0 and first[13, 14] == 240
    assert palm_at_master(writer.f, 1)[2] == 0
    assert palm_at_master(writer.f, 2)[0] is None
    assert palm_at_master(writer.f, 2, max_age_ms=200)[2] == 1
    assert len(writer.f["dexterous_hand/palm_stream/matrix"]) == 2
    writer.f.close()


def test_recorder_writes_palm_once_and_master_index_points_to_it(tmp_path):
    config = CollectionConfig(
        control_source="linkerforce", target_hand_model="o6", hand_type="o6",
        hand_side="left", image_width=8, image_height=6,
        enable_right_hand=False, enable_tactile=True,
        output_dir=tmp_path, selected_channels=["d405", "tactile"],
    )
    app = RecorderApp(config, "mock", 0, enable_keyboard=False)
    app.start_recording()
    timestamp = time.monotonic_ns()
    palm = np.zeros((14, 15), np.int16)
    palm[5, 8] = 177
    assert app._add_palm_sample(palm, timestamp, timestamp, timestamp, 42)
    camera = {
        "rgb": np.zeros((6, 8, 3), np.uint8),
        "depth": np.zeros((6, 8), np.uint16),
        "unified_timestamp_ns": timestamp + 20_000_000,
        "host_receive_timestamp_ns": timestamp + 20_000_000,
        "frame_number": 1,
        "device_timestamp": 0.0,
        "valid": True,
    }
    row = app._record_row(app._aligned_row(camera))
    assert row["dexterous_hand/palm/index"] == 0
    assert row["dexterous_hand/palm/age_ms"] == 20
    app.writer.append(row)
    app.stop_recording()
    path = next(tmp_path.rglob("episode_*.hdf5"))
    with h5py.File(path, "r") as episode:
        assert episode["dexterous_hand/palm_stream/matrix"].shape == (1, 14, 15)
        assert episode["dexterous_hand/palm_stream/matrix"][0, 5, 8] == 177
        assert episode["dexterous_hand/palm_stream/sequence"][0] == 42
        assert episode["dexterous_hand/palm/valid"][0]
    assert validate(path)["ok"]


def test_palm_udp_packet_reaches_collector_buffer(tmp_path):
    config = CollectionConfig(
        control_source="linkerforce", target_hand_model="o6", hand_type="o6",
        hand_side="left", image_width=8, image_height=6,
        enable_right_hand=False, enable_tactile=True, output_dir=tmp_path,
    )
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as socket_probe:
        socket_probe.bind(("127.0.0.1", 0))
        port = socket_probe.getsockname()[1]
    app = RecorderApp(config, "mock", port, enable_keyboard=False)
    worker = threading.Thread(target=app._telemetry_worker, daemon=True)
    worker.start()
    matrix = np.zeros((14, 15), np.int16)
    matrix[6, 8] = 201
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
            for sequence in range(5):
                timestamp = time.monotonic_ns()
                packet = make_packet("palm", matrix, "o6_palm_b1", "o6", "left",
                                     timestamp, timestamp, sequence, timestamp)
                sender.sendto(json.dumps(packet).encode(), ("127.0.0.1", port))
                time.sleep(.01)
        latest = app.buffers["palm"].latest()
        assert latest is not None
        assert latest.value.shape == (14, 15)
        assert latest.value[6, 8] == 201
    finally:
        app.stop_event.set()
        worker.join(timeout=1)


def test_tactile_loader_keeps_v2_compatibility(tmp_path):
    path = tmp_path / "legacy_v2.hdf5"
    matrix = np.arange(360, dtype=np.float32).reshape(5, 12, 6)
    with h5py.File(path, "w") as episode:
        episode.attrs["schema_version"] = "quality-first-v2.0"
        episode.create_dataset("timestamps/master_timestamp_ns", data=[1_000_000_000])
        episode.create_dataset("dexterous_hand/tactile/matrix", data=matrix[None])
        episode.create_dataset("dexterous_hand/tactile/valid", data=[True])
        episode.create_dataset("sync/tactile_dt_ms", data=[-4.0])
    with h5py.File(path, "r") as episode:
        resolved, age_ms, index = tactile_at_master(episode, 0, max_age_ms=10)
    assert index == 0 and age_ms == -4.0
    assert np.array_equal(resolved, matrix)
