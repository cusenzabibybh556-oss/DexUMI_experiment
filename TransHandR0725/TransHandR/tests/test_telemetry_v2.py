from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "config"))

from multisource_core import TimedSample, interpolate_vector, latest_before, nearest
from telemetry_protocol import PROTOCOL_V2, legacy_messages, make_packet, parse_v2_packet, tactile_summary
from teleoperation_profiles import get_profile


def test_v2_roundtrip_and_dimension_validation():
    packet = make_packet("operator", np.arange(21), "linkerforce", "o6", "left", 10, 20, 3)
    message = parse_v2_packet(packet, {"operator": 21})
    assert packet["protocol"] == PROTOCOL_V2
    assert message.kind == "operator"
    assert message.values.shape == (21,)
    assert message.host_receive_timestamp_ns == 20
    assert message.unified_timestamp_ns == 20
    mapped = make_packet("state", [1], "sdk", "o6", "left", 10, 20, 4, 15)
    assert parse_v2_packet(mapped).unified_timestamp_ns == 15
    packet["values"] = [1, 2]
    with pytest.raises(ValueError, match="dimension"):
        parse_v2_packet(packet, {"operator": 21})


def test_inspire_force_packet_is_six_dimensional():
    packet = make_packet("force", [0, 10, 20, 30, 40, -2],
                         "inspire_force_act", "inspire", "right", 10, 20, 7)
    assert parse_v2_packet(packet, {"force": 6}).values.tolist() == [0, 10, 20, 30, 40, -2]
    packet["values"] = [float("nan")] * 6
    with pytest.raises(ValueError, match="non-finite"):
        parse_v2_packet(packet, {"force": 6})


def test_tactile_shape_and_summary():
    matrix = np.arange(360, dtype=np.float32).reshape(5, 12, 6)
    packet = make_packet("tactile", matrix, "o6_tactile", "o6", "left", 10, 20, 4)
    message = parse_v2_packet(packet)
    summary = tactile_summary(message.values, contact_threshold=100)
    assert summary["matrix"].shape == (5, 12, 6)
    assert summary["max"].shape == (5,)
    assert summary["contact_area"].dtype == np.int16
    packet["values"] = [[1, 2]]
    with pytest.raises(ValueError, match="tactile shape"):
        parse_v2_packet(packet)


def test_b1_palm_packet_is_complete_and_validated():
    matrix = np.zeros((14, 15), dtype=np.int16)
    matrix[13, 14] = 255
    packet = make_packet("palm", matrix, "o6_palm_b1", "o6", "left", 10, 20, 4)
    assert parse_v2_packet(packet).values[13, 14] == 255
    packet["values"] = [[0]]
    with pytest.raises(ValueError, match="palm shape"):
        parse_v2_packet(packet)


def test_tactile_summary_ignores_sensor_padding_and_rejects_sentinel_data():
    matrix = np.full((5, 12, 6), -1, dtype=np.float32)
    matrix[:, :10, :4] = 0
    matrix[2, 3, 1] = 25
    summary = tactile_summary(matrix, contact_threshold=1)
    assert np.allclose(summary["mean"], [0, 0, 25 / 40, 0, 0])
    assert summary["max"].tolist() == [0, 0, 25, 0, 0]
    assert summary["contact_area"].tolist() == [0, 0, 1, 0, 0]

    with pytest.raises(ValueError, match="no valid cells"):
        tactile_summary(np.full((5, 12, 6), -1, dtype=np.float32))


def test_legacy_flat_command_and_state_are_supported():
    messages = legacy_messages({"command": [1, 2], "command_timestamp_ns": 50,
                                "state": [3, 4], "state_timestamp_ns": 60}, 70)
    assert [message.kind for message in messages] == ["action", "state"]
    assert [message.host_receive_timestamp_ns for message in messages] == [50, 60]


def test_synthetic_alignment_policies():
    ms = 1_000_000
    operator = [TimedSample(np.array([i], np.float32), i * 10 * ms, i * 10 * ms, i * 10 * ms)
                for i in range(11)]
    state = [TimedSample(np.array([i], np.float32), i * 33 * ms, i * 33 * ms, i * 33 * ms)
             for i in range(5)]
    action = [TimedSample(np.array([i], np.float32), i * 20 * ms, i * 20 * ms, i * 20 * ms)
              for i in range(6)]
    tactile = [TimedSample(np.array([i], np.float32), i * 100 * ms, i * 100 * ms, i * 100 * ms)
                for i in range(3)]
    target = 66 * ms
    op, _ = interpolate_vector(operator, target, 20)
    hand, _ = interpolate_vector(state, target, 35)
    command, age = latest_before(action, target, 100)
    touch, dt = latest_before(tactile, target, 120)
    assert np.allclose(op.value, [6.6])
    assert np.allclose(hand.value, [2.0])
    assert command.unified_timestamp_ns == 60 * ms
    assert command.unified_timestamp_ns <= target and age == 6
    assert touch.unified_timestamp_ns == 0 and dt == 66


def test_stateful_action_does_not_expire_when_max_age_is_disabled():
    ms = 1_000_000
    action = [TimedSample(np.array([321], np.float32), 10 * ms, 10 * ms, 10 * ms)]

    command, age = latest_before(action, 5_010 * ms, 0)

    assert command is action[0]
    assert age == 5_000


def test_profile_matrix_rejects_unsupported_combinations():
    o6 = get_profile("linkerforce", "o6", "left")
    assert o6.tactile_shape == (5, 12, 6)
    assert o6.requires_calibration
    inspire = get_profile("linkerforce", "inspire", "right")
    assert inspire.action_dim == 6
    assert inspire.tactile_shape is None
    assert inspire.requires_hand_serial and inspire.requires_calibration
    assert inspire.topic("action_topic", "right") == "/robot/inspire/right/action_applied"
    with pytest.raises(ValueError):
        get_profile("linkerforce", "o6", "right")
