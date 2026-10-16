import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
ROS_PACKAGE = (
    ROOT.parents[1] / "linkerhand_telop_python-master/ros2/src/linkerhand_retarget"
)
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROS_PACKAGE))

from inspire_bridge import (
    CommandStabilizer,
    ControlDiagnostics,
    apply_limits,
    commands_differ,
    expand_channels,
    map_linkerforce_to_inspire,
    write_acknowledged_command,
)
from linkerhand_retarget.motion.linkerforce.calibration_store import CalibrationStore
from linkerhand_retarget.motion.linkerforce.inspire_feedback import InspireFeedbackMapper
from linkerhand_retarget.motion.linkerforce.config.inspire_config import (
    InspireCommandGuard,
    extract_six_features,
    map_calibrated_command,
)


def test_inspire_force_feedback_reorders_filters_and_clears_stale_data():
    mapper = InspireFeedbackMapper(baseline_samples=2, threshold=0, gain=1,
                                   alpha=1, max_output=30, max_step=30)
    assert mapper.update([0] * 6, 1_000_000_000) == [0] * 5
    assert mapper.update([0] * 6, 1_050_000_000) == [0] * 5
    assert mapper.update([1, 2, 3, 4, 5, 999], 1_100_000_000) == [5, 4, 3, 2, 1]
    assert mapper.current(1_200_000_000) == [5, 4, 3, 2, 1]
    assert mapper.current(1_300_000_000) == [0] * 5
    with pytest.raises(ValueError, match="six"):
        mapper.update([1, 2], 1_400_000_000)


def test_inspire_channel_reorder_scale_and_limits():
    assert map_linkerforce_to_inspire([0, 51, 102, 153, 204, 255]) == [
        1000, 800, 600, 400, 0, 200,
    ]
    assert apply_limits([0, 100, 200, 300, 400, 500],
                        [50, 0, 0, 0, 0, 0],
                        [900, 900, 900, 900, 350, 450]) == [50, 100, 200, 300, 350, 450]
    with pytest.raises(ValueError, match="6"):
        map_linkerforce_to_inspire([1, 2])


def test_inspire_command_stabilizer_deadband_and_step_limit():
    stabilizer = CommandStabilizer(deadband=8, alpha=0.5, max_step=100)
    stabilizer.reset([500] * 6)
    assert stabilizer.update([505] * 6) == [500] * 6
    assert stabilizer.update([900] * 6) == [600] * 6
    assert stabilizer.update([900] * 6) == [700] * 6


def test_inspire_channel_deadbands_and_adaptive_response():
    stabilizer = CommandStabilizer(
        deadband=[12, 12, 12, 12, 16, 60],
        alpha_min=0.25,
        alpha_max=0.75,
        max_step=100,
    )
    stabilizer.reset([500] * 6)

    assert stabilizer.update([511, 511, 511, 511, 515, 559]) == [500] * 6
    assert stabilizer.last_deadband_suppressed == 6
    small = stabilizer.update([520, 520, 520, 520, 520, 570])
    assert all(500 < value < 520 for value in small[:5])
    assert 500 < small[5] < 570
    large = stabilizer.update([900] * 6)
    assert all(value - before == 100 for value, before in zip(large, small))


def test_inspire_stabilizer_converges_to_calibrated_endpoints():
    stabilizer = CommandStabilizer(
        deadband=[12, 12, 12, 12, 16, 60], max_step=100)
    stabilizer.reset([1000] * 6)
    for _ in range(40):
        closed = stabilizer.update([0] * 6)
    assert closed == [0] * 6


def test_inspire_duplicate_write_suppression_and_scalar_compatibility():
    assert expand_channels(8, "deadband", int) == [8] * 6
    assert commands_differ(None, [500] * 6, 2)
    assert not commands_differ([500] * 6, [498, 499, 500, 501, 502, 500], 2)
    assert commands_differ([500] * 6, [497, 500, 500, 500, 500, 500], 2)


def test_inspire_grasp_noise_produces_no_repeated_commands():
    stabilizer = CommandStabilizer(
        deadband=[12, 12, 12, 12, 16, 60], max_step=100)
    stabilizer.reset([400] * 6)
    queued = [400] * 6
    writes = 0
    for offset in (5, -7, 9, -11, 4, -6) * 20:
        command = stabilizer.update([400 + offset] * 5 + [400 + 2 * offset])
        if commands_differ(queued, command, 2):
            queued = command
            writes += 1
    assert writes == 0


def test_inspire_failed_write_does_not_publish_and_can_retry():
    class FakeSdk:
        def __init__(self):
            self.results = [False, True]

        def write(self, _command):
            return self.results.pop(0)

    sdk = FakeSdk()
    published = []
    command = [500] * 6

    with pytest.raises(RuntimeError, match="not acknowledged"):
        write_acknowledged_command(sdk.write, published.append, command)
    assert published == []

    write_acknowledged_command(sdk.write, published.append, command)
    assert published == [command]


def test_inspire_control_diagnostics_reports_rates_and_peak_to_peak():
    diagnostics = ControlDiagnostics()
    diagnostics.started_at = 10.0
    diagnostics.note_input([100, 200, 300, 400, 500, 600], 4)
    diagnostics.note_input([110, 190, 300, 420, 480, 605], 3)
    diagnostics.note_state([90, 200, 310, 400, 500, 610])
    diagnostics.note_state([100, 205, 300, 430, 490, 600])
    diagnostics.writes = 1
    diagnostics.duplicate_suppressed = 1
    stats = diagnostics.snapshot_and_reset(12.0)
    assert stats["input_hz"] == pytest.approx(1.0)
    assert stats["write_hz"] == pytest.approx(0.5)
    assert stats["request_p2p"] == [10, 10, 0, 20, 20, 5]
    assert stats["state_p2p"] == [10, 5, 10, 30, 10, 10]


def test_inspire_maps_each_calibrated_open_and_fist_endpoint_directly():
    opened = [float(index) for index in range(21)]
    fist = list(opened)
    for index in (1, 2, 4, 6, 10, 14, 18):
        fist[index] += 2.0 if index % 2 else -2.0
    middle = [(start + end) / 2.0 for start, end in zip(opened, fist)]

    assert extract_six_features(opened)[1] == opened[1]
    assert map_calibrated_command(opened, opened, fist) == [255] * 6
    assert map_calibrated_command(middle, opened, fist) == [128] * 6
    assert map_calibrated_command(fist, opened, fist) == [0] * 6
    assert map_linkerforce_to_inspire(
        map_calibrated_command(fist, opened, fist)) == [0] * 6


def test_inspire_rejects_a_calibration_channel_without_motion():
    opened = [0.0] * 21
    fist = [1.0] * 21
    fist[1] = 0.0
    with pytest.raises(ValueError, match="channel 1"):
        map_calibrated_command(opened, opened, fist)


def test_inspire_command_guard_rejects_single_frame_thumb_rotate_spike():
    guard = InspireCommandGuard()
    assert guard.update([200, 120, 200, 200, 200, 200]) == [200, 120, 200, 200, 200, 200]

    assert guard.update([200, 20, 200, 200, 200, 200]) == [200, 120, 200, 200, 200, 200]
    assert guard.update([200, 121, 200, 200, 200, 200]) == [200, 121, 200, 200, 200, 200]
    assert guard.rejected_spikes[1] == 1


def test_inspire_command_guard_accepts_sustained_thumb_rotate_motion_slowly():
    guard = InspireCommandGuard()
    guard.update([200, 120, 200, 200, 200, 200])

    assert guard.update([200, 20, 200, 200, 200, 200])[1] == 120
    assert guard.update([200, 20, 200, 200, 200, 200])[1] == 120
    assert guard.update([200, 20, 200, 200, 200, 200])[1] == 114
    assert guard.update([200, 20, 200, 200, 200, 200])[1] == 108


def test_calibration_profiles_are_isolated_and_legacy_is_o6_only(tmp_path):
    legacy = tmp_path / "jointangle_data.tmp"
    opened = [float(index) for index in range(21)]
    middle = [value + 1.0 for value in opened]
    closed = [value + 2.0 for value in opened]
    legacy.write_text(json.dumps({
        "timestamp": "2026-01-01T00:00:00",
        "jointangleoriginal_l": opened,
        "jointangleopose_l": middle,
        "jointanglefist_l": closed,
    }), encoding="utf-8")

    store = CalibrationStore(tmp_path / "calibration.json")
    assert store.migrate_legacy(legacy)
    assert store.status("o6", "left")["valid"] is True
    assert store.status("inspire", "left")["valid"] is False

    store.save("inspire", "left", closed, middle, opened, "2.1.5")
    store.save("inspire", "right", opened, middle, closed, "2.1.6")
    assert store.get("inspire", "left")["original"] == closed
    assert store.get("inspire", "right")["original"] == opened
    assert store.get("o6", "left")["original"] == opened
