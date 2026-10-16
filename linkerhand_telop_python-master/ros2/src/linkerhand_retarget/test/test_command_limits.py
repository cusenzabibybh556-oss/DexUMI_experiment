import math

import pytest

from linkerhand_retarget.command_limits import CommandLimiter
from linkerhand_retarget.hand_profiles import O6_LEFT_PROFILE


def test_default_limits_preserve_existing_commands():
    limiter = CommandLimiter([0.0] * 6, [255.0] * 6)

    command, count = limiter.apply([0, 50, 100, 150, 200, 255])

    assert command == [0, 50, 100, 150, 200, 255]
    assert count == 0


def test_thumb_minimum_reduces_maximum_flexion():
    limiter = CommandLimiter([120.0, 0, 0, 0, 0, 0], [255.0] * 6)

    command, count = limiter.apply([0, 10, 20, 30, 40, 50])

    assert command == [120, 10, 20, 30, 40, 50]
    assert count == 1


@pytest.mark.parametrize(
    "minimum,maximum",
    [([100], [99]), ([-1], [255]), ([0], [256]), ([math.nan], [255])],
)
def test_invalid_limits_are_rejected(minimum, maximum):
    with pytest.raises(ValueError):
        CommandLimiter(minimum, maximum)


def test_o6_profile_has_one_mapping_per_command_channel():
    profile = O6_LEFT_PROFILE

    assert profile.command_names[0] == "thumb_flexion"
    assert len(profile.command_names) == len(profile.command_to_model)
    assert len(profile.command_names) == len(profile.default_open_pose)
