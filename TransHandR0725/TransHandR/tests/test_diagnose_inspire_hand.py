from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from diagnose_inspire_hand import probe_target, signed_short, trajectory_summary


def test_signed_short_decodes_force_feedback():
    assert signed_short(0) == 0
    assert signed_short(1000) == 1000
    assert signed_short(65535) == -1
    assert signed_short(65423) == -113


def test_probe_target_is_small_and_bounded():
    assert probe_target(500, -20) == 480
    assert probe_target(990, 20) == 1000
    assert probe_target(10, -20) == 0
    with pytest.raises(ValueError, match="between 1 and 50"):
        probe_target(500, 0)
    with pytest.raises(ValueError, match="between 1 and 50"):
        probe_target(500, 51)


def test_trajectory_summary_only_uses_selected_channel():
    samples = [
        {"angles": [100, 200, 300, 400, 500, 600]},
        {"angles": [100, 200, 300, 400, 500, 580]},
        {"angles": [100, 200, 300, 400, 500, 590]},
    ]
    assert trajectory_summary(samples, 5) == {
        "sample_count": 3,
        "first": 600,
        "last": 590,
        "minimum": 580,
        "maximum": 600,
    }
