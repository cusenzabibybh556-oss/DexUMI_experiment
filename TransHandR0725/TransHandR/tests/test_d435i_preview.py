from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "config"))

from multisource_collection import CollectionConfig
from collect_transhandr_multisource import RecorderApp
from multisource_core import TimedSample


def camera_frame(timestamp_ns, rgb_value=0):
    return {
        "rgb": np.full((6, 8, 3), rgb_value, np.uint8),
        "depth": np.full((6, 8), rgb_value, np.uint16),
        "unified_timestamp_ns": timestamp_ns,
        "host_receive_timestamp_ns": timestamp_ns,
        "frame_number": 1,
        "device_timestamp": 0.0,
        "valid": True,
    }


def test_d435i_preview_uses_fresh_raw_frame_even_when_master_alignment_fails(tmp_path):
    config = CollectionConfig(
        control_source="linkerforce", target_hand_model="o6", hand_type="o6",
        hand_side="left", image_width=8, image_height=6,
        head_image_width=8, head_image_height=6,
        enable_right_hand=False, enable_head_camera=True,
        output_dir=tmp_path, selected_channels=["d405", "d435i"],
    )
    app = RecorderApp(config, "mock", 0, enable_keyboard=False)
    now = time.monotonic_ns()
    head = camera_frame(now, 77)
    app._add_sample("head_camera", TimedSample(head, 0.0, now, now))
    row = app._aligned_row(camera_frame(now + 100_000_000))
    assert not row["camera/d435i/valid"]
    app._update_status(row)
    status = app.public_status()
    assert not status["valid"]["d435i"]
    assert status["head_preview"]["valid"]
    assert np.all(status["head_rgb"] == 77)
    assert np.all(status["head_depth"] == 77)

    app._stream_last["head_camera"] = time.monotonic() - 1
    app._update_status(row)
    assert not app.public_status()["head_preview"]["valid"]
    assert app.public_status()["head_rgb"] is None


def test_d435i_preview_is_disabled_when_channel_does_not_start(tmp_path):
    config = CollectionConfig(
        control_source="linkerforce", target_hand_model="o6", hand_type="o6",
        hand_side="left", image_width=8, image_height=6,
        enable_right_hand=False, enable_head_camera=False,
        output_dir=tmp_path, selected_channels=["d405"],
    )
    app = RecorderApp(config, "mock", 0, enable_keyboard=False)
    app._update_status(app._aligned_row(camera_frame(time.monotonic_ns())))
    status = app.public_status()
    assert status["head_preview"] == {"enabled": False, "valid": False, "error": ""}
    assert status["head_rgb"] is None
