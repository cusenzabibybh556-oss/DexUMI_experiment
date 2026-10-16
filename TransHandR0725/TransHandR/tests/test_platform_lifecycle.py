from pathlib import Path
import json
import os
import signal
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest
import h5py


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "config"))

from run_teleoperation_platform import (
    DEFAULT_RECORDING_DIR,
    O6_BRIDGE_SEND_INTERVAL_S,
    O6_BRIDGE_SPEED,
    O6_BRIDGE_TORQUE,
    TeleoperationPlatform,
)
import run_teleoperation_platform as platform_module
from validate_multisource_episode import validate


def free_udp_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def test_mock_platform_lifecycle_and_crash_rollback(tmp_path):
    platform = TeleoperationPlatform("mock", "127.0.0.1", 0, free_udp_port(), tmp_path)
    platform.update_system_config({"avp_ip": "vision-pro.local"})
    saved = platform.update_recording_config({
        "control_source": "linkerforce", "target_hand_model": "o6", "hand_side": "left",
        "channels": ["d405", "operator_exoskeleton", "hand_state", "action", "tactile"],
    })
    assert {item["id"] for item in saved["channels"] if item["selected"]} == {
        "d405", "operator_exoskeleton", "hand_state", "action", "tactile",
    }
    with pytest.raises(ValueError, match="IP"):
        platform.update_system_config({"avp_ip": "https://bad address"})
    platform.start_system()
    assert platform.state == "RUNNING"
    with pytest.raises(RuntimeError, match="运行中"):
        platform.update_system_config({"target_hand_model": "o6"})
    with pytest.raises(RuntimeError, match="运行中"):
        platform.update_recording_config({"channels": ["d405"]})

    platform.record_control("start")
    time.sleep(.35)
    platform.record_control("stop")
    episodes = list(tmp_path.rglob("episode_*.hdf5"))
    assert episodes
    with h5py.File(episodes[0], "r") as episode:
        assert episode["operator/exoskeleton/joint_positions"].shape[1:] == (21,)
        assert "operator_exoskeleton" in json.loads(episode.attrs["selected_channels"])
        assert episode.attrs["schema_version"] == "quality-first-v3.2"
        assert "dexterous_hand/tactile/matrix" not in episode
        tactile_frames = len(episode["dexterous_hand/tactile_stream/timestamp_ns"])
        assert 0 < tactile_frames < len(episode["timestamps/master_timestamp_ns"])
        valid = episode["dexterous_hand/tactile/valid"][:]
        indices = episode["dexterous_hand/tactile/index"][:]
        assert ((indices[valid] >= 0) & (indices[valid] < tactile_frames)).all()
    assert validate(episodes[0])["ok"]

    platform.processes["simulated_crash"] = subprocess.Popen(["/bin/false"])
    deadline = time.monotonic() + 3
    while (platform.state != "ERROR" or platform.recorder is not None) and time.monotonic() < deadline:
        time.sleep(.05)
    assert platform.state == "ERROR"
    assert platform.recorder is None
    platform.stop_system()
    assert platform.state == "STOPPED"


def test_recording_directory_is_separate_from_platform_settings(tmp_path):
    settings_dir = tmp_path / "settings"
    recording_dir = tmp_path / "other_disk" / "episodes"
    platform = TeleoperationPlatform("mock", "127.0.0.1", 0, free_udp_port(), settings_dir)
    assert platform.recording_config()["output_dir"] == str(settings_dir)

    result = platform.update_system_config({"recording_dir": str(recording_dir)})
    assert result["config"]["recording_dir"] == str(recording_dir)
    assert platform.dashboard.episode_root == recording_dir
    assert platform._collection_config().episode_dir.is_relative_to(recording_dir)
    assert platform.config_path.parent == settings_dir
    assert platform.channel_config_path.parent == settings_dir

    restored = TeleoperationPlatform("mock", "127.0.0.1", 0, free_udp_port(), settings_dir)
    assert restored.recording_config()["output_dir"] == str(recording_dir)
    restored.start_system()
    try:
        restored.record_control("start")
        time.sleep(.25)
        restored.record_control("stop")
    finally:
        restored.stop_system()
    assert list(recording_dir.rglob("episode_*.hdf5"))
    assert not list(settings_dir.rglob("episode_*.hdf5"))


def test_recording_directory_validation_and_runtime_lock(tmp_path):
    platform = TeleoperationPlatform("mock", "127.0.0.1", 0, free_udp_port(), tmp_path)
    old_root = platform.dashboard.episode_root
    for invalid in ("", "relative/episodes"):
        with pytest.raises(ValueError, match="数据保存目录"):
            platform.update_system_config({"recording_dir": invalid})
    file_path = tmp_path / "not-a-directory"
    file_path.write_text("already a file", encoding="utf-8")
    with pytest.raises(RuntimeError, match="不可写"):
        platform.update_system_config({"recording_dir": str(file_path)})
    assert platform.dashboard.episode_root == old_root
    platform.state = "RUNNING"
    with pytest.raises(RuntimeError, match="运行中"):
        platform.update_system_config({"recording_dir": str(tmp_path / "new")})


def test_hardware_legacy_config_uses_data_disk_for_new_recordings(tmp_path):
    settings_dir = tmp_path / "settings"
    settings_dir.mkdir()
    (settings_dir / ".platform_config.json").write_text(
        json.dumps({"task_name": "existing_task"}), encoding="utf-8")
    platform = TeleoperationPlatform("hardware", "127.0.0.1", 0, free_udp_port(), settings_dir)
    assert platform.config["task_name"] == "existing_task"
    assert platform.recording_config()["output_dir"] == str(DEFAULT_RECORDING_DIR)
    assert platform.dashboard.episode_root == DEFAULT_RECORDING_DIR


def test_unmounted_data_disk_does_not_fall_back_to_system_disk(monkeypatch):
    monkeypatch.setattr(Path, "is_mount", lambda self: False)
    with pytest.raises(RuntimeError, match="数据盘未挂载"):
        TeleoperationPlatform._check_recording_dir(DEFAULT_RECORDING_DIR)


def test_recording_channels_are_profile_scoped_and_persistent(tmp_path):
    platform = TeleoperationPlatform("mock", "127.0.0.1", 0, free_udp_port(), tmp_path)
    platform.update_recording_config({
        "control_source": "linkerforce", "target_hand_model": "o6", "hand_side": "left",
        "channels": ["d405", "operator_exoskeleton", "tactile"],
    })
    platform.update_recording_config({
        "control_source": "vision_pro", "target_hand_model": "inspire", "hand_side": "right",
        "channels": ["d405", "vision_hand", "action"],
    })

    restored = TeleoperationPlatform("mock", "127.0.0.1", 0, free_udp_port(), tmp_path)
    exo = restored.recording_config({
        "control_source": "linkerforce", "target_hand_model": "o6", "hand_side": "left",
    })
    vision = restored.recording_config({
        "control_source": "vision_pro", "target_hand_model": "inspire", "hand_side": "right",
    })
    assert {item["id"] for item in exo["channels"] if item["selected"]} == {
        "d405", "operator_exoskeleton", "tactile",
    }
    assert {item["id"] for item in vision["channels"] if item["selected"]} == {
        "d405", "vision_hand", "action",
    }
    with pytest.raises(RuntimeError, match="不支持"):
        restored.update_recording_config({
            "control_source": "vision_pro", "target_hand_model": "inspire", "hand_side": "right",
            "channels": ["d405", "tactile"],
        })

    inspire_exo = restored.update_recording_config({
        "control_source": "linkerforce", "target_hand_model": "inspire", "hand_side": "left",
        "channels": ["d405", "operator_exoskeleton", "hand_state", "action"],
    })
    available = {item["id"]: item["available"] for item in inspire_exo["channels"]}
    assert available["operator_exoskeleton"] is True
    assert available["tactile"] is False


def test_process_whitelist_is_exact():
    assert TeleoperationPlatform._is_managed_command([
        "/usr/bin/python3", str(ROOT / "scripts/main_visionpro_multi_hand.py"),
    ])
    assert TeleoperationPlatform._is_managed_command([
        "/usr/bin/python3", str(ROOT / "scripts/inspire_bridge.py"),
    ])
    assert TeleoperationPlatform._is_managed_command([
        "/opt/ros/humble/bin/ros2", "run", "linkerhand_retarget", "o6_bridge",
    ])
    assert not TeleoperationPlatform._is_managed_command([
        "/usr/bin/python3", "-c", "from ros2cli.daemon.daemonize import main; main()",
    ])
    assert not TeleoperationPlatform._is_managed_command([
        "/usr/bin/python3", "/tmp/unrelated_o6_bridge.py",
    ])


def test_vision_stream_preflight_rejects_unreachable_headset(monkeypatch):
    def unreachable(command, **kwargs):
        assert command[-1] == "172.20.10.10:12345"
        assert kwargs["timeout"] == 5
        return subprocess.CompletedProcess(command, 1)

    monkeypatch.setattr(subprocess, "run", unreachable)
    with pytest.raises(RuntimeError, match="Vision Pro 172.20.10.10:12345 的 gRPC 数据流未就绪"):
        TeleoperationPlatform._check_vision_stream("172.20.10.10")


def test_vision_stream_preflight_accepts_reachable_headset(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda command, **_kwargs: subprocess.CompletedProcess(command, 0))
    TeleoperationPlatform._check_vision_stream("172.20.10.10")


def test_o6_bridge_uses_fast_motion_with_low_torque():
    assert O6_BRIDGE_SPEED == [255] * 6
    assert O6_BRIDGE_TORQUE == [40] * 6
    assert O6_BRIDGE_SEND_INTERVAL_S == 0.005


def test_replay_only_o6_requires_stopped_platform(tmp_path):
    platform = TeleoperationPlatform("mock", "127.0.0.1", 0, free_udp_port(), tmp_path)
    platform._allow_replay_only_o6_bridge()
    platform.state = "RUNNING"
    with pytest.raises(RuntimeError, match="已停止"):
        platform._allow_replay_only_o6_bridge()


def test_hardware_platform_cannot_clean_up_during_replay_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(platform_module, "DEXUMI_ROOT", tmp_path)
    platform = TeleoperationPlatform("hardware", "127.0.0.1", 0,
                                    free_udp_port(), tmp_path)
    with platform._replay_boundary():
        with pytest.raises(RuntimeError, match="真机恢复/回放正在运行"):
            platform.stop_system()
        with pytest.raises(RuntimeError, match="真机恢复/回放正在运行"):
            platform.start_system()


def test_inspire_device_paths_must_be_distinct_but_o6_keeps_legacy_compatibility(tmp_path):
    platform = TeleoperationPlatform("mock", "127.0.0.1", 0, free_udp_port(), tmp_path)
    shared = "/dev/ttyUSB0"
    platform.update_system_config({"left_glove_serial": shared, "hand_serial": shared})
    with pytest.raises(ValueError, match="不能重复"):
        platform.update_system_config({
            "control_source": "linkerforce", "target_hand_model": "inspire",
            "hand_side": "left",
        })


def test_inspire_process_order_and_profile_topics(tmp_path, monkeypatch):
    platform = TeleoperationPlatform("hardware", "127.0.0.1", 0, free_udp_port(), tmp_path)
    platform.config.update({
        "control_source": "linkerforce", "target_hand_model": "inspire", "hand_side": "right",
        "right_glove_serial": "/dev/right-glove", "hand_serial": "/dev/inspire-hand",
    })
    spawned = []
    monkeypatch.setattr(platform, "_ros_command", lambda command: command)
    monkeypatch.setattr(platform, "_spawn", lambda name, command: spawned.append((name, command)))

    platform._start_device_processes()

    assert [name for name, _ in spawned] == [
        "ros_telemetry_adapter", "inspire_bridge", "handretarget",
    ]
    adapter = spawned[0][1]
    assert "/robot/inspire/right/action_applied" in adapter
    assert "/robot/inspire/right/force_states" in adapter
    assert "--tactile-topic" not in adapter
    assert "active_hand_side:=right" in spawned[2][1]
    assert "calibration_profile:=inspire" in spawned[2][1]
    assert "enable_force_feedback:=false" in spawned[2][1]
    platform.config["feedback_enabled_profiles"] = {"linkerforce:inspire:right": True}
    spawned.clear()
    platform._start_device_processes()
    assert "enable_force_feedback:=true" in spawned[2][1]


def test_inspire_feedback_setting_is_side_scoped_and_locked_while_running(tmp_path):
    platform = TeleoperationPlatform("mock", "127.0.0.1", 0, free_udp_port(), tmp_path)
    assert platform.system_config()["config"]["feedback_enabled_profiles"] == {}
    platform.update_system_config({"feedback_enabled_profiles": {"linkerforce:inspire:left": True}})
    restored = TeleoperationPlatform("mock", "127.0.0.1", 0, free_udp_port(), tmp_path)
    assert restored.config["feedback_enabled_profiles"] == {"linkerforce:inspire:left": True}
    restored.update_system_config({"feedback_enabled_profiles": {"linkerforce:inspire:right": False}})
    assert restored.config["feedback_enabled_profiles"]["linkerforce:inspire:left"] is True
    with pytest.raises(ValueError, match="布尔值"):
        restored.update_system_config({"feedback_enabled_profiles": {"linkerforce:inspire:left": "true"}})
    restored.state = "RUNNING"
    with pytest.raises(RuntimeError, match="运行中"):
        restored.update_system_config({"feedback_enabled_profiles": {"linkerforce:inspire:left": False}})


def test_inspire_force_events_are_recorded_once_even_with_feedback_disabled(tmp_path):
    platform = TeleoperationPlatform("mock", "127.0.0.1", 0, free_udp_port(), tmp_path)
    platform.update_system_config({
        "control_source": "linkerforce", "target_hand_model": "inspire", "hand_side": "right",
    })
    platform.update_recording_config({
        "control_source": "linkerforce", "target_hand_model": "inspire", "hand_side": "right",
        "channels": ["d405", "inspire_force"],
    })
    platform.start_system()
    try:
        platform.record_control("start")
        for sequence in range(5):
            timestamp = time.monotonic_ns()
            message = SimpleNamespace(values=[sequence] * 6, source_timestamp_ns=timestamp,
                                      unified_timestamp_ns=timestamp, sequence=sequence)
            platform.recorder._add_force_sample(message, timestamp)
            time.sleep(.015)
        time.sleep(.1)
        platform.record_control("stop")
    finally:
        platform.stop_system()
    path = next(tmp_path.rglob("episode_*.hdf5"))
    with h5py.File(path, "r") as episode:
        assert episode.attrs["schema_version"] == "quality-first-v3.2"
        assert episode["dexterous_hand/force_stream/value"].shape == (5, 6)
        assert episode["dexterous_hand/force_stream/value"][:, 0].tolist() == list(range(5))
        assert len(episode["timestamps/master_timestamp_ns"]) != 5
        assert not len(episode["dexterous_hand/tactile_stream/matrix"])
    assert validate(path)["ok"]


def test_linkerforce_inspire_episode_keeps_operator_and_tactile_placeholder(tmp_path):
    platform = TeleoperationPlatform("mock", "127.0.0.1", 0, free_udp_port(), tmp_path)
    platform.update_system_config({
        "control_source": "linkerforce", "target_hand_model": "inspire", "hand_side": "right",
    })
    platform.update_recording_config({
        "control_source": "linkerforce", "target_hand_model": "inspire", "hand_side": "right",
        "channels": ["d405", "operator_exoskeleton", "hand_state", "action"],
    })
    platform.start_system()
    platform.record_control("start")
    time.sleep(.2)
    platform.record_control("stop")
    platform.stop_system()

    episode_path = next(tmp_path.rglob("episode_*.hdf5"))
    with h5py.File(episode_path, "r") as episode:
        assert episode.attrs["control_source"] == "linkerforce"
        assert episode.attrs["target_hand_model"] == "inspire"
        assert episode.attrs["hand_side"] == "right"
        assert episode["operator/exoskeleton/joint_positions"].shape[1:] == (21,)
        assert episode["action/command"].shape[1:] == (6,)
        assert episode["dexterous_hand/joint_positions"].shape[1:] == (6,)
        assert not episode["dexterous_hand/tactile/valid"][:].any()
        assert not len(episode["dexterous_hand/tactile_stream/matrix"])
        assert (episode["dexterous_hand/tactile/index"][:] == -1).all()


def test_stale_process_cleanup_targets_only_whitelisted_process(monkeypatch):
    fake_pid = 999_991
    snapshots = iter((
        {
            os.getpid(): {"ppid": 0, "args": ["python3", "platform.py"]},
            fake_pid: {"ppid": 1, "args": [
                "/opt/ros/humble/bin/ros2", "run", "linkerhand_retarget", "handretarget",
            ]},
            999_992: {"ppid": 1, "args": ["python3", "unrelated.py"]},
        },
        {os.getpid(): {"ppid": 0, "args": ["python3", "platform.py"]}},
    ))
    killed = []
    monkeypatch.setattr(TeleoperationPlatform, "_proc_snapshot", staticmethod(lambda: next(snapshots)))
    monkeypatch.setattr(os, "kill", lambda pid, sig: killed.append((pid, sig)))

    assert TeleoperationPlatform._cleanup_stale_processes() == []
    assert killed == [(fake_pid, signal.SIGINT)]
