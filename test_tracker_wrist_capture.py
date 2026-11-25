"""离线验证成对采集、未知状态保留和网页入口；不连接任何硬件。"""
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import urllib.error
import urllib.request

import numpy as np
import pytest

import capture_tron2_tracker_calibration as capture

SCRIPTS = Path(__file__).resolve().parent / "TransHandR0725/TransHandR/scripts"
sys.path.insert(0, str(SCRIPTS))
from wrist_calibration_web import WristCalibrationManager
from realtime_multisource_dashboard import DashboardServer


def wait_until(predicate, timeout=6):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.04)
    raise AssertionError("timed out")


def frames(status=2, moving=False):
    trackers, robots = [], []
    for i in range(5):
        now = 1_000_000_000 + i * 100_000_000
        p = [.123456789123, .2, .3 + (.02 * i if moving else 0)]
        # q 和 -q 是同一旋转，不能被判为抖动。
        q = [1 if i % 2 else -1, 0, 0, 0]
        trackers.append({**capture.pose_values(p, q), "received_monotonic_ns": now,
                         "tracking_status": status, "mac": "A"})
        robots.append({**capture.pose_values(p, q), "received_monotonic_ns": now + 1_000_000,
                       "request_started_monotonic_ns": now, "request_duration_ms": 1})
    return trackers, robots


def test_status20_kept_but_not_eligible_and_quaternion_sign_invariant():
    trackers, robots = frames(20)
    summary = capture.summarize_sample(trackers, robots)
    assert summary["tracking_statuses"] == [20]
    assert summary["pair_count"] == 5
    assert not summary["eligible_for_calibration"]
    assert not summary["calibration_validated"]
    assert summary["tracker_statistics"]["rotation_radius_deg"] == 0
    assert trackers[0]["position_m"][0] == .123456789123
    trackers, robots = frames(2)
    assert capture.summarize_sample(trackers, robots)["eligible_for_calibration"]


def test_moving_late_and_mismatched_samples_are_flagged():
    trackers, robots = frames(moving=True)
    assert not capture.summarize_sample(trackers, robots)["eligible_for_calibration"]
    trackers, robots = frames()
    for robot in robots:
        robot["received_monotonic_ns"] += 2_000_000_000
        robot["request_duration_ms"] = 2000
    summary = capture.summarize_sample(trackers, robots)
    assert summary["pair_count"] == 0
    assert not summary["eligible_for_calibration"]


def test_no_cached_pose_reused_when_source_disappears(tmp_path):
    class DisconnectedSource(capture.MockSource):
        def robot_pose(self):
            raise TimeoutError("no fresh reply")

    session = capture.CaptureSession(DisconnectedSource(), tmp_path, lambda _: None, {"mock": True})
    old_trackers, old_robots = frames()
    session.trackers.extend(old_trackers)
    session.robots.extend(old_robots)
    session.start()
    try:
        session.sample(0, .5)
        wait_until(lambda: len(session.groups) == 1)
        group = session.groups[0]
        assert group["tracker_frames"] == group["robot_frames"] == group["pair_count"] == 0
        assert not group["eligible_for_calibration"]
        session.exclude(1)
    finally:
        session.close()
    records = [json.loads(line) for line in session.path.read_text().splitlines()]
    assert records[1]["tracker"] == records[1]["robot"] == []
    assert records[2] == {"event": "exclude", "id": 1}


def test_readonly_client_cannot_send_motion(monkeypatch):
    import replay_tron2_tracker

    class FakeClient:
        accid = "TEST"
        def __init__(self, ip):
            self.sent = []
        def wait_for_identity(self):
            pass
        def send(self, title, data, **kwargs):
            self.sent.append(title)
            return "ok"

    monkeypatch.setattr(replay_tron2_tracker, "Tron2JsonClient", FakeClient)
    source = capture.HardwareSource("127.0.0.1", 0, kinematics=SimpleNamespace(metadata={}))
    for title in ("request_servop", "request_servoj", "request_movej", "request_moveh"):
        with pytest.raises(RuntimeError, match="只允许"):
            source.client.send(title, {})
    with pytest.raises(RuntimeError, match="只允许"):
        source.client.send("request_get_servo_pose", {})
    source.client.send("request_get_joint_state", {})
    assert source.client.sent == ["request_get_joint_state"]


def test_measured_joints_drive_capture_and_duplicate_feedback_is_rejected(monkeypatch):
    import replay_tron2_tracker

    class FakeClient:
        accid = "TEST"
        stamp = 10
        q = [0.] * 16
        def __init__(self, ip):
            pass
        def wait_for_identity(self):
            pass
        def request(self, title, data, response, timeout):
            assert title == "request_get_joint_state"
            return {"data": {"result": "success", "timestamp": self.stamp, "q": self.q}}

    monkeypatch.setattr(replay_tron2_tracker, "Tron2JsonClient", FakeClient)
    fk = SimpleNamespace(metadata={}, pose=lambda q: (q[:3], [1., 0., 0., 0.]))
    source = capture.HardwareSource("127.0.0.1", 0, kinematics=fk)
    a = source.robot_pose()
    source.client.q = [.1] + [0.] * 15
    source.client.stamp = 11
    b = source.robot_pose()
    assert a["position_m"] != b["position_m"]
    assert b["joint_position_rad"][0] == .1
    assert b["pose_source"] == capture.JOINT_FK_SOURCE
    with pytest.raises(RuntimeError, match="时间戳未更新"):
        source.robot_pose()


def test_web_capture_roundtrip_and_export(tmp_path):
    manager = WristCalibrationManager(lambda: None, tmp_path / "captures", mock=True, python=sys.executable)
    server = DashboardServer("127.0.0.1", 0, lambda: {}, lambda _: None, lambda *_: {}, lambda _: {}, tmp_path)
    server.wrist_calibration = manager
    server.start()
    base = f"http://127.0.0.1:{server.httpd.server_port}"

    def request(path, payload=None, origin=True):
        headers = {"Content-Type": "application/json"}
        if origin:
            headers["Origin"] = base
        req = urllib.request.Request(base + path, headers=headers,
                                     data=json.dumps(payload).encode() if payload is not None else None)
        return urllib.request.urlopen(req, timeout=5).read()

    try:
        assert b"wrist-calibrationView" in request("/wrist-calibration")
        assert b"sampleWristPose" in request("/assets/wrist_calibration.js")
        with pytest.raises(urllib.error.HTTPError) as exc:
            request("/api/wrist-calibration/start", {}, origin=False)
        assert exc.value.code == 403
        request("/api/wrist-calibration/start", {"map_label": "test map", "mount_label": "left mount"})
        wait_until(lambda: manager.status()["phase"] == "ready")
        request("/api/wrist-calibration/sample", {"countdown": 0, "duration": .6, "label": "参考姿态"})
        with pytest.raises(urllib.error.HTTPError):
            request("/api/wrist-calibration/sample", {})
        wait_until(lambda: len(manager.status()["groups"]) == 1)
        data = [json.loads(line) for line in request("/api/wrist-calibration/download").splitlines()]
        assert data[0]["mock"] and data[0]["map_label"] == "test map"
        assert data[1]["tracker"][0]["tracking_status"] == 20
        assert data[1]["robot"] and data[1]["pairs"]
        assert not data[1]["eligible_for_calibration"]
        request("/api/wrist-calibration/exclude", {"id": 1})
        wait_until(lambda: manager.status()["groups"][0]["excluded"])
        request("/api/wrist-calibration/stop", {})
        wait_until(lambda: not manager.status()["active"])
        assert json.loads(request("/api/wrist-calibration/download").splitlines()[-1])["event"] == "session_end"
    finally:
        server.stop()


def test_platform_requires_stopped_collector(tmp_path):
    from run_teleoperation_platform import TeleoperationPlatform
    platform = TeleoperationPlatform("mock", "127.0.0.1", 0, 0, tmp_path)
    platform.state = "RUNNING"
    with pytest.raises(RuntimeError, match="停止采集系统"):
        platform.dashboard.wrist_calibration.start({})
    assert not platform.dashboard.wrist_calibration.status()["active"]


def test_capture_lock_excludes_replay_without_connecting_devices(tmp_path, monkeypatch):
    # 测试自己的锁文件，不占用或干扰正在使用的真机采集锁。
    monkeypatch.setattr(capture, "ROOT", tmp_path)
    handle = capture.acquire_capture_lock()
    try:
        with pytest.raises(RuntimeError, match="已有"):
            capture.acquire_capture_lock()
    finally:
        handle.close()
