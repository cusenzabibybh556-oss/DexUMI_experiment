"""用已知刚性变换验证求解方向、误差统计、退化姿态和留出验证；无硬件访问。"""
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from solve_tron2_tracker_calibration import (
    CalibrationError, PoseGroup, transform, inverse, load_samples, solve_groups, solve_file,
)


def synthetic_groups(n=16, noise=0., kind="diverse"):
    rng = np.random.default_rng(19)
    X = transform(Rotation.from_euler("xyz", [174, -21, 68], degrees=True).as_matrix(), [.045, -.07, .085])
    Y = transform(Rotation.from_euler("xyz", [70, 16, -31], degrees=True).as_matrix(), [1.2, -.9, .6])
    groups = []
    for i in range(n):
        vector = rng.normal(size=3) * .4
        if kind == "translation":
            vector[:] = 0
        elif kind == "single_axis":
            vector[:2] = 0
        P = transform(Rotation.from_rotvec(vector).as_matrix(), rng.normal(size=3) * .1 + [.5, .2, -.25])
        Q = Y @ P @ X
        if noise:
            Q[:3, 3] += rng.normal(size=3) * noise
            Q[:3, :3] = Rotation.from_rotvec(rng.normal(size=3) * noise).as_matrix() @ Q[:3, :3]
        groups.append(PoseGroup(i + 1, f"姿态 {i + 1}", P, Q, [2], 6))
    return groups, X, Y


def write_session(path, groups=None, status=2, mock=False):
    groups = synthetic_groups()[0] if groups is None else groups
    events = [{"event": "session", "schema": "tracker-tron2-pairs-v1", "arm": "left",
               "mock": mock, "robot_id": "SYNTHETIC_TEST", "tracker_index": 0,
               "map_label": "synthetic", "mount_label": "synthetic", "position_unit": "m",
               "quaternion_order": "wxyz", "robot_pose_source": "response_get_servo_pose",
               "transform_convention": "T_WT = T_WB @ T_BE @ T_ET"}]
    for group in groups:
        start = group.id * 2_000_000_000
        def frames(T, tracker):
            q = Rotation.from_matrix(T[:3, :3]).as_quat()
            q = q[[3, 0, 1, 2]]
            result = []
            for j in range(6):
                t = start + 50_000_000 + j * 100_000_000
                frame = {"position_m": T[:3, 3].tolist(), "quaternion_wxyz": (q * (-1 if j % 2 else 1)).tolist(),
                         "received_monotonic_ns": t + (0 if tracker else 2_000_000)}
                if tracker:
                    frame.update(tracking_status=status, mac="TEST", tracker_index=0)
                else:
                    frame.update(request_started_monotonic_ns=t, request_duration_ms=2)
                result.append(frame)
            return result
        events.append({"event": "sample", "id": group.id, "label": group.label,
                       "started_monotonic_ns": start, "ended_monotonic_ns": start + 600_000_000,
                       "tracker": frames(group.tracker, True), "robot": frames(group.robot, False),
                       "eligible_for_calibration": status == 2})
    events.append({"event": "session_end", "groups": len(groups)})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in events), encoding="utf-8")
    return events


def test_exact_solution_direction_and_group_validation():
    groups, X, Y = synthetic_groups()
    report = solve_groups(groups)
    np.testing.assert_allclose(report["T_ET"], X, atol=1e-8)
    np.testing.assert_allclose(report["T_WB"], Y, atol=1e-8)
    np.testing.assert_allclose(report["T_BW"], inverse(Y), atol=1e-8)
    assert report["numerical_checks_passed"]
    assert set(report["fit_group_ids"]).isdisjoint(report["validation_group_ids"])
    assert report["metrics"]["validation"]["wrist_position_error_mm"]["max"] < 1e-6
    assert report["observability"]["jacobian_rank"] == 12


def test_joint_fk_schema_keeps_provenance_and_rejects_cached_joint_samples(tmp_path):
    from capture_tron2_tracker_calibration import JOINT_FK_SOURCE
    path = tmp_path / "samples.jsonl"
    events = write_session(path)
    events[0].update(schema="tracker-tron2-pairs-v2", robot_pose_source=JOINT_FK_SOURCE,
                     robot_kinematics={"model_sha256": "a" * 64, "base_frame": "base_Link",
                                       "wrist_frame": "wrist_roll_L_Link", "tcp_offset_m": [0, 0, -.2],
                                       "joint_names": [str(i) for i in range(16)]})
    for e in events[1:-1]:
        for i, f in enumerate(e["robot"]):
            f.update(joint_position_rad=[0.] * 16, robot_timestamp_ms=i + 1, pose_source=JOINT_FK_SOURCE)
    def save():
        path.write_text("\n".join(json.dumps(e) for e in events) + "\n")
    save()
    metadata, groups, rejected, _ = load_samples(path)
    assert len(groups) == 16 and not rejected
    assert metadata["robot_kinematics"]["model_sha256"] == "a" * 64
    events[1]["robot"][2]["robot_timestamp_ms"] = 1
    save()
    _, groups, rejected, _ = load_samples(path)
    assert len(groups) == 15 and rejected[0]["id"] == 1


def test_map_change_preserves_mount_transform():
    groups, X, Y = synthetic_groups()
    G = transform(Rotation.from_euler("xyz", [-20, 60, 110], degrees=True).as_matrix(), [-3, 2, 1])
    for group in groups:
        group.tracker = G @ group.tracker
    report = solve_groups(groups)
    np.testing.assert_allclose(report["T_ET"], X, atol=1e-8)
    np.testing.assert_allclose(report["T_WB"], G @ Y, atol=1e-8)


def test_noisy_data_has_nonzero_error_and_recovers_offset():
    groups, X, _ = synthetic_groups(noise=.0003)
    report = solve_groups(groups)
    error = report["metrics"]["validation"]["wrist_position_error_mm"]
    assert 0.05 < error["rmse"] < 2
    assert error["max"] >= error["rmse"] >= error["mean"]
    np.testing.assert_allclose(np.asarray(report["T_ET"])[:3, 3], X[:3, 3], atol=.002)


@pytest.mark.parametrize("kind", ["translation", "single_axis"])
def test_degenerate_motion_cannot_return_mount(kind):
    groups, _, _ = synthetic_groups(kind=kind)
    with pytest.raises(CalibrationError, match="姿态变化不足"):
        solve_groups(groups)


def test_held_out_error_is_not_fitted_away():
    groups, X, _ = synthetic_groups()
    for group in groups:
        if group.id in {3, 7, 12}:
            group.tracker[:3, 3] += [.04, 0, 0]
    report = solve_groups(groups, validation_ids={3, 7, 12})
    np.testing.assert_allclose(report["T_ET"], X, atol=1e-8)
    assert report["metrics"]["fit"]["wrist_position_error_mm"]["max"] < 1e-6
    assert report["metrics"]["validation"]["wrist_position_error_mm"]["rmse"] == pytest.approx(40)
    assert not report["numerical_checks_passed"]


def test_small_session_is_only_fit_not_independent_validation():
    groups, _, _ = synthetic_groups(5)
    report = solve_groups(groups)
    assert report["metrics"]["validation"] is None
    assert not report["numerical_checks_passed"]
    assert report["checks"]["fit_residuals_passed"]


def test_status20_diagnostic_and_complete_reports(tmp_path):
    path = tmp_path / "samples.jsonl"
    write_session(path, status=20)
    failed = solve_file(path)
    assert failed["status"] == "failed"
    assert len(failed["rejected_groups"]) == 16
    assert not (Path(failed["output_dir"]) / "calibration.json").exists()
    report = solve_file(path, include_status20=True)
    folder = Path(report["output_dir"])
    assert report["status"] == "review_required" and report["diagnostic_only"]
    assert report["numerical_checks_passed"] and not report["applied_to_replay"]
    assert {"calibration.json", "report.json", "report.txt", "errors.csv"} <= {f.name for f in folder.iterdir()}
    assert "独立验证组" in (folder / "report.txt").read_text()
    assert len((folder / "errors.csv").read_text(encoding="utf-8-sig").splitlines()) == 17


def test_input_rechecks_frames_and_honors_exclusion(tmp_path):
    path = tmp_path / "samples.jsonl"
    events = write_session(path)
    events[1]["eligible_for_calibration"] = True
    events[1]["tracker"][0]["tracking_status"] = 3
    events.insert(-1, {"event": "exclude", "id": 2})
    path.write_text("".join(json.dumps(e) + "\n" for e in events))
    _, groups, rejected, _ = load_samples(path, include_status20=True)
    assert [g.id for g in groups] == list(range(3, 17))
    assert {g["id"] for g in rejected} == {1, 2}


@pytest.mark.parametrize("change,match", [("unit", "单位"), ("map", "会话"), ("identity", "身份"), ("unfinished", "结束采集")])
def test_incompatible_or_unfinished_input_rejected(tmp_path, change, match):
    path = tmp_path / "samples.jsonl"
    events = write_session(path)
    if change == "unit":
        events[0]["position_unit"] = "mm"
    elif change == "map":
        events.insert(2, copy.deepcopy(events[0]))
    elif change == "identity":
        for f in events[2]["tracker"]:
            f["mac"] = "ANOTHER"
    else:
        events.pop()
    path.write_text("".join(json.dumps(e) + "\n" for e in events))
    with pytest.raises(CalibrationError, match=match):
        load_samples(path)


def test_cli_offline_result_and_existing_output_not_overwritten(tmp_path):
    path = tmp_path / "samples.jsonl"
    write_session(path)
    output = tmp_path / "result"
    command = [sys.executable, "solve_tron2_tracker_calibration.py", "--input", str(path), "--output-dir", str(output)]
    run = subprocess.run(command, capture_output=True, text=True, timeout=20)
    assert run.returncode == 0, run.stdout + run.stderr
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "passed"
    assert "RMSE" in run.stdout and "T_ET" in run.stdout
    again = subprocess.run(command, capture_output=True, text=True, timeout=20)
    assert again.returncode != 0
    assert json.loads((output / "report.json").read_text()) == report


def test_web_solver_uses_same_cli_and_exports_zip(tmp_path):
    scripts = Path(__file__).resolve().parent / "TransHandR0725/TransHandR/scripts"
    sys.path.insert(0, str(scripts))
    from wrist_calibration_web import WristCalibrationManager
    path = tmp_path / "session/samples.jsonl"
    write_session(path, status=20, mock=True)
    manager = WristCalibrationManager(lambda: pytest.fail("offline solver must not connect hardware"), tmp_path,
                                      mock=True, python=sys.executable)
    manager.state.update(file=str(path), phase="stopped")
    try:
        manager.command("solve", {"include_status20": True})
        deadline = time.monotonic() + 10
        while manager.status()["solution"]["phase"] == "running" and time.monotonic() < deadline:
            time.sleep(.05)
        solution = manager.status()["solution"]
        assert solution["phase"] == "review_required", solution
        assert solution["report"]["numerical_checks_passed"]
        with zipfile.ZipFile(io.BytesIO(manager.export_solution())) as zipped:
            assert len(zipped.namelist()) == 4
            assert json.loads(zipped.read("calibration.json"))["diagnostic_only"]
    finally:
        manager.shutdown()
