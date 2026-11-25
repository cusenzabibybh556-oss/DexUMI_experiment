#!/usr/bin/env python3
"""从完整原始会话生成严格 status=2 原配对子集，仅作离线辅助对比。

不改原文件、不重配被丢弃的帧、不接受 status18 参与位姿均值、不按残差选样本。
固定规则：至少10对、双方时间跨度至少1秒、接收时差<=100ms、静止半径<=3mm/1°。
同时继承原load_samples的身份、模型、时间戳、查询耗时、0.5秒缺帧检查。
"""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import uuid

import numpy as np

from capture_tron2_tracker_calibration import pose_spread
from solve_tron2_tracker_calibration import load_samples


POLICY = {
    "allowed_retained_status": 2, "allowed_source_statuses_for_quality_check": [2, 18],
    "minimum_pair_count": 10, "minimum_each_stream_span_seconds": 1.0,
    "maximum_receive_gap_ms": 100.0, "maximum_stream_gap_seconds": .5,
    "maximum_position_radius_mm": 3.0, "maximum_rotation_radius_deg": 1.0,
    "pairing_rule": "validate original nearest-receive pairs, discard non2 whole pairs, do not rematch",
    "thresholds_selected_using_residuals": False,
}


def original_pairs(sample):
    """核对采集时缓存的索引，再按原配对顺序去重；没有重新寻找status2邻帧。"""
    tracker, robot = sample["tracker"], sample["robot"]
    expected = []
    for ri, frame in enumerate(robot):
        if not tracker:
            continue
        ti = min(range(len(tracker)), key=lambda i: abs(tracker[i]["received_monotonic_ns"] - frame["received_monotonic_ns"]))
        gap = abs(tracker[ti]["received_monotonic_ns"] - frame["received_monotonic_ns"]) / 1e6
        if gap <= POLICY["maximum_receive_gap_ms"]:
            expected.append({"robot_index": ri, "tracker_index": ti, "receive_gap_ms": gap})
    saved = sample.get("pairs")
    if not isinstance(saved, list) or len(saved) != len(expected):
        raise ValueError("原始pairs缺失或数量与原始最近时间配对不一致")
    for cached, computed in zip(saved, expected):
        if (cached.get("robot_index") != computed["robot_index"]
                or cached.get("tracker_index") != computed["tracker_index"]
                or not np.isclose(cached.get("receive_gap_ms", float("nan")), computed["receive_gap_ms"], atol=1e-6, rtol=0)):
            raise ValueError("原始pairs索引或接收时差与原帧不一致")
    unique, used = [], set()
    for pair in expected:
        if pair["tracker_index"] not in used:
            used.add(pair["tracker_index"])
            unique.append(pair)
    return unique


def filter_sample(sample, source_index, source_quality_reason=None):
    original = deepcopy(sample)
    pairs = []
    diagnostic = {
        "id": sample["id"], "source_sample_index": source_index,
        "source_frame_status_counts": dict(Counter(str(t["tracking_status"]) for t in sample.get("tracker", []))),
        "source_pair_status_counts": {}, "kept_pair_count": 0, "kept_source_pairs": [],
        "source_started_monotonic_ns": sample["started_monotonic_ns"],
        "source_ended_monotonic_ns": sample["ended_monotonic_ns"], "reasons": [],
    }
    if source_quality_reason:
        diagnostic["reasons"].append("原窗口质量检查未通过：" + source_quality_reason)
    else:
        try:
            pairs = original_pairs(sample)
            diagnostic["source_pair_status_counts"] = dict(Counter(
                str(sample["tracker"][p["tracker_index"]]["tracking_status"]) for p in pairs))
        except (ValueError, KeyError, TypeError) as exc:
            diagnostic["reasons"].append(str(exc))
    retained = [pair for pair in pairs if sample["tracker"][pair["tracker_index"]]["tracking_status"] == 2]
    trackers = [deepcopy(sample["tracker"][pair["tracker_index"]]) for pair in retained]
    robots = [deepcopy(sample["robot"][pair["robot_index"]]) for pair in retained]
    diagnostic.update(kept_pair_count=len(retained), kept_source_pairs=retained,
                      discarded_pair_count=len(pairs) - len(retained))
    if len(retained) < POLICY["minimum_pair_count"]:
        diagnostic["reasons"].append("保留的原始status2配对不足10对")
    for name, frames in (("tracker", trackers), ("robot", robots)):
        times = np.asarray([f["received_monotonic_ns"] for f in frames], dtype=np.int64)
        span = float((times[-1] - times[0]) / 1e9) if len(times) >= 2 else 0.0
        diagnostic[name + "_retained_span_seconds"] = span
        if span < POLICY["minimum_each_stream_span_seconds"]:
            diagnostic["reasons"].append(name + "保留帧时间跨度不足1秒")
        if len(times):
            with_edges = np.r_[sample["started_monotonic_ns"], times, sample["ended_monotonic_ns"]]
            max_gap = float(np.diff(with_edges).max() / 1e9)
            diagnostic[name + "_maximum_gap_seconds"] = max_gap
            if max_gap > POLICY["maximum_stream_gap_seconds"]:
                diagnostic["reasons"].append(name + "子集在原窗口内有超过0.5秒的缺帧")
        stats = pose_spread(frames)
        diagnostic[name + "_retained_statistics"] = stats
        if stats and (stats["position_radius_mm"] > POLICY["maximum_position_radius_mm"]
                      or stats["rotation_radius_deg"] > POLICY["maximum_rotation_radius_deg"]):
            diagnostic["reasons"].append(name + "保留帧未静止：位置半径>3mm或旋转>1°")
    eligible = not diagnostic["reasons"] and not original.get("excluded", False)
    diagnostic["eligible"] = eligible
    original.update(tracker=trackers, robot=robots,
                    pairs=[{"robot_index": i, "tracker_index": i, "receive_gap_ms": pair["receive_gap_ms"]}
                           for i, pair in enumerate(retained)],
                    tracking_statuses=[2] if retained else [], tracker_frames=len(trackers), robot_frames=len(robots),
                    pair_count=len(retained), tracker_statistics=diagnostic["tracker_retained_statistics"],
                    robot_statistics=diagnostic["robot_retained_statistics"], reasons=diagnostic["reasons"],
                    eligible_for_calibration=eligible, calibration_validated=False,
                    excluded=not eligible, status2_subset=diagnostic)
    return original, diagnostic


def derive(input_path, output_dir):
    input_path, output_dir = Path(input_path).resolve(), Path(output_dir).resolve()
    # 允许18只为核查原窗口所有原帧的质量；其均值不进入派生数据。
    metadata, source_groups, source_rejected, source_digest = load_samples(input_path, diagnostic_statuses={18})
    source_bytes = input_path.read_bytes()
    if hashlib.sha256(source_bytes).hexdigest() != source_digest:
        raise ValueError("原始文件在质量检查期间发生变化；等待采集完全结束")
    source_good = {g.id for g in source_groups}
    reasons = {g["id"]: g["reason"] for g in source_rejected}
    events = [json.loads(line) for line in source_bytes.decode().splitlines() if line.strip()]
    result_events, diagnostics = [], []
    header = deepcopy(metadata)
    header["derived_data"] = {"kind": "strict_status2_original_pair_subset", "auxiliary_only": True,
                              "source_file": str(input_path), "source_sha256": source_digest,
                              "source_modified": False, "policy": POLICY}
    result_events.append(header)
    for event in events[1:]:
        if event["event"] != "sample":
            result_events.append(deepcopy(event))
            continue
        filtered, diagnostic = filter_sample(event, len(diagnostics),
            None if event["id"] in source_good else reasons.get(event["id"], "原始组不可用"))
        result_events.append(filtered)
        diagnostics.append(diagnostic)
    output_dir.mkdir(parents=True, exist_ok=False)
    output = output_dir / "samples_status2.jsonl"
    with output.open("x") as f:
        for event in result_events:
            f.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
    derived_metadata, groups, rejected, digest = load_samples(output)
    expected_ids = {row["id"] for row in diagnostics if row["eligible"]}
    actual_ids = {g.id for g in groups}
    if actual_ids != expected_ids:
        raise ValueError("派生数据未通过原求解器的严格复核：" + str(rejected))
    if any(g.statuses != [2] for g in groups):
        raise ValueError("派生数据出现非status2记录")
    report = {"auxiliary_only": True, "source_file": str(input_path), "source_sha256": source_digest,
              "output_file": str(output), "output_sha256": digest, "policy": POLICY,
              "groups": diagnostics, "source_group_quality_rejections": source_rejected,
              "retained_group_ids": sorted(actual_ids), "retained_group_count": len(groups),
              "source_sample_count": len(diagnostics), "raw_statuses_rewritten": False,
              "rematched_pairs": False, "hardware_connected": False, "applied_to_mapping": False,
              "strict_loader_rechecked": True}
    (output_dir / "filter_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or args.input.parent / "status2_subsets" / (
        datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8])
    try:
        report = derive(args.input, output_dir)
        print(f"严格status2原配对子集：{report['retained_group_count']}/{report['source_sample_count']}组")
        print(report["output_file"])
    except (ValueError, KeyError, TypeError, OSError, np.linalg.LinAlgError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
