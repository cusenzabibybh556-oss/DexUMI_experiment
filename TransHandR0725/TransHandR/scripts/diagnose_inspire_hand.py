#!/usr/bin/env python3
"""Read and conservatively probe an Inspire RH56 hand over Modbus RTU."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
INSPIRE_SDK = ROOT / "inspire_hands"
if str(INSPIRE_SDK) not in sys.path:
    sys.path.insert(0, str(INSPIRE_SDK))

from inspire_hand import InspireHand  # noqa: E402
from inspire_hand.hand import Register  # noqa: E402


JOINT_NAMES = ("little", "ring", "middle", "index", "thumb_bend", "thumb_rotate")
MIN_ANGLE = 0
MAX_ANGLE = 1000
MAX_PROBE_DELTA = 50


def signed_short(value: int) -> int:
    """Interpret a Modbus 16-bit value as a signed short."""
    value = int(value) & 0xFFFF
    return value - 0x10000 if value & 0x8000 else value


def probe_target(current: int, delta: int) -> int:
    if not MIN_ANGLE <= current <= MAX_ANGLE:
        raise ValueError(f"invalid current angle: {current}")
    if delta == 0 or abs(delta) > MAX_PROBE_DELTA:
        raise ValueError(f"probe delta must be between 1 and {MAX_PROBE_DELTA} in magnitude")
    return min(MAX_ANGLE, max(MIN_ANGLE, current + delta))


def read_snapshot(hand: InspireHand) -> dict:
    angles = hand.get_finger_angles()
    raw_forces = hand.get_finger_forces()
    errors = hand.get_finger_errors()
    if angles is None or len(angles) != 6:
        raise RuntimeError("invalid ANGLE_ACT response")
    if raw_forces is None or len(raw_forces) != 6:
        raise RuntimeError("invalid FORCE_ACT response")
    if errors is None or len(errors) != 6:
        raise RuntimeError("invalid ERROR response")
    return {
        "timestamp_ns": time.time_ns(),
        "angles": [int(value) for value in angles],
        # Some RH56 firmware reports FORCE_ACT as a signed short.
        "forces_raw": [int(value) for value in raw_forces],
        "forces_signed": [signed_short(value) for value in raw_forces],
        "errors": [int(value) for value in errors],
    }


def wait_for_target(hand: InspireHand, channel: int, target: int,
                    timeout_s: float, tolerance: int = 8) -> tuple[dict, list[dict], bool]:
    deadline = time.monotonic() + timeout_s
    samples = []
    while True:
        snapshot = read_snapshot(hand)
        samples.append(snapshot)
        if any(snapshot["errors"]):
            raise RuntimeError(f"hand reported an error: {snapshot['errors']}")
        if abs(snapshot["angles"][channel] - target) <= tolerance:
            return snapshot, samples, True
        if time.monotonic() >= deadline:
            return snapshot, samples, False
        time.sleep(0.05)


def trajectory_summary(samples: Sequence[dict], channel: int) -> dict:
    values = [sample["angles"][channel] for sample in samples]
    return {
        "sample_count": len(values),
        "first": values[0] if values else None,
        "last": values[-1] if values else None,
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
    }


def run_probe(hand: InspireHand, channel: int, delta: int, speed: int,
              force: int, hold_s: float, timeout_s: float) -> dict:
    baseline = read_snapshot(hand)
    if any(baseline["errors"]):
        raise RuntimeError(f"refusing to move while errors are active: {baseline['errors']}")

    start = baseline["angles"][channel]
    target = probe_target(start, delta)
    if target == start:
        raise RuntimeError("probe target equals current angle; reverse the delta direction")

    old_speeds = hand.modbus.read_holding_registers(Register.SPEED_SET, 6)
    old_forces = hand.modbus.read_holding_registers(Register.FORCE_SET, 6)
    if old_speeds is None or len(old_speeds) != 6:
        raise RuntimeError("could not read current speed settings")
    if old_forces is None or len(old_forces) != 6:
        raise RuntimeError("could not read current force settings")

    moved = None
    returned = None
    move_samples: list[dict] = []
    return_samples: list[dict] = []
    try:
        hand.set_finger_speed(channel, speed)
        hand.set_finger_force(channel, force)
        hand.set_finger_angle(channel, target)
        moved, move_samples, move_reached = wait_for_target(
            hand, channel, target, timeout_s)
        time.sleep(max(0.0, hold_s))
    finally:
        try:
            # Return with the device's original force threshold while retaining
            # the conservative probe speed. Low probe force may not overcome
            # actuator friction in the reverse direction.
            hand.modbus.write_multiple_registers(Register.FORCE_SET, old_forces)
            hand.set_finger_angle(channel, start)
            returned, return_samples, return_reached = wait_for_target(
                hand, channel, start, timeout_s)
        finally:
            hand.modbus.write_multiple_registers(Register.SPEED_SET, old_speeds)
            hand.modbus.write_multiple_registers(Register.FORCE_SET, old_forces)

    return {
        "joint": JOINT_NAMES[channel],
        "channel": channel,
        "start": start,
        "target": target,
        "delta": target - start,
        "speed": speed,
        "force_limit": force,
        "moved": moved,
        "returned": returned,
        "move_reached": move_reached,
        "return_reached": return_reached,
        "move_summary": trajectory_summary(move_samples, channel),
        "return_summary": trajectory_summary(return_samples, channel),
        "move_samples": move_samples,
        "return_samples": return_samples,
    }


def parse_channel(value: str) -> int:
    try:
        channel = int(value)
    except ValueError:
        if value not in JOINT_NAMES:
            raise argparse.ArgumentTypeError(
                "channel must be 0..5 or one of " + ", ".join(JOINT_NAMES))
        channel = JOINT_NAMES.index(value)
    if channel not in range(6):
        raise argparse.ArgumentTypeError("channel must be in range 0..5")
    return channel


def bounded_int(name: str, value: int, minimum: int, maximum: int) -> int:
    value = int(value)
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be in range {minimum}..{maximum}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read an Inspire RH56 hand; motion is locked behind explicit safety flags.")
    parser.add_argument("--port", required=True, help="stable /dev/serial/by-path/... device")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--slave-id", type=int, default=1)
    parser.add_argument("--samples", type=int, default=3,
                        help="number of read-only snapshots")
    parser.add_argument("--sample-interval", type=float, default=0.1)
    parser.add_argument("--probe-channel", type=parse_channel)
    parser.add_argument("--delta", type=int, default=-20,
                        help="signed probe step, limited to +/-50")
    parser.add_argument("--speed", type=int, default=100)
    parser.add_argument("--force", type=int, default=100)
    parser.add_argument("--hold", type=float, default=0.2)
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument("--enable-motion", action="store_true")
    parser.add_argument("--confirm-free-space", default="",
                        help="must be exactly YES for a motion probe")
    parser.add_argument("--include-samples", action="store_true",
                        help="include every probe sample instead of summaries only")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.port.startswith("/dev/"):
        raise SystemExit("--port must refer to a device under /dev")
    if args.samples < 1:
        raise SystemExit("--samples must be at least 1")
    if args.probe_channel is not None:
        if not args.enable_motion or args.confirm_free_space != "YES":
            raise SystemExit(
                "motion probe refused: pass --enable-motion --confirm-free-space YES")
        try:
            probe_target(500, args.delta)
            bounded_int("speed", args.speed, 1, 200)
            bounded_int("force", args.force, 1, 200)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc

    hand = InspireHand(args.port, args.baudrate, args.slave_id)
    result = {"port": args.port, "joint_names": JOINT_NAMES, "snapshots": []}
    try:
        hand.open()
        for index in range(args.samples):
            result["snapshots"].append(read_snapshot(hand))
            if index + 1 < args.samples:
                time.sleep(max(0.0, args.sample_interval))
        if args.probe_channel is not None:
            result["probe"] = run_probe(
                hand, args.probe_channel, args.delta, args.speed,
                args.force, args.hold, args.timeout)
    finally:
        hand.close()

    probe = result.get("probe")
    if probe is not None and not args.include_samples:
        probe.pop("move_samples", None)
        probe.pop("return_samples", None)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if probe is not None and (not probe["move_reached"] or not probe["return_reached"]):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
