#!/usr/bin/env python3
"""Read-only O6 CAN/tactile probe.

This script does not send motion commands. It sends the same kind of empty
query frames used by the LinkerHand SDK, plus optional tactile candidate
queries, then prints raw CAN replies so we can identify newer tactile firmware.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import can
except ImportError as exc:  # pragma: no cover - hardware environment only
    raise SystemExit(
        "python-can is not installed. Try: python3 -m pip install python-can"
    ) from exc


HAND_IDS = {
    "left": 0x28,
    "right": 0x27,
}

KNOWN_READ_QUERIES = [
    ("version", 0x64, []),
    ("version_alt", 0xC2, []),
    ("serial_number", 0xC0, []),
    ("touch_type", 0xB0, []),
    ("normal_force", 0x20, []),
    ("tangential_force", 0x21, []),
    ("tangential_force_dir", 0x22, []),
    ("approach_inc", 0x23, []),
]

FINGER_TOUCH_FRAMES = [
    ("thumb_touch", 0xB1),
    ("index_touch", 0xB2),
    ("middle_touch", 0xB3),
    ("ring_touch", 0xB4),
    ("little_touch", 0xB5),
]

PALM_CANDIDATE_FRAMES = [
    ("palm_candidate_b6", 0xB6),
    ("touch_candidate_b7", 0xB7),
    ("touch_candidate_b8", 0xB8),
]

TOUCH_CODES = [0xC6, 0xA4]


@dataclass(frozen=True)
class Query:
    name: str
    command: int
    payload: tuple[int, ...]

    @property
    def data(self) -> list[int]:
        return [self.command, *self.payload]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe LinkerHand O6 SN/version/tactile CAN replies without moving the hand."
    )
    parser.add_argument("--hand", choices=sorted(HAND_IDS), default="left")
    parser.add_argument("--can", default="can0", help="SocketCAN interface")
    parser.add_argument("--bitrate", type=int, default=1_000_000)
    parser.add_argument(
        "--listen",
        type=float,
        default=0.08,
        help="Seconds to listen after each query",
    )
    parser.add_argument(
        "--passive",
        type=float,
        default=0.5,
        help="Seconds to passively listen before active queries",
    )
    parser.add_argument(
        "--touch",
        action="store_true",
        help="Query known five-finger tactile matrix frames B1-B5",
    )
    parser.add_argument(
        "--probe-palm",
        action="store_true",
        help="Also query candidate palm/new tactile frames B6-B8",
    )
    parser.add_argument(
        "--include-state",
        action="store_true",
        help="Also query current joint state with empty 0x01 read frame",
    )
    return parser.parse_args()


def make_queries(args: argparse.Namespace) -> list[Query]:
    queries = [Query(name, command, tuple(payload)) for name, command, payload in KNOWN_READ_QUERIES]
    if args.include_state:
        queries.append(Query("joint_state", 0x01, ()))
    if args.touch:
        for name, command in FINGER_TOUCH_FRAMES:
            queries.append(Query(f"{name}_scalar", command, ()))
            for code in TOUCH_CODES:
                queries.append(Query(f"{name}_matrix_{code:02X}", command, (code,)))
    if args.probe_palm:
        for name, command in PALM_CANDIDATE_FRAMES:
            queries.append(Query(f"{name}_scalar", command, ()))
            for code in TOUCH_CODES:
                queries.append(Query(f"{name}_matrix_{code:02X}", command, (code,)))
    return queries


def drain(bus: "can.BusABC", duration_s: float, expected_id: int) -> list["can.Message"]:
    deadline = time.monotonic() + max(duration_s, 0.0)
    messages: list[can.Message] = []
    while time.monotonic() < deadline:
        timeout = max(0.0, deadline - time.monotonic())
        msg = bus.recv(timeout=timeout)
        if msg is None:
            break
        if msg.arbitration_id == expected_id:
            messages.append(msg)
    return messages


def format_data(data: Iterable[int]) -> str:
    return " ".join(f"{value:02X}" for value in data)


def ascii_tail(data: bytes) -> str:
    chars = []
    for value in data:
        if 32 <= value <= 126:
            chars.append(chr(value))
        else:
            chars.append(".")
    return "".join(chars)


def print_rx(label: str, messages: list["can.Message"]) -> None:
    if not messages:
        print(f"{label}: no reply")
        return
    print(f"{label}: {len(messages)} replies")
    for msg in messages:
        data = bytes(msg.data)
        command = data[0] if data else -1
        print(
            f"  rx id=0x{msg.arbitration_id:X} cmd=0x{command:02X} "
            f"len={len(data)} data=[{format_data(data)}] ascii='{ascii_tail(data)}'"
        )


def summarize(all_messages: list["can.Message"]) -> None:
    by_command: dict[int, list[bytes]] = defaultdict(list)
    serial_chunks: dict[int, bytes] = {}

    for msg in all_messages:
        data = bytes(msg.data)
        if not data:
            continue
        command = data[0]
        payload = data[1:]
        by_command[command].append(payload)
        if command == 0xC0 and len(payload) >= 2:
            serial_chunks[payload[0]] = payload[1:]

    print("\n=== Summary ===")
    if serial_chunks:
        serial_bytes = b"".join(serial_chunks[index] for index in sorted(serial_chunks))
        serial = serial_bytes.rstrip(b"\x00\xff").decode("ascii", errors="replace")
        print(f"serial_from_0xC0: {serial}")
    else:
        print("serial_from_0xC0: not found")

    for command in sorted(by_command):
        payloads = by_command[command]
        lengths = sorted({len(payload) + 1 for payload in payloads})
        print(
            f"cmd=0x{command:02X}: replies={len(payloads)} frame_lengths={lengths} "
            f"first_payload=[{format_data(payloads[0])}]"
        )

    for command in (0xB6, 0xB7, 0xB8):
        status = "FOUND" if command in by_command else "not found"
        print(f"candidate_palm_or_extra_touch_0x{command:02X}: {status}")


def main() -> int:
    args = parse_args()
    hand_id = HAND_IDS[args.hand]
    queries = make_queries(args)

    print("Read-only O6 CAN probe")
    print(f"hand={args.hand} can={args.can} expected_reply_id=0x{hand_id:X}")
    print("Stop o6_bridge/vendor SDK before running this, otherwise another process may consume replies.")

    bus = can.interface.Bus(
        channel=args.can,
        interface="socketcan",
        bitrate=args.bitrate,
        receive_own_messages=False,
    )
    all_messages: list[can.Message] = []
    try:
        passive = drain(bus, args.passive, hand_id)
        print_rx("passive_listen", passive)
        all_messages.extend(passive)

        for query in queries:
            msg = can.Message(
                arbitration_id=hand_id,
                data=query.data,
                is_extended_id=False,
            )
            print(f"\nTX {query.name}: id=0x{hand_id:X} data=[{format_data(query.data)}]")
            bus.send(msg)
            replies = drain(bus, args.listen, hand_id)
            print_rx(query.name, replies)
            all_messages.extend(replies)
            time.sleep(0.01)
    finally:
        bus.shutdown()

    summarize(all_messages)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
