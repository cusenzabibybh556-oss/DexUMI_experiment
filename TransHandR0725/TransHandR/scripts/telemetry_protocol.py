"""Validation helpers for TransHandR UDP telemetry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np


PROTOCOL_V2 = "transhandr.telemetry.v2"
KINDS = {"operator", "action", "state", "tactile", "palm", "force"}


@dataclass(frozen=True)
class TelemetryMessage:
    kind: str
    values: np.ndarray
    source: str
    robot_model: str
    hand_side: str
    source_timestamp_ns: int
    unified_timestamp_ns: int
    host_receive_timestamp_ns: int
    sequence: int


def make_packet(
    kind: str,
    values: Any,
    source: str,
    robot_model: str,
    hand_side: str,
    source_timestamp_ns: int,
    host_receive_timestamp_ns: int,
    sequence: int,
    unified_timestamp_ns: Optional[int] = None,
) -> Dict[str, Any]:
    if kind not in KINDS:
        raise ValueError(f"unsupported telemetry kind: {kind}")
    packet = {
        "protocol": PROTOCOL_V2,
        "kind": kind,
        "values": np.asarray(values).tolist(),
        "source": str(source),
        "robot_model": str(robot_model).lower(),
        "hand_side": str(hand_side).lower(),
        "source_timestamp_ns": int(source_timestamp_ns),
        "host_receive_timestamp_ns": int(host_receive_timestamp_ns),
        "sequence": int(sequence),
    }
    if unified_timestamp_ns is not None:
        packet["unified_timestamp_ns"] = int(unified_timestamp_ns)
    return packet


def parse_v2_packet(
    packet: Mapping[str, Any],
    expected_dims: Optional[Mapping[str, int]] = None,
) -> TelemetryMessage:
    if packet.get("protocol") != PROTOCOL_V2:
        raise ValueError("not a transhandr.telemetry.v2 packet")
    kind = str(packet.get("kind", ""))
    if kind not in KINDS:
        raise ValueError(f"unsupported telemetry kind: {kind}")
    values = np.asarray(packet.get("values"), dtype=np.float32)
    if kind == "tactile":
        if values.shape != (5, 12, 6):
            raise ValueError(f"tactile shape must be (5, 12, 6), got {values.shape}")
    elif kind == "palm":
        if values.shape != (14, 15):
            raise ValueError(f"palm shape must be (14, 15), got {values.shape}")
    elif values.ndim != 1:
        raise ValueError(f"{kind} values must be one-dimensional")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{kind} contains a non-finite value")
    if kind == "palm" and (
        not np.any(values >= 0) or np.any(values < -1) or np.any(values > 255)
        or not np.all(values == np.rint(values))
    ):
        raise ValueError("palm must contain valid integer taxels in [-1, 255]")
    expected = (expected_dims or {}).get(kind)
    if expected is not None and values.size != expected:
        raise ValueError(f"{kind} dimension must be {expected}, got {values.size}")
    side = str(packet.get("hand_side", "")).lower()
    if side not in {"left", "right"}:
        raise ValueError(f"invalid hand_side: {side}")
    source_ts = int(packet["source_timestamp_ns"])
    host_ts = int(packet["host_receive_timestamp_ns"])
    unified_ts = int(packet.get("unified_timestamp_ns", host_ts))
    if source_ts < 0 or unified_ts <= 0 or host_ts <= 0:
        raise ValueError("timestamps must be non-negative")
    return TelemetryMessage(
        kind=kind,
        values=values,
        source=str(packet.get("source", "")),
        robot_model=str(packet.get("robot_model", "")).lower(),
        hand_side=side,
        source_timestamp_ns=source_ts,
        unified_timestamp_ns=unified_ts,
        host_receive_timestamp_ns=host_ts,
        sequence=int(packet.get("sequence", 0)),
    )


def tactile_summary(matrix: Any, contact_threshold: float = 1.0) -> Dict[str, np.ndarray]:
    values = np.asarray(matrix, dtype=np.float32)
    if values.shape != (5, 12, 6):
        raise ValueError(f"tactile shape must be (5, 12, 6), got {values.shape}")
    flat = values.reshape(5, -1)
    valid = np.isfinite(flat) & (flat >= 0)
    if not np.all(np.any(valid, axis=1)):
        raise ValueError("tactile sample contains a finger with no valid cells")
    valid_count = np.sum(valid, axis=1)
    return {
        "matrix": values,
        "max": np.max(np.where(valid, flat, -np.inf), axis=1).astype(np.float32),
        "mean": (np.sum(np.where(valid, flat, 0), axis=1) / valid_count).astype(np.float32),
        "contact_area": np.sum(
            valid & (flat > float(contact_threshold)), axis=1
        ).astype(np.int16),
    }


def legacy_messages(
    packet: Mapping[str, Any],
    host_receive_timestamp_ns: int,
) -> List[TelemetryMessage]:
    """Translate the existing flat VR packet into the v2 in-memory model."""
    messages: List[TelemetryMessage] = []
    side = str(packet.get("vision_hand_side", "right")).lower()
    common = dict(source="legacy_udp", robot_model=str(packet.get("hand_type", "")),
                  hand_side=side, sequence=int(packet.get("sequence", 0)))
    if packet.get("command") is not None:
        ts = int(packet.get("command_timestamp_ns", host_receive_timestamp_ns))
        messages.append(TelemetryMessage("action", np.asarray(packet["command"], np.float32),
                                         source_timestamp_ns=ts,
                                         unified_timestamp_ns=ts,
                                         host_receive_timestamp_ns=ts, **common))
    if packet.get("state") is not None:
        ts = int(packet.get("state_timestamp_ns", host_receive_timestamp_ns))
        messages.append(TelemetryMessage("state", np.asarray(packet["state"], np.float32),
                                         source_timestamp_ns=ts,
                                         unified_timestamp_ns=ts,
                                         host_receive_timestamp_ns=ts, **common))
    return messages
