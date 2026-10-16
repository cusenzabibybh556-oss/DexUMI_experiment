"""Versioned, per-target calibration storage for LinkerForce gloves."""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


STORE_VERSION = 2
POSE_KEYS = ("original", "opose", "fist")


def _valid_pose(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 21
        and all(isinstance(item, (int, float)) and math.isfinite(float(item)) for item in value)
    )


def valid_entry(entry: Any) -> bool:
    if not isinstance(entry, dict) or not all(_valid_pose(entry.get(key)) for key in POSE_KEYS):
        return False
    return math.dist(entry["original"], entry["fist"]) > 3.0


class CalibrationStore:
    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser()

    @staticmethod
    def key(target_model: str, hand_side: str) -> str:
        model = str(target_model).strip().lower()
        side = str(hand_side).strip().lower()
        if not model or side not in {"left", "right"}:
            raise ValueError("invalid calibration profile")
        return f"{model}:{side}"

    def _read(self) -> Dict[str, Any]:
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, TypeError, json.JSONDecodeError):
            value = {}
        if value.get("version") != STORE_VERSION or not isinstance(value.get("profiles"), dict):
            return {"version": STORE_VERSION, "profiles": {}}
        return value

    def _write(self, value: Dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.path)

    def get(self, target_model: str, hand_side: str) -> Optional[Dict[str, Any]]:
        entry = self._read()["profiles"].get(self.key(target_model, hand_side))
        return dict(entry) if valid_entry(entry) else None

    def status(self, target_model: str, hand_side: str) -> Dict[str, Any]:
        key = self.key(target_model, hand_side)
        entry = self._read()["profiles"].get(key)
        return {
            "key": key,
            "valid": valid_entry(entry),
            "timestamp": entry.get("timestamp", "") if isinstance(entry, dict) else "",
            "glove_version": entry.get("glove_version", "") if isinstance(entry, dict) else "",
        }

    def save(
        self,
        target_model: str,
        hand_side: str,
        original: list,
        opose: list,
        fist: list,
        glove_version: str = "",
    ) -> Dict[str, Any]:
        entry = {
            "timestamp": datetime.now().astimezone().isoformat(),
            "glove_version": str(glove_version or ""),
            "original": [float(value) for value in original],
            "opose": [float(value) for value in opose],
            "fist": [float(value) for value in fist],
        }
        if not valid_entry(entry):
            raise ValueError("calibration poses are incomplete or too similar")
        data = self._read()
        data["profiles"][self.key(target_model, hand_side)] = entry
        self._write(data)
        return entry

    def migrate_legacy(self, legacy_path: str | Path) -> bool:
        """Import the old shared cache as O6-only profiles without overwriting v2 data."""
        try:
            legacy = json.loads(Path(legacy_path).read_text(encoding="utf-8"))
        except (OSError, TypeError, json.JSONDecodeError):
            return False
        data = self._read()
        changed = False
        for side, suffix in (("left", "l"), ("right", "r")):
            key = self.key("o6", side)
            if key in data["profiles"]:
                continue
            entry = {
                "timestamp": str(legacy.get("timestamp", "")),
                "glove_version": "legacy",
                "original": legacy.get(f"jointangleoriginal_{suffix}"),
                "opose": legacy.get(f"jointangleopose_{suffix}"),
                "fist": legacy.get(f"jointanglefist_{suffix}"),
            }
            if valid_entry(entry):
                data["profiles"][key] = entry
                changed = True
        if changed:
            self._write(data)
        return changed
