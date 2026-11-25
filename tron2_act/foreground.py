"""Validate reviewed, lossless RGB caches against their original camera rows."""
import json
from pathlib import Path

import h5py
import numpy as np

FOREGROUND_CONTRACT = "sam3-top-causal-tracking-fusion-v2"
FOREGROUND_KEY = "foreground/rgb"


def validate_foreground(source_path, source, root, source_stamp, allow_experimental=False):
    path = Path(root) / Path(source_path).stem / "foreground.hdf5"
    with h5py.File(path, "r") as cache:
        contract = cache.attrs.get("contract")
        if contract not in (FOREGROUND_CONTRACT, "top-foreground-student-v1") or not cache.attrs.get("causal", False):
            raise ValueError("Unsupported foreground preprocessing contract")
        status = cache.attrs.get("review_status")
        if status != "accepted" and not (allow_experimental and status == "experimental"):
            raise ValueError("Foreground cache has not passed review")
        if contract == "top-foreground-student-v1":
            digest = str(cache.attrs.get("model_sha256", ""))
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("Missing foreground model fingerprint")
            metadata = json.loads(cache.attrs["model_metadata"])
            if len(metadata["input_size"]) != 2:
                raise ValueError("Invalid foreground model preprocessing")
        if json.loads(cache.attrs["source_stamp"]) != source_stamp:
            raise ValueError("Foreground source fingerprint does not match original episode")
        images = cache[FOREGROUND_KEY]
        if images.shape != source["observations/images/d435i_rgb"].shape or images.dtype != np.uint8:
            raise ValueError("Foreground RGB shape/dtype differs from original")
        if not np.array_equal(cache["source/frame_index"][:], np.arange(len(images))):
            raise ValueError("Foreground rows differ from original camera rows")
        for cached, original in (("frame_number", "frame_number"), ("unified_timestamp_ns", "unified_timestamp_ns"), ("camera_valid", "valid")):
            if not np.array_equal(cache[f"source/{cached}"][:], source[f"camera/d435i/{original}"][:]):
                raise ValueError(f"Foreground camera {original} differs from original episode")
    return str(path.resolve())
