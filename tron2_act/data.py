"""Read-only HDF5 sampling with real camera anchors and jointly retimed labels."""
from functools import lru_cache
from pathlib import Path
import hashlib
import json

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from tracker_replay_mapping import load_calibration
from .config import CONTRACT, Config
from .geometry import decode_quaternions, hold_commands, interpolate, matrix_to_rotation6d
from .foreground import FOREGROUND_KEY, validate_foreground

IMAGE_KEY = "observations/images/d435i_rgb"
HAND_NAMES = ["thumb_flexion", "thumb_abduction", "index_flexion", "middle_flexion", "ring_flexion", "pinky_flexion"]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def source_stamp(path):
    p = Path(path)
    s = p.stat()
    return {"path": str(p.resolve()), "size": s.st_size, "mtime_ns": s.st_mtime_ns}


def verify_duplicate_master_rows(source, master):
    """Only accept exact copies of the observations/commands consumed by ACT."""
    repeated = np.flatnonzero(np.diff(master) == 0)
    keys = [IMAGE_KEY, "camera/d435i/frame_number", "camera/d435i/unified_timestamp_ns", "camera/d435i/valid",
            "vive_tracker/pos", "vive_tracker/quat", "vive_tracker/valid", "vive_tracker/unified_timestamp_ns",
            "dexterous_hand/joint_positions", "dexterous_hand/state_valid", "dexterous_hand/unified_timestamp_ns",
            "action/command", "action/command_timestamp_ns", "action/valid"]
    keys += [key for key in ["camera/d405/frame_number", "camera/d405/unified_timestamp_ns", "observations/images/rgb", "observations/images/d405_rgb"] if key in source]
    for index in repeated:
        for key in keys:
            if not np.array_equal(source[key][index], source[key][index + 1], equal_nan=True):
                raise ValueError(f"Conflicting duplicate master timestamp at rows {index},{index + 1}: {key}")
    return (repeated + 1).tolist()


def unique_state_times(name, times, arrays, allow_duplicates):
    delta = np.diff(times)
    if len(times) < 2 or np.any(delta < 0):
        raise ValueError(f"{name}: need >=2 ordered valid states")
    repeated = np.flatnonzero(delta == 0)
    if len(repeated) and not allow_duplicates:
        raise ValueError(f"{name}: need >=2 ordered valid states")
    for index in repeated:
        if any(not np.array_equal(a[index], a[index + 1]) for a in arrays):
            raise ValueError(f"{name}: conflicting states share one timestamp")
    keep = np.r_[True, delta > 0]
    if int(keep.sum()) < 2:
        raise ValueError(f"{name}: need >=2 distinct valid states")
    return times[keep], [a[keep] for a in arrays], len(repeated)


class Episode:
    def __init__(self, path, config, mapping):
        self.path, self.config = str(path), config
        self.foreground_file = None
        self.timestamp_audit = {"duplicate_master_rows": [], "tracker_samples_removed": 0, "o6_samples_removed": 0}
        with h5py.File(path, "r") as f:
            if str(f.attrs.get("hand_type")) != "o6" or str(f.attrs.get("hand_side")) != "left":
                raise ValueError("Expected O6 left-hand episode")
            master = f["timestamps/master_timestamp_ns"][:]
            if len(master) < 2 or np.any(np.diff(master) < 0):
                raise ValueError("Master timestamps are not strictly increasing")
            if np.any(np.diff(master) == 0):
                if not config.deduplicate_identical_timestamps:
                    raise ValueError("Master timestamps are not strictly increasing")
                self.timestamp_audit["duplicate_master_rows"] = verify_duplicate_master_rows(f, master)
            self.origin = int(master[0])
            relative = lambda a: (np.asarray(a, np.int64) - self.origin) / 1e9
            self.image_times = relative(f["camera/d435i/unified_timestamp_ns"][:])
            image_valid = f["camera/d435i/valid"][:].astype(bool)
            frame_ids = f["camera/d435i/frame_number"][:]
            if f[IMAGE_KEY].shape != (len(master), 480, 640, 3):
                shape = f[IMAGE_KEY].shape
                if len(shape) != 4 or shape[0] != len(master) or shape[-1] != 3:
                    raise ValueError("Expected raw RGB frames [T,H,W,3]")
            if f[IMAGE_KEY].dtype != np.uint8:
                raise ValueError("Expected uint8 RGB images")
            raw_order = str(f["vive_tracker"].attrs.get("quat_order", ""))
            if "".join(x for x in raw_order.lower() if x.isalpha()) != "wxyz":
                raise ValueError("Tracker storage order must explicitly declare WXYZ")
            unit = str(f["vive_tracker"].attrs.get("position_unit", ""))
            if unit not in {"meter", "meters", "m"}:
                raise ValueError("Tracker positions must declare meters")
            p = f["vive_tracker/pos"][:, 0].astype(float)
            q = f["vive_tracker/quat"][:, 0].astype(float)
            valid = f["vive_tracker/valid"][:, 0].astype(bool)
            valid &= np.isfinite(p).all(1) & np.isfinite(q).all(1) & (np.linalg.norm(q, axis=1) > 1e-6)
            self.pose_t = relative(f["vive_tracker/unified_timestamp_ns"][:, 0])[valid]
            axis = np.asarray(mapping["T_BW"], float)[:3, :3]
            self.p = p[valid] @ axis.T
            self.r = axis @ decode_quaternions(q[valid], config.quaternion_format)
            h = f["dexterous_hand/joint_positions"][:].astype(float)
            hv = f["dexterous_hand/state_valid"][:].astype(bool)
            hv &= np.isfinite(h).all(1) & (h >= 0).all(1) & (h <= 255).all(1)
            self.hand_t = relative(f["dexterous_hand/unified_timestamp_ns"][:])[hv]
            self.hand = h[hv]
            commands = f["action/command"][:].astype(float)
            command_ns = f["action/command_timestamp_ns"][:]
            command_valid = f["action/valid"][:].astype(bool)
            if commands.shape != (len(master), 6) or command_ns.shape != master.shape:
                raise ValueError("Expected O6 command [T,6] and command timestamps [T]")
            command_valid &= np.isfinite(commands).all(1) & (commands >= 0).all(1) & (commands <= 255).all(1)
            command_valid &= (command_ns >= 0) & (command_ns <= master)
            command_ns, commands = command_ns[command_valid], commands[command_valid]
            if not len(command_ns) or np.any(np.diff(command_ns) < 0):
                raise ValueError("Need valid time-ordered O6 action commands")
            repeated = np.flatnonzero(np.diff(command_ns) == 0)
            if len(repeated) and np.any(commands[repeated] != commands[repeated + 1]):
                raise ValueError("Conflicting O6 commands share one timestamp")
            keep = np.r_[True, np.diff(command_ns) > 0]
            self.command_t = relative(command_ns[keep])
            self.commands = commands[keep]
            self.record_end = float(relative(master)[-1])
            self.command_max_age_s = float(f.attrs.get("action_max_age_ms", 0)) / 1000
            if not np.isfinite(self.command_max_age_s) or self.command_max_age_s < 0:
                raise ValueError("Invalid recorded action_max_age_ms")
            self.source_task = str(f.attrs.get("task_name", "unknown"))
            if config.foreground_root:
                self.foreground_file = validate_foreground(path, f, config.foreground_root, source_stamp(path), config.allow_experimental_foreground)
        self.pose_t, pose_arrays, count = unique_state_times("Tracker", self.pose_t, [self.p, self.r], config.deduplicate_identical_timestamps)
        self.p, self.r = pose_arrays
        self.timestamp_audit["tracker_samples_removed"] = count
        self.hand_t, hand_arrays, count = unique_state_times("O6", self.hand_t, [self.hand], config.deduplicate_identical_timestamps)
        self.hand = hand_arrays[0]
        self.timestamp_audit["o6_samples_removed"] = count
        # Reused camera frames are not additional observations. Keep their first valid row.
        self.rows = []
        seen = set()
        last_time = -np.inf
        for i in np.flatnonzero(image_valid):
            if int(frame_ids[i]) not in seen and self.image_times[i] > last_time:
                self.rows.append(int(i))
                seen.add(int(frame_ids[i]))
                last_time = self.image_times[i]

    def sample(self, row):
        c = self.config
        t = self.image_times[row]
        step = c.time_scale / c.action_hz  # source seconds per physical action step
        offsets = np.r_[-np.arange(c.history_steps, 0, -1), 0, np.arange(1, c.chunk_size + 1)]
        times = t + offsets * step
        p, pv = interpolate(self.pose_t, self.p, times, c.max_source_gap_s)
        r, rv = interpolate(self.pose_t, self.r, times, c.max_source_gap_s, rotations=True)
        # q_pos is actual O6 feedback at the observation. It is never the action label.
        current_hand, current_hand_valid = interpolate(self.hand_t, self.hand, np.array([t]), c.max_source_gap_s)
        good = pv & rv
        current = c.history_steps
        if not good[:current + 1].all() or not current_hand_valid[0]:
            return None
        commands, command_valid = hold_commands(self.command_t, self.commands, times[current + 1:],
                                                self.record_end, self.command_max_age_s)
        future = good[current + 1:] & command_valid
        # Never bridge an invalid interval or resume labels beyond it.
        prefix = np.logical_and.accumulate(future)
        if int(prefix.sum()) < c.min_future_steps:
            return None
        dp = p - p[current]
        dr = r @ r[current].T
        pose = np.concatenate([dp, matrix_to_rotation6d(dr)], axis=-1)
        state = np.concatenate([pose[:current].reshape(-1), current_hand[0]]).astype(np.float32)
        action = np.concatenate([pose[current + 1:], commands], axis=-1).astype(np.float32)
        action[~prefix] = 0
        return state, action, ~prefix


def prepare(config, output):
    c = config.validate()
    out = Path(output)
    if out.exists():
        raise FileExistsError(f"Output already exists: {out}")
    out.mkdir(parents=True)
    mapping = load_calibration(c.calibration, "left")
    if c.quaternion_format == "pyvut-legacy" and mapping["tracker_quaternion_convention"] != "raw-xyzw":
        raise ValueError("Legacy decoder requires the matching raw-xyzw calibration")
    paths = sorted(Path(c.data_root).glob("episode_*.hdf5"))
    if c.episodes is not None:
        wanted = set(c.episodes)
        paths = [p for p in paths if int(p.stem.split("_")[-1]) in wanted]
        if {int(p.stem.split("_")[-1]) for p in paths} != wanted:
            raise ValueError("Some explicitly requested episode files are missing")
    if not paths:
        raise ValueError("No episode files selected")
    accepted, audit = [], []
    for path in paths:
        row = {"episode_id": int(path.stem.split("_")[-1]), "source": source_stamp(path)}
        try:
            ep = Episode(path, c, mapping)
            anchors = [i for i in ep.rows if ep.sample(i) is not None]
            if not anchors:
                raise ValueError("No complete observation with a valid future prefix")
            row.update(status="accepted", image_rows=anchors, samples=len(anchors), source_task=ep.source_task)
            row["timestamp_audit"] = ep.timestamp_audit
            if ep.foreground_file:
                row["foreground"] = source_stamp(ep.foreground_file)
            accepted.append(row)
        except (ValueError, KeyError, OSError) as exc:
            row.update(status="rejected", reason=str(exc))
        audit.append(row)
        print(f"episode {row['episode_id']}: {row['status']} ({row.get('samples', 0)} samples)", flush=True)
    write_json(out / "audit.json", audit)
    if len(accepted) < 3:
        raise ValueError("Need >=3 usable episodes for independent train/validation/test splits; see audit.json")
    shuffled = np.random.default_rng(c.seed).permutation(len(accepted))
    ntest = max(1, int(len(accepted) * .1))
    nval = max(1, int(len(accepted) * .1))
    split = {}
    for name, indices in [("test", shuffled[:ntest]), ("val", shuffled[ntest:ntest+nval]),
                          ("train", shuffled[ntest+nval:])]:
        split[name] = [accepted[int(i)] for i in indices]
    semantic = {"contract": CONTRACT, "config": c.to_dict(), "mapping": mapping,
                "rotation6d": "first two matrix columns, column-major concatenation",
                "wrist_source": "tracker-origin proxy, R_BW only; T_ET and world translation not applied",
                "action": "spatial delta from current camera anchor; rotations left-multiply anchor",
                "o6_state_source": "dexterous_hand/joint_positions: real O6 SDK feedback",
                "o6_action_source": "action/command: glove-retargeted, bridge-limited sent O6 targets",
                "o6_action_sampling": "latest-before at command_timestamp_ns; zero-order hold, recorded expiry",
                "o6_names": HAND_NAMES,
                "image": "real D435i RGB frame at its unified timestamp; no synthesized frames",
                "time": "source_offset = physical_offset * time_scale; no second slowdown",
                "splits": split}
    if c.foreground_root:
        semantic["image"] = "automatic causal foreground; original RGB pixels and camera timestamps, black background"
        semantic["foreground_quality"] = "experimental allowed" if c.allow_experimental_foreground else "accepted only"
    semantic["manifest_sha256"] = hashlib.sha256(json.dumps(semantic, sort_keys=True).encode()).hexdigest()
    write_json(out / "manifest.json", semantic)
    train = ACTDataset(out, "train", normalized=False, images=False)
    stats = compute_stats(train, c.normalization_floor)
    write_json(out / "stats.json", stats)
    summary = {"selected": len(paths), "accepted": len(accepted), "rejected": len(paths)-len(accepted),
               "splits": {k: {"episodes": len(v), "samples": sum(x["samples"] for x in v)} for k, v in split.items()},
               "state_dim": c.state_dim, "action_shape": [c.chunk_size, c.action_dim],
               "physical_horizon_s": c.chunk_size / c.action_hz}
    write_json(out / "summary.json", summary)
    return summary


class ACTDataset(Dataset):
    def __init__(self, prepared, split, normalized=True, images=True):
        self.prepared = Path(prepared)
        self.manifest = json.loads((self.prepared / "manifest.json").read_text())
        if self.manifest["contract"] != CONTRACT:
            raise ValueError("Unsupported action contract")
        self.config = Config(**self.manifest["config"]).validate()
        self.mapping = self.manifest["mapping"]
        self.normalized, self.images = normalized, images
        self.stats = json.loads((self.prepared / "stats.json").read_text()) if normalized else None
        self.items = []
        self.foreground_files = {}
        for episode in self.manifest["splits"][split]:
            stamp = episode["source"]
            if source_stamp(stamp["path"]) != stamp:
                raise ValueError(f"Source file changed since prepare: {stamp['path']}")
            if self.config.foreground_root:
                foreground = episode.get("foreground")
                if not foreground or source_stamp(foreground["path"]) != foreground:
                    raise ValueError("Foreground cache changed or is missing; run prepare again")
                self.foreground_files[stamp["path"]] = foreground["path"]
            self.items.extend((stamp["path"], row) for row in episode["image_rows"])
        if not self.items:
            raise ValueError(f"No samples in split {split}")

    @lru_cache(maxsize=4)
    def episode(self, path):
        return Episode(path, self.config, self.mapping)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        path, row = self.items[index]
        sample = self.episode(path).sample(row)
        if sample is None:
            raise RuntimeError("Prepared sample no longer valid; rebuild dataset")
        state, action, pad = sample
        if self.normalized:
            state = (state - np.asarray(self.stats["state_mean"], np.float32)) / np.asarray(self.stats["state_std"], np.float32)
            action = (action - np.asarray(self.stats["action_mean"], np.float32)) / np.asarray(self.stats["action_std"], np.float32)
            action[pad] = 0
        result = {"state": torch.from_numpy(state), "action": torch.from_numpy(action), "is_pad": torch.from_numpy(pad)}
        if self.images:
            image_path = self.foreground_files.get(path, path)
            image_key = FOREGROUND_KEY if self.config.foreground_root else IMAGE_KEY
            with h5py.File(image_path, "r") as f:
                rgb = f[image_key][row]
            if not np.any(rgb):
                raise ValueError(f"All-zero RGB placeholder: {path} row {row}")
            height, width = self.config.image_size
            rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
            result["image"] = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float().div_(255)
        return result


def compute_stats(dataset, floor):
    totals = {}
    for sample in dataset:
        values = {"state": sample["state"].numpy()[None],
                  "action": sample["action"].numpy()[~sample["is_pad"].numpy()]}
        for key, a in values.items():
            a = a.astype(np.float64)
            if key not in totals:
                totals[key] = [0, np.zeros(a.shape[-1]), np.zeros(a.shape[-1])]
            totals[key][0] += len(a)
            totals[key][1] += a.sum(0)
            totals[key][2] += (a*a).sum(0)
    result = {"source_split": "train", "floor": floor}
    for key, (n, summed, squared) in totals.items():
        mean = summed / n
        result[key + "_mean"] = mean.tolist()
        result[key + "_std"] = np.maximum(np.sqrt(np.maximum(0, squared/n-mean*mean)), floor).tolist()
        result[key + "_count"] = n
    return result
