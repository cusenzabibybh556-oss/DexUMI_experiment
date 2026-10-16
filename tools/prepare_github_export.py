#!/usr/bin/env python3
"""Export all Dexumi projects into a separate, flattened source repository.

Never deletes or changes nested repositories in the working source tree.
Writes an explicit file manifest, preserving source/resource files even when a
third-party .gitignore would normally hide them.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess


SKIP_DIRS = {
    ".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ".cache", ".idea", ".vscode", "node_modules", "artifacts",
}
OUTPUT_DIRS = {
    "linkerhand_telop_python-master/ros2/build",
    "linkerhand_telop_python-master/ros2/install",
    "linkerhand_telop_python-master/ros2/log",
    "tron2_replay_traces",
    "TransHandR0725/TransHandR/data/multisource",
}
ARTIFACT_SUFFIXES = {
    ".h5", ".hdf5", ".pth", ".pt", ".ckpt", ".safetensors", ".onnx",
    ".zip", ".tar", ".tgz", ".mp4", ".avi", ".bag", ".db3",
}
SECRET_KEYS = {
    "password", "passwd", "passphrase", "pwd", "psk", "ssid",
    "api_key", "apikey", "access_token", "secret_key", "wifi_password",
}
SECRET_KEY = "(?:" + "|".join(sorted(SECRET_KEYS, key=len, reverse=True)) + ")"
YAML_SECRET = re.compile(
    rf"^(\s*[\"']?({SECRET_KEY})[\"']?\s*:\s*)([^\r\n]*)(\r?\n|$)",
    re.IGNORECASE | re.MULTILINE,
)
SOURCE_SUFFIXES = {".py", ".sh", ".bash", ".zsh", ".c", ".cc", ".cpp", ".h", ".hpp", ".js", ".ts", ".tsx", ".jsx", ".ino", ".swift"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scrub_json(value, keys: list[str]):
    if isinstance(value, dict):
        result = {}
        for key, child in value.items():
            if key.lower() in SECRET_KEYS and isinstance(child, str) and child:
                keys.append(key)
                result[key] = "REPLACE_WITH_LOCAL_VALUE" if key.lower() == "ssid" else ""
            else:
                result[key] = scrub_json(child, keys)
        return result
    if isinstance(value, list):
        return [scrub_json(child, keys) for child in value]
    return value


def copy_config(source: Path, target: Path) -> list[str]:
    """Scrub known local credentials in the export only, preserving originals."""
    keys: list[str] = []
    if source.suffix.lower() not in {".json", ".yaml", ".yml"}:
        shutil.copy2(source, target)
        return keys
    content = source.read_text(encoding="utf-8")
    if source.suffix.lower() == ".json":
        try:
            value = scrub_json(json.loads(content), keys)
        except json.JSONDecodeError:
            shutil.copy2(source, target)
            return keys
        if keys:
            target.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    else:
        def replace(match):
            raw = match[3].strip()
            if not raw or raw.startswith("#") or raw in {"null", "None", "''", '\"\"'}:
                return match[0]
            keys.append(match[2])
            replacement = '\"REPLACE_WITH_LOCAL_VALUE\"' if match[2].lower() == "ssid" else '\"\"'
            return match[1] + replacement + match[4]
        replacement = YAML_SECRET.sub(replace, content)
        if keys:
            target.write_text(replacement, encoding="utf-8")
    if not keys:
        shutil.copy2(source, target)
    return keys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    source, destination = args.source.resolve(), args.destination.resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        parser.error("Source and destination must be separate, non-overlapping directories")
    if destination.exists():
        parser.error("Destination already exists; choose a fresh directory")
    destination.mkdir(parents=True)
    included, excluded, sanitized, repos = [], [], [], []
    projects: Counter = Counter()
    for current, dirs, files in os.walk(source, followlinks=False):
        current = Path(current)
        relative_dir = current.relative_to(source)
        if "pyvenv.cfg" in files or "conda-meta" in dirs:
            excluded.append({"path": relative_dir.as_posix() + "/", "reason": "virtual environment"})
            dirs[:] = []
            continue
        for name in list(dirs):
            path = current / name
            relative = path.relative_to(source).as_posix()
            reason = None
            if name == ".git":
                repos.append(relative_dir.as_posix())
                reason = "nested Git metadata; source files retained"
            elif name in SKIP_DIRS or name.endswith(".egg-info"):
                reason = "cache, IDE, or generated artifact directory"
            elif relative in OUTPUT_DIRS:
                reason = "build or runtime output"
            if reason:
                dirs.remove(name)
                excluded.append({"path": relative + "/", "reason": reason})
            elif path.is_symlink():
                raise RuntimeError(f"Directory symlink requires explicit handling: {relative}")
        for name in files:
            path = current / name
            relative = path.relative_to(source).as_posix()
            reason = None
            if name == ".git":
                reason = "nested Git metadata"
                repos.append(relative_dir.as_posix())
            elif name == ".env" or (name.startswith(".env.") and name != ".env.example") or name in {"id_rsa", "id_ed25519"}:
                reason = "local environment or private key"
            elif path.suffix.lower() in {".pyc", ".pyo", ".log", ".lock"}:
                reason = "cache or runtime output"
            elif path.suffix.lower() in ARTIFACT_SUFFIXES or name.endswith(".tar.gz"):
                reason = "recording, model artifact, or distribution archive"
            if reason:
                entry = {"path": relative, "reason": reason}
                if path.is_file() and not path.is_symlink():
                    entry["bytes"] = path.stat().st_size
                excluded.append(entry)
                continue
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if path.is_symlink():
                resolved = path.resolve(strict=True)
                if source not in resolved.parents:
                    raise RuntimeError(f"External symlink requires explicit handling: {relative}")
                target.symlink_to(os.path.relpath(resolved, path.parent))
                included.append({"path": relative, "symlink": os.readlink(target)})
            else:
                if path.stat().st_size > 100 * 1024 * 1024:
                    raise RuntimeError(f"Unexpected file over GitHub's regular-file limit: {relative}")
                keys = copy_config(path, target)
                if keys:
                    sanitized.append({"path": relative, "keys": sorted(set(keys))})
                included.append({"path": relative, "bytes": target.stat().st_size, "sha256": sha256(target)})
            parts = Path(relative).parts
            project = "/".join(parts[:2]) if parts[0] == "project1yu" and len(parts) > 2 else parts[0]
            projects[project] += 1
    report_dir = destination / "github-export"
    report_dir.mkdir()
    manifest = {"source": str(source), "source_repositories": sorted(repos), "projects": dict(sorted(projects.items())),
                "included": included, "excluded": excluded, "sanitized_configs": sanitized,
                "notes": ["Source working files are copied as a snapshot; nested Git histories are not imported.",
                          "This targeted config scrubbing is not a comprehensive secret audit.",
                          "Original working directories and credential values are not modified."]}
    (report_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    subprocess.run(["git", "init", "-b", "main", str(destination)], check=True, capture_output=True)
    # Force only the explicit export manifest, so vendor ignores cannot omit code.
    paths = [entry["path"] for entry in included] + ["github-export/manifest.json"]
    subprocess.run(["git", "--literal-pathspecs", "-C", str(destination), "add", "--force", "--pathspec-from-file=-", "--pathspec-file-nul"],
                   input=b"\0".join(path.encode() for path in paths) + b"\0", check=True, capture_output=True)
    index = subprocess.check_output(["git", "-C", str(destination), "ls-files", "--stage", "-z"])
    if any(entry.startswith(b"160000 ") for entry in index.split(b"\0")):
        raise RuntimeError("Export contains unexpected gitlinks")
    source_files = sum(Path(entry["path"]).suffix.lower() in SOURCE_SUFFIXES for entry in included)
    print(json.dumps({"destination": str(destination), "files": len(included), "source_files": source_files,
                      "bytes": sum(entry.get("bytes", 0) for entry in included), "excluded_entries": len(excluded),
                      "sanitized_configs": len(sanitized), "nested_histories_preserved_in_original": len(repos),
                      "gitlinks": 0, "state": "staged; not committed or pushed"}, indent=2))


if __name__ == "__main__":
    main()
