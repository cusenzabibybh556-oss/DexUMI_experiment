"""Revision, label integrity and causal model-input checks for the vision chain."""
import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
from PIL import Image
import torch

from test_tron2_act import fixture, foreground_fixture
from tron2_act.foreground import validate_foreground
from tron2_vision.annotations import apply_polygons, export_project, fingerprint, load_project, save_label
from tron2_vision.model import ForegroundNet, ForegroundPredictor, MODEL_CONTRACT
from tron2_vision.pipeline import train


class AnnotationTests(unittest.TestCase):
    def test_edit_preserves_source_identity_and_rejects_stale_revision(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "episode_000000.hdf5"
            fixture(source)
            cache = foreground_fixture(source, root / "cache")
            with h5py.File(cache, "a") as f:
                f["masks/hand"] = np.ones((35, 32, 32), np.uint8)
                f["masks/objects"] = np.zeros((35, 32, 32), np.uint8)
            before = fingerprint(source)
            project = root / "project"
            exported = export_project(root, root / "cache", [0], project, 2)
            frame = exported["frames"][0]
            corrected = apply_polygons(project, frame["id"], [{"label": 0, "points": [[0, 0], [10, 0], [10, 10], [0, 10]]}], "test", True)
            labels = np.asarray(Image.open(project / "labels" / f"{frame['id']}.png"))
            self.assertEqual(int(labels[5, 5]), 0)
            self.assertEqual(int(labels[20, 20]), 1)
            self.assertEqual(corrected["revision"], 1)
            self.assertEqual(corrected["timestamp_ns"], frame["timestamp_ns"])
            self.assertEqual(fingerprint(source), before)
            with self.assertRaisesRegex(ValueError, "Revision changed"):
                save_label(project, frame["id"], labels, 0, "test", True)
            invalid = labels.copy(); invalid[0, 0] = 3
            with self.assertRaisesRegex(ValueError, "Labels must"):
                save_label(project, frame["id"], invalid, 1, "test", True)
            with h5py.File(source, "a") as f:
                f.attrs["modified"] = True
            with self.assertRaisesRegex(ValueError, "source has changed"):
                load_project(project)

    def test_experimental_opt_in_never_accepts_rejected_or_misaligned_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "episode_000000.hdf5"; fixture(source)
            cache = foreground_fixture(source, root / "cache")
            with h5py.File(cache, "a") as f:
                f.attrs.update(contract="top-foreground-student-v1", review_status="experimental", model_sha256="a" * 64, model_metadata=json.dumps({"input_size": [240, 320]}))
            with h5py.File(source, "r") as f:
                with self.assertRaisesRegex(ValueError, "review"):
                    validate_foreground(source, f, root / "cache", fingerprint(source))
                validate_foreground(source, f, root / "cache", fingerprint(source), True)
            with h5py.File(cache, "a") as f:
                f["source/frame_number"][1] = 999
            with h5py.File(source, "r") as f:
                with self.assertRaisesRegex(ValueError, "frame_number"):
                    validate_foreground(source, f, root / "cache", fingerprint(source), True)
            with h5py.File(cache, "a") as f:
                f.attrs["review_status"] = "rejected"
            with h5py.File(source, "r") as f:
                with self.assertRaisesRegex(ValueError, "review"):
                    validate_foreground(source, f, root / "cache", fingerprint(source), True)

    def test_unreviewed_labels_cannot_train(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "project").mkdir()
            (root / "project/project.json").write_text(json.dumps({"contract": "top-mask-annotations-v1", "frames": []}))
            with self.assertRaisesRegex(ValueError, "reviewed train frames"):
                train(root / "project", root / "model", [1], steps=1, device="cpu", pretrained=False)

    def test_predictor_has_no_previous_frame_dependency(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.pt"
            torch.manual_seed(5)
            model = ForegroundNet()
            torch.save({"contract": MODEL_CONTRACT, "model": model.state_dict(), "metadata": {"input_size": [32, 32]}}, path)
            predictor = ForegroundPredictor(path, "cpu")
            rgb = np.random.default_rng(1).integers(0, 256, (48, 64, 3), dtype=np.uint8)
            first = predictor.predict(rgb)
            predictor.predict(255 - rgb)
            again = predictor.predict(rgb)
            np.testing.assert_array_equal(first["labels"], again["labels"])
            self.assertEqual(first["labels"].shape, rgb.shape[:2])
            np.testing.assert_array_equal(first["rgb"], np.where(first["union"][..., None], rgb, 0))
