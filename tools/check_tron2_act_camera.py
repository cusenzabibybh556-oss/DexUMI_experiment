#!/usr/bin/env python3
"""Inspect a saved Top ROS snapshot using the deployment camera and segmentation path."""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.deploy_tron2_act import load_runtime
from tron2_act.live_policy import LivePolicy, mask_geometry


def inspect_snapshot(policy, snapshot, output):
    metadata = json.loads((snapshot / 'calibration.json').read_text())
    raw = cv2.imread(str(snapshot / 'top_raw.png'))
    if raw is None:
        raise ValueError(f'Missing or unreadable Top RGB in {snapshot}')
    aligned, fg = policy.prepare_vision(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB), metadata['source'])
    geometry=mask_geometry(fg)
    metrics={'mask_centroids':{k:v['centroid'] for k,v in geometry.items()},
             'mask_bboxes':{k:v['bbox'] for k,v in geometry.items()}}
    try:
        policy.check_masks(fg)
        passed = True
    except RuntimeError:
        passed = False
    overlay = aligned.copy()
    for key, color in [('hand', [0, 200, 255]), ('objects', [255, 120, 0])]:
        mask = fg[key]
        overlay[mask] = (.45 * overlay[mask] + .55 * np.array(color)).astype(np.uint8)
    output.mkdir(parents=True, exist_ok=False)
    panel = cv2.cvtColor(np.hstack([aligned, overlay, fg['rgb']]), cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(output / 'vision.jpg'), panel):
        raise OSError('Could not save vision.jpg')
    report = dict(snapshot=str(snapshot.resolve()), panel='aligned | cyan:hand, orange:objects | ACT input',
                  hand_pixels=int(fg['hand'].sum()), object_pixels=int(fg['objects'].sum()),
                  mask_centroids=metrics['mask_centroids'],mask_bboxes=metrics['mask_bboxes'],
                  pixel_guard_passed=passed, valid_view_fraction=policy.alignment.valid_fraction if policy.alignment else None,
                  foreground_sha256=policy.runtime_config['foreground_sha256'],
                  virtual_look_down_deg=policy.runtime_config.get('virtual_look_down_deg', 0.),
                  virtual_look_right_deg=policy.runtime_config.get('virtual_look_right_deg', 0.))
    (output / 'report.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/tron2_act_deploy.json')
    args = parser.parse_args()
    policy = LivePolicy(load_runtime(args.config), ROOT)
    report = inspect_snapshot(policy, args.snapshot_dir, args.output)
    print(json.dumps(report, indent=2))
    return 0 if report['pixel_guard_passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
