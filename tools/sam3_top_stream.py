"""Stateful, causal RGB segmentation using local SAM3 detection and mask tracking.

Only the current image enters the model. Memory contains the first annotation,
at most three recent corrections, and 32 past tracking outputs. No video loader
or backward propagation is used. This adapter uses local SAM3 internal APIs.
"""
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch

from try_sam3_top_foreground import select_hand


def overlap(a, b):
    return float((a & b).sum()) / max(1, int((a | b).sum()))


class TopForegroundStream:
    def __init__(self, sam3_root, checkpoint=None, redetect_every=30):
        sys.path.insert(0, str(sam3_root))
        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        checkpoint = checkpoint or Path(sam3_root) / "checkpoints/modelscope_sam3/sam3.pt"
        self.model = build_sam3_image_model(checkpoint_path=str(checkpoint), load_from_HF=False, enable_inst_interactivity=True)
        self.processor = Sam3Processor(self.model, confidence_threshold=0.5)
        self.tracker = self.model.inst_interactive_predictor.model
        self.tracker.clear_non_cond_mem_around_input = False
        self.redetect_every = redetect_every
        self.reset()

    def reset(self):
        self.state = None
        self.index = -1
        self.previous_frame_number = None
        self.previous_result = None

    def detect(self, image_state, rgb):
        detected = {}
        for prompt in ("white robot hand", "cup"):
            self.processor.reset_all_prompts(image_state)
            output = self.processor.set_text_prompt(prompt, image_state)
            detected[prompt] = (output["masks"][:, 0].cpu().numpy(), output["scores"].float().cpu().numpy())
        hand, _, _ = select_hand(rgb, detected["white robot hand"][0], False)
        return hand, detected["cup"]

    def trim_memory(self):
        s = self.state
        keep_cond = sorted(s["output_dict"]["cond_frame_outputs"])
        keep_cond = set(keep_cond[:1] + keep_cond[-3:])
        for key in ("cond_frame_outputs", "non_cond_frame_outputs"):
            outputs = s["output_dict"][key]
            remove = [i for i in outputs if i not in keep_cond] if key == "cond_frame_outputs" else [i for i in outputs if i < self.index - 32]
            for i in remove:
                outputs.pop(i, None)
                s["consolidated_frame_inds"][key].discard(i)
                for obj_idx, obj in s["output_dict_per_obj"].items():
                    obj[key].pop(i, None)
                    s["point_inputs_per_obj"][obj_idx].pop(i, None)
                    s["mask_inputs_per_obj"][obj_idx].pop(i, None)
        s["frames_already_tracked"] = {i: value for i, value in s["frames_already_tracked"].items() if i >= self.index - 32}

    @torch.inference_mode()
    def step(self, rgb, frame_number):
        if self.previous_frame_number == frame_number:
            result = dict(self.previous_result)
            result["duplicate"] = True
            result["total_ms"] = 0.0
            return result
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            image = Image.fromarray(rgb)
            image_state = self.processor.set_image(image)
            # The processor already projects the tracker high-resolution features.
            tensor = self.processor.transform(torch.from_numpy(rgb).permute(2, 0, 1).cuda()).unsqueeze(0)
            self.index += 1
            if self.state is None:
                self.state = self.tracker.init_state(video_height=rgb.shape[0], video_width=rgb.shape[1], num_frames=0)
            self.state["num_frames"] = self.index + 1
            self.state["cached_features"] = {self.index: (tensor, image_state["backbone_out"]["sam2_backbone_out"])}
            initialized = self.index == 0
            corrections = []
            # Reuse the same current-image features for detection and tracking.
            detected_hand, (detected_cups, scores) = self.detect(image_state, rgb)
            if initialized:
                hand, cups = detected_hand, detected_cups
                if hand.sum() < 1000 or len(cups) != 2:
                    raise ValueError(f"Initialization needs a visible robot hand and two cups, got hand={hand.sum()}, cups={len(cups)}")
                for obj_id, mask in enumerate([hand, *cups], 1):
                    self.tracker.add_new_mask(self.state, self.index, obj_id, torch.from_numpy(mask))
                self.tracker.propagate_in_video_preflight(self.state)
            outputs = list(self.tracker.propagate_in_video(self.state, start_frame_idx=self.index, max_frame_num_to_track=0, reverse=False, tqdm_disable=True))
            assert len(outputs) == 1 and outputs[0][0] == self.index
            _, obj_ids, _, logits, object_scores = outputs[0]
            masks = (logits[:, 0] > 0).cpu().numpy()
            if not initialized and self.redetect_every and self.index % self.redetect_every == 0:
                proposals = []
                # Accept only a substantial mask that agrees with the existing track.
                # A missing semantic detection never erases a tracked object.
                hand_ratio = detected_hand.sum() / max(1, masks[0].sum())
                if 0.7 <= hand_ratio <= 1.4 and overlap(detected_hand, masks[0]) >= 0.65:
                    proposals.append((1, detected_hand))
                used = set()
                for slot in range(1, len(masks)):
                    choices = [(overlap(masks[slot], mask), j) for j, mask in enumerate(detected_cups) if j not in used and scores[j] >= 0.75]
                    if choices:
                        agreement, j = max(choices)
                        ratio = detected_cups[j].sum() / max(1, masks[slot].sum())
                        if agreement >= 0.65 and 0.7 <= ratio <= 1.4:
                            proposals.append((int(obj_ids[slot]), detected_cups[j]))
                            used.add(j)
                if proposals:
                    for obj_id, mask in proposals:
                        self.tracker.add_new_mask(self.state, self.index, obj_id, torch.from_numpy(mask))
                    self.tracker.propagate_in_video_preflight(self.state)
                    output = list(self.tracker.propagate_in_video(self.state, start_frame_idx=self.index, max_frame_num_to_track=0, reverse=False, tqdm_disable=True))[0]
                    masks = (output[3][:, 0] > 0).cpu().numpy()
                    object_scores = output[4]
                    corrections = [obj_id for obj_id, _ in proposals]
            self.trim_memory()
            hand = masks[0].copy()
            objects = masks[1:].any(axis=0)
            # Tracking prevents semantic hand dropouts. Current detections recover
            # visible cup regions after nesting; no hidden RGB is generated.
            if detected_hand.sum() < 1.5 * max(1, hand.sum()) and overlap(hand, detected_hand) >= 0.4:
                hand |= detected_hand
            for candidate in detected_cups:
                if any(overlap(candidate, tracked) >= 0.15 for tracked in masks[1:]):
                    objects |= candidate
            union = hand | objects
            foreground = np.where(union[..., None], rgb, 0).astype(np.uint8)
            torch.cuda.synchronize()
            result = {"hand": hand, "objects": objects, "instances": masks, "union": union, "rgb": foreground,
                      "object_scores": object_scores.float().sigmoid().flatten().cpu().tolist(), "corrections": corrections,
                      "total_ms": (time.perf_counter() - start) * 1000, "duplicate": False, "initialized": initialized}
            self.previous_frame_number = frame_number
            self.previous_result = result
            return result
