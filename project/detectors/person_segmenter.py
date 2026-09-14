"""
Module role:
    Segment person / body regions with SAM2.
    Turns PersonDetector boxes into per-instance pixel masks.

Inputs:
    - images: same views as the detector (paths / RGB arrays / tensors)
    - prompts: detector output {bboxes: List[Nx4 xyxy]} (or DetBoxes / a bboxes list)
    - checkpoint_path, device, mask_threshold

Outputs:
    - masks: per-image [N,H,W] uint8 (N=0 when no box)
    - mask_scores: per-mask quality scores
    - seg_meta: model_cfg, mask_threshold
    - vis images (save_mask_images): per-view PNG with person-pixel regions overlaid
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Distinct colors so overlapping people stay separable in the mask overlay.
_MASK_COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 128, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (255, 128, 0),
    (128, 0, 255),
]


class PersonSegmenter:
    def __init__(
        self,
        checkpoint_path,
        device,
        mask_threshold,
        model_cfg="configs/sam2.1/sam2.1_hiera_l.yaml",
        multimask_output=False,
    ):
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.mask_threshold = mask_threshold
        self.model_cfg = model_cfg
        self.multimask_output = multimask_output

        self.model = build_sam2(self.model_cfg, self.checkpoint_path, device=self.device)
        self.predictor = SAM2ImagePredictor(
            self.model, mask_threshold=self.mask_threshold
        )

    def predict(self, images, prompts):
        """Box prompts -> one binary mask per box, aligned with images."""
        image_list = self._as_image_list(images)
        box_list = self._as_box_list(prompts, n_images=len(image_list))

        use_cuda = str(self.device).startswith("cuda") and torch.cuda.is_available()
        autocast_ctx = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if use_cuda
            else contextlib.nullcontext()
        )

        masks, mask_scores = [], []
        with torch.inference_mode(), autocast_ctx:
            for image, boxes in zip(image_list, box_list):
                rgb = self._to_rgb_hwc(image)
                if boxes is None:
                    masks.append(np.zeros((0, *rgb.shape[:2]), dtype=np.uint8))
                    mask_scores.append(np.zeros((0,), dtype=np.float32))
                    continue

                self.predictor.set_image(rgb)
                # One box -> several masks -> chose the best mask
                pred_masks, pred_scores, _ = self.predictor.predict(
                    box=boxes,
                    multimask_output=self.multimask_output and len(boxes) == 1,
                )
                inst_masks, inst_scores = self._pick_best(pred_masks, pred_scores)
                masks.append(inst_masks.astype(np.uint8))
                mask_scores.append(inst_scores)

        return {
            "masks": masks,
            "mask_scores": mask_scores,
            "seg_meta": {
                "model_cfg": self.model_cfg,
                "mask_threshold": self.mask_threshold,
            },
        }

    def save_mask_images(self, images, masks, output_dir, mask_scores=None):
        """
        Overlay SAM2 person pixels on each view and write seg_mask_XXX.png.
        Used to check whether the extracted region is actually a person.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        image_list = self._as_image_list(images)
        vis_paths = []
        n = max(len(image_list), len(masks))
        for i in range(n):
            rgb = self._to_rgb_hwc(image_list[i] if i < len(image_list) else image_list[-1])
            inst = masks[i] if i < len(masks) else np.zeros((0, *rgb.shape[:2]), dtype=np.uint8)
            scores_i = None
            if mask_scores is not None and i < len(mask_scores):
                scores_i = mask_scores[i]
            vis = self._draw_masks(rgb, inst, scores_i)
            path = out / f"seg_mask_{i:03d}.png"
            Image.fromarray(vis).save(path)
            vis_paths.append(str(path))
        return vis_paths

    @staticmethod
    def _draw_masks(rgb, inst, scores=None):
        inst = np.asarray(inst)
        if inst.size == 0:
            inst = np.zeros((0, *rgb.shape[:2]), dtype=np.uint8)
        elif inst.ndim == 2:
            inst = inst[None]
        vis = rgb.copy()
        h, w = vis.shape[:2]
        for k, mask in enumerate(inst):
            m = np.asarray(mask)
            if m.shape != (h, w):
                m_img = Image.fromarray((m > 0).astype(np.uint8) * 255)
                m = np.array(m_img.resize((w, h), Image.NEAREST))
            sel = m > 0
            if not np.any(sel):
                continue
            color = np.asarray(_MASK_COLORS[k % len(_MASK_COLORS)], dtype=np.float32)
            vis[sel] = (vis[sel].astype(np.float32) * 0.45 + color * 0.55).astype(np.uint8)
        im = Image.fromarray(vis)
        draw = ImageDraw.Draw(im)
        font = ImageFont.load_default()
        if inst.shape[0] == 0:
            draw.text((8, 8), "no person mask", fill=(255, 0, 0), font=font)
            return np.array(im)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1) if scores is not None and len(scores) else None
        y = 8
        for k in range(inst.shape[0]):
            color = _MASK_COLORS[k % len(_MASK_COLORS)]
            score_txt = ""
            if scores is not None and k < len(scores):
                score_txt = f" {float(scores[k]):.2f}"
            label = f"person {k}{score_txt}"
            bbox = draw.textbbox((8, y), label, font=font)
            draw.rectangle(bbox, fill=color)
            draw.text((8, y), label, fill=(0, 0, 0), font=font)
            y = bbox[3] + 4
        return np.array(im)

    @staticmethod
    def _as_image_list(images):
        if isinstance(images, (str, Path)):
            return [images]
        if isinstance(images, (torch.Tensor, np.ndarray)) and images.ndim == 4:
            return [images[i] for i in range(images.shape[0])]
        if isinstance(images, (list, tuple)):
            return list(images)
        return [images]

    @staticmethod
    def _as_box_list(prompts, n_images):
        """Accept detector dict / DetBoxes / per-image box list."""
        if prompts is None:
            return [None] * n_images
        if isinstance(prompts, dict):
            prompts = prompts["bboxes"]
        elif hasattr(prompts, "bboxes"):
            prompts = prompts.bboxes

        # One (N, 4) array for a single image — do not take only the first row.
        if n_images == 1 and isinstance(prompts, (np.ndarray, torch.Tensor)):
            return [PersonSegmenter._normalize_boxes(prompts)]

        return [
            PersonSegmenter._normalize_boxes(prompts[i] if i < len(prompts) else None)
            for i in range(n_images)
        ]

    @staticmethod
    def _normalize_boxes(box):
        if box is None:
            return None
        if isinstance(box, torch.Tensor):
            box = box.detach().cpu().numpy()
        box = np.asarray(box, dtype=np.float32)
        if box.size == 0:
            return None
        if box.ndim == 1:
            box = box[None, :]
        if box.shape[-1] != 4:
            raise ValueError(f"box prompts must be xyxy with last dim 4, got {box.shape}")
        return box.reshape(-1, 4)

    @staticmethod
    def _to_rgb_hwc(image):
        """SAM2 set_image expects RGB HWC uint8."""
        if isinstance(image, (str, Path)):
            return np.array(Image.open(image).convert("RGB"))
        if isinstance(image, Image.Image):
            return np.array(image.convert("RGB"))
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        arr = np.asarray(image)
        if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
            arr = np.transpose(arr, (1, 2, 0))
        if arr.dtype != np.uint8:
            scale = 255.0 if arr.size and float(arr.max()) <= 1.0 else 1.0
            arr = np.clip(arr * scale, 0, 255).astype(np.uint8)
        return arr

    @staticmethod
    def _pick_best(pred_masks, pred_scores):
        """Keep one mask per box (highest score if SAM2 returned several)."""
        pred_masks = np.asarray(pred_masks)
        pred_scores = np.asarray(pred_scores, dtype=np.float32)
        if pred_masks.ndim == 2:  # (H, W)
            return pred_masks[None], np.atleast_1d(pred_scores)
        if pred_masks.ndim == 3:  # (C, H, W)
            best = int(np.argmax(pred_scores))
            return pred_masks[best][None], np.atleast_1d(pred_scores.reshape(-1)[best])
        # (N, C, H, W)
        best = np.argmax(pred_scores, axis=-1)
        rows = np.arange(pred_masks.shape[0])
        return pred_masks[rows, best], pred_scores[rows, best]
