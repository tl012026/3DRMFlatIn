"""
Module role:
    Detect person instances in each view (Ultralytics YOLO or equivalent).
    Responsible for bounding boxes only, not fine masks.

Inputs:
    - images: batched image tensors or ndarray list
      OR image_paths: List[str]
    - weights_path: detector checkpoint path (from config)
    - conf_thres: confidence threshold
    - device: "cuda" | "cpu"
    - person_class_id: class index for "person" (if multi-class model)

Outputs:
    # MODIFIED: dropped track_ids / det_meta (optional fields removed)
    - bboxes: per-image boxes, List[Nx4] in xyxy; list length = num images
    - scores: per-box confidence, List[N]; aligned with bboxes
    - vis images (save_bbox_images): per-view PNG with person xyxy boxes drawn

Variables (contract):
    - self.model: loaded detector weights
    - self.conf_thres, self.device, self.person_class_id
    - images / image_paths
    # MODIFIED: only bboxes, scores
    - bboxes, scores
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

# Distinct colors so overlapping people stay separable in the bbox overlay.
_BOX_COLORS = [
    (0, 255, 0),
    (255, 0, 0),
    (0, 128, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (255, 128, 0),
    (128, 0, 255),
]


class PersonDetector:
    """
    Role:
        Thin wrapper around the detection backend.

    Inputs (constructor):
        weights_path, conf_thres, device, person_class_id

    Outputs (predict):
        # MODIFIED: only bboxes, scores
        bboxes, scores
    """

    def __init__(self, weights_path, conf_thres, device, person_class_id):
        self.weights_path = weights_path
        self.conf_thres = conf_thres
        self.device = device
        self.person_class_id = person_class_id

        self.model = YOLO(self.weights_path).to(self.device)

    def predict(self, images):
        """
        Inputs:
            images: tensors / ndarrays / paths for one or more views
        Outputs:
            # MODIFIED: one dict for the whole batch (not one dict per image).
            #   bboxes[i] / scores[i] = detections for image i
            dict with keys: bboxes, scores
        Variables used:
            self.model, self.conf_thres, self.device, self.person_class_id
        """
        # MODIFIED: filter person class inside predict (classes=...); also apply conf/device here
        results = self.model(
            images,
            conf=self.conf_thres,
            classes=[self.person_class_id],
            device=self.device,
            verbose=False,
        )

        bboxes = []
        scores = []

        # MODIFIED: aggregate all images into list-of-per-image arrays
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                bboxes.append([])
                scores.append([])
                continue
            # xyxy float arrays; conf aligned with boxes
            bboxes.append(boxes.xyxy.cpu().numpy())
            scores.append(boxes.conf.cpu().numpy())

        # MODIFIED: return only bboxes + scores (track_ids / det_meta removed)
        return {
            "bboxes": bboxes,
            "scores": scores,
        }

    def save_bbox_images(self, images, bboxes, scores, output_dir):
        """
        Draw person xyxy boxes on each view and write det_bbox_XXX.png.
        Used to check whether YOLO actually boxed a person.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        image_list = self._as_image_list(images)
        vis_paths = []
        n = max(len(image_list), len(bboxes), len(scores))
        for i in range(n):
            rgb = self._to_rgb_hwc(image_list[i] if i < len(image_list) else image_list[-1])
            boxes_i = bboxes[i] if i < len(bboxes) else []
            scores_i = scores[i] if i < len(scores) else []
            vis = self._draw_boxes(rgb, boxes_i, scores_i)
            path = out / f"det_bbox_{i:03d}.png"
            Image.fromarray(vis).save(path)
            vis_paths.append(str(path))
        return vis_paths

    @staticmethod
    def _as_image_list(images):
        if isinstance(images, (str, Path)):
            return [images]
        if isinstance(images, (list, tuple)):
            return list(images)
        return [images]

    @staticmethod
    def _to_rgb_hwc(image):
        """Load a view as RGB HWC uint8 for overlay drawing."""
        if isinstance(image, (str, Path)):
            return np.array(Image.open(image).convert("RGB"))
        if isinstance(image, Image.Image):
            return np.array(image.convert("RGB"))
        if hasattr(image, "detach"):
            image = image.detach().cpu().numpy()
        arr = np.asarray(image)
        if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
            arr = np.transpose(arr, (1, 2, 0))
        if arr.dtype != np.uint8:
            scale = 255.0 if arr.size and float(arr.max()) <= 1.0 else 1.0
            arr = np.clip(arr * scale, 0, 255).astype(np.uint8)
        return arr

    @staticmethod
    def _draw_boxes(rgb, boxes, scores):
        im = Image.fromarray(rgb)
        draw = ImageDraw.Draw(im)
        font = ImageFont.load_default()
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4) if len(boxes) else np.zeros((0, 4), np.float32)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1) if len(scores) else np.zeros((0,), np.float32)
        if boxes.shape[0] == 0:
            draw.text((8, 8), "no person bbox", fill=(255, 0, 0), font=font)
            return np.array(im)
        h, w = rgb.shape[:2]
        for k, box in enumerate(boxes):
            color = _BOX_COLORS[k % len(_BOX_COLORS)]
            x1, y1, x2, y2 = [int(round(float(v))) for v in box]
            x1, x2 = max(0, min(w - 1, x1)), max(0, min(w - 1, x2))
            y1, y2 = max(0, min(h - 1, y1)), max(0, min(h - 1, y2))
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            score = float(scores[k]) if k < len(scores) else float("nan")
            label = f"person {k} {score:.2f}"
            tx, ty = x1 + 2, max(0, y1 - 12)
            bbox = draw.textbbox((tx, ty), label, font=font)
            draw.rectangle(bbox, fill=color)
            draw.text((tx, ty), label, fill=(0, 0, 0), font=font)
        return np.array(im)
