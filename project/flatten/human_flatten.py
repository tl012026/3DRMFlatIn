"""
Judge whether VGGT-Omega flattened a person onto the background.

Uses reconstruct depth maps (full image, not mask-cropped) plus SAM2 person
masks. No 3D-to-2D unfolding: A/B/C are computed on the 2D depth grid.

A: median depth inside the person mask
B: median depth in a dilated ring just outside the mask
C: |A - B|
score: C / B
is_flattened: score < score_thres  (person depth looks like the surround)
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Union

import numpy as np
from scipy.ndimage import binary_dilation

try:
    import torch
except ImportError:
    torch = None


DepthLike = Union[np.ndarray, Sequence, Any]


class HumanFlatten:
    def __init__(self, score_thres, ring_width, merge_instances=True):
        self.score_thres = float(score_thres)
        self.ring_width = int(ring_width)
        self.merge_instances = bool(merge_instances)

    def run(self, depth_maps, masks):
        """
        Inputs:
            depth_maps: reconstruct per-view depth, [S,H,W] or list of HxW
            masks: SAM2 person masks, list of [N,H,W] or HxW (any resolution)
        Outputs:
            dict: A, B, C, scores, is_flattened, valid,
                  human_masks, surround_masks, flat_meta
        """
        depths = self._as_depth_list(depth_maps)
        mask_list = self._as_mask_list(masks, n_views=len(depths))

        A, B, C, scores, is_flat, valid = [], [], [], [], [], []
        human_masks, surround_masks = [], []

        for depth, inst_masks in zip(depths, mask_list):
            person = self._person_mask(inst_masks, depth.shape[-2:])
            ring = self._surround_ring(person, self.ring_width)
            a = self._median_depth(depth, person)
            b = self._median_depth(depth, ring)
            c = abs(a - b) if np.isfinite(a) and np.isfinite(b) else float("nan")
            score = (c / b) if (np.isfinite(c) and np.isfinite(b) and b > 0) else float("nan")
            ok = bool(np.isfinite(score))

            A.append(a)
            B.append(b)
            C.append(c)
            scores.append(score)
            valid.append(ok)
            is_flat.append(bool(ok and score < self.score_thres))
            human_masks.append(person.astype(np.uint8))
            surround_masks.append(ring.astype(np.uint8))

        return {
            "A": A,
            "B": B,
            "C": C,
            "scores": scores,
            "is_flattened": is_flat,
            "valid": valid,
            "human_masks": human_masks,
            "surround_masks": surround_masks,
            "flat_meta": {
                "score_thres": self.score_thres,
                "ring_width": self.ring_width,
                "merge_instances": self.merge_instances,
                "num_views": len(depths),
            },
        }

    def _person_mask(self, inst_masks: Optional[np.ndarray], hw: tuple) -> np.ndarray:
        h, w = hw
        if inst_masks is None or inst_masks.size == 0:
            return np.zeros((h, w), dtype=bool)
        masks = inst_masks
        if masks.ndim == 2:
            masks = masks[None]
        resized = np.stack([self._resize_mask(m, (h, w)) for m in masks], axis=0)
        if self.merge_instances:
            return np.any(resized > 0, axis=0)
        return resized[0] > 0

    @staticmethod
    def _surround_ring(person: np.ndarray, ring_width: int) -> np.ndarray:
        if ring_width <= 0 or not np.any(person):
            return np.zeros_like(person, dtype=bool)
        r = int(ring_width)
        yy, xx = np.ogrid[-r : r + 1, -r : r + 1]
        disk = (xx * xx + yy * yy) <= r * r
        dilated = binary_dilation(person.astype(bool), structure=disk)
        return dilated & (~person.astype(bool))

    @staticmethod
    def _median_depth(depth: np.ndarray, region: np.ndarray) -> float:
        vals = np.asarray(depth, dtype=np.float64)[region.astype(bool)]
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if vals.size == 0:
            return float("nan")
        return float(np.median(vals))

    @staticmethod
    def _resize_mask(mask: np.ndarray, hw: tuple) -> np.ndarray:
        h, w = hw
        mask = np.asarray(mask)
        if mask.shape[-2:] == (h, w):
            return mask
        ys = (np.arange(h) * mask.shape[0] / h).astype(np.int64)
        xs = (np.arange(w) * mask.shape[1] / w).astype(np.int64)
        return mask[ys[:, None], xs[None, :]]

    @staticmethod
    def _as_depth_list(depth_maps: DepthLike) -> List[np.ndarray]:
        if torch is not None and isinstance(depth_maps, torch.Tensor):
            depth_maps = depth_maps.detach().cpu().numpy()
        if isinstance(depth_maps, (list, tuple)):
            return [HumanFlatten._squeeze_hw(d) for d in depth_maps]
        arr = np.asarray(depth_maps)
        arr = HumanFlatten._drop_channel(arr)
        if arr.ndim == 2:
            return [arr]
        if arr.ndim == 3:
            return [arr[i] for i in range(arr.shape[0])]
        raise ValueError(f"depth_maps must be HxW or SxHxW, got {arr.shape}")

    @staticmethod
    def _as_mask_list(masks: Any, n_views: int) -> List[Optional[np.ndarray]]:
        if masks is None:
            return [None] * n_views
        if torch is not None and isinstance(masks, torch.Tensor):
            masks = masks.detach().cpu().numpy()
        if isinstance(masks, np.ndarray):
            if masks.ndim == 2:
                items = [masks]
            elif masks.ndim == 3:
                # S,H,W or N,H,W — treat leading dim as views if it matches
                items = [masks[i] for i in range(masks.shape[0])] if masks.shape[0] == n_views else [masks]
            elif masks.ndim == 4:
                items = [masks[i] for i in range(masks.shape[0])]
            else:
                raise ValueError(f"masks have unsupported shape {masks.shape}")
        elif isinstance(masks, (list, tuple)):
            items = list(masks)
        else:
            items = [masks]
        if len(items) != n_views:
            raise ValueError(f"masks views {len(items)} != depth views {n_views}")
        out = []
        for m in items:
            if m is None:
                out.append(None)
                continue
            if torch is not None and isinstance(m, torch.Tensor):
                m = m.detach().cpu().numpy()
            m = np.asarray(m)
            out.append(None if m.size == 0 else m)
        return out

    @staticmethod
    def _squeeze_hw(depth: DepthLike) -> np.ndarray:
        if torch is not None and isinstance(depth, torch.Tensor):
            depth = depth.detach().cpu().numpy()
        return HumanFlatten._drop_channel(np.asarray(depth, dtype=np.float32))

    @staticmethod
    def _drop_channel(arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 4 and arr.shape[1] == 1:
            return arr[:, 0]
        if arr.ndim == 4 and arr.shape[-1] == 1:
            return arr[..., 0]
        if arr.ndim == 3 and arr.shape[0] == 1:
            return arr[0]
        if arr.ndim == 3 and arr.shape[-1] == 1:
            return arr[..., 0]
        return arr
