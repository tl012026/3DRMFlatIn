"""
Judge whether VGGT-Omega flattened a person onto the background.

A: median depth inside the person mask
B: median depth in a dilated ring just outside the mask
C: |A - B|
score: C / B; flattened if score < score_thres
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation


def _numpy(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


class HumanFlatten:
    def __init__(self, score_thres, ring_width, merge_instances=True):
        self.score_thres = float(score_thres)
        self.ring_width = int(ring_width)
        self.merge_instances = bool(merge_instances)

    def run(self, depth_maps, masks):
        """
        depth_maps: [S,H,W] (or HxW)
        masks: list of per-view [N,H,W] (SAM2); resized to depth if needed
        """
        depth = np.squeeze(_numpy(depth_maps))
        if depth.ndim == 2:
            depth = depth[None]
        if depth.ndim != 3:
            raise ValueError(f"depth_maps must be HxW or SxHxW, got {depth.shape}")
        if len(masks) != len(depth):
            raise ValueError(f"masks views {len(masks)} != depth views {len(depth)}")

        A, B, C, scores, is_flat, valid = [], [], [], [], [], []
        human_masks, surround_masks = [], []

        for depth_i, inst in zip(depth, masks):
            person = self._person_mask(inst, depth_i.shape)
            ring = self._surround_ring(person)
            a = self._median(depth_i, person)
            b = self._median(depth_i, ring)
            c = abs(a - b) if np.isfinite(a) and np.isfinite(b) else float("nan")
            score = c / b if np.isfinite(c) and b > 0 else float("nan")
            ok = np.isfinite(score)

            A.append(a)
            B.append(b)
            C.append(c)
            scores.append(score)
            valid.append(bool(ok))
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
            },
        }

    def _person_mask(self, inst, hw):
        inst = _numpy(inst)
        if inst.size == 0:
            return np.zeros(hw, dtype=bool)
        if inst.ndim == 2:
            inst = inst[None]
        person = np.any(inst > 0, axis=0) if self.merge_instances else inst[0] > 0
        if person.shape != hw:
            ys = (np.arange(hw[0]) * person.shape[0] / hw[0]).astype(np.int64)
            xs = (np.arange(hw[1]) * person.shape[1] / hw[1]).astype(np.int64)
            person = person[ys[:, None], xs[None, :]]
        return person.astype(bool)

    def _surround_ring(self, person):
        r = self.ring_width
        if r <= 0 or not np.any(person):
            return np.zeros_like(person, dtype=bool)
        yy, xx = np.ogrid[-r : r + 1, -r : r + 1]
        disk = xx * xx + yy * yy <= r * r
        return binary_dilation(person, structure=disk) & ~person

    @staticmethod
    def _median(depth, region):
        vals = np.asarray(depth, dtype=np.float64)[region]
        vals = vals[np.isfinite(vals) & (vals > 0)]
        return float(np.median(vals)) if vals.size else float("nan")
