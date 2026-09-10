"""
Judge whether VGGT-Omega flattened a person onto the background.

Each SAM2 instance is scored. A view is flattened if any person is.

A: median depth inside that person's mask
B: median depth in a ring around that mask (other people excluded)
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
    def __init__(self, score_thres, ring_width):
        self.score_thres = float(score_thres)
        self.ring_width = int(ring_width)

    def run(self, depth_maps, masks):
        """
        depth_maps: [S,H,W] (or HxW)
        masks: list of per-view [N,H,W]
        """
        depth = np.squeeze(_numpy(depth_maps))
        if depth.ndim == 2:
            depth = depth[None]
        if depth.ndim != 3:
            raise ValueError(f"depth_maps must be HxW or SxHxW, got {depth.shape}")
        if len(masks) != len(depth):
            raise ValueError(f"masks views {len(masks)} != depth views {len(depth)}")

        people, view_flat = [], []
        human_masks, surround_masks = [], []

        for depth_i, inst in zip(depth, masks):
            inst = self._instances(inst, depth_i.shape)
            union = np.zeros(depth_i.shape, dtype=bool)
            for person in inst:
                union |= person
            recs, person_maps, ring_maps = [], [], []
            for person in inst:
                ring = self._surround_ring(person, union)
                rec = self._score(depth_i, person, ring)
                recs.append(rec)
                person_maps.append(person.astype(np.uint8))
                ring_maps.append(ring.astype(np.uint8))
            people.append(recs)
            view_flat.append(any(r["is_flattened"] for r in recs))
            human_masks.append(np.stack(person_maps) if person_maps else np.zeros((0, *depth_i.shape), np.uint8))
            surround_masks.append(np.stack(ring_maps) if ring_maps else np.zeros((0, *depth_i.shape), np.uint8))

        return {
            "people": people,
            "view_is_flattened": view_flat,
            "human_masks": human_masks,
            "surround_masks": surround_masks,
            "flat_meta": {
                "score_thres": self.score_thres,
                "ring_width": self.ring_width,
            },
        }

    def _score(self, depth, person, ring):
        a = self._median(depth, person)
        b = self._median(depth, ring)
        c = abs(a - b) if np.isfinite(a) and np.isfinite(b) else float("nan")
        score = c / b if np.isfinite(c) and b > 0 else float("nan")
        ok = bool(np.isfinite(score))
        return {
            "A": a,
            "B": b,
            "C": c,
            "score": score,
            "valid": ok,
            "is_flattened": bool(ok and score < self.score_thres),
        }

    def _instances(self, inst, hw):
        inst = _numpy(inst)
        if inst.size == 0:
            return []
        if inst.ndim == 2:
            inst = inst[None]
        out = []
        for m in inst:
            if m.shape != hw:
                ys = (np.arange(hw[0]) * m.shape[0] / hw[0]).astype(np.int64)
                xs = (np.arange(hw[1]) * m.shape[1] / hw[1]).astype(np.int64)
                m = m[ys[:, None], xs[None, :]]
            out.append(m.astype(bool))
        return out

    def _surround_ring(self, person, union):
        r = self.ring_width
        if r <= 0 or not np.any(person):
            return np.zeros_like(person, dtype=bool)
        yy, xx = np.ogrid[-r : r + 1, -r : r + 1]
        disk = xx * xx + yy * yy <= r * r
        return binary_dilation(person, structure=disk) & ~union

    @staticmethod
    def _median(depth, region):
        vals = np.asarray(depth, dtype=np.float64)[region]
        vals = vals[np.isfinite(vals) & (vals > 0)]
        return float(np.median(vals)) if vals.size else float("nan")
