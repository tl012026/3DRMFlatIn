"""
Judge whether VGGT-Omega flattened a person onto the background.

Each SAM2 instance is scored. A view is flattened if any person is.

A: median depth inside that person's mask
B: median depth in a ring around that mask (other people excluded)
C: |A - B|
score: C / B; flattened if score < score_thres

Per-person dumps also report depth_human (=A) vs depth_background (=B).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_dilation

# Person mask vs surround-ring colors on flatten_vis overlays.
_HUMAN_COLOR = (255, 0, 0)
_BG_COLOR = (0, 255, 255)


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

        for view_i, (depth_i, inst) in enumerate(zip(depth, masks)):
            inst = self._instances(inst, depth_i.shape)
            union = np.zeros(depth_i.shape, dtype=bool)
            for person in inst:
                union |= person
            recs, person_maps, ring_maps = [], [], []
            for pid, person in enumerate(inst):
                ring = self._surround_ring(person, union)
                rec = self._score(depth_i, person, ring)
                rec["view_id"] = view_i
                rec["person_id"] = pid
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
            # Named aliases for the verification dump: human vs background depth.
            "depth_human": a,
            "depth_background": b,
        }

    def save_depth_compare(self, people, output_dir, images=None, human_masks=None, surround_masks=None, view_is_flattened=None):
        """
        Write per-person depth_human vs depth_background (txt+json).
        If images/masks are given, also write flatten_vis_XXX.png overlays.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        rows = self._flatten_rows(people)
        (out / "flatten_depth_compare.json").write_text(json.dumps(rows, indent=2))
        (out / "flatten_depth_compare.txt").write_text(
            self._format_depth_txt(people, view_is_flattened)
        )
        vis_paths = []
        if images is not None and human_masks is not None and surround_masks is not None:
            vis_paths = self.save_vis_images(
                images, people, human_masks, surround_masks, out
            )
        return {
            "json_path": str(out / "flatten_depth_compare.json"),
            "txt_path": str(out / "flatten_depth_compare.txt"),
            "vis_paths": vis_paths,
        }

    def save_vis_images(self, images, people, human_masks, surround_masks, output_dir):
        """Overlay person pixels (red) and background ring (cyan) with A vs B text."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        if isinstance(images, (str, Path)):
            images = [images]
        vis_paths = []
        n = len(people)
        for i in range(n):
            rgb = self._to_rgb_hwc(images[i] if i < len(images) else images[-1])
            inst = human_masks[i] if i < len(human_masks) else np.zeros((0, *rgb.shape[:2]), np.uint8)
            rings = surround_masks[i] if i < len(surround_masks) else np.zeros((0, *rgb.shape[:2]), np.uint8)
            recs = people[i] if i < len(people) else []
            vis = self._draw_depth_compare(rgb, inst, rings, recs)
            path = out / f"flatten_vis_{i:03d}.png"
            Image.fromarray(vis).save(path)
            vis_paths.append(str(path))
        return vis_paths

    @staticmethod
    def _flatten_rows(people):
        rows = []
        for recs in people:
            for rec in recs:
                rows.append(
                    {
                        "view_id": rec.get("view_id"),
                        "person_id": rec.get("person_id"),
                        "depth_human": HumanFlatten._json_num(rec.get("depth_human", rec.get("A"))),
                        "depth_background": HumanFlatten._json_num(rec.get("depth_background", rec.get("B"))),
                        "C": HumanFlatten._json_num(rec.get("C")),
                        "score": HumanFlatten._json_num(rec.get("score")),
                        "valid": rec.get("valid"),
                        "is_flattened": rec.get("is_flattened"),
                    }
                )
        return rows

    @staticmethod
    def _format_depth_txt(people, view_is_flattened=None):
        lines = []
        for i, recs in enumerate(people):
            view_flag = None if view_is_flattened is None else bool(view_is_flattened[i])
            head = f"view {i}"
            if view_flag is not None:
                head += f"  view_is_flattened={view_flag}"
            lines.append(head)
            if not recs:
                lines.append("  (no person)")
                continue
            for rec in recs:
                pid = rec.get("person_id")
                a = rec.get("depth_human", rec.get("A"))
                b = rec.get("depth_background", rec.get("B"))
                lines.append(
                    "  person {pid}  depth_human={a}  depth_background={b}  "
                    "C={c}  score={score}  is_flattened={flat}".format(
                        pid=pid,
                        a=HumanFlatten._fmt_num(a),
                        b=HumanFlatten._fmt_num(b),
                        c=HumanFlatten._fmt_num(rec.get("C")),
                        score=HumanFlatten._fmt_num(rec.get("score")),
                        flat=rec.get("is_flattened"),
                    )
                )
        return "\n".join(lines) + "\n"

    @staticmethod
    def _draw_depth_compare(rgb, inst, rings, recs):
        inst = np.asarray(inst)
        rings = np.asarray(rings)
        if inst.size == 0:
            inst = np.zeros((0, *rgb.shape[:2]), dtype=np.uint8)
        elif inst.ndim == 2:
            inst = inst[None]
        if rings.size == 0:
            rings = np.zeros((0, *rgb.shape[:2]), dtype=np.uint8)
        elif rings.ndim == 2:
            rings = rings[None]
        h, w = rgb.shape[:2]
        vis = rgb.copy()
        target_hw = (h, w)
        human_col = np.asarray(_HUMAN_COLOR, dtype=np.float32)
        bg_col = np.asarray(_BG_COLOR, dtype=np.float32)
        for k in range(max(inst.shape[0], rings.shape[0])):
            if k < rings.shape[0]:
                ring = HumanFlatten._resize_mask(rings[k], target_hw)
                vis[ring] = (vis[ring].astype(np.float32) * 0.45 + bg_col * 0.55).astype(np.uint8)
            if k < inst.shape[0]:
                person = HumanFlatten._resize_mask(inst[k], target_hw)
                vis[person] = (vis[person].astype(np.float32) * 0.4 + human_col * 0.6).astype(np.uint8)
        im = Image.fromarray(vis)
        draw = ImageDraw.Draw(im)
        font = ImageFont.load_default()
        if not recs:
            draw.text((8, 8), "no person  (no depth_human vs depth_background)", fill=(255, 0, 0), font=font)
            return np.array(im)
        y = 8
        for rec in recs:
            pid = rec.get("person_id")
            a = HumanFlatten._fmt_num(rec.get("depth_human", rec.get("A")))
            b = HumanFlatten._fmt_num(rec.get("depth_background", rec.get("B")))
            score = HumanFlatten._fmt_num(rec.get("score"))
            label = f"p{pid} human={a} bg={b} score={score} flat={rec.get('is_flattened')}"
            bbox = draw.textbbox((8, y), label, font=font)
            draw.rectangle(bbox, fill=(0, 0, 0))
            draw.text((8, y), label, fill=(255, 255, 255), font=font)
            y = bbox[3] + 4
        return np.array(im)

    @staticmethod
    def _resize_mask(mask, hw):
        m = np.asarray(mask) > 0
        if m.shape == hw:
            return m
        m_img = Image.fromarray(m.astype(np.uint8) * 255)
        return np.array(m_img.resize((hw[1], hw[0]), Image.NEAREST)) > 0

    @staticmethod
    def _to_rgb_hwc(image):
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
    def _json_num(x):
        if x is None:
            return None
        x = float(x)
        return None if not np.isfinite(x) else x

    @staticmethod
    def _fmt_num(x):
        if x is None:
            return "nan"
        x = float(x)
        return "nan" if not np.isfinite(x) else f"{x:.6f}"

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
