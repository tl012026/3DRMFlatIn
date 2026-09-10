"""
Module role:
    Framework / orchestrator for the full pipeline.
    Loads config, calls detectors -> reconstruct -> flatten in order,
    passes intermediate results, and writes final outputs.
    Does NOT implement detection, reconstruction, or flatten-score algorithms.

Inputs:
    - config: runtime settings (paths, thresholds, switches)
    - image_paths: list of input image file paths (multi-view)
    - output_dir: directory for intermediate and final artifacts

Outputs:
    - pipeline_result: detection, reconstruction, and flatten verdicts
    - side-effect files under output_dir (boxes, masks, depth, scores)

Variables (contract):
    - config: full config object / dict
    - image_paths: List[str]
    - output_dir: str | Path
    - det_result: output from detectors (bboxes, masks, scores)
    - recon_result: output from reconstruct (depth_maps, poses, ...)
    - flat_result: output from flatten (A, B, C, scores, is_flattened)
    - pipeline_result: final bundled return value
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .detectors.person_detector import PersonDetector
from .detectors.person_segmenter import PersonSegmenter
from .flatten.human_flatten import HumanFlatten
from .reconstruct.vggt_reconstructor import VGGTReconstructor


def run_pipeline(config, image_paths, output_dir):
    """
    Role:
        Single entry that wires all stages.

    Inputs:
        config, image_paths, output_dir  (see module docstring)

    Outputs:
        pipeline_result

    Variables used inside (planned):
        det_result, recon_result, flat_result, pipeline_result
    """
    device = config["device"]
    dcfg, scfg = config["detector"], config["segmenter"]
    rcfg, fcfg = config["reconstruct"], config["flatten"]

    detector = PersonDetector(
        dcfg["weights_path"], dcfg["conf_thres"], device, dcfg["person_class_id"]
    )
    segmenter = PersonSegmenter(
        scfg["checkpoint_path"],
        device,
        scfg["mask_threshold"],
        model_cfg=scfg["model_cfg"],
        multimask_output=scfg["multimask_output"],
    )
    reconstructor = VGGTReconstructor(
        rcfg["checkpoint_path"],
        rcfg["image_resolution"],
        device,
        rcfg.get("use_mask_filter", False),
    )
    flatten = HumanFlatten(fcfg["score_thres"], fcfg["ring_width"])

    # detectors -> reconstruct -> flatten
    # flatten.run(depth_maps=recon_result.depth_maps, masks=det_result.masks)
    det_result = detector.predict(image_paths)
    seg_result = segmenter.predict(image_paths, det_result)
    recon_result = reconstructor.run(image_paths)
    flat_result = flatten.run(recon_result["depth_maps"], seg_result["masks"])

    # save artifacts to output_dir
    _save(output_dir, config.get("save") or {}, det_result, seg_result, recon_result, flat_result)

    pipeline_result = {
        "det": det_result,
        "seg": seg_result,
        "recon": recon_result,
        "flat": flat_result,
        "image_paths": image_paths,
        "output_dir": str(output_dir),
    }
    return pipeline_result


def _save(output_dir, save, det, seg, recon, flat):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if save.get("flatten_scores"):
        payload = {k: _jsonable(flat[k]) for k in ("people", "view_is_flattened")}
        (out / "flatten_scores.json").write_text(json.dumps(payload, indent=2))
    if save.get("det_boxes"):
        (out / "det_boxes.json").write_text(
            json.dumps(
                {"bboxes": _jsonable(det["bboxes"]), "scores": _jsonable(det["scores"])},
                indent=2,
            )
        )
    if save.get("masks"):
        for i, m in enumerate(seg["masks"]):
            np.save(out / f"mask_{i:03d}.npy", m)
    if save.get("depth_maps"):
        depth = recon["depth_maps"]
        if hasattr(depth, "detach"):
            depth = depth.detach().cpu().numpy()
        np.save(out / "depth_maps.npy", depth)


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if hasattr(x, "tolist"):
        return x.tolist()
    if isinstance(x, (float, int, bool)) or x is None:
        return x
    return float(x)
