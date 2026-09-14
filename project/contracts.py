"""
Module role:
    Shared naming contract for pipeline I/O variables.
    Typed dataclasses that stages return / pipeline passes between stages.

Inputs:
    None (reference + type definitions).

Outputs:
    Dataclass types imported by detectors / reconstruct / flatten / pipeline.

Variables (cross-stage contract):
    Stage: detectors
        - images / image_paths
        - bboxes, scores
        - masks, mask_scores
        - seg_meta
        - det_result: DetResult bundle for pipeline

    Stage: reconstruct
        - images, masks
        - poses, intrinsics, depth_maps, point_cloud, conf_maps
        - recon_meta
        - recon_result: ReconResult bundle for pipeline

    Stage: flatten
        - depth_maps from reconstruct (full image), masks from detectors
        - people[view][person]: A, B, C, score, is_flattened, depth_human, depth_background
        - view_is_flattened, human_masks, surround_masks, flat_meta
        - flat_result: FlatResult bundle for pipeline

    Stage: pipeline
        - config, image_paths, output_dir
        - det_result, recon_result, flat_result
        - pipeline_result: PipelineResult aggregate
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


# ---------------------------------------------------------------------------
# detectors
# ---------------------------------------------------------------------------

@dataclass
class DetBoxes:
    """Output of PersonDetector.predict (per-view lists aligned by image index)."""

    bboxes: List[Any]
    """Per-image boxes, each Nx4 in xyxy."""

    scores: List[Any]
    """Per-box confidence, aligned with bboxes."""


@dataclass
class SegMasks:
    """Output of PersonSegmenter.predict."""

    masks: List[Any]
    """Per-image masks: HxW or [N,H,W], aligned with views."""

    mask_scores: Optional[List[Any]] = None
    """Optional quality scores per mask."""

    seg_meta: Dict[str, Any] = field(default_factory=dict)
    """Optional: prompt type, resolution, etc."""


@dataclass
class DetResult:
    """
    Stage-level bundle from detectors for pipeline / reconstruct.
    Combines DetBoxes + SegMasks (+ optional image refs).
    """

    bboxes: List[Any]
    scores: List[Any]
    masks: List[Any]
    mask_scores: Optional[List[Any]] = None
    seg_meta: Dict[str, Any] = field(default_factory=dict)
    image_paths: Optional[List[str]] = None
    images: Optional[Any] = None

    @classmethod
    def from_parts(
        cls,
        boxes: DetBoxes,
        segs: SegMasks,
        image_paths: Optional[List[str]] = None,
        images: Optional[Any] = None,
    ) -> "DetResult":
        return cls(
            bboxes=boxes.bboxes,
            scores=boxes.scores,
            masks=segs.masks,
            mask_scores=segs.mask_scores,
            seg_meta=segs.seg_meta,
            image_paths=image_paths,
            images=images,
        )


# ---------------------------------------------------------------------------
# reconstruct
# ---------------------------------------------------------------------------

@dataclass
class ReconResult:
    """Stage-level bundle from reconstruct. Flatten consumes depth_maps only."""

    poses: Any
    """Camera extrinsics / decoded poses."""

    depth_maps: Any
    """Per-view full-image depth. Required by flatten (do not mask-crop)."""

    point_cloud: Any
    """Optional 3D points from VGGT; not used by flatten."""

    intrinsics: Optional[Any] = None
    conf_maps: Optional[Any] = None
    recon_meta: Dict[str, Any] = field(default_factory=dict)
    images: Optional[Any] = None
    masks: Optional[Any] = None


# ---------------------------------------------------------------------------
# flatten
# ---------------------------------------------------------------------------

@dataclass
class FlatResult:
    """Flatten verdicts: one record per person, plus a per-view flag."""

    people: List[List[Any]]
    """people[view][person] = {A, B, C, score, valid, is_flattened, depth_human, depth_background}."""

    view_is_flattened: List[bool]
    """True if any person in that view is flattened."""

    human_masks: Optional[List[Any]] = None
    """Per-view [N,H,W] person masks on the depth grid."""

    surround_masks: Optional[List[Any]] = None
    """Per-view [N,H,W] rings on the depth grid."""

    flat_meta: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# pipeline
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    """Final aggregate returned by run_pipeline."""

    det_result: DetResult
    recon_result: ReconResult
    flat_result: FlatResult
    image_paths: List[str]
    output_dir: Union[str, Any]
    config: Optional[Dict[str, Any]] = None


__all__ = [
    "DetBoxes",
    "SegMasks",
    "DetResult",
    "ReconResult",
    "FlatResult",
    "PipelineResult",
]
