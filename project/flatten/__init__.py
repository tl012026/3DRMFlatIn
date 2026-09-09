"""
Subpackage role:
    Detect VGGT-Omega human-flatten artifacts on per-view depth maps.
    Compare person-mask depth (A) vs a surrounding ring (B); score = |A-B|/B.

Inputs:
    - depth_maps from reconstruct (full image depth, not mask-cropped)
    - person masks from detectors (resized to the depth grid)
    - flatten config (score_thres, ring_width, merge_instances)

Outputs:
    - flat_result: A, B, C, scores, is_flattened, valid, aligned masks

Variables / symbols (contract):
    - HumanFlatten: depth-based flatten detector (human_flatten.py)
"""

from .human_flatten import HumanFlatten

__all__ = ["HumanFlatten"]
