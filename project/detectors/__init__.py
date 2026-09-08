"""
Subpackage role:
    Person / body region detection and segmentation.
    Produces 2D localization cues for reconstruction and flattening.

Inputs:
    - images or image_paths
    - detector / segmenter config (weights, conf, device)

Outputs:
    - det_result consumed by pipeline and downstream stages

Variables / symbols (contract):
    - PersonDetector: YOLO-style box detector (person_detector.py)
    - PersonSegmenter: SAM2-style mask segmenter (person_segmenter.py)
"""

# Export after implementation, e.g.:
# from .person_detector import PersonDetector
# from .person_segmenter import PersonSegmenter
# __all__ = ["PersonDetector", "PersonSegmenter"]
