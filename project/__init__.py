"""
Package role:
    Top-level package: detect VGGT-Omega human-flatten artifacts
    (YOLO + SAM2 masks, VGGT depth, A/B/C score).
    Exposes the orchestration entry only; algorithm details live in subpackages.

Inputs:
    None (package init).

Outputs:
    Public symbols for external import (to be exported after implementation).

Variables / symbols (contract):
    - run_pipeline: unified pipeline entry (defined in pipeline.py)
"""

# Export after implementation, e.g.:
# from .pipeline import run_pipeline
# __all__ = ["run_pipeline"]
