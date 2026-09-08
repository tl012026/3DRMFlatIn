"""
Subpackage role:
    Multi-view geometric reconstruction for the human region.
    Wraps VGGT-Omega (or similar) to produce poses, depth, and 3D points.

Inputs:
    - images / image_paths
    - optional masks from detectors (to focus on human region)
    - reconstructor config (checkpoint, resolution, device)

Outputs:
    - recon_result for pipeline and flatten stages

Variables / symbols (contract):
    - VGGTReconstructor: main reconstruction wrapper (vggt_reconstructor.py)
"""

# Export after implementation, e.g.:
# from .vggt_reconstructor import VGGTReconstructor
# __all__ = ["VGGTReconstructor"]
