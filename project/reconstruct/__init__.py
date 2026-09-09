"""
Subpackage role:
    Multi-view geometric reconstruction via VGGT-Omega.

Inputs:
    - images / image_paths
    - reconstructor config (checkpoint, resolution, device)

Outputs:
    - recon_result for pipeline; flatten uses depth_maps only

Variables / symbols (contract):
    - VGGTReconstructor: main reconstruction wrapper (vggt_reconstructor.py)
"""

# Export after implementation, e.g.:
# from .vggt_reconstructor import VGGTReconstructor
# __all__ = ["VGGTReconstructor"]
