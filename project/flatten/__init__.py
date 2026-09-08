"""
Subpackage role:
    Map reconstructed human surface geometry into a 2D flattened representation
    (unfolded map / UV / texture atlas style output).

Inputs:
    - recon_result (point cloud / depth / poses) from reconstruct
    - optional masks / images for coloring the flat map
    - flatten config (method switches, resolution, seam policy)

Outputs:
    - flat_result consumed by pipeline for saving / visualization

Variables / symbols (contract):
    - HumanFlatten: main flattening module (human_flatten.py)
"""

# Export after implementation, e.g.:
# from .human_flatten import HumanFlatten
# __all__ = ["HumanFlatten"]
