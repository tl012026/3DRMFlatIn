"""
Module role:
    Human surface flattening: convert 3D human-region geometry into a 2D map.
    Handles parameterization / unfolding, optional texture baking, and export.

Inputs:
    - point_cloud: xyz [, rgb] from reconstruct
    - depth_maps / poses / intrinsics: optional helpers for projection
    - masks: optional human masks
    - images: optional source views for texture
    - flatten_resolution: output 2D map size (H, W) or single int
    - method: placeholder for flattening strategy name (string)
    - seam_policy: how to handle cuts / discontinuities (string)

Outputs:
    - flat_map: 2D array / image of the unfolded surface
    - uv_coords: optional per-point or per-vertex UVs in [0,1]^2
    - flat_mesh: optional mesh with UVs if mesh path is used
    - texture: optional baked RGB texture aligned to flat_map / UV
    - flat_meta: dict (method, resolution, valid pixel count, etc.)

Variables (contract):
    - self.flatten_resolution, self.method, self.seam_policy
    - point_cloud, depth_maps, poses, intrinsics, masks, images
    - flat_map, uv_coords, flat_mesh, texture, flat_meta
"""


class HumanFlatten:
    """
    Role:
        Core flattening stage (algorithm body left empty for now).

    Inputs (constructor):
        flatten_resolution, method, seam_policy

    Outputs (run):
        flat_map, uv_coords, flat_mesh, texture, flat_meta
    """

    def __init__(self, flatten_resolution, method, seam_policy):
        # TODO: store config vars only
        pass

    def run(self, point_cloud, depth_maps=None, poses=None, intrinsics=None,
            masks=None, images=None):
        """
        Inputs:
            point_cloud (required); optional depth_maps, poses, intrinsics,
            masks, images for projection / texturing
        Outputs:
            dict with keys:
                flat_map, uv_coords, flat_mesh, texture, flat_meta
        Variables used:
            self.flatten_resolution, self.method, self.seam_policy,
            point_cloud, depth_maps, poses, intrinsics, masks, images
        """
        # TODO: parameterize surface; rasterize flat_map; optional texture bake
        pass
