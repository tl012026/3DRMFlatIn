"""
Module role:
    Run VGGT-Omega feed-forward reconstruction on multi-view images.
    Produces per-view depth maps that flatten scores against person masks.
    Keep full-image depth (do not mask-crop): flatten needs the surround ring.

Inputs:
    - images: preprocessed multi-view tensor (e.g. [S,3,H,W])
      OR image_paths + preprocess settings
    - checkpoint_path: VGGT-Omega weights
    - image_resolution: e.g. 512
    - device: "cuda" | "cpu"
    - masks: optional person masks from detectors (align to image size)
    - use_mask_filter: if true, drop non-human depth (breaks flatten surround B)

Outputs:
    - poses: camera extrinsics / pose encodings decoded to cameras
    - intrinsics: camera intrinsics if available
    - depth_maps: per-view depth
    - point_cloud: fused or per-view 3D points (xyz [, rgb])
    - conf_maps: optional confidence maps from the model
    - recon_meta: dict (resolution, num_views, scale notes, etc.)

Variables (contract):
    - self.model: VGGTOmega instance
    - self.checkpoint_path, self.image_resolution, self.device
    - self.use_mask_filter
    - images, masks
    - poses, intrinsics, depth_maps, point_cloud, conf_maps, recon_meta
"""

import torch

from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera

class VGGTReconstructor:
    """
    Role:
        Thin wrapper around vggt_omega for this project.

    Inputs (constructor):
        checkpoint_path, image_resolution, device, use_mask_filter

    Outputs (run):
        poses, intrinsics, depth_maps, point_cloud, conf_maps, recon_meta
    """

    def __init__(self, checkpoint_path, image_resolution, device, use_mask_filter):
        # TODO: store config vars; build self.model; load checkpoint
        self.checkpoint_path = checkpoint_path
        self.image_resolution = image_resolution
        self.device = device
        self.use_mask_filter = use_mask_filter

        self.model = VGGTOmega().to("cude").eval

    def run(self, images, masks=None):
        """
        Inputs:
            images: multi-view preprocessed images
            masks: optional person masks aligned to images
        Outputs:
            dict with keys:
                poses, intrinsics, depth_maps, point_cloud, conf_maps, recon_meta
        Variables used:
            self.model, self.image_resolution, self.device,
            self.use_mask_filter, images, masks
        """
        # TODO: forward VGGT-Omega; decode poses; return full-image depth_maps
        # What is the sequence of outputs from the model?
        predictions = self.model(images)
        poses, intrinsics = encoding_to_camera(predictions["pose_enc"], predictions["images"].shape[-2:])
        depth_maps = predictions["depth"]
        conf_maps = predictions["conf_maps"]
        recon_meta = predictions["recon_meta"]
        return {
            "poses": poses,
            "intrinsics": intrinsics,
            "depth_maps": depth_maps,
            "conf_maps": conf_maps,
            "recon_meta": recon_meta
        }
        pass
