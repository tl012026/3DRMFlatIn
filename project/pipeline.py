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
    # TODO: load config fields needed by each stage
    # TODO: call detectors -> reconstruct -> flatten
    # flatten.run(depth_maps=recon_result.depth_maps, masks=det_result.masks)
    # TODO: save artifacts to output_dir
    # TODO: return pipeline_result
    pass
