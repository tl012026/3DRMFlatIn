"""
Module role:
    Framework / orchestrator for the full pipeline.
    Loads config, calls detectors -> reconstruct -> flatten in order,
    passes intermediate results, and writes final outputs.
    Does NOT implement detection, reconstruction, or flattening algorithms.

Inputs:
    - config: runtime settings from configs/ (paths, thresholds, switches)
    - image_paths: list of input image file paths (multi-view)
    - output_dir: directory for intermediate and final artifacts

Outputs:
    - pipeline_result: aggregated result dict / object containing
      detection, reconstruction, and flattening products
    - side-effect files under output_dir (masks, depth, meshes, flat maps)

Variables (contract):
    - config: full config object / dict
    - image_paths: List[str]
    - output_dir: str | Path
    - det_result: output from detectors (bboxes, masks, scores)
    - recon_result: output from reconstruct (poses, depth, point cloud)
    - flat_result: output from flatten (2D unfolded map / UV / texture)
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
    # TODO: save artifacts to output_dir
    # TODO: return pipeline_result
    pass
