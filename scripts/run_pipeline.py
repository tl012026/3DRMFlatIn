"""
Script role:
    CLI / entry script to launch the pipeline framework.
    Parses args, loads configs/default.yaml (or a user config),
    and calls flatten.pipeline.run_pipeline.
    No algorithm logic here.

Inputs:
    - --config: path to YAML config (default: configs/default.yaml)
    - --input_dir or --image_paths: multi-view image source
    - --output_dir: output directory
    (CLI values override config keys when provided)

Outputs:
    - pipeline_result from run_pipeline
    - files written under output_dir by the pipeline stages

Variables (contract):
    - args: parsed CLI namespace
    - config: loaded YAML dict / object
    - image_paths: List[str] resolved from args/config
    - output_dir: str | Path
    - pipeline_result: return value of run_pipeline
"""


def main():
    """
    Inputs:
        CLI args + config file
    Outputs:
        runs pipeline; exits with status code later
    Variables used:
        args, config, image_paths, output_dir, pipeline_result
    """
    # TODO: parse CLI
    # TODO: load YAML into config
    # TODO: resolve image_paths and output_dir
    # TODO: pipeline_result = run_pipeline(config, image_paths, output_dir)
    pass


if __name__ == "__main__":
    main()
