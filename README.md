## Overview
We first want to find instances where VGGT-Omega flattens objects onto the background. In this case, we focus on humans. After collecting multiple examples, we want to see if we can find correlations inside the model as to why objects are being flattened.

## Method
Three models are used: YOLO, SAM2, and VGGT-Omega. We first use YOLO to detect boxes labeled as person, and send those boxes to SAM2 as prompts, where pixel-wise human masks are produced. Then we run VGGT-Omega on the same views to get per-view depth. Person masks must be aligned to the VGGT image size before they are applied.

We can get:
A. the depth of the human (pixels inside the mask)
B. the depth of the surrounding area of the human (a ring just outside the mask)
C. the depth difference between human and background, `|A − B|`

If humans are not flattened, then `C / B` should be no less than a threshold.

## Choice of model
- **YOLO11s** (`yolo11s.pt`): COCO person class `0`. Enough for boxes; SAM2 does the masks.
- **SAM 2.1 Large** (`sam2.1_hiera_large.pt` + `configs/sam2.1/sam2.1_hiera_l.yaml`): box-prompted masks. One mask per box (`multimask_output: false`). Each person is scored; a view is flattened if any person is.
- **VGGT-Omega-1B-512** (`vggt_omega_1b_512.pt`, resolution 512): official camera + depth model. Do not use the 256 text-alignment checkpoint.

## Choice of threshold
- **YOLO `conf_thres`**: `0.25`. Raise it if extra people / false boxes pollute the mask.
- **SAM2 `mask_threshold`**: `0.0` (official default: logits > 0).
- **Flatten score `C / B`**: start at `0.05`. Below this, treat the human as flattened onto the background. Tune on a small labeled set; `A` / `B` use median depth so outliers hurt less. `B` is a dilated-mask ring, not the whole image.

## Environment
We run from a conda env with torch. This repo holds the code. Vendored `sam2`, `vggt-omega`, and `ultralytics` are added to `PYTHONPATH`, with `sam2` first. Weights and run outputs stay outside the git tree (`checkpoint_dir` in `default.yaml`). Pip cache and temp files go on scratch.

```bash
conda activate $ENV
export TMPDIR=$SCRATCH/tmp
export PIP_CACHE_DIR=$SCRATCH/.cache/pip
python -m pip install einops safetensors iopath polars fiftyone

export PYTHONPATH=$PWD/sam2:$PWD/vggt-omega:$PWD/ultralytics:$PWD
```

Place `sam2.1_hiera_large.pt` and `vggt_omega_1b_512.pt` in `checkpoint_dir`. `yolo11s.pt` is pulled there on first run if it is missing.

## Dataset
`--input_dir` is one VGGT multi-view batch, so each example needs its own folder. For COCO-2017 val (person only, up to 50 images), `prepare_coco.py` uses FiftyOne and writes:

```
images/coco/<index>/<file>.jpg
```

FiftyOne zoo and db go under a scratch `fiftyone` directory (see `prepare_coco.py`). Then one index is one job:

```bash
python prepare_coco.py
sbatch run.sbatch 1
```

Other datasets follow the same layout: one folder per example, then `--input_dir` or `sbatch run.sbatch <index>`.
