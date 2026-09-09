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
- **SAM 2.1 Large** (`sam2.1_hiera_large.pt` + `configs/sam2.1/sam2.1_hiera_l.yaml`): box-prompted masks. One mask per box (`multimask_output: false`). Merge instances with OR when a view has several people.
- **VGGT-Omega-1B-512** (`vggt_omega_1b_512.pt`, resolution 512): official camera + depth model. Do not use the 256 text-alignment checkpoint.

## Choice of threshold
- **YOLO `conf_thres`**: `0.25`. Raise it if extra people / false boxes pollute the mask.
- **SAM2 `mask_threshold`**: `0.0` (official default: logits > 0).
- **Flatten score `C / B`**: start at `0.05`. Below this, treat the human as flattened onto the background. Tune on a small labeled set; `A` / `B` use median depth so outliers hurt less. `B` is a dilated-mask ring, not the whole image.
