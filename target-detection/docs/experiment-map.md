# Experiment Map

## Known runs

| Run | Evidence | Interpretation |
|---|---|---|
| `train` through `train6` | `args.yaml` only | Early YOLOv8n setup attempts. Dataset YAML paths move between Linux and Windows locations; `train4`–`train6` specify CPU. |
| `train7` | 4 metric rows, checkpoints | Short/incomplete run using `Documents/dataset.yaml`; CPU. |
| `train8` | 50 metric rows, plots and checkpoints | Completed GPU run; best mAP50–95 `0.966` at epoch 43. |
| `train9` | `args.yaml` only | Config points to relative `data.yaml`; no result files found. |
| `train10` | 100 metric rows, plots and checkpoints | Completed GPU run on `Documents/dataset.yaml`; best mAP50–95 `0.756` at epoch 51. |
| `predict` | 61 JPG files | Saved prediction outputs; the exact checkpoint used is not recorded in this folder. |

## Candidate pipeline versions

1. **Owner-recalled design:** OpenCV threshold/shape candidate filter → candidate score cutoff → YOLO.
2. **CNN prefilter experiment:** CNN classifies full camera frame; only positive frames go to YOLO. The notebook sets a positive-class score threshold of `0.55` in one version.
3. **CNN verification experiment:** YOLO produces boxes → crop each box → CNN calls target/background.
4. **YOLO-only webcam experiment:** YOLO confidence cutoff `0.7`.

The code for versions 2 through 4 is visible in `CNN.ipynb`. The OpenCV threshold, candidate scoring formula, and handoff to YOLO for version 1 were not found in the reviewed notebooks. Keep these versions separate in the eventual README until they are reconciled.

## Publishing checklist

- Confirm the actual dataset, class names, and split used for each run.
- Identify which checkpoint `predict` used.
- Recover the OpenCV scoring source and define its cutoff.
- Use a small sample set with permission to publish; do not upload personal/private images.
- Exclude `.pt`, `.pth`, full run directories, and dataset assets until size, rights, and intended visibility are decided.
