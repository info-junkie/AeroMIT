# Bullseye Detection Pipeline

An efficient target-detection project intended to reduce expensive detector work by filtering candidate frames or regions first.

## Pipeline map

The intended first version, as recalled by the project owner, is:

1. Use an OpenCV threshold-based stage to find likely bullseye candidates and calculate a candidate score.
2. Send only candidates above a score cutoff to YOLO for object detection.
3. Review the candidate filter and YOLO together on held-out images/video, including false positives and misses.

The old `data manipulation/CNN.ipynb` also contains later/alternate experiments: a CNN screen before YOLO with a configurable positive-class threshold of `0.55`, YOLO followed by CNN classification of each detected crop, and a webcam YOLO-only run using confidence `0.7`. These should be described as alternate experiments until the owner confirms which version was actually used.

## Evidence and current status

- `runs/detect/train8` is the strongest completed YOLO run found: 50 epochs, best recorded mAP50–95 `0.966` at epoch 43. Its arguments point to a local `Documents/dataset.yaml`.
- `train10` ran 100 epochs on the same recorded dataset path, with best mAP50–95 `0.756` at epoch 51 and `0.619` at the final epoch.
- `train7` stopped after 4 epochs. `train`, `train2`–`train6`, and `train9` retain settings only.
- `runs/detect/predict` contains 61 annotated prediction images.
- The reviewed OpenCV notebooks show general HSV thresholding, contours, and template matching. No bullseye-specific OpenCV scoring implementation was identified in those notebooks.

These metrics are copied from local run CSVs; they have not been independently reproduced. The dataset YAML and training images were not in the reviewed folder, so the exact labels, class names, split, and possible leakage still need confirmation.

## Suggested repository layout

```text
bullseye-detection/
├── README.md
├── docs/
│   └── experiment-map.md
├── src/
│   ├── opencv_candidates.py
│   └── detect.py
├── configs/
│   └── pipeline.yaml
├── examples/             # small, licensed sample images only
├── requirements.txt
└── .gitignore
```

Keep large weights and full datasets out of Git until their provenance and intended release are confirmed.

## Original material to recover

Look for the OpenCV candidate-scoring code, its threshold/score definition, and the YOLO invocation in the project source folders or notebook history. The current `CNN.ipynb` is a mixed experiment notebook and contains machine-specific paths; it should be cleaned before publishing.
