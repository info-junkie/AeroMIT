# GitHub Upload Kit

This kit groups the local work into three clear project candidates. It contains new project documentation and clean starter scripts; it does not alter the source folders on the PC.

See [notebook-audit.md](notebook-audit.md) for the notebook-by-notebook findings.

## Suggested repositories

| Folder | GitHub repository | Status |
|---|---|---|
| `target-detection/` | `bullseye-detection` | Project map and experiment notes; pipeline implementation still needs to be recovered from the original scripts/notebooks. |
| `insurance-cost-regression/` | `insurance-cost-regression` | Clean linear and degree-2 polynomial baselines. The original `insurance.csv` is missing from its recorded location. |
| `house-price-regression/` | `house-price-regression` | Clean linear and polynomial baselines with the found 1,168-row `TRAIN_ROHAN.csv`, copied to `data/train.csv`. |

Each folder can become its own GitHub repository. Upload one folder at a time, after checking whether the dataset may be redistributed and whether the desired repo visibility is public or private.

## Existing GitHub profile snapshot

The public `info-junkie` profile currently shows three repositories:

- `cleaners`: an OpenEnv echo environment; a separate, substantial project.
- `Shourya`: currently contains only `OOP.py`.
- `ideal-tribble`: currently contains only `.gitignore`.

The profile has no bullseye or regression repository yet. The new folders here give those projects clearer names and entry points. The `cleaners` repo should remain separate; the other two need fuller descriptions or can be archived by the owner if they are no longer useful.

## Suggested portfolio presentation

Feature `bullseye-detection`, `insurance-cost-regression`, and `house-price-regression` once their README details and data rights are settled. Add a profile README that introduces the portfolio and links to those projects. Avoid uploading model checkpoints, full training runs, or unrelated notebook dumps by default.

## Local evidence reviewed

- `runs/detect`: YOLOv8 runs `train` through `train10` and saved `predict` images.
- `data manipulation/CNN.ipynb`: several target classifier / YOLO experiments.
- `data manipulation/LR_FINAL.ipynb`: medical insurance charge regression experiments.
- `data manipulation/LR_ROHAN.ipynb` and the matching `TRAIN_ROHAN.csv`: house price features and sale price; original notebook is exploratory only.

The starter scripts are cleaned, reproducible baselines, not byte-for-byte copies of the old notebooks. Recorded notebook metrics are historical results and should not be presented as results from these new scripts.
