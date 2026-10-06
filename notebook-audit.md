# Notebook and Dataset Audit

I read notebook cell source and saved output text without executing notebook cells.

## Bullseye / computer vision

| Local notebook | What is in it | Project use |
|---|---|---|
| `CNN.ipynb` | Target/background classifier work, image augmentation, YOLOv8 training, webcam prediction, CNN-before-YOLO and CNN-after-YOLO variants. One CNN gate uses a positive score threshold of `0.55`; a YOLO-only example uses confidence `0.7`. | Main source notebook to mine, then split into clean scripts. It does not show the owner-recalled OpenCV bullseye scoring stage. |
| `opencv.ipynb` | General OpenCV learning examples: thresholding, HSV masks, morphology, template matching, webcam filters, and feature demos. | Useful background, but no bullseye-specific scoring flow found. |
| `OPENCV_2.ipynb` | HSV/contour and box-ratio filtering for tracking red/black players in a football video. | Separate tracking experiment, not the bullseye detector. |
| `OPEN_CV_TASK.ipynb` | HSV thresholding and contours for red/purple fruit in an image. | General thresholding practice, not the bullseye detector. |

The OpenCV-to-YOLO pipeline in `target-detection/README.md` is based on the owner's description. The notebook evidence is documented separately as alternate CNN/YOLO experiments.

## Regression candidates

### Insurance charges

`LR_FINAL.ipynb` loads `insurance.csv` and works with `age`, `bmi`, `sex`, `region`, `children`, `smoker`, and target `charges`. It has a hand-written gradient-descent model, a closed-form linear model, and degree-2 feature-wise powers without cross terms. The notebook records roughly `0.785` test R-squared for one linear version and `0.776` to `0.786` across polynomial cells; different cells show different reruns. Treat these as historical, not comparable final scores. A second copy of the notebook was found in the referenced `ADR` folder.

`insurance.csv` was not found at the path recorded in the notebook, so the clean project script needs a licensed dataset copy before it can run.

### House prices

`LR_ROHAN.ipynb` loads `TRAIN_ROHAN.csv`, then inspects missing values and numeric distributions; it does not fit a regression model. The matching CSV was found and copied to `house-price-regression/data/train.csv`. It has 1,168 rows, ten numeric features, and target `SalePrice`.

`PAKKA_DATA.ipynb` is a different exploratory notebook for a `house_prices.csv` file. It parses Indian rupee Lac/Cr price strings and experiments with category encoding, but has no fitted regression model. That CSV was not found at its recorded path. Keep this as a separate future cleanup unless you confirm it is the same housing task.

### Other data work

- `DataPreProc.ipynb` explores a life-expectancy CSV, but the file it references was not found at its recorded path.
- `Diabetes_Missing_Data.csv` is also in the `ADR` folder, but the reviewed regression notebooks do not reference it; I did not attach it to a project without evidence that it belongs there.
- `CNN_Leaves.ipynb` appears to be a separate leaf-image classification project.
- The other notebooks in the folder are general data/ML practice or unrelated work based on their sources; they did not surface the remembered bullseye pipeline or a second completed regression model.

## What is included here

- `insurance-cost-regression/src/train_models.py`: reproducible linear and degree-2 polynomial baselines using a single train/test split and training-only preprocessing.
- `house-price-regression/src/train_models.py`: matching baselines for the recovered numeric housing dataset.
- `house-price-regression/data/train.csv`: local copy of the CSV, retained for your upload decision. Its source/licence was not established during this review.

The original notebooks remain on the PC; they were not copied into the upload kit because they mix experiments, saved output, and machine-specific paths.
