# Insurance Cost Regression

Compare a linear regression baseline with a degree-2 polynomial regression model for predicting insurance charges from age, BMI, sex, smoker status, region, and number of children.

## Local source found

The work is in `data manipulation/LR_FINAL.ipynb`. It includes exploratory analysis, a hand-written gradient-descent linear model, a closed-form linear model, and manual degree-2 polynomial features made from individual feature powers (the polynomial section explicitly omits cross terms).

The notebook's last recorded outputs show a linear closed-form test R² of about `0.785` and a polynomial test R² of about `0.776` in one cell. Another polynomial cell reports about `0.786`; notebook cells have been rerun/edited over time, so treat these as historical, inconsistent outputs rather than a single authoritative benchmark.

## Data status

The notebook expects `insurance.csv` in a local `ADR` folder, but that file was not present at its recorded path when reviewed. Put a licensed copy at `data/insurance.csv` before running. The data is not included in this kit.

## Run

```bash
python -m pip install -r requirements.txt
python src/train_models.py --data data/insurance.csv
```

The clean script uses one fixed train/test split, fits preprocessing on training data only, and reports MAE, RMSE, and R² for both models. Its results will differ from the historical notebook because preprocessing and evaluation are made reproducible.

## Layout

```text
insurance-cost-regression/
├── README.md
├── data/README.md
├── src/train_models.py
├── requirements.txt
└── .gitignore
```
