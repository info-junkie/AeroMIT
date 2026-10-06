# House Price Regression

Predict `SalePrice` from ten numeric house features in the recovered `TRAIN_ROHAN.csv` dataset. The project includes a clean linear baseline and a degree-2 polynomial comparison.

## Local source found

- `data manipulation/LR_ROHAN.ipynb` reads `TRAIN_ROHAN.csv` and performs initial data inspection/boxplots; it does not contain a trained regression model.
- `data manipulation/PAKKA_DATA.ipynb` is a separate exploratory/preprocessing notebook for a different `house_prices.csv` file. That file was not found at its recorded path.

The recovered CSV has 1,168 rows and 10 numeric input columns: `OverallQual`, `GrLivArea`, `GarageCars`, `GarageArea`, `TotalBsmtSF`, `1stFlrSF`, `FullBath`, `TotRmsAbvGrd`, `YearBuilt`, and `YearRemodAdd`. `SalePrice` is the target. It is copied into `data/train.csv` for this local starter project. Confirm dataset provenance and redistribution rights before publishing it.

## Run

```bash
python -m pip install -r requirements.txt
python src/train_models.py --data data/train.csv
```

This is a new, reproducible baseline, not a result reproduced from the original notebook. It uses a fixed 80/20 split, fits polynomial expansion and scaling inside the training pipeline, then reports MAE, RMSE, and R².
