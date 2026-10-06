"""Compare linear and degree-2 polynomial house-price baselines."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


TARGET = "SalePrice"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/train.csv"))
    args = parser.parse_args()
    if not args.data.is_file():
        raise SystemExit(f"Dataset not found: {args.data}")

    data = pd.read_csv(args.data)
    if TARGET not in data:
        raise SystemExit(f"Dataset must contain target column {TARGET!r}.")
    x = data.drop(columns=TARGET).apply(pd.to_numeric, errors="raise")
    y = pd.to_numeric(data[TARGET], errors="raise")
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    models = [
        ("Linear", make_pipeline(StandardScaler(), LinearRegression())),
        (
            "Polynomial degree 2",
            make_pipeline(
                PolynomialFeatures(degree=2, include_bias=False),
                StandardScaler(),
                LinearRegression(),
            ),
        ),
    ]
    for name, model in models:
        model.fit(x_train, y_train)
        prediction = model.predict(x_test)
        rmse = mean_squared_error(y_test, prediction) ** 0.5
        print(
            f"{name}: MAE={mean_absolute_error(y_test, prediction):,.2f}, "
            f"RMSE={rmse:,.2f}, R2={r2_score(y_test, prediction):.4f}"
        )


if __name__ == "__main__":
    main()
