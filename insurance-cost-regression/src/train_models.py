"""Compare linear and degree-2 polynomial models for insurance charges."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler


TARGET = "charges"
CATEGORICAL = ["sex", "smoker", "region"]
NUMERIC = ["age", "bmi", "children"]


def make_model(polynomial: bool) -> Pipeline:
    numeric_steps = []
    if polynomial:
        # Matches the notebook's feature-wise powers (no pairwise cross terms).
        numeric_steps.append(("powers", PolynomialFeatures(degree=2, include_bias=False)))
    numeric_steps.append(("scale", StandardScaler()))
    numeric_pipeline = Pipeline(numeric_steps)
    preprocess = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, NUMERIC),
            ("categorical", OneHotEncoder(handle_unknown="ignore", drop="first"), CATEGORICAL),
        ]
    )
    return Pipeline([("preprocess", preprocess), ("model", LinearRegression())])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/insurance.csv"))
    args = parser.parse_args()
    if not args.data.is_file():
        raise SystemExit(f"Dataset not found: {args.data}. See data/README.md.")

    data = pd.read_csv(args.data)
    required = set(NUMERIC + CATEGORICAL + [TARGET])
    missing = sorted(required - set(data.columns))
    if missing:
        raise SystemExit(f"Dataset is missing required columns: {', '.join(missing)}")

    x = data[NUMERIC + CATEGORICAL]
    y = pd.to_numeric(data[TARGET], errors="raise")
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    for name, polynomial in (("Linear", False), ("Polynomial degree 2", True)):
        model = make_model(polynomial)
        model.fit(x_train, y_train)
        prediction = model.predict(x_test)
        rmse = mean_squared_error(y_test, prediction) ** 0.5
        print(
            f"{name}: MAE={mean_absolute_error(y_test, prediction):.2f}, "
            f"RMSE={rmse:.2f}, R2={r2_score(y_test, prediction):.4f}"
        )


if __name__ == "__main__":
    main()
