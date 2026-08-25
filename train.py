"""Train and compare models on the on-time delivery dataset.

Run from the project root:
    python src/train.py
"""

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from evaluate import (
    print_class_balance,
    print_coefficients,
    print_feature_importances,
    print_metrics,
)
from preprocessing import build_preprocessor, load_data, split_data

MODEL_DIR = Path("models")


def build_models(random_state=5):
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000, random_state=random_state
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            random_state=random_state,
            n_jobs=-1,
        ),
    }


def main(data_path, save):
    df = load_data(data_path)
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns\n")
    print_class_balance(df)

    X_train, X_test, y_train, y_test = split_data(df)

    results = {}

    for name, estimator in build_models().items():
        print(f"\n{'=' * 60}\n{name.replace('_', ' ').upper()}\n{'=' * 60}")

        pipe = Pipeline(
            [("prep", build_preprocessor()), ("model", estimator)]
        )
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        acc = print_metrics(y_test, y_pred)
        results[name] = acc

        feature_names = pipe.named_steps["prep"].get_feature_names_out()

        if name == "logistic_regression":
            print_coefficients(pipe.named_steps["model"], feature_names)
        else:
            print_feature_importances(pipe.named_steps["model"], feature_names)

        if save:
            MODEL_DIR.mkdir(exist_ok=True)
            joblib.dump(pipe, MODEL_DIR / f"{name}.pkl")
            print(f"\nSaved -> {MODEL_DIR / f'{name}.pkl'}")

    print(f"\n{'=' * 60}\nSUMMARY\n{'=' * 60}")
    for name, acc in results.items():
        print(f"{name:25s} {acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/On_Time_Delivery.csv")
    parser.add_argument("--save", action="store_true", help="Persist fitted pipelines")
    args = parser.parse_args()

    main(args.data, args.save)
