"""Metrics, coefficient inspection and diagnostic plots."""

from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

REPORTS_DIR = Path("reports")


def print_class_balance(df, target="Reached.on.Time_Y.N"):
    """Baseline check — accuracy is misleading when classes are skewed."""
    balance = df[target].value_counts(normalize=True).sort_index()
    print("Class balance:")
    for label, share in balance.items():
        name = "on time" if label == 0 else "late"
        print(f"  {label} ({name:8s}) {share:.1%}")
    print(f"  Majority-class baseline: {balance.max():.1%}")


def print_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    print(f"\nAccuracy: {acc:.4f}")
    print("\nConfusion matrix (rows = actual, cols = predicted):")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=["on time", "late"]))
    return acc


def print_coefficients(model, feature_names, top_n=15):
    """Coefficients are comparable in magnitude because features are scaled."""
    coefs = (
        pd.DataFrame(
            {"feature": feature_names, "coefficient": model.coef_[0]}
        )
        .sort_values("coefficient", key=abs, ascending=False)
        .head(top_n)
    )
    print("\nCoefficients (positive -> increases odds of being late):")
    print(coefs.to_string(index=False))
    return coefs


def print_feature_importances(model, feature_names, top_n=15):
    imp = (
        pd.DataFrame(
            {"feature": feature_names, "importance": model.feature_importances_}
        )
        .sort_values("importance", ascending=False)
        .head(top_n)
    )
    print("\nFeature importances:")
    print(imp.to_string(index=False))
    return imp


def plot_discount_effect(df, save=True):
    """P(late) against discount level.

    The point of this plot: the curve has a sharp elbow rather than a gradual
    slope. That threshold shape is why logistic regression underperforms here
    and why a tree, which splits on thresholds natively, does better.
    """
    import matplotlib.pyplot as plt

    grouped = (
        df.groupby("Discount_offered")["Reached.on.Time_Y.N"]
        .agg(["mean", "count"])
        .query("count >= 20")
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(grouped.index, grouped["mean"], marker="o", linewidth=1.5)
    ax.axhline(0.5, linestyle="--", color="grey", linewidth=1)
    ax.set_xlabel("Discount offered (%)")
    ax.set_ylabel("P(late delivery)")
    ax.set_title("Late-delivery rate by discount level")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()

    if save:
        REPORTS_DIR.mkdir(exist_ok=True)
        fig.savefig(REPORTS_DIR / "discount_effect.png", dpi=150)
        print(f"Saved -> {REPORTS_DIR / 'discount_effect.png'}")

    return fig, ax
