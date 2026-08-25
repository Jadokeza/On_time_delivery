"""Data loading and preprocessing for the on-time delivery dataset.

All transformers are bundled into a single ColumnTransformer so the exact
same fitted state is applied to train and test — no manual reapplication,
no leakage.
"""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder

TARGET = "Reached.on.Time_Y.N"
DROP_COLS = ["ID"]

ONEHOT_FEATURES = ["Warehouse_block"]
ORDINAL_FEATURES = ["Product_importance", "Mode_of_Shipment", "Gender"]
NUMERIC_FEATURES = [
    "Cost_of_the_Product",
    "Discount_offered",
    "Weight_in_gms",
    "Customer_care_calls",
    "Prior_purchases",
    "Customer_rating",
]

# Explicit ordering — sklearn defaults to alphabetical, which would map
# high=0, low=1, medium=2 and hand the model a nonsense ordering.
IMPORTANCE_ORDER = ["low", "medium", "high"]


def load_data(path="data/On_Time_Delivery.csv"):
    """Load the raw CSV and drop identifier columns."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Place On_Time_Delivery.csv in data/."
        )

    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    return df


def build_preprocessor():
    """Return an unfitted ColumnTransformer covering all feature types."""
    return ColumnTransformer(
        transformers=[
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ONEHOT_FEATURES,
            ),
            (
                "importance",
                OrdinalEncoder(
                    categories=[IMPORTANCE_ORDER],
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
                ["Product_importance"],
            ),
            (
                "ordinal",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value", unknown_value=-1
                ),
                ["Mode_of_Shipment", "Gender"],
            ),
            ("numeric", MinMaxScaler(), NUMERIC_FEATURES),
        ],
        remainder="drop",
    )


def split_data(df, train_size=0.8, random_state=5):
    """Split into train/test, stratified on the target to preserve class balance."""
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    return train_test_split(
        X,
        y,
        train_size=train_size,
        random_state=random_state,
        stratify=y,
    )


def get_feature_names(fitted_preprocessor):
    """Readable feature names after transformation, for coefficient inspection."""
    return list(fitted_preprocessor.get_feature_names_out())
