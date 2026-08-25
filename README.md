# On-Time Delivery Prediction

Predicting whether an e-commerce shipment arrives on time, and — more usefully — identifying *which* operational factors drive lateness.

**Headline finding:** discount level is by far the strongest predictor of late delivery, and its effect is non-linear. Logistic regression tops out at ~65% accuracy because it can only fit a straight line to what is effectively a threshold effect. Tree-based models capture the threshold directly and perform better.

---

## Dataset

E-commerce shipping data: 10,999 orders, 12 features, binary target.

| Column | Type | Notes |
|---|---|---|
| `Warehouse_block` | categorical | A–F, one-hot encoded |
| `Mode_of_Shipment` | categorical | Flight / Ship / Road |
| `Customer_care_calls` | numeric | Number of support calls |
| `Customer_rating` | numeric | 1–5 |
| `Cost_of_the_Product` | numeric | USD |
| `Prior_purchases` | numeric | |
| `Product_importance` | ordinal | low < medium < high |
| `Gender` | categorical | |
| `Discount_offered` | numeric | % |
| `Weight_in_gms` | numeric | |
| `Reached.on.Time_Y.N` | **target** | 1 = late, 0 = on time |

Class balance is roughly 60/40 toward late, so accuracy alone is a weak metric — a model predicting "late" every time already scores ~60%. Per-class precision and recall are reported instead.

The CSV is not committed to this repo. Place `On_Time_Delivery.csv` in `data/` before running.

---

## Results

| Model | Accuracy | Recall (on-time) | Recall (late) |
|---|---|---|---|
| Logistic Regression | 0.654 | 0.58 | 0.71 |
| Random Forest | see `reports/` | | |

Logistic regression coefficients (MinMax-scaled, so magnitudes are comparable):

```
Discount_offered           6.42
Weight_in_gms             -1.61
Prior_purchases           -0.60
Customer_care_calls       -0.56
Cost_of_the_Product       -0.48
Warehouse_block_F          0.15
Product_importance         0.10
Mode_of_Shipment           0.01
```

`Discount_offered` dominates everything else by a factor of four. Inspecting P(late) against discount level shows a sharp elbow rather than a gradual slope — which is precisely the shape a linear model cannot represent, and the reason a tree does better here.

---

## Structure

```
.
├── data/                  # raw CSV (gitignored)
├── models/                # saved model artifacts (gitignored)
├── notebooks/             # exploratory analysis
├── reports/               # figures and output
├── src/
│   ├── preprocessing.py   # encoding + scaling pipeline
│   ├── train.py           # trains and evaluates models
│   └── evaluate.py        # metrics, coefficients, plots
├── requirements.txt
└── README.md
```

---

## Running it

```bash
git clone https://github.com/YOUR_USERNAME/shipment-delivery-prediction.git
cd shipment-delivery-prediction

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# place On_Time_Delivery.csv in data/, then:
python src/train.py
```

---

## Method notes

**Encoding.** `Warehouse_block` is one-hot encoded (no natural order). `Product_importance` is ordinal-encoded with an explicit `low < medium < high` ordering — scikit-learn's default is alphabetical, which would have mapped `high=0, low=1, medium=2` and fed the model a meaningless ordering. `Mode_of_Shipment` and `Gender` are label-encoded.

**Scaling.** MinMax applied to numeric columns only; one-hot columns are already 0/1 and are left alone.

**Leakage.** All transformers are fit on the training split only and applied to test via `transform`. Encoders and scaler are bundled into a single `ColumnTransformer` so the same fitted state applies to both splits.

**Dropped.** `ID` is a row identifier with no predictive value.

---

## What I'd do next

- Threshold tuning rather than the default 0.5 cutoff — the business cost of a missed late shipment is probably not equal to the cost of a false alarm
- Gradient boosting (XGBoost / LightGBM) as a stronger baseline
- SHAP values to confirm the discount threshold effect at the individual-prediction level
- Investigate whether the discount effect is causal or an artifact of how the dataset was constructed — a discount that large may be *applied because* an order is already known to be delayed

---

## License

MIT
