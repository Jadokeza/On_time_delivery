import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, MinMaxScaler

## LOAD
df = pd.read_csv("/Users/jado-1/Desktop/DATA_PROJECTS/SHIP/On_Time_Delivery.csv")

## ENCODING  (all fit on df, not df_encoded)
OHE = OneHotEncoder(sparse_output=False)
warehouse_encoded = OHE.fit_transform(df[["Warehouse_block"]])

Ord_encoder = OrdinalEncoder(categories=[["low", "medium", "high"]])
Product_Importance_encoded = Ord_encoder.fit_transform(df[["Product_importance"]])

ship_encoder = OrdinalEncoder()
shipmode_encoded = ship_encoder.fit_transform(df[["Mode_of_Shipment"]])

Gender_encoder = OrdinalEncoder()
Gender_encoded = Gender_encoder.fit_transform(df[["Gender"]])

## WRAP INTO DATAFRAMES
warehouse_df = pd.DataFrame(
    warehouse_encoded,
    columns=OHE.get_feature_names_out(["Warehouse_block"]),
    index=df.index
)
importance_df = pd.DataFrame(
    Product_Importance_encoded,
    columns=["Product_importance_encoded"],
    index=df.index
)
ship_mode_df = pd.DataFrame(
    shipmode_encoded,
    columns=["Mode_of_Shipment_encoded"],
    index=df.index
)
gender_df = pd.DataFrame(
    Gender_encoded,
    columns=["Gender_encoded"],
    index=df.index
)

## DROP ORIGINALS AND CONCAT
df_encoded = pd.concat(
    [df.drop(columns=["ID", "Warehouse_block", "Product_importance",
                      "Mode_of_Shipment", "Gender"]),
     warehouse_df,
     importance_df,
     ship_mode_df,
     gender_df],
    axis=1
)

## SPLITTING
X = df_encoded.drop("Reached.on.Time_Y.N", axis=1)
Y = df_encoded["Reached.on.Time_Y.N"]
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, train_size=0.8, random_state=5
)

## SCALING
numeric_features = ["Cost_of_the_Product", "Discount_offered", "Weight_in_gms",
                    "Customer_care_calls", "Prior_purchases"]

scaler = MinMaxScaler()
X_train_scaled = X_train.copy()
X_test_scaled  = X_test.copy()

X_train_scaled[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_test_scaled[numeric_features]  = scaler.transform(X_test[numeric_features])

print(X_train_scaled.dtypes)
print(X_train_scaled.head())

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

## TRAIN  (X_train_scaled already MinMax-scaled above)
log_reg = LogisticRegression(max_iter=1000, random_state=5)
log_reg.fit(X_train_scaled, Y_train)

## PREDICT
Y_pred = log_reg.predict(X_test_scaled)

## EVALUATE
print("Accuracy:", accuracy_score(Y_test, Y_pred))
print("\nConfusion matrix:\n", confusion_matrix(Y_test, Y_pred))
print("\nClassification report:\n", classification_report(Y_test, Y_pred))

## COEFFICIENTS — which features actually drive late delivery
coefs = pd.DataFrame({
    "feature": X_train_scaled.columns,
    "coefficient": log_reg.coef_[0]
}).sort_values("coefficient", key=abs, ascending=False)

print("\n", coefs)
