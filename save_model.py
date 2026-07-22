import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import pandas as pd

# Load dataset and train your best model
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
if "customerID" in df.columns:
    df.drop(columns=["customerID"], inplace=True)
df = pd.get_dummies(df, drop_first=True)

X = df.drop(columns=["Churn_Yes"])
y = df["Churn_Yes"]

imputer = SimpleImputer(strategy="median")
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Save the trained model and feature columns to disk
joblib.dump(model, "model.pkl")
joblib.dump(list(X.columns), "model_columns.pkl")
print("Model successfully saved as model.pkl!")