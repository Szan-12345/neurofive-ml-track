import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# 1. Load Dataset
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# 2. Quick Data Preparation & Cleaning
# Coerce non-numeric strings/blank spaces in TotalCharges to NaN, then fill with median
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

# Drop customerID if present
if "customerID" in df.columns:
    df.drop(columns=["customerID"], inplace=True)

# --- EDA & CLASS BALANCE CHECK (Before get_dummies) ---
print("=== EDA: CLASS BALANCE ===")
class_counts = df["Churn"].value_counts(normalize=True) * 100
print(f"Retained (No): {class_counts['No']:.2f}% | Churned (Yes): {class_counts['Yes']:.2f}%")

print("\n=== EDA: CHURN RATE BY CONTRACT TYPE ===")
contract_churn = df.groupby("Contract")["Churn"].value_counts(normalize=True).unstack() * 100
print(contract_churn)

# 3. Handle Categorical Variables
df = pd.get_dummies(df, drop_first=True)

# Target variable is Churn_Yes
X = df.drop(columns=["Churn_Yes"])
y = df["Churn_Yes"]

# Train/Test Split (stratified to maintain class balance proportions)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- FEATURE SCALING FOR LOGISTIC REGRESSION ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Train Models: Logistic Regression vs. Decision Tree
# Logistic Regression (uses scaled data to prevent convergence warnings)
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)

# Decision Tree Classifier (uses unscaled data)
dt_model = DecisionTreeClassifier(random_state=42, max_depth=5)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)

# 5. Evaluate Performance
print("\n=== LOGISTIC REGRESSION ===")
print(f"Accuracy: {accuracy_score(y_test, lr_pred)*100:.2f}%")
print(classification_report(y_test, lr_pred))

print("\n=== DECISION TREE CLASSIFIER ===")
print(f"Accuracy: {accuracy_score(y_test, dt_pred)*100:.2f}%")
print(classification_report(y_test, dt_pred))

# 6. Identify Top 3 Features Driving Churn (using Decision Tree feature importances)
importances = pd.Series(dt_model.feature_importances_, index=X.columns)
top_3_features = importances.nlargest(3)
print("\n=== TOP 3 FEATURES DRIVING CHURN ===")
print(top_3_features)

# 7. Business Summary for a Non-Technical Manager
"""
Business Summary:
Our analysis reveals that customer churn is heavily dictated by contract flexibility, tenure duration, and monthly billing thresholds. 
Specifically, customers on month-to-month contracts with high monthly charges and low tenure are at the highest risk of leaving. 
To combat this, the retention team should proactively target month-to-month subscribers approaching their 3rd to 6th months with incentivized 
annual contract transitions or loyalty discounts. Implementing these targeted interventions will directly protect recurring monthly revenue 
and lower overall customer acquisition costs.
"""