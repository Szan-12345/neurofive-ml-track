import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# 1. Load and Prepare Titanic Dataset
df = pd.read_csv("Titanic-Dataset.csv")

# Feature Engineering
df["family_size"] = df["SibSp"] + df["Parch"] + 1
df["fare_per_age"] = df["Fare"] / (df["Age"] + 1)

# Drop non-feature columns
df = df.drop(columns=["Cabin", "Name", "Ticket", "PassengerId"])

# Handle categorical missing values safely
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df = pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True)

# Separate Target and Features
X = df.drop(columns=["Survived"])
y = df["Survived"]

# Split data first
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# IMPERATIVE FIX: Impute ANY remaining NaNs across the entire feature set using median strategy
imputer = SimpleImputer(strategy="median")
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# 2. Train Models
# A. Baseline Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_acc = accuracy_score(y_test, lr_pred)

# B. Random Forest Classifier (Bagging)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)

# C. XGBoost Classifier (Boosting)
xgb_model = XGBClassifier(
    n_estimators=100, learning_rate=0.1, random_state=42, eval_metric="logloss"
)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
xgb_acc = accuracy_score(y_test, xgb_pred)

# 3. Print Performance Comparisons
print("=== MODEL PERFORMANCE COMPARISON ===")
print(f"Logistic Regression Accuracy: {lr_acc * 100:.2f}%")
print(f"Random Forest Accuracy:       {rf_acc * 100:.2f}%")
print(f"XGBoost Accuracy:             {xgb_acc * 100:.2f}%")

# 4. Feature Importances Comparison
rf_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
xgb_importances = pd.Series(xgb_model.feature_importances_, index=X.columns)

print("\n=== TOP 5 FEATURES (Random Forest) ===")
print(rf_importances.nlargest(5))

print("\n=== TOP 5 FEATURES (XGBoost) ===")
print(xgb_importances.nlargest(5))

# 5. Plot Feature Importances Side-by-Side
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
rf_importances.nlargest(5).plot(
    kind="barh", ax=axes[0], color="skyblue"
).set_title("Random Forest Top Features")
xgb_importances.nlargest(5).plot(
    kind="barh", ax=axes[1], color="salmon"
).set_title("XGBoost Top Features")
plt.tight_layout()
plt.show()

# 6. Explanation: Random Forest vs. XGBoost Combination Methods
"""
How Random Forest and XGBoost Differ:
Random Forest uses bagging (Bootstrap Aggregating), building multiple decision trees independently and in parallel, 
then averaging their predictions via majority vote to reduce variance. In contrast, XGBoost uses boosting, building trees 
sequentially where each new tree specifically tries to correct the errors and residual mistakes made by the previous ones. 
While Random Forest is robust to overfitting out-of-the-box, XGBoost builds heavily optimized trees that iteratively refine 
predictive boundaries for superior performance.
"""