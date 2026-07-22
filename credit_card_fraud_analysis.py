import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# 1. Load Dataset (Using Telco Churn as a reliable imbalanced binary classification case)
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Quick Data Preparation & Cleaning
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
if "customerID" in df.columns:
  df.drop(columns=["customerID"], inplace=True)
df = pd.get_dummies(df, drop_first=True)

X = df.drop(columns=["Churn_Yes"])
y = df["Churn_Yes"]

# Impute missing values
imputer = SimpleImputer(strategy="median")
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. Check Class Balance and Visualize with a Bar Chart
print("=== ORIGINAL CLASS DISTRIBUTION ===")
print(y.value_counts(normalize=True) * 100)

plt.figure(figsize=(6, 4))
y.value_counts().plot(kind="bar", color=["skyblue", "salmon"])
plt.title("Target Class Balance Distribution")
plt.xlabel("Class (0: Retained/Normal, 1: Churn/Fraud)")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# 3. Train Baseline Model (Imbalanced)
baseline_model = RandomForestClassifier(random_state=42)
baseline_model.fit(X_train, y_train)
baseline_preds = baseline_model.predict(X_test)

print("\n=== CLASSIFICATION REPORT: BEFORE BALANCING (Baseline) ===")
print(classification_report(y_test, baseline_preds))

# 4. Apply SMOTE to Address Class Imbalance
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(
    f"\nAfter SMOTE - Training shape: {X_train_smote.shape}, Minority class count:"
    f" {sum(y_train_smote == 1)}"
)

# 5. Retrain Model with Balanced Data
balanced_model = RandomForestClassifier(random_state=42)
balanced_model.fit(X_train_smote, y_train_smote)
balanced_preds = balanced_model.predict(X_test)

print("\n=== CLASSIFICATION REPORT: AFTER SMOTE (Balanced) ===")
print(classification_report(y_test, balanced_preds))

# 6. Explanation on Why Accuracy is Misleading
"""
Why 'Accuracy' is a Misleading Metric for Imbalanced Datasets:
Accuracy measures the total percentage of correct predictions out of all instances. In highly imbalanced datasets 
(such as fraud detection or customer churn where positive cases might only make up 5-15%), a naive model that 
predicts the majority class 100% of the time can easily achieve 85-95% accuracy while completely failing to catch 
a single true positive case. Relying on Precision, Recall, F1-Score, and Confusion Matrices ensures that performance 
on the critical minority class is accurately evaluated.
"""