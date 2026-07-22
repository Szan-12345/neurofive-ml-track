import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split

#  Load and prepare dataset (same pipeline as before)
df = pd.read_csv("Titanic-Dataset.csv")
df = df.drop(columns=["Cabin", "Name", "Ticket", "PassengerId"])
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df = pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True)

X = df.drop(columns=["Survived"])
y = df["Survived"]

#  Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# BASELINE MODEL 
base_model = LogisticRegression(max_iter=500)
base_model.fit(X_train, y_train)
base_pred = base_model.predict(X_test)

base_acc = accuracy_score(y_test, base_pred)
print("=== ORIGINAL (BASELINE) MODEL ===")
print(f"Baseline Accuracy: {base_acc * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, base_pred))

# HYPERPARAMETER TUNING USING GridSearchCV 
# Tuning 'C' (regularization strength) and 'solver' (optimization algorithm)
param_grid = {
    "C": [0.01, 0.1, 1.0, 10.0, 100.0],
    "solver": ["liblinear", "lbfgs"],
}

grid_search = GridSearchCV(
    estimator=LogisticRegression(max_iter=1000),
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
)
grid_search.fit(X_train, y_train)

# Best Tuned Model
best_model = grid_search.best_estimator_
tuned_pred = best_model.predict(X_test)
tuned_acc = accuracy_score(y_test, tuned_pred)

print("\n=== TUNED MODEL (GridSearchCV) ===")
print(f"Best Hyperparameters: {grid_search.best_params_}")
print(f"Tuned Model Accuracy: {tuned_acc * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, tuned_pred))

#  EXPLANATION OF PRECISION, RECALL, F1 & ACCURACY LIMITATIONS 
"""
Why accuracy alone can be misleading for imbalanced datasets:
If a dataset has 95% of samples belonging to one class and 5% to another, a dummy model that 
predicts the majority class every time will achieve 95% accuracy while completely failing to detect 
the minority class. Precision, Recall, and F1-score fix this by evaluating how well the model handles 
false positives and false negatives individually.
- Precision: Out of all positive predictions, how many were actually correct?
- Recall: Out of all actual positive cases, how many did the model find?
- F1-score: The harmonic mean of Precision and Recall, balancing both metrics.
"""
# Titanic Survival Prediction - Advanced ML Track

## Updates & Hyperparameter Tuning
# Evaluation Metrics:Added `classification_report` tracking Precision, Recall, and F1-score to handle class distribution balance safely.
# Hyperparameter Tuning: Implemented `GridSearchCV` to optimize the regularization parameter `C` and solver algorithms.
# Results: Documented performance variations before and after tuning.