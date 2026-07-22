import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# 1. Load dataset
df = pd.read_csv("Titanic-Dataset.csv")

# 2. Handle missing values and clean columns
df = df.drop(columns=["Cabin", "Name", "Ticket", "PassengerId"])
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# 3. Encode categorical columns using pd.get_dummies
df = pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True)

# 4. Define Features (X) and Target (y)
X = df.drop(columns=["Survived"])
y = df["Survived"]

# 5. Split dataset into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Train Logistic Regression Model
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

# 7. Evaluate Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%\n")

# 8. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Confusion Matrix Explanation:
"""
Confusion Matrix Breakdown:
- True Negatives (Top-Left): Passengers correctly predicted as did not survive.
- False Positives (Top-Right): Passengers incorrectly predicted as survived (Type I error).
- False Negatives (Bottom-Left): Passengers incorrectly predicted as did not survive (Type II error).
- True Positives (Bottom-Right): Passengers correctly predicted as survived.
"""