import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# 1. Load Dataset
df = pd.read_csv("Titanic-Dataset.csv")

# 2. Feature Engineering (Creating 2 new features)
# - family_size: Total family members onboard (SibSp + Parch + 1 for the passenger themselves)
# - fare_per_age: Ratio of ticket fare to passenger age
df["family_size"] = df["SibSp"] + df["Parch"] + 1
df["fare_per_age"] = df["Fare"] / (df["Age"] + 1)  # +1 to avoid division by zero

# Drop irrelevant identifiers and high-missing columns
df = df.drop(columns=["Cabin", "Name", "Ticket", "PassengerId"])

# Define Target and Features
X = df.drop(columns=["Survived"])
y = df["Survived"]

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Identify Numerical and Categorical Columns
numerical_cols = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()
categorical_cols = X.select_dtypes(
    include=["object", "bool", "category"]
).columns.tolist()

# 4. Create Preprocessing Pipelines for Numerical and Categorical data
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

# Combine preprocessing steps using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols),
    ]
)

# 5. Chain Preprocessing and Model into a Single Pipeline
# Using RandomForestClassifier to leverage the newly engineered features effectively
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42, n_estimators=100)),
    ]
)

# 6. Fit and Evaluate the Pipeline
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("=== PIPELINE EVALUATION (WITH FEATURE ENGINEERING) ===")
print(f"Pipeline Test Accuracy: {accuracy * 100:.2f}%\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# 7. Save Final Pipeline Using Joblib
joblib.dump(pipeline, "titanic_pipeline_model.pkl")
print(
    "Pipeline successfully trained and saved as 'titanic_pipeline_model.pkl'!"
)
