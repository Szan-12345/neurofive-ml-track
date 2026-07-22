import pandas as pd

# Load the Titanic dataset (make sure 'Titanic-Dataset.csv' is in the same folder)
df = pd.read_csv("Titanic-Dataset.csv")

# 1. Basic Info & Dimensions (Rows and Columns)
print("--- DATASET INFO ---")
df.info()

# 2. Statistical Summary
print("\n--- STATISTICAL SUMMARY ---")
print(df.describe())

# 3. First Few Rows
print("\n--- HEAD ---")
print(df.head())

# 4. Identify Missing Values
print("\n--- MISSING VALUES PER COLUMN ---")
print(df.isnull().sum())

# 5. Identify Categorical vs Numerical Columns
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

print(f"\nNumerical Columns ({len(numerical_cols)}): {numerical_cols}")
print(f"Categorical Columns ({len(categorical_cols)}): {categorical_cols}")
