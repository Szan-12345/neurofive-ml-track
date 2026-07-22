import pandas as pd

df = pd.read_csv("Titanic-Dataset.csv")


print("--- DATASET INFO ---")
df.info()

print("\n--- STATISTICAL SUMMARY ---")
print(df.describe())

print("\n--- HEAD ---")
print(df.head())


print("\n--- MISSING VALUES PER COLUMN ---")
print(df.isnull().sum())


numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

print(f"\nNumerical Columns ({len(numerical_cols)}): {numerical_cols}")
print(f"Categorical Columns ({len(categorical_cols)}): {categorical_cols}")
# The Titanic dataset consists of 891 rows and 12 columns detailing passenger characteristics and survival outcomes. It features a mix of numerical features (such as Age, Fare, and SibSp) and categorical variables (including Sex, Ticket, and Embarked). Missing values are heavily concentrated in the Cabin column, with moderate gaps in Age and a couple of missing values in Embarked. Recognizing these missing patterns and data types provides a solid foundation for the upcoming data cleaning and feature engineering workflow.