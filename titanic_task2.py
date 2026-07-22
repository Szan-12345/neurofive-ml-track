import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Load dataset
df = pd.read_csv("Titanic-Dataset.csv")

# HANDLE MISSING VALUES 
# Justification:
# - Cabin: Has >70% missing data, so dropping the column is better than imputing false data.
# - Age: Imputing with the median age preserves data distribution without letting extreme age values skew it.
# - Embarked: Only 2 missing values, so filling them with the mode (most common port) is safe and efficient.

df = df.drop(columns=["Cabin"])
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
# DETECT OUTLIERS USING A BOXPLOT (Fare column) 
plt.figure(figsize=(6, 4))
sns.boxplot(x=df["Fare"], color="orange")
plt.title("Outlier Detection in Passenger Fare")
plt.show()

# VISUALIZATIONS (Histogram, Boxplot, Bar Chart, Correlation Heatmap) 

# 1. Histogram (Age Distribution)
plt.figure(figsize=(8, 5))
sns.histplot(df["Age"], bins=30, kde=True, color="blue")
plt.title("Histogram of Passenger Ages")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

# 2. Boxplot (Fare vs Pclass)
plt.figure(figsize=(8, 5))
sns.boxplot(x="Pclass", y="Fare", data=df, palette="Set2")
plt.title("Fare Distribution Across Passenger Classes")
plt.show()

# 3. Bar Chart (Survival Rate by Sex)
plt.figure(figsize=(6, 4))
sns.barplot(x="Sex", y="Survived", data=df, palette="coolwarm")
plt.title("Survival Rate by Gender")
plt.ylabel("Survival Probability")
plt.show()

# 4. Correlation Heatmap (Numerical columns only)
plt.figure(figsize=(8, 6))
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Numerical Features")
plt.show()

# --- STEP 4: WRITTEN ANSWER ---
"""
Question: Which feature do you think most affects survival, and why?
Answer: 
Based on exploratory data analysis and visualization (such as the survival bar chart), 
'Sex' (gender) most significantly affects survival. Historically and statistically, 
female passengers had a drastically higher survival rate compared to male passengers 
due to the 'women and children first' protocol enforced during the evacuation of the Titanic.
"""