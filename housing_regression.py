import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# 1. Load California Housing Dataset from scikit-learn
housing = fetch_california_housing(as_frame=True)
df = housing.frame

# 2. Select 3-5 features that most affect price
# Features: MedInc (Median Income), HouseAge (House Age), AveRooms (Average Rooms), AveOccup (Average Occupancy)
selected_features = ["MedInc", "HouseAge", "AveRooms", "AveOccup"]
X = df[selected_features]
y = df["MedHouseVal"]  # Target variable: House value in hundreds of thousands

# 3. Split dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Evaluate Performance
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Model Evaluation Metrics:")
print(f"--- RMSE (Root Mean Squared Error): {rmse:.4f}")
print(f"--- R² Score: {r2:.4f}")

# 6. Scatter Plot: Predicted vs. Actual Prices
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.3, color="teal")
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    "r--",
    lw=2,
    label="Ideal Fit",
)
plt.title("Actual vs. Predicted House Prices")
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.legend()
plt.show()

# 7. Non-Technical Explanation of R² Score (Plain English)
"""
Plain English R² Explanation:
Think of the R² score as a measure of how well our model explains the ups and downs of housing prices compared to just guessing an average value. 
If an R² score is around 0.60, it means that roughly 60% of the variation in house prices can be successfully explained by our selected features 
like median income and room count. The remaining 40% comes from other hidden factors our model isn't tracking yet.
"""