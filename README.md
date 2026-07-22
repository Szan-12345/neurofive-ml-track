# Neurofive Solutions - Machine Learning Internship Track

Welcome to my machine learning internship repository! This project portfolio demonstrates hands-on implementation of core data science, supervised learning, data cleaning pipelines, ensemble methods, and handling imbalanced datasets using Python and Scikit-Learn.

---

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [Milestones & Key Projects](#milestones--key-projects)
3. [Tech Stack & Libraries](#tech-stack--libraries)
4. [Project Structure](#project-structure)

---

## 🚀 Project Overview

Throughout this internship track, I progressed from foundational exploratory data analysis and single predictive models to advanced ensemble architectures (Bagging and Boosting) and production-grade techniques for handling real-world imbalanced data (SMOTE).

---

## 🛠️ Milestones & Key Projects

### 1. Exploratory Data Analysis & Baseline Models
* **Objective:** Clean datasets, engineer meaningful features, and build initial baseline models (Logistic Regression & Linear Regression).
* **Key Tasks:** Missing value imputation, categorical encoding, feature scaling, and performance benchmarking.

### 2. Ensemble Methods (Random Forest vs. XGBoost)
* **Objective:** Compare single-model baselines against powerful ensemble architectures to reduce variance and optimize predictive boundaries.
* **Key Techniques:**
  * **Random Forest (Bagging):** Built independent trees in parallel and averaged predictions via majority vote.
  * **XGBoost (Boosting):** Sequentially built optimized trees to correct residual errors from previous iterations.
* **Feature Importance Analysis:** Visualized and compared how each algorithm weighs critical variables (e.g., gender, passenger class, fare ratios).

### 3. Imbalanced Data Handling & Evaluation Metrics
* **Objective:** Address real-world class imbalance (such as fraud detection and customer churn) where traditional accuracy becomes misleading.
* **Techniques Used:** Applied **SMOTE (Synthetic Minority Over-sampling Technique)** to balance training class distributions.
* **Evaluation Shift:** Evaluated performance using **Precision, Recall, F1-Score, and Confusion Matrices** rather than relying solely on accuracy.

---

## 💻 Tech Stack & Libraries
* **Language:** Python
* **Data Manipulation & Analysis:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn, XGBoost, Imbalanced-Learn (`imblearn`)
* **Data Visualization:** Matplotlib, Seaborn
* **Version Control:** Git & GitHub

---

## 📂 Repository Structure

```text
├── credit_card_fraud_analysis.py   # Imbalanced data handling & SMOTE pipeline
├── ensemble_comparison.py          # Random Forest & XGBoost model comparison
├── .gitignore                      # Git exclusion rules (large datasets excluded)
└── README.md                       # Project documentation
## 🌐 Live Web App
Check out the live interactive machine learning application here:
🔗 [Customer Churn Predictor App](http://localhost:8501)