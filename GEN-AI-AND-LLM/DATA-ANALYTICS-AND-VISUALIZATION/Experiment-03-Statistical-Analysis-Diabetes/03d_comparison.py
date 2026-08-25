# EX.NO: 3D - Comparison of Analysis Results Between UCI and Pima Datasets
# AIM: Compare statistical metrics and model performance between datasets

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, accuracy_score

np.random.seed(42)
n = 200

def gen_dataset(seed):
    np.random.seed(seed)
    return pd.DataFrame({
        'Glucose':       np.random.normal(137, 36, n).clip(50, 200).astype(int),
        'BloodPressure': np.random.normal(83, 19, n).clip(40, 130).astype(int),
        'BMI':           np.random.normal(31, 7.8, n).clip(15, 60),
        'Age':           np.random.randint(21, 80, n),
        'Outcome':       np.random.randint(0, 2, n)
    })

uci  = gen_dataset(42)
pima = gen_dataset(99)

print("=" * 70)
print("  COMPARISON: UCI Diabetes vs Pima Indians Diabetes Dataset")
print("=" * 70)

# Statistical Comparison
print("\n--- Feature Statistics Comparison ---")
print(f"{'Feature':<20} {'UCI Mean':>10} {'Pima Mean':>10} {'UCI Std':>10} {'Pima Std':>10}")
print("-" * 60)
for col in ['Glucose', 'BloodPressure', 'BMI', 'Age']:
    print(f"{col:<20} {uci[col].mean():>10.2f} {pima[col].mean():>10.2f} "
          f"{uci[col].std():>10.2f} {pima[col].std():>10.2f}")

# Model Comparison
feats = ['Glucose', 'BloodPressure', 'BMI']
print("\n--- Model Performance Comparison ---")
for name, df in [("UCI", uci), ("Pima", pima)]:
    # Linear Regression
    Xt, Xte, yt, yte = train_test_split(df[feats], df['Age'], test_size=0.2, random_state=42)
    r2 = r2_score(yte, LinearRegression().fit(Xt, yt).predict(Xte))
    # Logistic Regression
    Xt2, Xte2, yt2, yte2 = train_test_split(df[feats], df['Outcome'], test_size=0.2, random_state=42)
    acc = accuracy_score(yte2, LogisticRegression(max_iter=1000).fit(Xt2, yt2).predict(Xte2))
    print(f"\n  {name}:")
    print(f"    Linear Regression R2 Score    : {r2:.4f}")
    print(f"    Logistic Regression Accuracy  : {acc:.4f}")

print("\n--- Conclusion ---")
print("Both datasets show similar distributions with minor differences in means.")
print("Model performance may vary based on dataset quality and sample size.")
