# EX.NO: 3A - Univariate Analysis on Diabetes Datasets
# AIM: Calculate Mean, Median, Mode, Variance, Std Dev, Skewness, Kurtosis

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

np.random.seed(42)
n = 200

# Simulated UCI Diabetes Dataset
uci = pd.DataFrame({
    'Glucose':       np.random.normal(137, 36, n).clip(50, 200).astype(int),
    'BloodPressure': np.random.normal(83, 19, n).clip(40, 130).astype(int),
    'SkinThickness': np.random.normal(29, 12, n).clip(5, 60).astype(int),
    'Insulin':       np.random.normal(146, 93, n).clip(10, 400).astype(int),
    'BMI':           np.random.normal(31, 7.8, n).clip(15, 60),
    'Age':           np.random.randint(21, 80, n),
    'Outcome':       np.random.randint(0, 2, n)
})

# Simulated Pima Indians Dataset
pima = pd.DataFrame({
    'Glucose':       np.random.normal(136, 32, n).clip(50, 200).astype(int),
    'BloodPressure': np.random.normal(82, 21, n).clip(40, 130).astype(int),
    'SkinThickness': np.random.normal(30, 12, n).clip(5, 60).astype(int),
    'Insulin':       np.random.normal(148, 89, n).clip(10, 400).astype(int),
    'BMI':           np.random.normal(32.5, 6.9, n).clip(15, 60),
    'Age':           np.random.randint(21, 80, n),
    'Outcome':       np.random.randint(0, 2, n)
})

cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Age']

def univariate_analysis(df, name):
    print(f"\n{'='*60}")
    print(f"  {name} - Univariate Statistics")
    print(f"{'='*60}")
    print(f"{'Feature':<20} {'Mean':>8} {'Median':>8} {'Std':>8} {'Skew':>8} {'Kurt':>8}")
    print("-"*60)
    for col in cols:
        print(f"{col:<20} {np.mean(df[col]):>8.2f} {np.median(df[col]):>8.2f} "
              f"{np.std(df[col]):>8.2f} {skew(df[col]):>8.3f} {kurtosis(df[col]):>8.3f}")

univariate_analysis(uci,  "UCI Diabetes Dataset")
univariate_analysis(pima, "Pima Indians Dataset")
