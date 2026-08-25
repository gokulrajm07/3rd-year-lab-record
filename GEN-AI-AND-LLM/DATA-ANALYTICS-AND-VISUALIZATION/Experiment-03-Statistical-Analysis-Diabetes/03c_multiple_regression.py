# EX.NO: 3C - Multiple Regression Analysis on Diabetes Datasets
# AIM: Apply multiple linear regression using multiple predictor variables

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

np.random.seed(42)
n = 200

def gen_dataset(seed):
    np.random.seed(seed)
    return pd.DataFrame({
        'Glucose':       np.random.normal(137, 36, n).clip(50, 200).astype(int),
        'BloodPressure': np.random.normal(83, 19, n).clip(40, 130).astype(int),
        'Age':           np.random.randint(21, 80, n),
        'BMI':           np.random.normal(31, 7.8, n).clip(15, 60),
    })

uci  = gen_dataset(42)
pima = gen_dataset(99)

features = ['Glucose', 'BloodPressure', 'Age']
target   = 'BMI'

for name, df in [("UCI Diabetes", uci), ("Pima Indians", pima)]:
    X = df[features]; y = df[target]
    Xt, Xte, yt, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression().fit(Xt, yt)
    y_pred = model.predict(Xte)

    print(f"\n{'='*50}")
    print(f"  {name} - Multiple Regression")
    print(f"{'='*50}")
    print(f"  R2 Score : {r2_score(yte, y_pred):.4f}")
    print(f"  MSE      : {mean_squared_error(yte, y_pred):.4f}")
    print(f"  MAE      : {mean_absolute_error(yte, y_pred):.4f}")
    print(f"  Coefficients:")
    for feat, coef in zip(features, model.coef_):
        print(f"    {feat}: {coef:.4f}")
    print(f"  Intercept: {model.intercept_:.4f}")
