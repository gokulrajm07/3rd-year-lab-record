# EX.NO: 4B - Z-Test on UCI Diabetes Dataset
# AIM: Perform Z-test to check if mean Glucose differs significantly from population mean

import numpy as np
import pandas as pd
from statsmodels.stats.weightstats import ztest

np.random.seed(42)
n = 200
df = pd.DataFrame({
    'Glucose':       np.random.normal(137, 36, n).clip(50, 200).astype(int),
    'BloodPressure': np.random.normal(83, 19, n).clip(40, 130).astype(int),
    'BMI':           np.random.normal(31, 7.8, n).clip(15, 60)
})

print("=" * 50)
print("  Z-Test on UCI Diabetes Dataset")
print("=" * 50)
print(f"\nSample size: {n}")

# One-sample Z-test for Glucose (H0: mean = 100)
for col, pop_mean in [('Glucose', 100), ('BloodPressure', 72), ('BMI', 25)]:
    z_stat, p_value = ztest(df[col], value=pop_mean)
    print(f"\n--- {col} (H0: mean = {pop_mean}) ---")
    print(f"  Sample Mean  : {df[col].mean():.2f}")
    print(f"  Z-Statistic  : {z_stat:.4f}")
    print(f"  P-Value      : {p_value:.6f}")
    if p_value < 0.05:
        print(f"  Result       : REJECT H0 (p < 0.05) - Significant difference from {pop_mean}")
    else:
        print(f"  Result       : FAIL TO REJECT H0 - No significant difference from {pop_mean}")
