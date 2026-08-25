# EX.NO: 4A - Normal Curves on UCI Diabetes Dataset
# AIM: Plot normal distribution curves for diabetes dataset features

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

np.random.seed(42)
n = 200
df = pd.DataFrame({
    'Glucose':       np.random.normal(137, 36, n).clip(50, 200).astype(int),
    'BloodPressure': np.random.normal(83, 19, n).clip(40, 130).astype(int),
    'SkinThickness': np.random.normal(29, 12, n).clip(5, 60).astype(int),
    'BMI':           np.random.normal(31, 7.8, n).clip(15, 60)
})

features = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI']
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

for ax, col in zip(axes.flatten(), features):
    mu, sigma = df[col].mean(), df[col].std()
    # Histogram
    ax.hist(df[col], bins=20, density=True, color='steelblue', alpha=0.7, edgecolor='white', label='Data')
    # Normal curve overlay
    x = np.linspace(df[col].min(), df[col].max(), 200)
    ax.plot(x, norm.pdf(x, mu, sigma), 'r-', linewidth=2.5, label=f'Normal (mu={mu:.1f})')
    ax.set_title(f'Normal Curve - {col}')
    ax.set_xlabel(col)
    ax.set_ylabel('Density')
    ax.legend()

plt.suptitle('Normal Curves - UCI Diabetes Dataset', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('normal_curves.png', dpi=150)
plt.show()
print("Saved: normal_curves.png")
