# EX.NO: 2D - Descriptive Analytics using the Iris Dataset
# AIM: Explore and visualize the Iris dataset using Pandas, Seaborn, Matplotlib

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load Iris dataset (built-in seaborn dataset)
df = sns.load_dataset('iris')

print("=== First 5 Rows ===")
print(df.head())

print("\n=== Dataset Info ===")
df.info()

print("\n=== Summary Statistics ===")
print(df.describe())

print("\n=== Species Count ===")
print(df['species'].value_counts())

print("\n=== Mean by Species ===")
print(df.groupby('species').mean())

# Histogram of all numeric features
df.drop('species', axis=1).hist(figsize=(10, 7), bins=15, edgecolor='black', color='steelblue')
plt.suptitle('Iris Dataset - Feature Distributions', fontsize=14)
plt.tight_layout()
plt.savefig('iris_histograms.png')
plt.show()
print("Saved: iris_histograms.png")

# Boxplot - Sepal Length by Species
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='species', y='sepal_length', palette='Set2')
plt.title('Sepal Length by Species')
plt.savefig('iris_boxplot.png')
plt.show()
print("Saved: iris_boxplot.png")

# Pairplot
sns.pairplot(df, hue='species', palette='husl')
plt.savefig('iris_pairplot.png')
plt.show()
print("Saved: iris_pairplot.png")

print("\nDescriptive analytics complete!")
