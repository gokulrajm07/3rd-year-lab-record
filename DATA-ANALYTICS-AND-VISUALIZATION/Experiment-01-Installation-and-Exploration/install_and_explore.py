# EX.NO: 1 - Installation and Exploration of Libraries
# AIM: Install and explore NumPy, SciPy, Jupyter, Statsmodels, Pandas, Matplotlib, Seaborn, Plotly, Bokeh
# Run: pip install numpy scipy jupyter statsmodels pandas matplotlib seaborn plotly bokeh

import numpy as np
import pandas as pd
import matplotlib
import seaborn as sns
import statsmodels.api as sm
import scipy
import plotly
import bokeh

print("=" * 40)
print("   Library Version Report")
print("=" * 40)
print(f"NumPy        : {np.__version__}")
print(f"Pandas       : {pd.__version__}")
print(f"Matplotlib   : {matplotlib.__version__}")
print(f"Seaborn      : {sns.__version__}")
print(f"Statsmodels  : {sm.__version__}")
print(f"SciPy        : {scipy.__version__}")
print(f"Plotly       : {plotly.__version__}")
print(f"Bokeh        : {bokeh.__version__}")
print("=" * 40)
print("All libraries installed and verified!")
