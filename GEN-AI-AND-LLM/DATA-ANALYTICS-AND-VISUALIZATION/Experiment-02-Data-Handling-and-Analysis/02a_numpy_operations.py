# EX.NO: 2A - Working with NumPy Arrays
# AIM: Demonstrate NumPy array creation, indexing, slicing, and operations

import numpy as np

print("NumPy Version:", np.__version__)

# --- Array Creation ---
arr_1d = np.array([1, 2, 3, 4, 5])
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
arr_zeros = np.zeros((3, 3))
arr_ones = np.ones((2, 4))
arr_range = np.arange(0, 10, 2)
arr_linspace = np.linspace(0, 1, 5)

print("\n--- Array Creation ---")
print("1D Array:", arr_1d)
print("2D Array:\n", arr_2d)
print("Zeros:\n", arr_zeros)
print("Ones:\n", arr_ones)
print("arange(0,10,2):", arr_range)
print("linspace(0,1,5):", arr_linspace)

# --- Indexing and Slicing ---
print("\n--- Indexing & Slicing ---")
print("arr_1d[2]:", arr_1d[2])
print("arr_2d[1,2]:", arr_2d[1, 2])
print("arr_1d[1:4]:", arr_1d[1:4])
print("arr_2d[0,:]:", arr_2d[0, :])
print("arr_2d[:,1]:", arr_2d[:, 1])

# --- Element-wise Operations ---
a = np.array([10, 20, 30])
b = np.array([1, 2, 3])
print("\n--- Element-wise Operations ---")
print("Add:", a + b)
print("Sub:", a - b)
print("Mul:", a * b)
print("Div:", a / b)
print("Scalar*2:", a * 2)
print("Square:", a ** 2)

# --- Aggregation Functions ---
print("\n--- Aggregations ---")
print("Sum:", np.sum(a))
print("Mean:", np.mean(a))
print("Std Dev:", np.std(a))
print("Min:", np.min(a))
print("Max:", np.max(a))

# --- Boolean Masking ---
print("\n--- Boolean Masking ---")
print("Elements > 15:", a[a > 15])

# --- Fancy Indexing ---
indices = [0, 2]
print("Fancy Indexing [0,2]:", a[indices])

# --- Reshape & Transpose ---
reshaped = arr_1d.reshape(5, 1)
print("\nReshaped (5,1):\n", reshaped)
print("Transpose of 2D:\n", arr_2d.T)

# --- Structured Array ---
structured = np.array(
    [(25, 90.5), (30, 85.2)],
    dtype=[('age', 'i4'), ('score', 'f4')]
)
print("\nStructured Array:", structured)
print("Ages:", structured['age'])
print("Scores:", structured['score'])
