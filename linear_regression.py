# Linear regression from scratch

import numpy as np

# Feature: House size
x = np.array([1, 2, 3, 4, 5], dtype=float)

# Target: House price
y = np.array([3, 5, 7, 9, 11], dtype=float)

# Initialize parameters
w = 0.0   # weight (Zero knowledge values)
b = 0.0   # bias   (Zero knowledge values)

lr = 0.01  # learning rate
n = len(x)

# Train using Batch Gradient Descent
for _ in range(5000):
    y_pred = w * x + b

    dw = (-2/n) * np.sum(x * (y - y_pred))
    db = (-2/n) * np.sum(y - y_pred)

    w = w - lr * dw
    b = b - lr * db

print("Final weight:", w)
print("Final bias:", b)

