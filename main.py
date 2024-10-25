import pandas as pd
import numpy as np
import functions as fn
import matplotlib.pyplot as plt


df = pd.read_csv('Nairobi Office Price Ex.csv')

x = df["SIZE"].values
y = df["PRICE"].values

w = np.random.rand()
b = np.random.rand()

for epoch in range(10):
    y_pred = w * x + b
    error = fn.mean_squared_error(y, y_pred)
    print(f"Epoch {epoch+1}: MSE = {error:.4f}")
    w, b = fn.gradient_descent(x, y, w, b)

plt.scatter(x, y, color="blue", label="Data")
plt.plot(x, w * x + b, color="red", label="Line of best fit")
plt.xlabel("Office Size (sq. ft.)")
plt.ylabel("Office Price")
plt.title("A graph of Office Size in sq. ft. against Office Price")
plt.legend()
plt.show()

# What will the office price be when the size is 100 sq. ft?
size_100_pred = w * 100 + b
print(f"For office size 100 sq. ft. the price will be: {size_100_pred:.4f}")