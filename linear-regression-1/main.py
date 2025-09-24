import linear_regression as lr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data.csv')

def main():
    x_original = data.SAT.values
    y_original = data.GPA.values
    
    x_normalized = lr.normalize(x_original)
    y_normalized = lr.normalize(y_original)
    
    m, b = lr.gradient_descent(0, 0, x_normalized, y_normalized, lrate=0.01, epochs=10000)
    
    y_pred = lr.predict(m, b, x_normalized)
    loss = lr.mse(y_normalized, y_pred)
    
    x_input = lr.normalize([1567, 1983, 1245, 2134, 1678, 1892, 1456, 2267, 1321, 2045, 1789, 1923, 1543, 2187, 1654, 1976, 1421, 2289, 1387, 2076])

    y_input = []

    for i in x_input:
        y_input.append(lr.predict(m, b, i))
    
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    
    plt.scatter(x_normalized, y_normalized, color='red', label='Data Points')
    plt.scatter(x_input, y_input, color='green', label='Input Points')
    x_range = np.linspace(x_normalized.min(), x_normalized.max(), 100)
    y_range = lr.predict(m, b, x_range)
    
    plt.plot(x_range, y_range, color='blue', label='Regression Line')
    plt.xlabel('SAT')
    plt.ylabel('GPA')
    
    plt.title('Linear Regression (GPA vs SAT)')
    plt.grid(True)
    plt.legend()
    plt.show()
    
if __name__ == '__main__':
    main()