import numpy as np

def normalize(data):
    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0)
    return (data - means) / stds

def predict(m, b, x):
    return m * x + b

def mse(y, y_pred):
    n = len(y)
    return np.mean((y - y_pred) ** 2)

def gradient_descent(m_start, b_start, x, y, lrate=0.001, epochs=1000, tol = 1e-6):
    m, b = m_start, b_start
    loss = float('inf')
    
    for epoch in range(epochs):
        y_pred = predict(m,b,x)
        current_loss = mse(y, y_pred)
        
        if abs(loss - current_loss) < tol:
            print(f"Converged after {epoch} epochs")
            break
        
        m_gradient = (-2/len(x)) * np.sum(x * (y - y_pred))
        b_gradient = (-2/len(x)) * np.sum(y - y_pred)
        
        m = m - lrate * m_gradient
        b = b - lrate * b_gradient
        
        loss = current_loss 
            
    return m, b