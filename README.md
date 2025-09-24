# machine-learning

## SAT to GPA Predictor

A simple linear regression implementation from scratch to predict college GPA based on SAT scores.

### Overview

This project demonstrates how to build a linear regression model completely from scratch using only NumPy and basic Python. The model learns the relationship between SAT scores and GPA, then can predict a student's expected GPA given their SAT score.

### Features

- **From Scratch Implementation**: No machine learning libraries used
- **Manual Gradient Descent**: Custom optimization algorithm
- **Data Normalization**: Proper preprocessing for stable training
- **Visualization**: Plot of data points and regression line
- **Convergence Tracking**: Monitors training progress

### Files

- `linear_regression.py` - Core implementation (normalization, prediction, gradient descent)
- `main.py` - Training and visualization script
- `data.csv` - SAT to GPA dataset

### Usage

```python
# Train the model
python main.py

# Make a prediction
sat_score = 1800
predicted_gpa = predict(sat_score)  # Returns estimated GPA