import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
X = pd.read_csv("predictor.csv", header=None).values  # Independent/Predictor Variable
y = pd.read_csv("response.csv", header=None).values  # Dependent/Response Variable

# Normalize the predictor variable (X)
X = (X - np.mean(X)) / np.std(X)

# Add a column of ones to X for the intercept term (theta0)
X = np.hstack((np.ones((X.shape[0], 1)), X))

# Initialize parameters (theta), learning rate, and number of iterations
theta = np.zeros((2, 1))  # [theta0, theta1]
learning_rate = 0.5
iterations = 1000
m = len(y)  # Number of training examples

# Define the cost function
def compute_cost(X, y, theta):
    predictions = X @ theta
    errors = predictions - y
    cost = (1 / (2 * m)) * np.sum(errors**2)
    return cost

# Gradient descent function
def gradient_descent(X, y, theta, learning_rate, iterations):
    cost_history = []
    for _ in range(iterations):
        gradients = (1 / m) * (X.T @ (X @ theta - y))
        theta -= learning_rate * gradients
        cost_history.append(compute_cost(X, y, theta))
    return theta, cost_history

# Train the model using batch gradient descent
theta, cost_history = gradient_descent(X, y, theta, learning_rate, iterations)

# Print final parameters and cost
print(f"Final parameters (theta0, theta1): {theta.flatten()}")
print(f"Final cost: {cost_history[-1]}")

# Plot cost vs iterations for the first 50 iterations
plt.figure(figsize=(8, 5))
plt.plot(range(50), cost_history[:50], 'b-')
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost vs Iterations (First 50)")
plt.grid()
plt.show()

# Plot dataset and the fitted line
plt.figure(figsize=(8, 5))
plt.scatter(X[:, 1], y, color='red', label='Data points')  # Original data
plt.plot(X[:, 1], X @ theta, color='blue', label='Fitted Line')  # Regression line
plt.xlabel("Normalized Predictor (X)")
plt.ylabel("Response (y)")
plt.title("Dataset and Fitted Line")
plt.legend()
plt.grid()
plt.show()

# Test different learning rates and plot cost vs iterations for each
learning_rates = [0.005, 0.5, 5]
plt.figure(figsize=(10, 6))
for lr in learning_rates:
    theta_test = np.zeros((2, 1))
    _, cost_history_test = gradient_descent(X, y, theta_test, lr, 50)
    plt.plot(range(50), cost_history_test, label=f"lr = {lr}")
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost vs Iterations for Different Learning Rates")
plt.legend()
plt.grid()
plt.show()

# Stochastic Gradient Descent (SGD)
def stochastic_gradient_descent(X, y, theta, learning_rate, iterations):
    cost_history = []
    for _ in range(iterations):
        for i in range(m):
            xi = X[i, :].reshape(1, -1)
            yi = y[i]
            gradients = xi.T @ (xi @ theta - yi)
            theta -= learning_rate * gradients
        cost_history.append(compute_cost(X, y, theta))
    return theta, cost_history

# Mini-Batch Gradient Descent
def mini_batch_gradient_descent(X, y, theta, learning_rate, iterations, batch_size=16):
    cost_history = []
    for _ in range(iterations):
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        for i in range(0, m, batch_size):
            X_batch = X_shuffled[i:i + batch_size]
            y_batch = y_shuffled[i:i + batch_size]
            gradients = (1 / len(X_batch)) * (X_batch.T @ (X_batch @ theta - y_batch))
            theta -= learning_rate * gradients
        cost_history.append(compute_cost(X, y, theta))
    return theta, cost_history

# Compare Gradient Descent Methods
theta_sgd, cost_history_sgd = stochastic_gradient_descent(X, y, np.zeros((2, 1)), 0.05, 50)
theta_mbgd, cost_history_mbgd = mini_batch_gradient_descent(X, y, np.zeros((2, 1)), 0.05, 50)

plt.figure(figsize=(10, 6))
plt.plot(range(50), cost_history[:50], label="Batch Gradient Descent")
plt.plot(range(50), cost_history_sgd[:50], label="Stochastic Gradient Descent")
plt.plot(range(50), cost_history_mbgd[:50], label="Mini-Batch Gradient Descent")
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Comparison of Gradient Descent Methods")
plt.legend()
plt.grid()
plt.show()

