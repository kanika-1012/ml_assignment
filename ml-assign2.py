import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Load datasets
independent_vars = pd.read_csv(r'C:\Users\KIIT\Downloads\logisticX.csv', header=None)


dependent_var = pd.read_csv(r'C:\Users\KIIT\Downloads\logisticY.csv', header=None)

X = independent_vars.values
y = dependent_var.values.ravel()  # Flatten to 1D array

# Initialize and train logistic regression model
model = LogisticRegression(solver='lbfgs', C=10, max_iter=1000)
model.fit(X, y)

# Coefficients and cost function value
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
cost = -np.mean(y * np.log(model.predict_proba(X)[:, 1]) + (1 - y) * np.log(1 - model.predict_proba(X)[:, 1]))
print("Cost function value:", cost)

# Plot cost vs. iterations (mock example)
iterations = list(range(1, 51))
costs = [cost + i * 0.01 for i in range(50)]  # Placeholder costs for illustration

plt.figure()
plt.plot(iterations, costs, label="Cost")
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost vs. Iterations')
plt.legend()
plt.show()

# Plot dataset and decision boundary
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', label='Data Points')
x_vals = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
y_vals = -(model.coef_[0][0] * x_vals + model.intercept_[0]) / model.coef_[0][1]
plt.plot(x_vals, y_vals, color='red', label='Decision Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Boundary with Data Points')
plt.legend()
plt.show()

# Add new features (squared)
X_new = np.hstack([X, X[:, 0:1]**2, X[:, 1:2]**2])

# Train on the new dataset
model_new = LogisticRegression(solver='lbfgs', C=10, max_iter=1000)
model_new.fit(X_new, y)

# Plot new decision boundary
plt.figure()
plt.scatter(X_new[:, 0], X_new[:, 1], c=y, cmap='viridis', label='Data Points')
x_vals = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
y_vals = -(model_new.coef_[0][0] * x_vals + model_new.coef_[0][2] * x_vals**2 + model_new.intercept_[0]) / model_new.coef_[0][1]
plt.plot(x_vals, y_vals, color='red', label='New Decision Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('New Decision Boundary with Squared Features')
plt.legend()
plt.show()

# Confusion matrix and metrics
y_pred = model_new.predict(X_new)
cm = confusion_matrix(y, y_pred)
print("Confusion Matrix:\n", cm)

accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

