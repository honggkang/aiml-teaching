# logistic_regression.ipynb

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
import numpy as np

# 1. data creation (binary classification)
X, y = make_classification(n_samples=200, n_features=2, n_redundant=0,
                           n_informative=2, n_clusters_per_class=1)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

plt.scatter(X[:, 0], X[:, 1], c=y.squeeze(), cmap='coolwarm')
plt.title("Classification Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# 2. Model (hypothesis) definition
class LogisticRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)  # input_dim=2, output_dim=1

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

model = LogisticRegressionModel()

# 3. loss function and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# 4. training loop
epochs = 100
for epoch in range(epochs):
    y_pred = model(X)
    loss = criterion(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        with torch.no_grad():
            predicted = (y_pred > 0.5).float()
            accuracy = (predicted == y).float().mean()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Accuracy: {accuracy.item():.4f}")

# 5. decision boundary visualization

def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min().numpy() - .5, X[:, 0].max().numpy() + .5
    y_min, y_max = X[:, 1].min().numpy() - .5, X[:, 1].max().numpy() + .5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    probs = model(grid).detach().numpy().reshape(xx.shape)
    
    plt.contourf(xx, yy, probs, levels=[0, 0.5, 1], alpha=0.5, cmap='coolwarm')
    plt.scatter(X[:, 0].numpy(), X[:, 1].numpy(), c=y.squeeze().numpy(), cmap='coolwarm')
    plt.title("Decision Boundary")
    plt.show()

plot_decision_boundary(model, X, y)
