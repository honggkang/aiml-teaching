# linear_regression.ipynb

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Data creation
torch.manual_seed(0)
X = torch.rand(100, 1) * 10  # [100, 1]
y = 3 * X + 5 + torch.randn(100, 1)  # 정답 y = 3x + 5 + noise

plt.scatter(X.numpy(), y.numpy())
plt.title("Generated Data")
plt.xlabel("X")
plt.ylabel("y")
plt.show()

# Model (hypothesis) definition
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)  # input=1, output=1

    def forward(self, x):
        return self.linear(x)

model = LinearRegressionModel()

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop
epochs = 100
losses = []

for epoch in range(epochs):
    y_pred = model(X)
    loss = criterion(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# Visualization
predicted = model(X).detach()

plt.scatter(X.numpy(), y.numpy(), label='True')
plt.plot(X.numpy(), predicted.numpy(), color='red', label='Predicted')
plt.legend()
plt.title("Linear Regression Result")
plt.show()

# 6. 학습된 파라미터 확인
[w, b] = model.parameters()
print(f"Learned weight: {w.item():.3f}, bias: {b.item():.3f}")
