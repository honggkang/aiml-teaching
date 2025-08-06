# linear_regression.ipynb

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Data creation
torch.manual_seed(0)
X = torch.rand(100, 1) * 10  # [100, 1]
y = 3 * X + 5 + torch.randn(100, 1)  # 정답 y = 3x + 5 + noise

# plt.scatter(X.numpy(), y.numpy())
# plt.title("Generated Data")
# plt.xlabel("X")
# plt.ylabel("y")
# plt.show()

# Model (hypothesis) definition
model = nn.Linear(1, 1)  # input_dim=1, output_dim=1


# Training loop
epochs = 100
learning_rate = 0.01

for epoch in range(epochs):
    y_pred = model(X)
    loss = ((y_pred - y) ** 2).mean()

    model.zero_grad()      # optimizer 대신 zero_grad()
    loss.backward()        # autograd로 gradient 계산
    with torch.no_grad():  # weight 업데이트 (no grad tracking)
        for param in model.parameters():
            param -= learning_rate * param.grad

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
