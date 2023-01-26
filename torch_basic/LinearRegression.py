import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])

W = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

optimizer = optim.SGD([W,b], lr=0.01)

nb_epochs = 1000
for epoch in range(1, nb_epochs +1):
    hypothesis = x_train*W +b
    cost = torch.mean((hypothesis - y_train)**2)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 ==0 :
        print(f'Epoch {epoch} W: {W.item():.3f} b: {b.item():.3f} Cost: {cost.item():.6f}')

x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[1], [2], [3]])

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1,1)

    def forward(self, x):
        return self.linear(x)

model = LinearRegressionModel()


optimizer = optim.SGD(model.parameters(), lr=0.01)
nb_epochs = 1000
for i in range(nb_epochs+1):
    hypothesis = model(x_train)

    cost = F.mse_loss(hypothesis, y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if i % 100 ==0:
        params = list(model.parameters())
        w = params[0].item()
        b = params[1].item()
        print(f' epochs : {i} W: {w:.3f} b: {b:.3f} Cost: {cost.item():.6f}')

