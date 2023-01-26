import torch

device = torch.device("mps") if torch.backends.mps.is_available() else "cpu"
print(device)
torch.manual_seed(777)

if device == 'mps':
    torch.manual_seed(777)

X = torch.FloatTensor([[0, 0], [0, 1], [1, 0], [1, 1]]).to(device)
Y = torch.FloatTensor([[0], [1], [1], [0]]).to(device)

# linear = torch.nn.Linear(2, 1, bias = True)
# sigmoid = torch.nn.Sigmoid()

# model = torch.nn.Sequential(linear, sigmoid).to(device)

# criterion = torch.nn.BCELoss().to(device)
# optimizer = torch.optim.SGD(model.parameters(), lr=1)

# for step in range(10001):
#     optimizer.zero_grad()
#     hypothesis = model(X)

#     cost = criterion(hypothesis, Y)
#     cost.backward()
#     optimizer.step()
    
#     if step % 1000 ==0:
#         print(f'step : {step} cost : {cost.item()}')

## MLP
linear1 = torch.nn.Linear(2, 2, bias = True)
linear2 = torch.nn.Linear(2, 1, bias = True)
sigmoid = torch.nn.Sigmoid()

model = torch.nn.Sequential(linear1, sigmoid, linear2, sigmoid).to(device)

criterion = torch.nn.BCELoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=1)

for step in range(10001):
    optimizer.zero_grad()
    hypothesis = model(X)

    cost = criterion(hypothesis, Y)
    cost.backward()
    optimizer.step()
    
    if step % 1000 ==0:
        print(f'step : {step} cost : {cost.item()}')