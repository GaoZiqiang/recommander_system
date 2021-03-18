import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch import optim

input_size = 1
output_size =1
num_time_steps = 50
hidden_size = 16


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        for p in self.rnn.parameters():
            nn.init.normal(p, mean=0.0, std=0.001)

        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden_prev):
        out, hidden_prev = self.rnn(x, hidden_prev)
        # [1,seq,h] -> [seq,h]
        out = out.view(-1, hidden_size)
        out = self.linear(out)  # [seq,h] -> [seq,1]
        out = out.unsqueeze(dim=0)  # [1,seq,1]
        return out, hidden_prev

# class Net(nn.Module):
#     def __int__(self):
#         super(Net, self).__init__()
#         self.rnn = nn.RNN(
#             input_size=input_size,
#             hidden_size=hidden_size,
#             num_layers=1,
#             batch_first=True,  # [b,seq,f]
#         )
#         for p in self.rnn.parameters():
#             nn.init.normal(p, mean=0.0, std=0.001)
#
#         self.linear = nn.Linear(hidden_size, output_size)
#
#     def forward(self, x, hidden_prev):
#         out, hidden_prev = self.rnn(x, hidden_prev)
#         out = out.view(-1, hidden_size)  # out打平，用于和y计算损失 [1,seq,h] => [seq,h] 降维
#         out = self.linear(out)  # [seq,h] => [seq,1] 最后一个维度变为1
#         out = out.unsqueeze(dim=0)  # turn out [seq,1] => [1,seq,1] 增维、
#
#         return out, hidden_prev

### 模型训练
model = Net()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-2)
hidden_prev = torch.zeros(1, 1, hidden_size)  # h0

for iter in range(6000):
    start = np.random.randint(3, size=1)[0]  # random in [0,3)
    time_steps = np.linspace(start, start + 10, num_time_steps)
    data = np.sin(time_steps)
    data = data.reshape(num_time_steps, 1)
    x = torch.tensor(data[:-1]).float().view(1, num_time_steps - 1, 1)  # 0~48
    y = torch.tensor(data[1:]).float().view(1, num_time_steps - 1, 1)  # 预测1~49

    output, hidden_prev = model(x, hidden_prev)
    hidden_prev = hidden_prev.detach()

    loss = criterion(output, y)
    model.zero_grad()
    loss.backward()
    optimizer.step()

    if iter % 100 == 0:
        print('Iteration: {} loss {}'.format(iter, loss.item()))

### 模型测试
start = np.random.randint(3,size=1)[0]  # random in [0,3)
time_steps = np.linspace(start,start+10,num_time_steps)
data = np.sin(time_steps)
data = data.reshape(num_time_steps,1)
x = torch.tensor(data[:-1]).float().view(1,num_time_steps-1,1) # 0~48
y = torch.tensor(data[1:]).float().view(1,num_time_steps-1,1) # 预测1~49

predictions = []
input = x[:,0,:]
for _ in range(x.shape[1]):
    input = input.view(1,1,1)
    (pred,hidden_prev) = model(input,hidden_prev)
    input = pred
    predictions.append(pred.detach().numpy().ravel()[0])

### 图示
x = x.data.numpy().ravel()
y = y.data.numpy()
plt.scatter(time_steps[:-1],x.ravel(),s=90)
plt.plot(time_steps[:-1],x.ravel())

plt.scatter(time_steps[1:],predictions)
plt.show()