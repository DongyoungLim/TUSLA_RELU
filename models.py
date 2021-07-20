import torch
import torch.nn as nn
import torch.nn.functional as F

class single_nn(nn.Module):
    def __init__(self, N=10, act_fn='relu'):
        super(single_nn, self).__init__()
        self.N = N
        self.act_fn = act_fn
        self.l1 = nn.Linear(2, N)
        self.l2 = nn.Linear(N, 1, bias=False)

    def forward(self, x):
        x = x.view(-1, x.shape[1])
        if self.act_fn == 'sigmoid':
            x = torch.sigmoid(self.l1(x))
        elif self.act_fn == 'relu':
            x = F.relu(self.l1(x))
        elif self.act_fn == 'tanh':
            x = torch.tanh(self.l1(x))
        x = self.l2(x)
        return x

class two_nn(nn.Module):
    def __init__(self, N=10, act_fn='relu'):
        super(two_nn, self).__init__()
        self.N = N
        self.act_fn = act_fn
        self.l1 = nn.Linear(2, N)
        self.l2 = nn.Linear(N, N)
        self.l3 = nn.Linear(N, 1, bias=False)

    def forward(self, x):
        x = x.view(-1, x.shape[1])
        x = torch.relu(self.l1(x))
        if self.act_fn == 'sigmoid':
            x = torch.sigmoid(self.l2(x))
        elif self.act_fn == 'relu':
            x = F.relu(self.l2(x))
        elif self.act_fn == 'tanh':
            x = torch.tanh(self.l2(x))
        x = self.l3(x)
        return x
