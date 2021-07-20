## package load
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from custom_optim import TUSLA
import os
import argparse

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


device = 'cuda' if torch.cuda.is_available() else 'cpu'


## argparse
parser = argparse.ArgumentParser(description = '1-D optimization with a normal distribution')
parser.add_argument('--lr', default=0.002, type=float, help='learning rate')
parser.add_argument('--epochs', default=500, type=int)
parser.add_argument('--eta', default='0', type=float)
parser.add_argument('--beta', default='1e10', type=float)
parser.add_argument('--r', default=14, type=int)

args = parser.parse_args()


lr = args.lr
eta = args.eta
beta = args.beta
epochs = args.epochs
r = args.r



theta_zero = 5.0
highest_degree = 30

hist = {'ADAM': [],
        'SGD': [],
        'TUSLA': [],
        'RMSprop' : [],
        'AMSGrad': [],
        }
opt_name = ['ADAM', 'AMSGrad', 'RMSprop', 'TUSLA', 'SGD']

x_path = np.random.randn(epochs, 1)

for name in opt_name:
    theta = torch.tensor(theta_zero, requires_grad=True, device=device, dtype=torch.float64)
    hist[name] += [theta.item()]

    if name == 'ADAM': opt = optim.Adam([theta], lr=0.001)
    elif name == 'AMSGrad': opt = optim.Adam([theta], lr=0.001, amsgrad=True)
    elif name == 'SGD': opt = optim.SGD([theta], lr=lr)
    elif name == 'RMSprop': opt = optim.RMSprop([theta], lr=0.01)
    elif name == 'TUSLA': opt = TUSLA([theta], lr=lr, eta=eta, beta=beta, r=r)

    for epoch in range(1, epochs):
        opt.zero_grad()
        x = torch.tensor(x_path[epoch-1], device=device)

        loss = (torch.abs(theta) <= 1) * (
                (x <= theta) * 2 * torch.pow(theta, 2)
                + (x > theta) * torch.pow(theta, 2)
                + torch.pow(theta, highest_degree)) + (
                torch.abs(theta) > 1) * (
                (x <= theta) * (4 * torch.abs(theta) - 2)
                + (x > theta) * (2 * torch.abs(theta) - 1)
                + torch.pow(theta, highest_degree))
        loss.backward()
        opt.step()
        hist[name] += [theta.item()]

    plt.plot(range(epochs), hist[name], label=name)


plt.plot(range(epochs), np.zeros(epochs))
plt.legend()
plt.xlabel('epoch')
plt.ylabel('theta')
plt.ylim([-0.1, 5.5])
plt.show()

