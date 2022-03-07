## package load
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from custom_optim import TUSLA
import os
import argparse

os.environ['KMP_DUPLICATE_LIB_OK']='True'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

## argparse
parser = argparse.ArgumentParser(description = '1-D optimization with a beta distribution')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--epochs', default=1000, type=int)
parser.add_argument('--eta', default=0, type=float)
parser.add_argument('--beta', default=1e10, type=float)
parser.add_argument('--r', default=14, type=int)

args = parser.parse_args()


lr = args.lr
eta = args.eta
beta = args.beta
epochs = args.epochs
r = args.r


theta_zero = 4.0
highest_degree = 30
r = 14

hist = {'ADAM': [],
        'SGD': [],
        'TUSLA': [],
        'RMSprop' : [],
        'AMSGrad': [],
        }
opt_name = ['ADAM', 'AMSGrad', 'RMSprop', 'TUSLA', 'SGD']

x_path = np.random.beta(2, 2, size=(epochs, 1))

for name in opt_name:
    theta = torch.tensor(theta_zero, requires_grad=True, device=device)
    hist[name] += [theta.item()]

    if name == 'ADAM': opt = optim.Adam([theta], lr=lr)
    elif name == 'AMSGrad': opt = optim.Adam([theta], lr=lr, amsgrad=True)
    elif name == 'SGD': opt = optim.SGD([theta], lr=lr)
    elif name == 'RMSprop': opt = optim.RMSprop([theta], lr=lr)
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

plt.plot(range(epochs), np.zeros(epochs), 'k')
plt.legend()
plt.xlabel('iterations')
plt.ylabel('theta')
plt.ylim([-0.5, 4.5])
plt.show()


# ## 3.1.2 beta case with different learning rate
plt.figure(2)
lr = 0.001
eta = 0
beta = 1e10
r = 14
theta_zero = 4.0
epochs = int(1.0 * 1e3)
high_order = 30


hist = {'ADAM': [],
        'SGD': [],
        'TUSLA': [],
        'RMSprop' : [],
        'AMSGrad': [],
        }
#opt_name = ['TUSLA', 'ADAM', 'AMSGrad', 'RMSprop', 'SGD' ]
opt_name = ['ADAM', 'AMSGrad', 'RMSprop']
lrs = [1, 0.1, 0.01]
#x_path = torch.randint(low=1, high=12, size=(epochs, 1), device=device)
#x_path = np.random.rand(epochs, 1) * 2 -4
x_path = np.random.beta(2,2, size=(epochs, 1))

for name in opt_name:
    for lr in lrs:
        theta = torch.tensor(theta_zero, requires_grad=True, device=device, dtype=torch.float64)
        hist[name] = []
        hist[name] += [theta.item()]

        if name == 'ADAM': opt = optim.Adam([theta], lr=lr, betas=(0.9, 0.999))
        elif name == 'AMSGrad': opt = optim.Adam([theta], lr=lr, betas=(0.9, 0.999), amsgrad=True)
        elif name == 'SGD': opt = optim.SGD([theta], lr=lr)
        elif name == 'RMSprop': opt = optim.RMSprop([theta], lr=lr)
        elif name == 'TUSLA': opt = TUSLA([theta], lr=lr, eta=eta, beta=beta, r=r)


        for epoch in range(1, epochs):
            opt.zero_grad()
            #x = torch.randint(low=1, high=12, size=(1,), device=device)
            x = torch.tensor(x_path[epoch-1], device=device)

            #f1 = torch.where(torch.abs(theta) <= 1, 5.5 * torch.pow(theta, 2), 11 * torch.abs(theta) - 5.5).to(device)
            loss = (torch.abs(theta) <= 1) * (
                    (x <= theta) * 2 * torch.pow(theta, 2)
                    + (x > theta) * torch.pow(theta, 2)
                    + torch.pow(theta, high_order)) + (
                    torch.abs(theta) > 1) * (
                    (x <= theta) * (4 * torch.abs(theta) - 2)
                    + (x > theta) * (2 * torch.abs(theta) - 1)
                    + torch.pow(theta, high_order))
            loss.backward()
            opt.step()
            hist[name] += [theta.item()]

        plt.plot(range(epochs), hist[name], label=name + '(the step size=' + str(lr) + ')')
    #
    # print(hist['ADAM'])
    # print(hist['AMSGrad'])
    # print(hist['TUSLA'])

plt.plot(range(epochs), np.zeros(epochs), 'k')
plt.legend()
plt.xlabel('iterations')
plt.ylabel('theta')
plt.ylim([-2, 4.5])
plt.show()

