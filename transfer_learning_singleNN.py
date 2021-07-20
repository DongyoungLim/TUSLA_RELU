## package load
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from models import single_nn, two_nn
from custom_optim import TUSLA
import os
import argparse

os.environ['KMP_DUPLICATE_LIB_OK']='True'

parser = argparse.ArgumentParser()
parser.add_argument('--eta', default=1e-25, type=float)
parser.add_argument('--lr', default=0.5, type=float)
parser.add_argument('--beta', default=1e10, type=float)
parser.add_argument('--epochs', default=3000, type=int)
args = parser.parse_args()


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

## generate dataset
n = 10000
n_train = int(n*0.8)

y = torch.rand(n, 2) #input variable
z = (np.abs(2 * y[:, 0] + 2 * y[:, 1] - 2)).pow(3) #target variable

y = y.to(device)
z = z.to(device)

class CustomDataset(Dataset):
  def __init__(self, y, z):
    self.y = y
    self.z = z
  def __len__(self):
    return len(self.y)

  def __getitem__(self, idx):
    return y[idx], z[idx]


## HYPERPARAMETER SETTING
print('==================learning TLFN==================')
lr = args.lr
eta = args.eta
beta = args.beta
epochs = args.epochs

r = 3
d = 15 # hidden size

batch_size = 2
eps = np.sqrt(lr)
act_fn = 'tanh'
ckpt_dir = './model_save/'

torch.manual_seed(2)
traindataset = CustomDataset(y[:n_train], z[:n_train])
trainloader = DataLoader(traindataset, batch_size=batch_size, shuffle=True)

testdataset = CustomDataset(y[n_train:], z[n_train:])
testloader = DataLoader(testdataset, batch_size=batch_size, shuffle=True)

file_name = 'lr_' + str(lr) + '_eta_' + str(eta) + '_'
fig_dir = './figures/' + file_name  +'/'


hist_train = {}
hist_test = {}
networks = {}
#
settings = [[eta, 'TUSLA']]
#
criterion = nn.MSELoss()

num_batch = np.ceil(traindataset.__len__()/batch_size).astype('int')
plt.figure(1)

for setting in settings:
    eta = setting[0]
    opt_name = setting[1]
    net = two_nn(d, act_fn).to(device)

    if opt_name == 'ADAM': opt = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))
    elif opt_name == 'AMSGrad': opt = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), amsgrad=True)
    elif opt_name == 'SGD': opt = optim.SGD(net.parameters(), lr=0.1)
    elif opt_name == 'RMSprop': opt = optim.RMSprop(net.parameters(), lr=0.01)
    elif opt_name == 'TUSLA': opt = TUSLA(net.parameters(), lr=lr, eta=eta, beta=beta, r=r)

    exp_name = opt_name + '_eta=' + str(eta)

    hist_train[exp_name] = []
    hist_test[exp_name] = []

    for epoch in range(1, epochs+1):
        train_loss = []
        net.train()
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            targets = targets.view(-1, 1)
            opt.zero_grad()
            output = net(inputs)
            loss = criterion(output, targets)
            loss.backward()
            opt.step()
            train_loss.append(loss.item())
        hist_train[exp_name].append(np.mean(train_loss))

        net.eval()
        test_loss = []
        for batch_idx, (inputs, targets) in enumerate(testloader):
            targets = targets.view(-1, 1)
            opt.zero_grad()
            output = net(inputs)
            loss = criterion(output, targets)
            loss.backward()
            opt.step()
            test_loss.append(loss.item())
        hist_test[exp_name].append(np.mean(test_loss))

        print('epoch: %d, training_loss: %.4f, test_loss: %.4f'%(epoch, np.mean(train_loss), np.mean(test_loss)))

    networks[exp_name] = net

    state = {
        'net': net.state_dict(),
    }

    if not os.path.isdir(ckpt_dir):
        os.mkdir(ckpt_dir)
    torch.save(state, '%s/%s.pth' % (ckpt_dir, exp_name))

    plt.plot(range(1, len(hist_train[exp_name]) + 1), hist_train[exp_name])
    plt.xlabel('epochs')
    plt.ylabel('loss')



plt.plot(range(1, len(hist_train[exp_name])+1), np.zeros(len(hist_train[exp_name])), 'k')

file_name = 'lr_' + str(lr) + '_eta_' + str(eta) + '_'
fig_dir = './figures/' + file_name  +'/'
if not os.path.isdir(fig_dir):
    os.mkdir(fig_dir)
plt.savefig(fig_dir + file_name + 'fist_nn_training.png')


plt.figure(2)
plt.plot(range(1, len(hist_train[exp_name])+1), np.zeros(len(hist_train[exp_name])), 'k')
for setting in settings:
    eta = setting[0]
    opt_name = setting[1]
    exp_name = opt_name + '_eta=' + str(eta)
    plt.plot(range(1, len(hist_train[exp_name]) + 1), hist_test[exp_name])#, label=name_lr)
plt.xlabel('epochs')
plt.ylabel('loss')

plt.savefig(fig_dir + file_name + 'fist_nn_test.png')

plt.figure(3)
ax = plt.axes(projection='3d')
net.eval()
grid = torch.FloatTensor([[x/30, y/30] for x in range(0, 31) for y in range(0, 31)])
true = np.power(np.abs(2*grid[:, 0] + 2*grid[:, 1] - 2), 3)
ax.scatter(grid[:, 0], grid[:, 1], true,  alpha=0.5, label='true')
for setting in settings:
    eta = setting[0]
    opt_name = setting[1]
    exp_name = opt_name + '_eta=' + str(eta)
    net = networks[exp_name]
    pred = net(grid.to(device)).cpu().data
    ax.scatter(grid[:, 0], grid[:, 1], pred, alpha=0.5, label='prediction')#, label=name_lr)

plt.legend()
plt.savefig(fig_dir + file_name + 'fist_nn_plot.png')

###################transfer learning##################################
print('=========================start transfer learning=========================')
#epochs = 100
lr = 0.5
r = 2
hist_train_tr = {}
hist_test_tr = {}
networks_tr = {}
act_fn = 'relu'
## generate dataset
n = 10000
n_train = int(n*0.8)

y = torch.rand(n, 2)
z = (np.abs(2 * y[:, 0] + 2 * y[:, 1] - 1.5)).pow(3)

y = y.to(device)
z = z.to(device)

traindataset = CustomDataset(y[:n_train], z[:n_train])
trainloader = DataLoader(traindataset, batch_size=batch_size, shuffle=True)

testdataset = CustomDataset(y[-n_train:], z[-n_train:])
testloader = DataLoader(testdataset, batch_size=batch_size, shuffle=True)


num_batch = np.ceil(traindataset.__len__()/batch_size).astype('int')
plt.figure(4)

for setting in settings:
    eta = setting[0]
    opt_name = setting[1]
    exp_name = opt_name + '_eta=' + str(eta)

    hist_train_tr[exp_name] = []
    hist_test_tr[exp_name] = []

    net = single_nn(d, act_fn).to(device)
    state = torch.load('%s/%s.pth' % (ckpt_dir, exp_name))

    net.l1.weight.data = state['net']['l1.weight'].data  #transering the first layer parameters
    net.l1.weight.requires_grad = False # fix the transferred parameters


    if opt_name == 'ADAM': opt = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))
    elif opt_name == 'AMSGrad': opt = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), amsgrad=True)
    elif opt_name == 'SGD': opt = optim.SGD(net.parameters(), lr=0.1)
    elif opt_name == 'RMSprop': opt = optim.RMSprop(net.parameters(), lr=0.01)
    elif opt_name == 'TUSLA': opt = TUSLA(net.parameters(), lr=lr, eta=eta, beta=beta, r=r)

    for epoch in range(1, epochs+1):
        train_loss = []
        net.train()
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            targets = targets.view(-1,1)
            opt.zero_grad()
            output = net(inputs)
            loss = criterion(output, targets)
            loss.backward()
            opt.step()
            train_loss += [loss.item()]
        hist_train_tr[exp_name] += [np.mean(train_loss)]

        net.eval()
        test_loss = []
        for batch_idx, (inputs, targets) in enumerate(testloader):
            targets = targets.view(-1,1)
            opt.zero_grad()
            output = net(inputs)
            loss = criterion(output, targets)
            loss.backward()
            opt.step()
            test_loss += [loss.item()]
        hist_test_tr[exp_name] += [np.mean(test_loss)]

        print('epoch: %d, training loss: %.4f, test_loss: %.4f'%(epoch, np.mean(train_loss), np.mean(test_loss)))

    networks_tr[exp_name] = net

    state = {
        'net': net.state_dict(),
    }
    plt.plot(range(1, len(hist_train_tr[exp_name]) + 1), hist_train_tr[exp_name])



plt.plot(range(1, len(hist_train_tr[exp_name])+1), np.zeros(len(hist_train_tr[exp_name])), 'k')
plt.xlabel('epochs')
plt.ylabel('loss')

plt.savefig(fig_dir + file_name + 'second_nn_training.png')


plt.figure(5)
plt.plot(range(1, len(hist_train_tr[exp_name])+1), np.zeros(len(hist_train_tr[exp_name])), 'k')


for setting in settings:
    eta = setting[0]
    opt_name = setting[1]
    exp_name = opt_name + '_eta=' + str(eta)
    plt.plot(range(1, len(hist_train_tr[exp_name]) + 1), hist_test_tr[exp_name])#, label=exp_name)
plt.xlabel('epochs')
plt.ylabel('loss')

plt.savefig(fig_dir + file_name + 'second_nn_test.png')


plt.figure(6)
ax = plt.axes(projection='3d')
net.eval()
grid = torch.FloatTensor([[x/30, y/30] for x in range(0, 31) for y in range(0, 31)])
true = np.power(np.abs(2*grid[:, 0] + 2*grid[:, 1]-1.5), 3)
ax.scatter(grid[:, 0], grid[:, 1], true,  alpha=0.5, label='true')
for setting in settings:
    eta = setting[0]
    opt_name = setting[1]
    exp_name = opt_name + '_eta=' + str(eta)

    net = networks_tr[exp_name]
    pred = net(grid.to(device)).cpu().data
    ax.scatter(grid[:, 0], grid[:, 1], pred, alpha=0.5, label='prediction')

plt.legend()
plt.savefig(fig_dir + file_name + 'second_nn_plot.png')

plt.show()