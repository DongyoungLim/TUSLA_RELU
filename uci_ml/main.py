from __future__ import print_function
#from sklearn.datasets import load_wine, load_breast_cancer
import torch.optim as optim
import torch.backends.cudnn as cudnn
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import argparse
import time
from models import *
from optimizers import *
from utils import UCI_Dataset, get_data
from sklearn.preprocessing import MinMaxScaler
import zipfile
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='PyTorch UCI_ML Training')
parser.add_argument('--total_epoch', default=5000, type=int, help='Total number of training epochs')
parser.add_argument('--seed', default=111, type=int)
parser.add_argument('--model', default='slfn', type=str, help='model', choices=['slfn', 'tlfn'])
parser.add_argument('--optimizer', default='tusla', type=str, help='optimizer', choices=['sgd', 'adam', 'amsgrad', 'rmsprop', 'tusla'])
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--dataset', default='concrete', type=str,
                    choices=['concrete'])
parser.add_argument('--r', default=1, type=float, help='r for tusla')
parser.add_argument('--eta', default=0, type=float, help='eta for tusla')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum term')
parser.add_argument('--beta', default=1e12, type=float)
parser.add_argument('--beta1', default=0.9, type=float, help='Adam coefficients beta_1')
parser.add_argument('--beta2', default=0.999, type=float, help='Adam coefficients beta_2')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--weight_decay', default=0, type=float, help='weight decay for optimizers')

args = parser.parse_args()
torch.manual_seed(args.seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def build_dataset(args):
    print('==> Preparing data..')

    data, input_size, output_size = get_data(args.dataset)
    scaler = MinMaxScaler()
    data[:, :input_size] = scaler.fit_transform(data[:, :input_size])

    train_data, test_data = train_test_split(data, test_size=0.1, random_state=0, shuffle=True)

    train_loader = torch.utils.data.DataLoader(UCI_Dataset(train_data), batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(UCI_Dataset(test_data), batch_size=args.batch_size, shuffle=True)

    return train_loader, test_loader, input_size, output_size


def get_ckpt_name(dataset='f_mnist', seed=111, model='resnet', optimizer='sgd', lr=0.1, momentum=0.9,
                  beta1=0.9, beta2=0.999, r=1, weight_decay=5e-4,beta=1e10, bs=16):
    name = {
        'sgd': 'seed{}-lr{}-momentum{}-wdecay{}'.format(seed, lr, momentum,weight_decay),
        'adam': 'seed{}-lr{}-betas{}-{}-wdecay{}'.format(seed, lr, beta1, beta2, weight_decay),
        'amsgrad': 'seed{}-lr{}-betas{}-{}-wdecay{}'.format(seed, lr, beta1, beta2, weight_decay),
        'rmsprop': 'seed{}-lr{}-wdecay{}'.format(seed, lr, weight_decay),
        'tusla': 'seed{}-lr{}-r{}-beta{:.1e}-wdecay{}'.format(seed, lr, r, beta, weight_decay),
    }[optimizer]
    return '{}-{}-{}-bs{}-{}'.format(dataset, optimizer, model, bs, name)



print('==> Building model..')
train_loader, test_loader, input_size, output_size = build_dataset(args)

if args.model == 'slfn':
    net = SLFN(input_size=input_size, hidden_size=50, output_size=output_size)
elif args.model == 'tlfn':
    net = TLFN(input_size=input_size, hidden_size=50, output_size=output_size)

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True



def train(net, epoch, device, data_loader, optimizer, criterion):
    #print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    #correct = 0
    total = 0
    for batch_idx, inputs in enumerate(data_loader):
        inputs, targets = inputs[:, :input_size].to(device), inputs[:, input_size:].to(device)

        optimizer.zero_grad()
        outputs = net(inputs)

        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        #_, predicted = outputs.max(1)
        total += targets.size(0)
        #correct += predicted.eq(targets).sum().item()

    #accuracy = 100. * correct / total
    loss = train_loss / total
    if epoch % 500 == 0:
        print('\nEpoch: %d' % epoch)
        print('train loss %.5f' % (loss))

    return loss


def test(net, device, data_loader, criterion):
    net.eval()
    test_loss = 0
    #correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, inputs in enumerate(data_loader):
            inputs, targets = inputs[:, :input_size].to(device), inputs[:, input_size:].to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            #_, predicted = outputs.max(1)
            total += targets.size(0)
            #correct += predicted.eq(targets).sum().item()

    #accuracy = 100. * correct / total
    loss = test_loss / total
    if epoch % 500 == 0:
        print('test loss %.5f' % (loss))

    return loss

def adjust_learning_rate(optimizer, epoch, step_size=150, gamma=0.1):
    for param_group in optimizer.param_groups:
        if epoch % step_size == 0 and epoch>0:
            param_group['lr'] *= gamma





print('\n==> Start training ')

history = {'train_loss': [],
           'test_loss': [],
           'train_acc': [],
           'test_acc': []
           }
state = {}
best_loss = 99999


print('\n==> Setting optimizer.. use {%s}'%args.optimizer)

optimizer = { 'sgd': optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay),
              'adam': optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay),
              'amsgrad': optim.Adam(net.parameters(), lr=args.lr, amsgrad=True, weight_decay=args.weight_decay),
              'rmsprop': optim.RMSprop(net.parameters(), lr=args.lr, weight_decay=args.weight_decay),
              'tusla': TUSLA(net.parameters(), lr=args.lr, r=args.r, eta=args.eta, beta=args.beta, weight_decay=args.weight_decay)
}[args.optimizer]


save = get_ckpt_name(dataset=args.dataset, seed=args.seed, model=args.model, optimizer=args.optimizer, lr=args.lr, momentum=args.momentum, beta1=args.beta1,
                     beta2=args.beta2, r=args.r, weight_decay=args.weight_decay, beta=args.beta, bs=args.batch_size)
print('setting: {}'.format(save))

print('dataset: {} - criterion: MSEloss'.format(args.dataset))
criterion = nn.MSELoss()

for epoch in range(1, args.total_epoch):

    train_loss = train(net, epoch, device, train_loader, optimizer, criterion)
    test_loss = test(net, device, test_loader, criterion)
    if test_loss < best_loss:
        state = {
            'net': net.state_dict(),
            'loss': test_loss,
            'epoch': epoch,
            'optimizer': optimizer
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, os.path.join('checkpoint', save))
        best_loss = test_loss

    history['train_loss'].append(train_loss)
    history['test_loss'].append(test_loss)

if not os.path.isdir('logs'):
    os.mkdir('logs')
torch.save(history, os.path.join('logs/'+args.dataset, save))
print('best_loss', best_loss)

plt.show()

