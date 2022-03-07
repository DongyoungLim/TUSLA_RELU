"""Train CIFAR10 with PyTorch."""
from __future__ import print_function

import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

import os
import argparse
import time
from models import *
from torch.optim import Adam, SGD, RMSprop
from optimizers import *
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='PyTorch F_MNIST Training')
parser.add_argument('--total_epoch', default=100, type=int, help='Total number of training epochs')
parser.add_argument('--seed', default=111, type=int)
parser.add_argument('--decay_epoch', default=150, type=int, help='Number of epochs to decay learning rate')
parser.add_argument('--model', default='slfn', type=str, help='model',
                        choices=['slfn', 'tlfn'])
parser.add_argument('--optimizer', default='tusla', type=str, help='optimizer', choices=['sgd', 'adam', 'amsgrad', 'rmsprop', 'tusla'])
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--dataset', default='f_mnist', type=str, choices=['mnist', 'f_mnist'])
parser.add_argument('--r', default=3, type=float, help='r for tusla')
parser.add_argument('--eta', default=0, type=float, help='eta for tusla')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum term')
parser.add_argument('--beta', default=1e12, type=float)
parser.add_argument('--beta1', default=0.9, type=float, help='Adam coefficients beta_1')
parser.add_argument('--beta2', default=0.999, type=float, help='Adam coefficients beta_2')
parser.add_argument('--batchsize', type=int, default=128, help='batch size')
parser.add_argument('--weight_decay', default=0, type=float, help='weight decay for optimizers')

args = parser.parse_args()
torch.manual_seed(args.seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def build_dataset(args):
    print('==> Preparing data..')

    transform_train = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.5,), std=(0.5,))])
    transform_test = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize(mean=(0.5,), std=(0.5,))])

    if args.dataset == 'f_mnist':
        train_dataset = torchvision.datasets.FashionMNIST(root='./data/', download=True, train=True, transform=transform_train)
        test_dataset = torchvision.datasets.FashionMNIST(root='./data/', download=True, train=False, transform=transform_test)

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batchsize, shuffle=False)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batchsize, shuffle=False)
    elif args.dataset == 'mnist':
        train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST('./data/', train=True, download=True,
                                       transform=torchvision.transforms.Compose([
                                           torchvision.transforms.ToTensor(),
                                           torchvision.transforms.Normalize(
                                               (0.1307,), (0.3081,))
                                       ])),
            batch_size=args.batchsize, shuffle=True)

        test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST('./data/', train=False, download=True,
                                       transform=torchvision.transforms.Compose([
                                           torchvision.transforms.ToTensor(),
                                           torchvision.transforms.Normalize(
                                               (0.1307,), (0.3081,))
                                       ])),
            batch_size=args.batchsize, shuffle=True)

    return train_loader, test_loader


def get_ckpt_name(dataset='f_mnist', seed=111, model='resnet', optimizer='sgd', lr=0.1, momentum=0.9,
                  beta1=0.9, beta2=0.999, r=1, weight_decay=5e-4,beta=1e10, eta=0):
    name = {
        'sgd': 'seed{}-lr{}-momentum{}-wdecay{}'.format(seed, lr, momentum,weight_decay),
        'adam': 'seed{}-lr{}-betas{}-{}-wdecay{}'.format(seed, lr, beta1, beta2, weight_decay),
        'amsgrad': 'seed{}-lr{}-betas{}-{}-wdecay{}'.format(seed, lr, beta1, beta2, weight_decay),
        'rmsprop': 'seed{}-lr{}-wdecay{}'.format(seed, lr, weight_decay),
        'tusla': 'seed{}-lr{}-r{}-beta{:.1e}-eta{}-wdecay{}'.format(seed, lr, r, beta, eta, weight_decay)
    }[optimizer]
    return '{}-{}-{}-{}'.format(dataset, optimizer, model, name)



print('==> Building model..')
num_class = 10
if args.model == 'slfn':
    net = SLFN(input_size=28*28, hidden_size=50, output_size=num_class)
elif args.model == 'tlfn':
    net = TLFN(input_size=28*28, hidden_size=50, output_size=num_class)

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

print('\n==> Setting optimizer.. use {%s}'%args.optimizer)


def train(net, epoch, device, data_loader, optimizer, criterion):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    accuracy = 100. * correct / total
    loss = train_loss / total
    print('train loss %.5f, train acc %.3f' % (loss, accuracy))

    return accuracy, loss


def test(net, device, data_loader, criterion):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100. * correct / total
    loss = test_loss / total
    print('test loss %.5f, test acc %.3f' % (loss, accuracy))

    return accuracy, loss

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
best_acc = 0

optimizer = { 'sgd': optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay),
              'adam': optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay),
              'amsgrad': optim.Adam(net.parameters(), lr=args.lr, amsgrad=True, weight_decay=args.weight_decay),
              'rmsprop': optim.RMSprop(net.parameters(), lr=args.lr, weight_decay=args.weight_decay),
              'tusla': TUSLA(net.parameters(), lr=args.lr, r=args.r, beta=args.beta, eta=args.eta, weight_decay=args.weight_decay)
}[args.optimizer]

train_loader, test_loader = build_dataset(args)
save = get_ckpt_name(dataset=args.dataset, seed=args.seed, model=args.model, optimizer=args.optimizer, lr=args.lr, momentum=args.momentum, beta1=args.beta1,
                     beta2=args.beta2, r=args.r, weight_decay=args.weight_decay, beta=args.beta, eta=args.eta)
print('setting: {}'.format(save))
criterion = nn.CrossEntropyLoss()



for epoch in range(1, args.total_epoch):
    start = time.time()
    #scheduler.step()
    adjust_learning_rate(optimizer, epoch)

    train_acc, train_loss = train(net, epoch, device, train_loader, optimizer, criterion)
    test_acc, test_loss = test(net, device, test_loader, criterion)
    print('Time: {}'.format(time.time()-start))
    # Save checkpoint.
    if test_acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': test_acc,
            'epoch': epoch,
            'optimizer': optimizer
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, os.path.join('checkpoint', save))
        best_acc = test_acc

    history['train_acc'].append(train_acc)
    history['test_acc'].append(test_acc)
    history['train_loss'].append(train_loss)
    history['test_loss'].append(test_loss)

if not os.path.isdir('curve'):
    os.mkdir('curve')
torch.save(history, os.path.join('logs', save))

