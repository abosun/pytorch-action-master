#! /home/ss/anaconda2/envs/py35/bin/python3
'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

# from models. import *
from utils import progress_bar
from torch.autograd import Variable
import sys

# print(sys.path)
from models.multi_level_attention import *
import numpy as np
import datasets
def stop_grad(net, grad_set):
    names = []
    tensors = []
    for name,t in net.named_parameters():
        names.append(name)
        tensors.append(t)
    for i,p in enumerate(net.parameters()):
        if not names[i].split('.')[0] in grad_set:
            p.requires_grad = False
        else:
            p.requires_grad = True
    return net
def adjust_learning_rate(optimizer, decay_rate=.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.005, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--dataset', default='UCF101')
args = parser.parse_args()
train_set = {'group1'}
use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
check_path = './checkpoint/ckpt.t7'
CenLoss_a = 0.00001
drop_rate = 0.5
epoch_num = 50
class_num = 101 if args.dataset=='UCF101' else 51
split_dir = 'ucfTrainTestlist' if args.dataset=='UCF101' else 'hmdbTrainTestlist'
base_dir = '/media/ss/38cfe914-26f2-4a22-9cf1-bea9684775ac/lmy/temporal-segment-networks/data/'
# Data
print('==> Preparing data..')


trainset = datasets.UCF101_Top_multi_v3(
    root_short=os.path.join(base_dir, args.dataset+'_flow_top_short_v3'),
    root_mid=os.path.join(base_dir, args.dataset+'_flow_top_v3'), 
    label=os.path.join(split_dir,'trainlist01.txt'), 
    is_training=True
    )
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=1)

testset = datasets.UCF101_Top_multi_v3(
    root_short=os.path.join(base_dir, args.dataset+'_flow_top_short_v3'),
    root_mid=os.path.join(base_dir, args.dataset+'_flow_top_v3'),
    label=os.path.join(split_dir,'testlist01.txt'), 
    is_training=False
    )
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=True, num_workers=1)

net = TimeNet(in_channels=2048, mid_channels=101, out_channels=2048, num_classes=class_num, drop_rate=drop_rate)
if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
#optimizer = optim.Adam( filter(lambda p: p.requires_grad, net.parameters()),lr = args.lr, weight_decay=5e-4)
# Training

def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = [x.cuda() for x in inputs], targets.cuda()
        optimizer.zero_grad()
        inputs, targets = [Variable(x) for x in inputs], Variable(targets)
        outputs, lossShort, lossMid, lossLong = net(inputs)
        loss = criterion(outputs, targets)
        loss = loss + CenLoss_a*(lossShort+lossMid+lossLong)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    adjust_learning_rate(optimizer, 0.9)

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = [x.cuda() for x in inputs], targets.cuda()
        inputs, targets = [Variable(x, volatile=True) for x in inputs], Variable(targets)
        outputs, _,_,_ = net(inputs)
        loss = criterion(outputs, targets)#+addloss

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..'+str(acc))
        state = {
            'net': net.module if use_cuda else net,
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        #torch.save(state, './checkpoint/time_shot_ckpt.t7')
        best_acc = acc


for epoch in range(start_epoch, start_epoch+epoch_num):
    #if epoch>1:
        #net = stop_grad(net, set())
    #    optimizer = optim.Adam( net.parameters(),lr = args.lr, weight_decay=5e-4)   
    train(epoch)
    test(epoch)
print(best_acc, CenLoss_a)
