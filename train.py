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
args = parser.parse_args()
train_set = {'group1'}
use_cuda = torch.cuda.is_available()
class_num = 101
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
check_path = './checkpoint/ckpt.t7'
CenLoss_a = 0.001
drop_rate = 0.5
epoch_num = 50
read_workers = 16
feat_list = ['mixed_8_join','mixed_7_join']#['mixed_7_join','mixed_8_join','mixed_10_join']
print(feat_list)
# Data
print('==> Preparing data..')

BaseDir = '/home/ss/feats'
trainset = datasets.UCF101_mixed_v3_dict(
    rootDict={'mixed_10_join':os.path.join(BaseDir,'UCF101_rgb_mix10_v3_npz'),
              'mixed_8_join' :os.path.join(BaseDir,'UCF101_rgb_mix8_v3_npz') ,
              'mixed_7_join' :os.path.join(BaseDir,'UCF101_rgb_mix7_v3_npz')},
    label='./ucfTrainTestlist/trainlist01.txt.top3', 
    ext = '_rgb.npz',
    is_training=True,
    feat_list = feat_list
    )
#['top_cls_global_pool', 'mixed_7_join', 'fc_action', 'mixed_10_join']
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=read_workers)

testset = datasets.UCF101_mixed_v3_dict(
    rootDict={'mixed_10_join':os.path.join(BaseDir,'UCF101_rgb_mix10_v3_npz'),
              'mixed_8_join' :os.path.join(BaseDir,'UCF101_rgb_mix8_v3_npz') ,
              'mixed_7_join' :os.path.join(BaseDir,'UCF101_rgb_mix7_v3_npz')},
#    rootDict={'mixed_10_join':'/media/ss/38cfe914-26f2-4a22-9cf1-bea9684775ac/lmy/temporal-segment-networks/data/UCF101_rgb_mix10_v3_npz',
#              'mixed_8_join' :'/media/ss/38cfe914-26f2-4a22-9cf1-bea9684775ac/lmy/temporal-segment-networks/data/UCF101_rgb_mix8_v3_npz' ,
#              'mixed_7_join' :'/media/ss/38cfe914-26f2-4a22-9cf1-bea9684775ac/lmy/temporal-segment-networks/data/UCF101_rgb_mix7_v3_npz'},
    label='./ucfTrainTestlist/testlist01.txt.top3',
    ext = '_rgb.npz',
    is_training=False,
    feat_list = feat_list
    )
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True, num_workers=read_workers)
dim = 2048
net = SpaNet(glo_channels=1280, loc_channels=768, out_channels=dim, drop_rate=drop_rate)
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
        tar_hot = torch.zeros(targets.size(0), class_num).scatter_(1, targets.unsqueeze(1), 1.0)
        if use_cuda:
            inputs, targets, tar_hot = [x.cuda() for x in inputs], targets.cuda(), tar_hot.cuda()
        optimizer.zero_grad()
        inputs, targets, tar_hot = [Variable(x) for x in inputs], Variable(targets), Variable(tar_hot)
        cat, outputs, addloss = net(inputs+[tar_hot])
        loss_2 = torch.mul(- tar_hot , torch.log(outputs)).sum(-1).mean()
        loss_cat = criterion(cat, targets)+addloss
        loss = loss_cat + loss_2
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
        tar_hot = torch.zeros(targets.size(0), class_num).scatter_(1, targets.unsqueeze(1), 1.0)
        if use_cuda:
            inputs, targets, tar_hot = [x.cuda() for x in inputs], targets.cuda(), tar_hot.cuda()
        inputs, targets, tar_hot = [Variable(x, volatile=True) for x in inputs], Variable(targets), Variable(tar_hot)
        outputs, addloss = net(inputs+[tar_hot])
        #loss_2 = torch.mul(- tar_hot , torch.log(outputs)).sum(-1).mean()
        loss_cat = criterion(outputs, targets)#+addloss
        loss = loss_cat# + addloss
        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
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
    if epoch>=start_epoch:
        test(epoch)
print(best_acc, CenLoss_a)
