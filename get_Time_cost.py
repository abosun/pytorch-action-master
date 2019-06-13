#! ../anaconda/envs/py35/bin/python3
'''Train action with PyTorch.'''
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
parser = argparse.ArgumentParser(description='PyTorch Action Training')
parser.add_argument('--lr', default=0.005, type=float, help='learning rate')
parser.add_argument('--dataset', default='UCF101', choices=['UCF101','HMDB51'])
parser.add_argument('--split', default='1', choices=['1','2','3'])
parser.add_argument('--base_dir', default=None)
parser.add_argument('--drop_rate', default=None, type=float)
parser.add_argument('--a', nargs='+', default=[0,0,0], type=float)
parser.add_argument('--type', default='time')
parser.add_argument('--log_file', default='log/time.csv')
parser.add_argument('--epoch_num', type=int, default=50)
args = parser.parse_args()
train_set = {'group1'}
use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
best_epoch = start_epoch
check_path = './checkpoint/ckpt.t7'
loss_a = [1e-5, 1e-5, 1e-5] if args.a is None else args.a
drop_rate = 0.4 if args.drop_rate is None else args.drop_rate
epoch_num = args.epoch_num
class_num = 101 if args.dataset=='UCF101' else 51
split_dir = 'ucfTrainTestlist' if args.dataset=='UCF101' else 'hmdbTrainTestlist'
base_dir = '/home/ss/feats/' if args.base_dir is None else args.base_dir
#base_dir = '/media/ss/38cfe914-26f2-4a22-9cf1-bea9684775ac/lmy/temporal-segment-networks/data/'
score_out = 0
# Data
print('==> Preparing data..')


trainset = datasets.UCF101_Top_multi_v3(
    root_short=os.path.join(base_dir, args.dataset+'_flow_top_short_v3'),
    root_mid=os.path.join(base_dir, args.dataset+'_flow_top_v3'), 
    label=os.path.join(split_dir,'alllist.txt'), 
    is_training=True
    )
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=1)

testset = datasets.UCF101_Top_multi_v3(
    root_short=os.path.join(base_dir, args.dataset+'_flow_top_short_v3'),
    root_mid=os.path.join(base_dir, args.dataset+'_flow_top_v3'),
    label=os.path.join(split_dir,'testlist0'+args.split+'.txt'), 
    is_training=False
    )
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=1)

net = TimeNet(in_channels=2048, mid_channels=101, out_channels=2048, num_classes=class_num, drop_rate=drop_rate, out_type=args.type)
if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
#optimizer = optim.Adam( filter(lambda p: p.requires_grad, net.parameters()),lr = args.lr, weight_decay=5e-4)
# Training
def test_with_score(score):
    true_num = 0.0
    ids = np.argmax(score, axis=1)
    for i in range(testset.__len__()):
        if testset.datas_mid[i][1]==ids[i]:
            true_num += 1.0
    return true_num/testset.__len__()
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
        loss = loss + loss_a[0] * lossShort + loss_a[1] * lossMid + loss_a[2] * lossLong
        loss = loss.mean()
        #loss = loss + CenLoss_a*(lossShort+lossMid+lossLong)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.2f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    adjust_learning_rate(optimizer, 0.9)

def test(epoch, get_score=False):
    global best_acc
    global score_out
    global best_epoch
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    if get_score :
        score_res = []
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = [x.cuda() for x in inputs], targets.cuda()
        inputs, targets = [Variable(x, volatile=True) for x in inputs], Variable(targets)
        outputs, _,_,_ = net(inputs)
        loss = criterion(outputs, targets)#+addloss
        loss = loss.mean()
        

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.2f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    

if __name__=='__main__':
    import time
    start = time.clock()
    test(0,get_score=False)
    end = time.clock()
    print(end-start)
