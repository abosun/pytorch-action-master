#! ../anaconda/envs/py35/bin/python3
'''Train Action with PyTorch.'''
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
def adjust_learning_rate(optimizer, decay_rate=.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate
parser = argparse.ArgumentParser(description='PyTorch Action Training')
parser.add_argument('--lr', default=0.005, type=float, help='learning rate')
parser.add_argument('--dataset', default='UCF101', choices=['UCF101','HMDB51'])
parser.add_argument('--split', default='1', choices=['1','2','3'])
parser.add_argument('--base_dir', default=None)
parser.add_argument('--drop_rate', default=None, type=float)
parser.add_argument('--a', nargs='+', default=None, type=float)
parser.add_argument('--type', default='all')
parser.add_argument('--log_file', default='log/spa.csv')
parser.add_argument('--epoch_num', type=int, default=50)

parser.add_argument('--n', type=int, default=5)
args = parser.parse_args()
use_cuda = torch.cuda.is_available()
class_num = 101
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
best_epoch = start_epoch
check_path = './checkpoint/ckpt.t7'
CenLoss_a = 0.001
loss_a = [0,0,0] if args.a is None else args.a
drop_rate = 0.5 if args.drop_rate is None else args.drop_rate
epoch_num = args.epoch_num
read_workers = 16
score_out = 0
class_num = 101 if args.dataset=='UCF101' else 51
split_dir = 'ucfTrainTestlist' if args.dataset=='UCF101' else 'hmdbTrainTestlist'
base_dir = '/home/ss/feats' if args.base_dir is None else args.base_dir
feat_list = ['mixed_8_join','mixed_7_join']#, 'top_cls_global_pool']#['mixed_7_join','mixed_8_join','mixed_10_join']
print(feat_list)
# Data
print('==> Preparing data..')

#base_dir = '/home/ss/feats'
trainset = datasets.UCF101_mixed_v3_dict(
    rootDict={'mixed_10_join':os.path.join(base_dir,args.dataset+'_rgb_mix10_v3_npz')
              ,'mixed_8_join' :os.path.join(base_dir,args.dataset+'_rgb_mix8_v3_npz') 
              ,'mixed_7_join' :os.path.join(base_dir,args.dataset+'_rgb_mix7_v3_npz') 
             # ,'top_cls_global_pool':os.path.join(base_dir,args.dataset+'_rgb_top_v3_npz')
             },
    label=os.path.join(split_dir,'alllist.txt'), 
    ext = '_rgb.npz',
    is_training=True,
    feat_list = feat_list,
    n = args.n
    )
#['top_cls_global_pool', 'mixed_7_join', 'fc_action', 'mixed_10_join']
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=read_workers)

testset = datasets.UCF101_mixed_v3_dict(
    rootDict={'mixed_10_join':os.path.join(base_dir,args.dataset+'_rgb_mix10_v3_npz')
              ,'mixed_8_join' :os.path.join(base_dir,args.dataset+'_rgb_mix8_v3_npz')
              ,'mixed_7_join' :os.path.join(base_dir,args.dataset+'_rgb_mix7_v3_npz')
              #'top_cls_global_pool':os.path.join(base_dir,args.dataset+'_rgb_top_v3')
             },
#    rootDict={'mixed_10_join':'/media/ss/38cfe914-26f2-4a22-9cf1-bea9684775ac/lmy/temporal-segment-networks/data/UCF101_rgb_mix10_v3_npz',
#              'mixed_8_join' :'/media/ss/38cfe914-26f2-4a22-9cf1-bea9684775ac/lmy/temporal-segment-networks/data/UCF101_rgb_mix8_v3_npz' ,
#              'mixed_7_join' :'/media/ss/38cfe914-26f2-4a22-9cf1-bea9684775ac/lmy/temporal-segment-networks/data/UCF101_rgb_mix7_v3_npz'},
    label=os.path.join(split_dir,'testlist0'+args.split+'.txt'),
    ext = '_rgb.npz',
    is_training=False,
    feat_list = feat_list,
    n = args.n
    )
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=read_workers)
dim = 2048
net = SpaNet(glo_channels=1280, loc_channels=768, out_channels=dim, num_classes=class_num, drop_rate=drop_rate, out_type=args.type, n=args.n)
if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
#optimizer = optim.Adam( filter(lambda p: p.requires_grad, net.parameters()),lr = args.lr, weight_decay=5e-4)
# Training
net.load_state_dict(torch.load('UCF101_split1_rgb_98.678_model'))

def test_with_score(score):
    true_num = 0.0
    ids = np.argmax(score, axis=1)
    for i in range(testset.__len__()):
        if testset.datas[i][1]==ids[i]:
            true_num += 1.0
    return true_num/testset.__len__()
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
        #loss_2 = torch.mul(- tar_hot , torch.log(outputs)).sum(-1).mean()
        loss_cat = criterion(cat, targets)#+addloss
        #loss = loss_cat + loss_2
        loss_cat.backward()
        optimizer.step()

        train_loss = loss_cat.data[0]#+= loss.data[0]
        _, predicted = torch.max(cat.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    adjust_learning_rate(optimizer, 0.92)

def test(epoch, get_score = False):
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
        tar_hot = torch.zeros(targets.size(0), class_num).scatter_(1, targets.unsqueeze(1), 1.0)
        if use_cuda:
            inputs, targets, tar_hot = [x.cuda() for x in inputs], targets.cuda(), tar_hot.cuda()
        inputs, targets, tar_hot = [Variable(x, volatile=True) for x in inputs], Variable(targets), Variable(tar_hot)
        outputs, score, addloss = net(inputs+[tar_hot])
        #loss_2 = torch.mul(- tar_hot , torch.log(outputs)).sum(-1).mean()
        loss_cat = criterion(outputs, targets)#+addloss
        loss = loss_cat# + addloss
        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        if get_score :
            score = F.softmax(outputs,dim=-1).cpu().data.numpy()
            score_res.append(score)
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..%.2f'%(acc),args.dataset, args.split, 'rgb')
        state = {
            'net': net.module if use_cuda else net,
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/spa_shot_ckpt.t7')
        best_acc = acc
        best_epoch = epoch
        score_out = np.vstack(score_res)
    if get_score: return np.vstack(score_res)
for epoch in range(start_epoch, start_epoch+epoch_num):
    #if epoch>1:
        #net = stop_grad(net, set())
    #    optimizer = optim.Adam( net.parameters(),lr = args.lr, weight_decay=5e-4)   
    train(epoch)
    if epoch>=start_epoch:
        a = test(epoch, get_score = True)
    if epoch-best_epoch>5:
        break
    torch.save(net.state_dict(), args.dataset+'_split'+args.split+'_rgb'+'_%.3f'%(best_acc)+'_model')

np.save("scores/"+args.dataset+'_split'+args.split+'_rgb'+'_%.3f'%(best_acc),score_out)
print(best_acc, CenLoss_a)
with open(args.log_file, 'a+') as f:
    import time
    local_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    f.write(','.join([local_time, '%.2f'%(best_acc),args.dataset,args.split,'rgb', args.type]+ [str(x) for x in  loss_a])+'\n')
