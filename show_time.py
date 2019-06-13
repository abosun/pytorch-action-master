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
import glob
# from models. import *
from utils import progress_bar
from torch.autograd import Variable
import sys
import cv2 as cv
# print(sys.path)
from models.multi_level_attention_show import *
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


showset = datasets.UCF101_Top_multi_v3(
    root_short=os.path.join(base_dir, args.dataset+'_flow_top_short_v3'),
    root_mid=os.path.join(base_dir, args.dataset+'_flow_top_v3'),
    label=os.path.join(split_dir,'alllist.txt'), 
    is_training=False
    )
showloader = torch.utils.data.DataLoader(showset, batch_size=32, shuffle=False, num_workers=1)

net = TimeNet(in_channels=2048, mid_channels=101, out_channels=2048, num_classes=class_num, drop_rate=drop_rate, out_type=args.type)
if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

net.load_state_dict(torch.load('UCF101_split1_flow_99.921_model'))

IMG_DIR = '/home/ss/lmy/temporal-segment-networks/data/UCF101_FLOW'
image_list = open(os.path.join(split_dir,'alllist.txt')).read().split('\n')
image_list = [x.split('.')[0] for x in image_list]
path_list = [os.path.basename(x) for x in image_list]

print(image_list[:10])
print(len(image_list))

def get_images(video):
    image_list = sorted(glob.glob(os.path.join(IMG_DIR,video,'img*')))
    stack_depth = 1
    frame_cnt = len(image_list)
    num_frame_per_video = 25
    step = (frame_cnt - stack_depth) // (num_frame_per_video-1)
    frame_ticks = range(1, min((2 + step * (num_frame_per_video-1)), frame_cnt+1), step)
    return [image_list[i-1] for i in frame_ticks]

def get_showing_img(img_list, ws, wm, wl):
    def fun(img_list,ws):
      ws = ws/ws.max()
      img_ws = [img_list[i]*ws[i] for i in range(25)]
      res = np.concatenate(img_ws, axis=1)
      return res
    res = np.concatenate([fun(img_list,ws), fun(img_list,wm), fun(img_list,wl)], axis=0)
    return res

def showall():
    global path_list
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(showloader):
        if use_cuda:
            inputs, targets = [x.cuda() for x in inputs], targets.cuda()
        inputs, targets = [Variable(x, volatile=True) for x in inputs], Variable(targets)

        ws, wm, wl = net(inputs)

        ws, wm, wl = ws.cpu().data.numpy().squeeze(), wm.cpu().data.numpy().squeeze(), wl.cpu().data.numpy().squeeze()

#        print(ws.size(), wm.size(), wl.size())
        for i in range(ws.shape[0]):

            #np.savetxt('ucftimeweight/'+path_list[batch_idx*32+i]+'.txt' ,np.stack([ws[i],wm[i],wl[i]]))
#            np.savetxt('ucftimeweight/'+path_list[batch_idx*32+i]+'_wm.txt' ,wm)
#            np.savetxt('ucftimeweight/'+path_list[batch_idx*32+i]+'_wl.txt' ,wl)
            image_list = get_images(path_list[batch_idx*32+i])
            img_list = [cv.imread(path) for path in image_list]
            res = get_showing_img(img_list, ws[i], wm[i], wl[i])
            cv.imwrite('ucftimeshow/'+path_list[batch_idx*32+i]+'.jpg',res)
            print( path_list[batch_idx*32+i], 'has done')

if __name__ == '__main__':
    showall()


