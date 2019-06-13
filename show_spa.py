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
import glob
# from models. import *
from utils import progress_bar
from torch.autograd import Variable
import sys
import cv2 as cv
import multiprocessing
# print(sys.path)
from models.multi_level_attention_show import *
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
BATCH_SIZE=32
print(feat_list)
# Data
print('==> Preparing data..')

#base_dir = '/home/ss/feats'
showset = datasets.UCF101_mixed_v3_dict(
    rootDict={'mixed_10_join':os.path.join(base_dir,args.dataset+'_rgb_mix10_v3_npz')
              ,'mixed_8_join' :os.path.join(base_dir,args.dataset+'_rgb_mix8_v3_npz') 
              ,'mixed_7_join' :os.path.join(base_dir,args.dataset+'_rgb_mix7_v3_npz') 
             # ,'top_cls_global_pool':os.path.join(base_dir,args.dataset+'_rgb_top_v3_npz')
             },
    label=os.path.join(split_dir,'alllist.txt'), 
    ext = '_rgb.npz',
    is_training=False,
    feat_list = feat_list,
    n = args.n
    )
showloader = torch.utils.data.DataLoader(showset, batch_size=BATCH_SIZE, shuffle=False, num_workers=read_workers)
dim = 2048
net = SpaNet(glo_channels=1280, loc_channels=768, out_channels=dim, num_classes=class_num, drop_rate=drop_rate, out_type=args.type, n=args.n)
if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True
net.load_state_dict(torch.load('UCF101_split1_rgb_98.678_model'))

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

def pinghua(img, s=11):
    w,h = img.shape
    img0 = img.copy()
    for i in range(w):
        for j in range(h):
            img[i,j] = img0[max(0,i-s):min(h-1,i+s),max(0,j-s):min(h-1,j+s)].mean()
    return img

def resize299(img8,size):
    big = np.max(img8)
    img = img8 / big
    img299 = np.zeros((315,315))
    s = 299 // size+1
    for i in range(size):
        for j in range(size):
            a = int(img[i,j])
            img299[i*s:(i+1)*s+1,j*s:(j+1)*s+1] = img[i,j]
    img = cv.resize(img299, (299,299))/4#*0.3
    return img

def pay_atte(frame0, w, a):
    frame = frame0.copy()
    w2 = w.copy()
    #w2[w<0.6]=0
    #w2[w>=0.6]=1
    #w2 = pinghua(w2,20)
    frame[:,:,0] = frame[:,:,0] * w2#np.maximum(w2, a)
    frame[:,:,1] = frame[:,:,1] * w2#np.maximum(w2, a)
    frame[:,:,2] = frame[:,:,2] * w2#np.maximum(w2, a)
    return frame

def CAM(frame, w, a):
    heatmap = np.uint8(255 * w)
    heatmap = cv.applyColorMap(heatmap, cv.COLORMAP_JET)
    res = heatmap *0.3 + frame * 0.5 
    return res

def get_showing_img(image, w_glo, w_loc):
    image = cv.resize(image,(299,299))
    w_glo = pinghua(cv.resize(w_glo,(299,299)))
    w_loc = pinghua(cv.resize(w_loc,(299,299)),s=7)
    heat_glo = CAM(image, w_glo,0.5)
    heat_loc = CAM(image, w_loc,0.5)
    heat_loc_glo = CAM(image, np.multiply(w_loc,w_glo),0.6)
    res = np.concatenate([image, heat_glo, heat_loc, heat_loc_glo], axis=1)
    return res

def _fun(heat_map_glo, heat_map_loc, path, ids):
            image_list = get_images(path)
            image = cv.imread(image_list[0])
            heat_map_glo = heat_map_glo/heat_map_glo.max()
            heat_map_glo = np.reshape(heat_map_glo,(8,8))
            heat_map_loc = heat_map_loc/heat_map_loc.max()
            heat_map_loc = np.reshape(heat_map_loc,(17,17))
            res = get_showing_img(image, heat_map_glo, heat_map_loc)
            cv.imwrite('ucf101show_JET/'+path_list[ids]+'.jpg',res)
            print( path_list[ids], 'has done')

def showall():
    global path_list
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(showloader):
        tar_hot = torch.zeros(targets.size(0), class_num).scatter_(1, targets.unsqueeze(1), 1.0)
        if use_cuda:
            inputs, targets, tar_hot = [x.cuda() for x in inputs], targets.cuda(), tar_hot.cuda()
        inputs, targets, tar_hot = [Variable(x, volatile=True) for x in inputs], Variable(targets), Variable(tar_hot)
        glo_w, loc_w = net(inputs+[tar_hot])
        glo_w = glo_w.data.cpu().numpy()
        loc_w = loc_w.data.cpu().numpy()
        #print(glo_w.size(), loc_w.size())
        pool = multiprocessing.Pool(processes = 16)
        for i in range(glo_w.shape[0]//25):
            heat_glo = glo_w[25*i]
            heat_loc = loc_w[25*i]
            np.savetxt('ucfspaweight/'+path_list[batch_idx*32+i]+'_glo.txt' ,heat_glo)
            np.savetxt('ucfspaweight/'+path_list[batch_idx*32+i]+'_loc.txt' ,heat_loc)
            path = path_list[batch_idx*BATCH_SIZE+i]
            ids = batch_idx*BATCH_SIZE+i
            pool.apply_async(_fun, (heat_glo, heat_loc, path, ids))
        pool.close()
        pool.join()

if __name__ == '__main__':
    showall()



