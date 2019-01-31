#! ../anaconda/envs/py35/bin/python3
'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function
import os
import argparse
import sys
import numpy as np
import datasets

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.005, type=float, help='learning rate')
parser.add_argument('--dataset', default='UCF101')
parser.add_argument('--spa', default='UCF101')
parser.add_argument('--tim', default='UCF101')
parser.add_argument('--split', default='1')
args = parser.parse_args()

class_num = 101 if args.dataset=='UCF101' else 51
split_dir = 'ucfTrainTestlist' if args.dataset=='UCF101' else 'hmdbTrainTestlist'

base_dir = '../temporal-segment-networks/data/'
feat_list = ['mixed_8_join','mixed_7_join']#['mixed_7_join','mixed_8_join','mixed_10_join']

testset = datasets.UCF101_mixed_v3_dict(
    rootDict={'mixed_10_join':os.path.join(base_dir,args.dataset+'_rgb_mix10_v3_npz'),
              'mixed_8_join' :os.path.join(base_dir,args.dataset+'_rgb_mix8_v3_npz') ,
              'mixed_7_join' :os.path.join(base_dir,args.dataset+'_rgb_mix7_v3_npz')},
    label=os.path.join(split_dir,'testlist0'+args.split+'.txt'),
    ext = '_rgb.npz',
    is_training=False,
    feat_list = feat_list
    )
def test_with_score(score):
    true_num = 0.0
    ids = np.argmax(score, axis=1)
    for i in range(testset.__len__()):
        if testset.datas[i][1]==ids[i]:
            true_num += 1.0
    return true_num/testset.__len__()
if __name__ == '__main__':
    a = 0.0
    acc_max, a_max = 0, 0
    spa_score = np.load(args.spa)
    tim_score = np.load(args.tim)
    acc_spa = test_with_score(spa_score)
    acc_tim = test_with_score(tim_score)
    while a<100:
      all_score = a*spa_score + tim_score
      acc_all = test_with_score(all_score)
      if acc_max<acc_all:
          acc_max=acc_all
          a_max = a
      a+=0.01
    print("spa:%.2f%% tim:%.2f%% sum:%.2f%%"%(acc_spa*100, acc_tim*100, acc_max*100))
