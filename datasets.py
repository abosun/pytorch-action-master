from __future__ import print_function
import torch
from PIL import Image
import os
import os.path
import errno
import numpy as np
import sys
import cv2
import imageio
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
import random
import torch.utils.data as data


def default_loader(path):
    return Image.open(path).convert('RGB')

class UCF101_IMG(data.Dataset):
    def __init__(self, root, label, class_info=None, transform = None, target_transform=None, loader=default_loader):
        fh = open(label)
        c=0
        imgs=[]
        class_names=[]
        for line in  fh.readlines():
            line = line.split(' ')
            imgs.append((line[0], int(line[1])))
        print(len(imgs))
        print(imgs[:3])
        self.root = root
        self.imgs = imgs
        self.classes = class_names
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(os.path.join(self.root, fn))
        #img = img.resize((299,299))
        if self.transform is not None:
            img = self.transform(img)
        return img, label#torch.Tensor(label)
    def __len__(self):
        return len(self.imgs)

def video_loader(path):
#    video = imageio.get_reader(path)
#    return Image.fromarray(video.get_data(random.randint(0,video.get_length()-1)))
    video = cv2.VideoCapture(path)
    frame_list = []
    ret = True
    while(ret):
        frame_list.append(0)
        ret, frame_list[-1] = video.read()
    return Image.fromarray(frame_list[random.randint(0,len(frame_list)-2)])
def top_loader(path, is_training=False):
    data = np.load(path)
    if path[-3:]=='npz':
        data = data['global_pool']
    res = []
    for i in range(5):
        if is_training:
          res.append(data[random.randint(0,4)+i*5])
#        else:
#          res.append(data[2+i*5])
    if not is_training:
      return data.transpose()
    return np.vstack(res).transpose()
def mixed_loader(path, feat_list, is_training=False):
  try:
    data = np.load(path,encoding="latin1")#.all()
    feats = []
    for feat_name in feat_list:
        feat = data[feat_name]
        tmp = []
        for i in range(5):
            if is_training:
                tmp.append(feat[random.randint(0,4)+i*5])
        if not is_training:
            feats.append(feat)
        else:
            feats.append(np.stack(tmp,axis=0))
    return feats[0].squeeze()
  except:
    print(path)
def mixed_loader1(path, feat_name, is_training=False):
    data = np.load(path,encoding="latin1")
    feat = data[feat_name]
    if not is_training:
        return feat
    tmp = []
    for i in range(5):
        tmp.append(feat[random.randint(0,4)+i*5])
    feat = np.stack(tmp,axis=0)
    return feat.squeeze()

class UCF101_VIDEO(data.Dataset):
    def __init__(self, root, label, class_info=None, transform = None, target_transform=None, loader=video_loader):
        fh = open(label)
        c=0
        imgs=[]
        class_names=[]
        for line in  fh.readlines():
            line = line.split(' ')
            imgs.append((line[0], int(line[1])-1))
        print(len(imgs))
        print(imgs[:3])
        self.root = root
        self.imgs = imgs
        self.classes = class_names
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(os.path.join(self.root, fn))
        #img = img.resize((299,299))
        if self.transform is not None:
            img = self.transform(img)
        return img, label#torch.Tensor(label)
    def __len__(self):
        return len(self.imgs)

class UCF101_Top_v3(data.Dataset):
    def __init__(self, root, label, class_info=None, transform = None, target_transform=None, loader=top_loader, is_training=False, ext='_flow.npy'):
        self.ext = ext
        fh = open(label)
        c=0
        pathes=[]
        class_names=[]
        for line in fh.readlines():
            line = line.split(' ')
            pathes.append((line[0][:-4]+ext, int(line[1])-1))
        print(len(pathes))
        print(pathes[-3:])
        self.root = root
        self.datas = pathes
        self.classes = class_names
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.is_training = is_training
    def __getitem__(self, index):
        fn, label = self.datas[index]
        data = self.loader(os.path.join(self.root, fn), self.is_training)
        return data, label
    def __len__(self):
        return len(self.datas)

class UCF101_Top_multi_v3(data.Dataset):
    def __init__(self, root_short, root_mid, label, class_info=None, transform = None, target_transform=None, loader=top_loader, is_training=False):
        fh = open(label)
        c=0
        pathes_short=[]
        pathes_mid=[]
        class_names=[]
        for line in fh.readlines():
            line = line.split(' ')
            pathes_short.append((line[0].split('.')[0]+"_flow.npy", int(line[1])-1))
            pathes_mid.append((line[0].split('.')[0]+"_flow.npy", int(line[1])-1))
        assert(len(pathes_short)==len(pathes_mid))
        print("train: "+str(len(pathes_mid)) if is_training else "test: "+str(len(pathes_mid)) )
        self.root_short = root_short
        self.root_mid = root_mid
        self.datas_short = pathes_short
        self.datas_mid = pathes_mid
        self.classes = class_names
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.is_training = is_training
    def __getitem__(self, index):
        fn, label = self.datas_short[index]
        data_short = self.loader(os.path.join(self.root_short, fn), self.is_training)
        fn, label = self.datas_mid[index]
        data_mid = self.loader(os.path.join(self.root_mid, fn), self.is_training)
        return [data_short, data_mid], label
    def __len__(self):
        return len(self.datas_short)
class UCF101_mixed_v3(data.Dataset):
    def __init__(self, root, label, class_info=None, transform = None, target_transform=None, loader=mixed_loader, is_training=False, ext='_flow.npy', feat_list=None):
        self.ext = ext
        fh = open(label)
        c=0
        pathes=[]
        class_names=[]
        for line in fh.readlines():
            line = line.split(' ')
            pathes.append((line[0].split('.')[0]+ext, int(line[1])-1))
        assert(not feat_list is None)
        print("train: "+str(len(pathes)) if is_training else "test: "+str(len(pathes)) )
        self.root = root
        self.datas = pathes
        self.classes = class_names
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.is_training = is_training
        self.feat_list = feat_list
    def __getitem__(self, index):
        fn, label = self.datas[index]
        data = self.loader(os.path.join(self.root, fn), self.feat_list, self.is_training)
        return data, label
    def __len__(self):
        return len(self.datas)
class UCF101_mixed_v3_dict(data.Dataset):
    def __init__(self, rootDict, label, class_info=None, transform = None, target_transform=None, 
                 loader=mixed_loader1, is_training=False, ext='_flow.npy', feat_list=None):
        
        self.feat_list = feat_list
        self.ext = ext
        fh = open(label)
        pathes=[]
        for line in fh.readlines():
            line = line.split(' ')
            pathes.append((line[0].split('.')[0]+ext, int(line[1])-1))
        assert(not feat_list is None)
        print("train: "+str(len(pathes)) if is_training else "test: "+str(len(pathes)) )
        self.rootDict = rootDict
        self.datas = pathes
        self.loader = loader
        self.is_training = is_training
    def __getitem__(self, index):
        fn, label = self.datas[index]
        data_list = []
        for feat_name in self.feat_list:
            data_list.append(self.loader(os.path.join(self.rootDict[feat_name], fn),feat_name, self.is_training))
        return data_list, label
    def __len__(self):
        return len(self.datas)
