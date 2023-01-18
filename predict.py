import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
#from torchsummary import summary
from torch.autograd import Variable
import torch.optim as optim
import argparse
from data_pre import myDataSet
import os, json
import cv2
from easydict import EasyDict as edict
import time
from torchsummary import summary
from tensorboardX import SummaryWriter
import numpy as np
# from torchvision.ops import nms
from unet_model import UNet
import imageio

def mkdir(path):
    path=path.strip()
    path=path.rstrip("\\")
    isExists=os.path.exists(path)
    if not isExists:
        os.makedirs(path) 
        return True
    else:
        return False


Transform = transforms.Compose([
    transforms.Resize([256, 256]),
    transforms.ToTensor(),
    ])

parser = argparse.ArgumentParser(description='wsddn Input:BatchSize initial LR EPOCH')
parser.add_argument('--config_path', type=str, help='config path')
args = parser.parse_args()
file_dir = args.config_path
with open(file_dir) as f:
        config = json.load(f)
config = edict(config)
train_dir = config.TRAIN.train_dir
test_dir = config.VALID.test_dir
train_cam_dir = config.TRAIN.cam_dir
test_cam_dir = config.VALID.cam_dir
arch = config.TRAIN.arch
model_path = config.VALID.model_path
save_dir_sod = os.path.join(config.VALID.pre_dir, 'predict_sod')
save_dir_seg = os.path.join(config.VALID.pre_dir, 'predict_seg')
mkdir(save_dir_seg)


net_wsddn = torch.load(os.path.join(model_path, '95.pkl'))  #需要你注意的，10.pkl中，10的意思是第10个epoch后训练得到的权重

testData = myDataSet(test_dir, test_cam_dir, Transform)

testLoader = torch.utils.data.DataLoader(dataset=testData, batch_size=1, shuffle=False,num_workers=1)

trainData = myDataSet(train_dir, train_cam_dir, Transform)

trainLoader = torch.utils.data.DataLoader(dataset=trainData, batch_size=1, shuffle=False,num_workers=1)

Loader = trainLoader
save_dir_train = config.VALID.save_dir
save_dir_test = save_dir_train[0:-5] + 'test'
print(save_dir_test)
if(not os.path.exists(save_dir_test)):
    os.mkdir(save_dir_test)
all = 0
y = 0
im = np.zeros([10, 256, 256], np.float32) #创建一个10*256*256的矩阵
for i, (imagename, images, cam) in enumerate(Loader):
    print(imagename)
    imagename = imagename[0]
    save_dir_temp = save_dir_seg
    images = Variable(images).cuda() #创建变量
    seg, _, _ = net_wsddn(images)
    img = cv2.imread(os.path.join(test_dir, imagename)) #从指定的文件加载图像
    seg = nn.functional.softmax(seg,dim = 1)
    seg = seg.cpu().detach().numpy()
    seg = np.squeeze(seg[:,1,:,:]) #!
    # im[j,:,:] = seg
    imageio.imsave(os.path.join(save_dir_train, imagename), seg)

Loader = testLoader
for i, (imagename, images, cam) in enumerate(Loader):
    print(imagename)
    imagename = imagename[0]
    save_dir_temp = save_dir_seg
    images = Variable(images).cuda() #创建变量
    seg, _, _ = net_wsddn(images)
    img = cv2.imread(os.path.join(test_dir, imagename)) #从指定的文件加载图像
    seg = nn.functional.softmax(seg,dim = 1)
    seg = seg.cpu().detach().numpy()
    seg = np.squeeze(seg[:,1,:,:]) #!
    # im[j,:,:] = seg
    imageio.imsave(os.path.join(save_dir_test, imagename), seg)   

