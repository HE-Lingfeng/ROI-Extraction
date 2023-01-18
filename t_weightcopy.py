import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
#from torchsummary import summary
from torch.autograd import Variable
import torch.optim as optim
import argparse
from data_pre import myDataSet, myDataSet_R
import os, json
from tensorboardX import SummaryWriter
from loss import *
from easydict import EasyDict as edict
import time
import cv2
import numpy as np
from tqdm import tqdm
import imageio
from unet_model import UNet


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


SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)

parser = argparse.ArgumentParser(description='Input:BatchSize initial LR EPOCH')
parser.add_argument('--config_path', type=str, help='config path')
args = parser.parse_args()
file_dir = args.config_path
with open(file_dir) as f:
        config = json.load(f)
config = edict(config)
train_dir = config.TRAIN.train_dir
train_cam_dir = config.TRAIN.cam_dir
test_dir = config.VALID.test_dir
test_cam_dir = config.VALID.cam_dir
arch = config.TRAIN.arch
print_fre = config.TRAIN.print_fre
save_fre = config.TRAIN.save_frequence
time_now = time.strftime("%Y%m%d%H%M%S", time.localtime())
model_path = os.path.join('./checkpoint/',train_dir.split('/')[-4], time_now)
log_dir = os.path.join('./log/',train_dir.split('/')[-4], time_now)
save_dir = os.path.join('./result/',train_dir.split('/')[-4], time_now,'train')
predict_dir = os.path.join('./result/',train_dir.split('/')[-4], time_now,'predict')
mkdir(model_path)
mkdir(log_dir)
mkdir(save_dir)
mkdir(predict_dir)
config.VALID.model_path = model_path
config.VALID.save_dir = save_dir
config.VALID.pre_dir = predict_dir



save_config = os.path.join(predict_dir, 'config.json')
with open(file_dir, 'w') as f:
        f.write(json.dumps(config, indent=4))

with open(save_config, 'w') as f:
        f.write(json.dumps(config, indent=4))

BATCH_SIZE=config.TRAIN.BATCH_SIZE
EPOCH=config.TRAIN.EPOCH
LR=config.TRAIN.LR
print('model_path:',model_path)
print('batch_size:',BATCH_SIZE)
print('initial LR:',LR)
print('epoch:',EPOCH)
torch.cuda.set_device(config.TRAIN.GPU)

net = UNet(3,2) #计算提取结果
T_net = UNet(4,4) #计算转移矩阵T

net.cuda()
T_net.cuda()


# loss = loss_R()
loss = torch.nn.CrossEntropyLoss()
loss_t = torch.nn.CrossEntropyLoss()

optimizer = optim.Adam(net.parameters(), lr = LR)
optimizer1 = optim.Adam(T_net.parameters(), lr =1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 20, gamma = 0.5)
writer = SummaryWriter(log_dir)
# train_weight_dir = '/home/student/hlf/PSL/weight'
train_weight_dir = '/student/hlf/hlf/PSL/weight_result3_bin'

trainData = myDataSet_R(train_dir,train_cam_dir, train_weight_dir, Transform)
print(len(trainData))
testData = myDataSet(test_dir, test_cam_dir,Transform)
trainLoader = torch.utils.data.DataLoader(dataset=trainData, batch_size=BATCH_SIZE, shuffle=True,num_workers=1)
testLoader = torch.utils.data.DataLoader(dataset=testData, batch_size=BATCH_SIZE, shuffle=False,num_workers=1)
# data_helper = DataHelper()
ce = nn.CrossEntropyLoss()
net.train()
for epoch in range(EPOCH):
    running_loss = 0.0
    test_loss = 0.0
    k = 0
    # print(epoch)
    scheduler.step()
    for (imagename, images, cam, weight) in trainLoader:
        images = Variable(images).cuda()
        cams = Variable(cam).cuda()
        # weight = Variable(weight).cuda()
        seg_out_logistic, up2_f = net(images)
        # seg_out = torch.softmax(seg_out_logistic, dim = 1)
        # seg_out_bin = seg_out[:, 0, :, :].clone().detach().unsqueeze(1)

        # seg_out_bin = torch.nn.functional.interpolate(seg_out_bin, [up2_f.size()[2], up2_f.size()[2]])
        # seg_out_flatten = seg_out_bin.view(1,-1).squeeze()
        # up2_flatten = up2_f.view(up2_f.size()[1],-1)
        # # print(seg_out_bin.size())
        # # print(up2_f.size())
        # # print(up2_flatten.size())
        # # print(seg_out_flatten.size())

        # # index = seg_out_bin>0.5
        # # print(index.size())
        
        # pos_feature = up2_flatten[:,seg_out_flatten>0.5]
        # neg_feature = up2_flatten[:,seg_out_flatten<0.5]
        # num = min(neg_feature.size()[1], pos_feature.size()[1])

        # # print('pos:', pos_feature.size())
        # # print('neg:', neg_feature.size())
        
        # # pos_feature = pos_feature[:,:num]
        # # neg_feature = neg_feature[:,:num]

        
        # # print(neg_feature.size()[1] + pos_feature.size()[1])
        # l_pos = torch.bmm(pos_feature.reshape(pos_feature.size(1), 1, pos_feature.size(0)), pos_feature.reshape(pos_feature.size(1), pos_feature.size(0), 1)).squeeze(-1)
        
        # l_neg  = torch.mm(pos_feature.T, neg_feature)
        # # print(l_pos.size())
        # # print(pos_feature.T.size(), neg_feature.size())
        # # print(l_neg.size())
        
        # f = torch.cat([l_pos, l_neg], dim = 1)
        # # print(f.size())
        # labels = torch.zeros(pos_feature.size(1)).long().cuda()

        # loss_cl = ce(f,labels)
        # print(loss)
        loss_f = loss(seg_out_logistic, cams.long())

        running_loss += loss_f.item()
        
        optimizer.zero_grad()
        loss_f.backward()
        optimizer.step()
    
    writer.add_scalar('Train/loss', running_loss / len(trainLoader), epoch)
    # writer.add_scalar('lr', scheduler.get_last_lr()[0], epoch)

    # writer.add_image('input', torch.squeeze(torch.squeeze(images[0,:,:,:])), epoch)
    # pre = torch.softmax(seg_out, dim = 1)
    # pre = torch.softmax(seg_out, dim=1) #pre是什么？ [8, 2, 256 , 256]
    # writer.add_image('predict', torch.unsqueeze(seg_out[0,0,:,:],0), epoch)
    # writer.add_image('Y~', torch.unsqueeze(out[0,0,:,:],0), epoch)
    # writer.add_image('cam', torch.unsqueeze(cams[0,:,:],0), epoch)

     

    for j, (imagename, images, cam) in enumerate(testLoader):
        
        images = Variable(images).cuda()
        cam = Variable(cam).cuda()
        # weight = Variable(weight)
        seg_out, _ = net(images.cuda()) # 负无穷到正无穷之间的
        # seg_out = torch.softmax(seg_out_logistic, dim = 1)
        #loss = criterion_seg(seg_out , cam)
        print('main:segout', seg_out.size()) # 8,2,256,256
        print('main:cam',cam.size()) #8, 256, 256  
        loss_g = loss_t(seg_out, cam)
        test_loss += loss_g.item()
    # out = torch.nn.functional.softmax(seg_out,dim=1)


            # break
    print('Train:[epoch: %1d] loss_all: %.5f \
    | Test: loss_all: %.5f' % (epoch + 1 , \
    running_loss / len(trainLoader),  
    test_loss / len(testLoader),))
    writer.add_scalar('Test/loss', test_loss / len(testLoader), epoch)
    writer.add_scalar('lr', scheduler.get_last_lr()[0], epoch)

    writer.add_image('input', torch.squeeze(torch.squeeze(images[0,:,:,:])), epoch)
    pre = torch.softmax(seg_out, dim=1) # [8, 2, 256 , 256]
    writer.add_image('predict', torch.unsqueeze(pre[0,0,:,:],0), epoch)
    writer.add_image('cam', torch.unsqueeze(cam[0,:,:],0), epoch)
        
    if epoch % save_fre == 0 and epoch>0:
        torch.save(net, os.path.join(model_path, str(epoch)+'.pkl'))

