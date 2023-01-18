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
# for name, param in net.named_parameters():
#   if param.requires_grad:
#     print(name)

# for name, param in T_net.named_parameters():
#     if param.requires_grad:
#         print(name)

criterion_seg_T = loss_ce()
criterion_seg = loss_ce()

optimizer = optim.Adam(net.parameters(), lr = LR)
optimizer1 = optim.Adam(T_net.parameters(), lr =1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 50, gamma = 0.9)
writer = SummaryWriter(log_dir)


trainData = myDataSet(train_dir,train_cam_dir, Transform)
print(len(trainData))
testData = myDataSet(test_dir, test_cam_dir,Transform)
trainLoader = torch.utils.data.DataLoader(dataset=trainData, batch_size=BATCH_SIZE, shuffle=True,num_workers=1)
testLoader = torch.utils.data.DataLoader(dataset=testData, batch_size=BATCH_SIZE, shuffle=False,num_workers=1)

def Correction(images, T_net, seg_out,cam):
    cat_tensor = torch.cat([cam.unsqueeze(1), images],1).cuda() #伪标签和image叠在一起 [8,1,256,256] + [8,3,256,256] -》 [8, 4,256,256]
    T = T_net(cat_tensor) #参数后续训练的

   

    # print(T.size())
    b,_,h,w = T.size()
    T_temp = torch.reshape(T, [-1,2,2]) #  [8,4,256,256]  -》 [524288, 2, 2]
    # print(T_temp.size())
    T_flatten = torch.softmax(T_temp,dim = 1) # 这里的T 代表 原文中 T‘


    # b = images.size()[0]
    # T_eye = torch.zeros([b*256*256,2,2]).cuda()
    # # print(_sys.size())
    # T_eye[:,0,0] = 1
    # T_eye[:,1,1] = 1
    # T=  torch.reshape(T_eye, [b,4,256,256])
    # b,_,h,w = T.size()
    # T_flatten = torch.reshape(T, [-1,2,2])


    # print(T_flatten[0,:,:])
    seg_out_temp = torch.reshape(seg_out, [-1, 2, 1]) #[8,2,256,256] -> [524288, 2, 1] 
    seg_out_re = torch.reshape(seg_out_temp, [-1, 2, 256,256])
    # print(seg_out[0,:,0,0])
    # print(seg_out_re[0,:,0,0])
    # print(seg_out_temp.size())
    seg_out_flatten = torch.softmax(seg_out_temp, dim = 1)  # T[2,2] * Y[2,1] = Y~[2,1]
    # print(T_flatten.size(), seg_out_flatten.size())
    # [524288, 2, 2] * [524288, 2, 1]  = [524288, 2, 1] ->[8,2,256,256]
    seg_out_re = torch.reshape(seg_out_flatten, [-1, 2, 256,256])
    # print('t:',T_flatten[0,:,:])
    # print('y:',seg_out_flatten[0,:,:])
    out_temp = torch.bmm(T_flatten, seg_out_flatten) # T*Y 对
    out = torch.reshape(out_temp, [-1, 2, h,w])
    # print(torch.mean(torch.abs(out - seg_out)))
    # print('y~：',out_temp[0,:,:])


    # print(out[0,:,0,0])
    # print(seg_out_re[0,:,0,0])
    return out


# 1. 把初始化步骤拉长，初始化得到的Y好一点，把初始化权重保存下来。
# 2. 对于T进行初始化，尽量初始化为单位阵
# 3. 现在我们给T 用的是网络给的，想想还有没有其他方法？

# A(f(x)) f(A(x))

net.train()
for epoch in range(200):
    running_loss = 0.0
    test_loss = 0.0
    k = 0
    # print(epoch)
    scheduler.step()
    for (imagename, images, cam) in trainLoader:
        images = Variable(images).cuda()
        b = images.size()[0]
        T_eye = torch.zeros([b*256*256,2,2]).cuda()
        # print(_sys.size())
        T_eye[:,0,0] = 1
        T_eye[:,1,1] = 1
        T_eye = torch.reshape(T_eye, [b,4,256,256])
        loss_teye =torch.nn.MSELoss().cuda()


        cam = Variable(cam).cuda() #伪标签
        # print(images.size(), cam.size())
        seg_out = net(images) #segout模型提取结果，我们希望这个是完美的 cam = T*segout -> y~ = T*y
        #label_noise
        out = Correction(images, T_net, seg_out, cam) #作用：T*y
        if epoch<200:
            loss = criterion_seg_T(torch.squeeze(torch.softmax(seg_out, dim=1)), cam) #Y 与 Y~
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # cat_tensor = torch.cat([cam.unsqueeze(1), images],1).cuda() #伪标签和image叠在一起 [8,1,256,256] + [8,3,256,256] -》 [8, 4,256,256]
            # T = T_net(cat_tensor) #参数后续训练的
            # loss_t = loss_teye(T,T_eye)
            # optimizer1.zero_grad()
            # loss_t.backward()
            # optimizer1.step()
        # else:
        #     loss = criterion_seg_T(out, cam) #TY 与 Y~
        #     if(epoch%200<100):
        #         optimizer1.zero_grad()
        #         loss.backward()
        #         optimizer1.step()
        #     else:
        #         optimizer.zero_grad()
        #         loss.backward()
        #         optimizer.step()
            
            
        running_loss += loss.item()
        

    for j, (imagename, images, cam) in enumerate(testLoader):

        images = Variable(images).cuda()
        cam = Variable(cam).cuda()
        seg_out = net(images.cuda())
        out = Correction(images, T_net, seg_out, cam) #作用：T*y
        loss = criterion_seg(seg_out , cam)
        
        test_loss += loss.item()
    # out = torch.nn.functional.softmax(seg_out,dim=1)


            # break
    print('Train:[epoch: %1d] loss_all: %.5f \
    | Test: loss_all: %.5f' % (epoch + 1 , \
    running_loss / len(trainLoader),  
    test_loss / len(testLoader),))
    writer.add_scalar('Test/loss', test_loss / len(testLoader), epoch)
    writer.add_scalar('lr', scheduler.get_last_lr()[0], epoch)

    writer.add_image('input', torch.squeeze(torch.squeeze(images[0,:,:,:])), epoch)
    pre = torch.squeeze(torch.softmax(seg_out, dim=1))
    writer.add_image('predict', torch.unsqueeze(pre[0,0,:,:],0), epoch)
    writer.add_image('Y~', torch.unsqueeze(out[0,0,:,:],0), epoch)
    writer.add_image('cam', torch.unsqueeze(cam[0,:,:],0), epoch)
    
    writer.add_image('res', torch.unsqueeze(out[0,0,:,:] - pre[0,0,:,:] ,0), epoch)
    
    # print(cam[0,:,:])

    torch.save(net, '200.pkl')
    if epoch % save_fre == 0 and epoch>0:
        torch.save(net, os.path.join(model_path, str(epoch)+'.pkl'))
