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
from loss import loss_ce, loss_similarity #*
from easydict import EasyDict as edict
import time
import cv2
import numpy as np
from tqdm import tqdm
import imageio
from unet_model import UNet #*

def mkdir(path):
    path=path.strip() #去除字符串两边的空格
    path=path.rstrip("\\") #去除字符串右边的“\\”
    isExists=os.path.exists(path) #判断括号中的文件是否存在
    if not isExists:
        os.makedirs(path) #用于递归创建目录，多层目录
        return True
    else:
        return False


# Transform = transforms.Compose([
#     transforms.Resize([256, 256]),
#     transforms.ToTensor(),
#     transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
#                          std  = [ 0.229, 0.224, 0.225 ]),
#     ])

Transform = transforms.Compose([ #将transforms组合在一起，而每一个transforms都有自己的功能。
    transforms.Resize([256, 256]), #resize成256*256大小
    transforms.ToTensor(), #转换为ensor格式，送到tensorboard
    ])


SEED = 0 #控制随机种子
torch.manual_seed(SEED) #设置CPU生成随机数的种子，方便下次复现实验结果
torch.cuda.manual_seed(SEED) #为特定GPU设置种子，生成随机数
np.random.seed(SEED) #用于生成指定随机数

parser = argparse.ArgumentParser(description='wsddn Input:BatchSize initial LR EPOCH') #创建一个解析对象
parser.add_argument('--config_path', type=str, help='config path') #type是要传入参数的数据类型，help是该参数的提示信息
args = parser.parse_args() #进行解析
file_dir = args.config_path
with open(file_dir) as f:
        config = json.load(f) #传入一个json格式的文件流，将其解码为python对象
config = edict(config) #可以使得以属性（键）的方式去访问字典的值
train_dir = config.TRAIN.train_dir
train_cam_dir = config.TRAIN.cam_dir
test_dir = config.VALID.test_dir
test_cam_dir = config.VALID.cam_dir
arch = config.TRAIN.arch
print_fre = config.TRAIN.print_fre
save_fre = config.TRAIN.save_frequence
time_now = time.strftime("%Y%m%d%H%M%S", time.localtime())
model_path = os.path.join('./checkpoint/',train_dir.split('/')[-4], time_now) #拼接路径
log_dir = os.path.join('./log/',train_dir.split('/')[-4], time_now)
save_dir = os.path.join('./result/',train_dir.split('/')[-4], time_now,'train')
predict_dir = os.path.join('./result/',train_dir.split('/')[-4], time_now,'predict')
mkdir(model_path) #创建目录
mkdir(log_dir)
mkdir(save_dir)
mkdir(predict_dir)
config.VALID.model_path = model_path
config.VALID.save_dir = save_dir
config.VALID.pre_dir = predict_dir



save_config = os.path.join(predict_dir, 'config.json') #用于路径拼接文件路径
with open(file_dir, 'w') as f:
        f.write(json.dumps(config, indent=4)) #对数据进行编码，形成json格式的数据

with open(save_config, 'w') as f:
        f.write(json.dumps(config, indent=4))

BATCH_SIZE=config.TRAIN.BATCH_SIZE
EPOCH=config.TRAIN.EPOCH
LR=config.TRAIN.LR
print('model_path:',model_path)
print('batch_size:',BATCH_SIZE)
print('initial LR:',LR)
print('epoch:',EPOCH)
torch.cuda.set_device(config.TRAIN.GPU) #设定指定的GPU

net_wsddn = UNet(3,2)


net_wsddn.cuda() #传到显卡上
for name, param in net_wsddn.named_parameters():
  if param.requires_grad:
    print(name)

criterion_seg = nn.CrossEntropyLoss() #交叉熵损失
# criterion_seg = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(net_wsddn.parameters(), lr = LR) #parameters返回参数列表
# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 20, gamma = 0.5) #每step_size个epoch衰减每个参数组的学习率
writer = SummaryWriter(log_dir) #运行产生log文件


trainData = myDataSet(train_dir,train_cam_dir, Transform)
print(len(trainData))
testData = myDataSet(test_dir, test_cam_dir,Transform)
trainLoader = torch.utils.data.DataLoader(dataset=trainData, batch_size=BATCH_SIZE, shuffle=True,num_workers=1) #数据加载器
testLoader = torch.utils.data.DataLoader(dataset=testData, batch_size=BATCH_SIZE, shuffle=False,num_workers=1)


net_wsddn.train() #训练模式
for epoch in range(EPOCH):
    running_loss = 0.0
    test_loss = 0.0
    k = 0
    # print(epoch)
    scheduler.step()
    for (imagename, images, cam) in tqdm(trainLoader):
        images = Variable(images).cuda()
        cam = Variable(cam).long().cuda()
        optimizer.zero_grad()
        seg_out, up1_f, up2_f = net_wsddn(images)

        seg_out = torch.softmax(seg_out, dim = 1)
        seg_out_bin = seg_out[:, 1, :, :].clone().detach().unsqueeze(1)

        seg_out_bin1 = torch.nn.functional.interpolate(seg_out_bin, [up1_f.size()[2], up1_f.size()[2]])
        seg_out_bin2 = torch.nn.functional.interpolate(seg_out_bin, [up2_f.size()[2], up2_f.size()[2]])
        seg_out_flatten1 = seg_out_bin1.view(1,-1).squeeze()
        seg_out_flatten2 = seg_out_bin2.view(1,-1).squeeze()
        up1_flatten = up1_f.view(up1_f.size()[1],-1)
        up2_flatten = up2_f.view(up2_f.size()[1],-1)

        pos_feature1 = up1_flatten[:,seg_out_flatten1>0.5]
        neg_feature1 = up1_flatten[:,seg_out_flatten1<=0.5]
        pos_feature2 = up2_flatten[:,seg_out_flatten2>0.5]
        neg_feature2 = up2_flatten[:,seg_out_flatten2<=0.5]

        l_pos1 = torch.bmm(pos_feature1.reshape(pos_feature1.size(1), 1, pos_feature1.size(0)), pos_feature1.reshape(pos_feature1.size(1), pos_feature1.size(0), 1)).squeeze(-1)
        
        l_neg1  = torch.mm(pos_feature1.T, neg_feature1)

        l_pos2 = torch.bmm(pos_feature2.reshape(pos_feature2.size(1), 1, pos_feature2.size(0)), pos_feature2.reshape(pos_feature2.size(1), pos_feature2.size(0), 1)).squeeze(-1)
        
        l_neg2  = torch.mm(pos_feature2.T, neg_feature2)

        f1 = torch.cat([l_pos1, l_neg1], dim = 1)
        f2 = torch.cat([l_pos2, l_neg2], dim = 1)

        labels1 = torch.zeros(pos_feature1.size(1)).long().cuda()
        labels2 = torch.zeros(pos_feature2.size(1)).long().cuda()

        loss_cl1 = criterion_seg(f1,labels1)
        loss_cl2 = criterion_seg(f2,labels2)



        loss = criterion_seg(seg_out, cam) +  loss_cl1 + loss_cl2
        running_loss += loss.item()
        loss.backward()
        optimizer.step()

    # seg_out = torch.nn.functional.sigmoid(seg_out)
    
    writer.add_scalar('Train/loss', running_loss / len(trainLoader), epoch)

    for j, (imagename, images, cam) in enumerate(testLoader):

        images = Variable(images).cuda()
        cam = Variable(cam).cuda()
        seg_out,_,_ = net_wsddn(images.cuda())
        loss = criterion_seg(seg_out , cam)
        test_loss += loss.item()

    seg_out = nn.functional.softmax(seg_out, dim=1)
    writer.add_scalar('Test/loss', test_loss / len(testLoader), epoch)
    writer.add_scalar('lr', scheduler.get_last_lr()[0], epoch)
    writer.add_image('input', torch.squeeze(torch.squeeze(images[0,:,:,:])), epoch)
    writer.add_image('predict', torch.unsqueeze(seg_out[0,1,:,:],0), epoch)
    writer.add_image('cam', torch.unsqueeze(cam[0,:,:],0), epoch)


            # break
    print('Train:[epoch: %1d] loss_all: %.5f \
    | Test: loss_all: %.5f' % (epoch + 1 , \
    running_loss / len(trainLoader),  
    test_loss / len(testLoader),))
    

        
    if epoch % save_fre == 0 and epoch>0:
        torch.save(net_wsddn, os.path.join(model_path, str(epoch)+'.pkl'))







# for epoch in range(EPOCH):
#     running_loss = 0.0
#     test_loss = 0.0
#     k = 0
#     # print(epoch)
#     scheduler.step()
#     for (imagename, images, cam) in trainLoader: #训练集上测试
#         images = Variable(images).cuda()
#         cam = Variable(cam).cuda()
#         optimizer.zero_grad() #128、133、134行做loss优化
   