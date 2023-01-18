from torch.utils import data
from PIL import Image
import torchvision.transforms as transforms
import torch
import numpy as np
from math import floor
import pickle
import os
import cv2
import imageio

Transform = transforms.Compose([
    transforms.Resize([256, 256]),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                         std  = [ 0.229, 0.224, 0.225 ]),
    ])

def file_name(file_dir):   
    L=[]   
    for root, dirs, files in os.walk(file_dir):  
        for file in files:  
            if os.path.splitext(file)[1] == '.png':  
                L.append(os.path.join(file))  
    return L  

class myDataSet(data.Dataset):
    def __init__(self, image_dir, cam_dir, transfrom):
        self.image_dir = image_dir
        self.cam_dir = cam_dir
        self.transform = transfrom
        self.imgs = file_name(image_dir)
                    
    def __getitem__(self, index):
        image_name = self.imgs[index]
        # print(image_name)
        cam_name = os.path.join(self.cam_dir, image_name)
        cur_img = Image.fromarray(cv2.imread(os.path.join(self.image_dir, self.imgs[index])))
        cam = cv2.imread(cam_name)
        # cam = cam.unsqueeze(2)
        cam = cv2.resize(cam, (256, 256))
        cam = cam[:,:,0]
        # cam[cam<=127] = 0.
        # cam[cam>127] = 1.

        save_dir  = './temp_cam/'
        _, cam = cv2.threshold(cam,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        cam[cam<=127] = 0.
        cam[cam>127] = 1.
        cv2.imwrite(save_dir + image_name,cam)
        # print(np.sum(cam))
        data_once = self.transform(cur_img)
        cam_tensor = torch.tensor(cam).long()
        return self.imgs[index], data_once, cam_tensor
    
    def __len__(self):
        return len(self.imgs)

class myDataSet_R(data.Dataset):
    def __init__(self, image_dir, cam_dir, weight_dir, transfrom):
        self.image_dir = image_dir
        self.cam_dir = cam_dir
        self.weight_dir = weight_dir
        self.transform = transfrom
        self.imgs = file_name(image_dir)
                    
    def __getitem__(self, index):
        image_name = self.imgs[index]
        # print(image_name)
        cam_name = os.path.join(self.cam_dir, image_name)
        cur_img = Image.fromarray(cv2.imread(os.path.join(self.image_dir, self.imgs[index])))
        cam = cv2.imread(cam_name)
        # cam = cam.unsqueeze(2)
        cam = cv2.resize(cam, (256, 256))
        cam = cam[:,:,0]
        cam[cam<=127] = 0.
        cam[cam>127] = 1.
        # print(np.sum(cam))
        data_once = self.transform(cur_img)
        cam_tensor = torch.tensor(cam)

        weight_name = os.path.join(self.weight_dir, image_name)
        # print(weight_name)
        weight = cv2.imread(weight_name, cv2.IMREAD_GRAYSCALE).astype(np.float32)/255. #weight [0,1]
        weight_tensor = torch.tensor(weight)
        
        return self.imgs[index], data_once, cam_tensor, weight_tensor
    
    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':
    image_dir = '/student/zj/Data/act_solar_f1881+b1524+t190/train/fore/'
    cam_dir = '/student/zj/label_noisy/label_noisy_step1/act/deform/0.2/20210615200511/pseudo_label/features.21'
    trainData = myDataSet(image_dir, cam_dir, Transform)
    print('trainData', len(trainData))
    # print(trainData[0][2])
    cv2.imwrite('1.png',trainData[0][2].numpy()*255.)
