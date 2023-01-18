import torch 
import torch.nn as nn
import numpy as np

eps = 1e-10

class loss_ce(nn.Module):
    def __init__(self):
        super(loss_ce, self).__init__()
    
    def forward(self, pred, labels):
        '''
        pred.size: [8, 2, 256, 256]
        labels.size: [8, 2, 256, 256]
        # '''
        # print(pred.size())
        # print(labels.size())
        pred = torch.squeeze(nn.functional.softmax(pred, dim=1))
        pred = torch.clamp(pred, 1e-5, 1-1e-5) #将pred限制在1e-5到1-1e-5之间
        # pos_num = torch.sum(labels)*1.0
        # pos_ratio = pos_num/labels.numel()
        # neg_ratio = 1.- pos_ratio
        # print(pos_ratio)
        # print(pre, labels)
        # loss = nn.CrossEntropyLoss(weight = weight)
        # print(pred.size(), labels.size())
        # print(labels.size(), pred.size())
        # print(pred.size(), labels.size())
        # loss = torch.sum(-1.*(neg_ratio*labels*torch.log(pred[:,0,:,:]) + pos_ratio*(1.-labels)*torch.log(pred[:,1,:,:])))/labels.numel()
        temp1 = torch.log(pred[:,0,:,:])
        temp2 = torch.log(pred[:,1,:,:])
        # print(labels.size())
        # print(temp1.size())
        # print(temp2.size())
        loss = torch.mean(-1.*(labels*temp1 + (1.-labels)*temp2))
        
        # print(loss.size())
        # image_level_scores = torch.clamp(pred, min=0.0, max=1.0)
        # loss = nn.functional.binary_cross_entropy(pred, labels, reduction="sum")


        return loss


class loss_T(nn.Module):
    def __init__(self):
        super(loss_T, self).__init__()
    
    def forward(self, pred, labels, weight):
        '''
        pred.size: [8, 2, 256, 256]
        labels.size: [8, 2, 256, 256]
        # '''
        # print(pred.size())
        # print(labels.size())
        pred = torch.squeeze(nn.functional.softmax(pred, dim=1))
        pred = torch.clamp(pred, 1e-5, 1-1e-5) #将pred限制在1e-5到1-1e-5之间
        
        loss = torch.mean(-1.*weight*(labels*torch.log(pred[:,0,:,:]) + (1.-labels)*torch.log(pred[:,1,:,:])))
        
        
        return loss

class loss_ce_T(nn.Module):
    def __init__(self):
        super(loss_ce_T, self).__init__()
    
    def forward(self, pred, labels):
        '''
        pred.size: [8, 2, 256, 256]
        labels.size: [8, 2, 256, 256]
        # '''
        # print(pred.size())
        # print(labels.size())
        # pred = torch.squeeze(nn.functional.softmax(pred, dim=1))
        pred = torch.clamp(pred, 1e-5, 1-1e-5)
        pos_num = torch.sum(labels)*1.0 #计算labels的和
        pos_ratio = pos_num/labels.numel() #numel()返回lables的个数
        neg_ratio = 1.- pos_ratio
        # print(pos_ratio)
        # print(pre, labels)
        # loss = nn.CrossEntropyLoss(weight = weight)
        # print(pred.size(), labels.size())
        # print(labels.size(), pred.size())
        # print(pred.size(), labels.size())
        # loss = torch.sum(-1.*(neg_ratio*labels*torch.log(pred[:,0,:,:]) + pos_ratio*(1.-labels)*torch.log(pred[:,1,:,:])))/labels.numel()
        loss = torch.mean(-1.*(labels*torch.log(pred[:,0,:,:]) + (1.-labels)*torch.log(pred[:,1,:,:])))
        
        # print(loss.size())
        # image_level_scores = torch.clamp(pred, min=0.0, max=1.0)
        # loss = nn.functional.binary_cross_entropy(pred, labels, reduction="sum")


        return loss



class loss_R(nn.Module):
    def __init__(self):
        super(loss_R, self).__init__()
    
    def forward(self, pred, labels, weight):
        '''
        pred.size: [8, 2, 256, 256]
        labels.size: [8, 2, 256, 256]
        # '''
        pred = torch.squeeze(nn.functional.softmax(pred, dim=1))
        pred = torch.clamp(pred, 1e-5, 1-1e-5)
        loss = torch.mean(-1.*weight*(labels*torch.log(pred[:,0,:,:]) + (1.-labels)*torch.log(pred[:,1,:,:])))

        return loss

class loss_similarity(nn.Module):
    def __init__(self):
        super(loss_similarity, self).__init__()
    
    def forward(self, ssw, seg, score):
        ssw_block = torch.squeeze(ssw, dim = 0)[:, 1:]
        ssw_block = ssw_block.cpu().numpy()
        ssw_block = ssw_block.astype(np.int16)
        seg = nn.functional.softmax(seg, dim=1)     
        score_map = torch.squeeze(seg[0,1,:,:])
        roi_score = torch.squeeze(score)
        roi_num = ssw_block.shape[0]
        iou = torch.empty(roi_num).cuda()
        for i in range(roi_num):
            x1, y1, x2, y2 = ssw_block[i,:]
            iou_region = score_map[y1:y2,x1:x2]
            iou[i] = iou_region.mean()
        iou = iou.expand(roi_num,roi_num).cuda()
        iouT = iou.T
        dis_iou = iou - iou.T
        sim_iou = torch.sqrt(dis_iou*dis_iou + eps)
        
        score = score.expand(roi_num,roi_num)
        scoreT = score.T
        dis_score = score - scoreT
        sim_score = torch.sqrt(dis_score*dis_score+eps)
        if torch.sum(torch.isnan(sim_iou))>0:
            print(iou, roi_num)
            print('sim_iou')
        if torch.sum(torch.isnan(sim_score))>0:
            print('sim_score')

        # print('sim_iou', torch.sum(torch.isnan(sim_iou)))
        # print('sim_score', torch.sum(torch.isnan(sim_score)))
        
        loss_fn = nn.MSELoss()
        loss = loss_fn(sim_iou, sim_score)
        # print(loss)
        
        return loss

class loss_adv(nn.Module):
    def __init__(self):
        super(loss_adv, self).__init__()
    
    def forward(self, ssw, seg, score):
        ssw_block = torch.squeeze(ssw, dim = 0)[:, 1:]
        ssw_block = ssw_block.cpu().numpy()
        ssw_block = ssw_block.astype(np.int16)
        seg = nn.functional.softmax(seg, dim=1)     
        score_map = torch.squeeze(seg[0,1,:,:])
        roi_score = torch.squeeze(score)
        roi_num = ssw_block.shape[0]
        iou = torch.empty(roi_num).cuda()
        for i in range(roi_num):
            x1, y1, x2, y2 = ssw_block[i,:]
            iou_region = score_map[y1:y2,x1:x2]
            iou[i] = iou_region.mean()
        iou = iou.expand(roi_num,roi_num).cuda()
        iouT = iou.T
        dis_iou = iou - iou.T
        sim_iou = torch.sqrt(dis_iou*dis_iou + eps)
        
        score = score.expand(roi_num,roi_num)
        scoreT = score.T
        dis_score = score - scoreT
        sim_score = torch.sqrt(dis_score*dis_score+eps)
        if torch.sum(torch.isnan(sim_iou))>0:
            print(iou, roi_num)
            print('sim_iou')
        if torch.sum(torch.isnan(sim_score))>0:
            print('sim_score')

        # print('sim_iou', torch.sum(torch.isnan(sim_iou)))
        # print('sim_score', torch.sum(torch.isnan(sim_score)))
        
        loss_fn = nn.MSELoss()
        loss = loss_fn(sim_iou, sim_score)
        # print(loss)
        
        return loss
