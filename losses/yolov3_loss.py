import torch
import torch.nn as nn
import cv2
import numpy as np
from utils.utils import *


class YOLOV3Loss(nn.Module):
    def __init__(self):
        super(YOLOV3Loss, self).__init__()

    def forward(self,
                predictions,
                bbox_annotations,
                input_dim,
                anchors,
                num_classes,
                num_anchors,
                grid_size,
                iou_thresh=0.5,
                conf_thresh=0.5):
        '''
        predictions是预测结果 predictions.shape=(n,num_anchors,grid_size,grid_size,num_classes+5)
        predictions[1,2,i,j,:4]是第2张图片的坐标为(i,j)的网格预测的第3个框向量
        anchors[2,i,j,:]是坐标为(i,j)的网格的第3个预设框向量
        predictions放的是每张图片的每个网格的3个尺度所预测的处理后的结果(tx,ty,tw,th)
        '''
        batch_size = predictions.shape[0]
        classification_losses = []
        regression_losses = []
        classifictions, regressions = predictions[..., 5:], predictions[..., :5]
        anchors = anchors.view(-1, 4)
        dtype = anchors.dtype

        for i in range(batch_size):
            classification, regression = classifictions[i, :, :], regressions[i, :, :]
            # 为了计算iou方便
            # 把(num_anchors,grid_size,grid_size,num_classes)的classification向量view成(num_anchors*grid_size*grid_size,num_classes)
            # 下面的regression和anchor也同样view一下 view不会改变相对的位置信息
            # 将一个shape为(2,3,3)的张量view成(18,1)的张量再view回(2,3,3) 张量不会发生变化
            classification = classification.view(-1, num_classes)
            regression = regression.view(-1, 5)
            bbox_annotation = bbox_annotations[i]
            bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]
            #把coco的x1y1wh格式转换为xcycwh的格式
            bbox_annotation[:, 0] = bbox_annotation[:, 0] + bbox_annotation[:, 2] / 2
            bbox_annotation[:, 1] = bbox_annotation[:, 1] + bbox_annotation[:, 3] / 2

            if bbox_annotation.shape[0] == 0:
                if torch.cuda.is_available():
                    regression_losses.append(torch.tensor(0).to(dtype).cuda())
                    classification_losses.append(torch.tensor(0).to(dtype).cuda())
                else:
                    regression_losses.append(torch.tensor(0).to(dtype))
                    classification_losses.append(torch.tensor(0).to(dtype))
                continue
            # yolov3损失只计算正样本的损失项 怎么找到正样本呢
            # 1 找到与每个网格的3个anchor框iou最大也就是最匹配的gt框 即为每个网格的每个anchor框分配一个gt框
            # 2 anchor框与gt框的iou>=iou_thresh的 认为该网格的该anchor框框中了物体 即有正样本
            # (num_anchors*grid_size*grid_size)
            ious = IOU(anchors, bbox_annotation, formatting='xcycwh')
            # ious_max是每个anchor框对应的一堆gt框中iou最大的 ious_argmax是每个anchor框对应的最匹配的gt框的id
            ious_max, ious_argmax = ious.max(dim=1)

            # 计算分类损失
            # 找到每个anchor对应的gt框的id后 得到每个anchor框对应的gt框预测向量
            gt_classification = torch.Tensor(classification.shape).fill_(0)
            # iou大于阈值的认为有正样本
            positive_indices = ious_max.ge(iou_thresh)
            num_positive_anchors = positive_indices.sum()
            # 匹配到正样本的anchor框
            # 下面是一种神奇的广播用法
            # 假设ious_argmax[0]=2代表第0个anchor框匹配到了第2个gt框
            # assigned_annotations=bbox_annotation[ious_argmax[0],:]=bbox_annotation[2,:]就是第2个gt框的预测向量
            # (num_anchors*grid_size*grid_size,5)
            assigned_annotations = bbox_annotation[ious_argmax, :]
            # one-hot码
            gt_classification[~positive_indices, :] = 0
            gt_classification[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1
            # 只计算正样本的损失项
            bce = -(gt_classification[positive_indices, :] * torch.log(classification[positive_indices, :]) +
                    (1. - gt_classification[positive_indices, :]) * torch.log(1. - classification[positive_indices, :]))
            cls_loss = bce.sum() / torch.clamp(num_positive_anchors, min=1)
            if torch.cuda.is_available():
                classification_losses.append(cls_loss.to(dtype).cuda())
            else:
                classification_losses.append(cls_loss.to(dtype))

            # 计算定位损失
            if positive_indices.sum() <= 0:
                if torch.cuda.is_available():
                    regression_losses.append(torch.tensor(0).to(dtype).cuda())
                else:
                    regression_losses.append(torch.tensor(0).to(dtype))
            else:
                gt_ctr_x = assigned_annotations[positive_indices, 0]
                gt_ctr_y = assigned_annotations[positive_indices, 1]
                gt_w = torch.clamp(assigned_annotations[positive_indices, 2], min=1)
                gt_h = torch.clamp(assigned_annotations[positive_indices, 3], min=1)
                #gt框没有置信度 正样本的置信度设为1
                gt_conf = torch.Tensor(gt_ctr_x.shape).fill_(1.)

                tx = regression[positive_indices, 0]
                ty = regression[positive_indices, 1]
                tw = regression[positive_indices, 2]
                th = regression[positive_indices, 3]
                pred_conf = regression[positive_indices, 4]
                #https://www.jianshu.com/p/86b8208f634f
                sigmoid_tx = torch.sigmoid(tx)
                sigmoid_ty = torch.sigmoid(ty)

                anchor_ctr_x = anchors[positive_indices, 0]
                anchor_ctr_y = anchors[positive_indices, 1]
                anchor_w = anchors[positive_indices, 2]
                anchor_h = anchors[positive_indices, 3]

                #gt框的偏移量
                sigmoid_tx_gt = torch.sigmoid(gt_ctr_x - anchor_ctr_x)
                sigmoid_ty_gt = torch.sigmoid(gt_ctr_y - anchor_ctr_y)
                tw_gt = torch.log(gt_w / anchor_w)
                th_gt = torch.log(gt_h / anchor_h)
                #用下面的代码可以验证从(num_anchors, grid_size, grid_size,4/5)展
                #开到(num_anchors*grid_size*grid_size,4/5)的过程中元素还是正确的对应着
                # regression = regression.view(num_anchors, grid_size, grid_size, 5)
                # assigned_annotations = assigned_annotations.view(num_anchors, grid_size, grid_size, 5)
                # anchors = anchors.view(num_anchors, grid_size, grid_size, 4)
                # positive_indices = positive_indices.view(num_anchors, grid_size, grid_size)
                mse_criterion = torch.nn.MSELoss()
                reg_x_loss = mse_criterion(sigmoid_tx, sigmoid_tx_gt)
                reg_y_loss = mse_criterion(sigmoid_ty, sigmoid_ty_gt)
                reg_w_loss = mse_criterion(tw, tw_gt)
                reg_h_loss = mse_criterion(th, th_gt)
                reg_conf_loss = mse_criterion(pred_conf, gt_conf)
                reg_loss = reg_x_loss + reg_y_loss + reg_w_loss + reg_h_loss + reg_conf_loss
                reg_loss = (reg_loss.sum()) / torch.clamp(num_positive_anchors, min=1)

                if torch.cuda.is_available():
                    regression_losses.append(reg_loss.to(dtype).cuda())
                else:
                    regression_losses.append(reg_loss.to(dtype))

        return torch.stack(classification_losses).mean(dim=0,
                                                       keepdim=True), torch.stack(regression_losses).mean(dim=0,
                                                                                                          keepdim=True)
