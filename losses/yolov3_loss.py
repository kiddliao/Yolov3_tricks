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
                before_scale_anchors,
                num_classes,
                num_anchors,
                grid_size,
                stride,
<<<<<<< Updated upstream
<<<<<<< Updated upstream
                iou_thresh=0.5):
        '''
        predictions是预测结果 predictions.shape=(n,num_anchors,grid_size,grid_size,num_classes+5)
        predictions[1,2,i,j,:4]是第2张图片的坐标为(i,j)的网格预测的第3个框向量
        anchors[2,i,j,:]是坐标为(i,j)的网格的第3个预设框向量
        predictions放的是每张图片的每个网格的3个尺度所预测的处理后的结果(tx,ty,tw,th)
        '''
        batch_size = predictions.shape[0]
        classification_losses = []
        regression_losses = []
=======
                scales,
                iou_thresh=0.5):

        batch_size = predictions.shape[0]
        classification_losses = []
=======
                scales,
                iou_thresh=0.5):

        batch_size = predictions.shape[0]
        classification_losses = []
>>>>>>> Stashed changes
        # regression_losses = []
        regression_losses_x = []
        regression_losses_y = []
        regression_losses_w = []
        regression_losses_h = []
<<<<<<< Updated upstream
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes
        confidence_losses = []
        classifictions, regressions = predictions[..., 5:], predictions[..., :4]
        confidences = predictions[..., 4:5]
        #损失函数直接返回原张量shape的形式 不需要mean()
        mse_criterion = torch.nn.MSELoss(reduction='none')
        bce_criterion = torch.nn.BCELoss(reduction='none')
<<<<<<< Updated upstream
<<<<<<< Updated upstream
        ce_criterion = torch.nn.CrossEntropyLoss(reduction='none')

        anchors = anchors.view(-1, 4)

        FloatTensor = torch.cuda.FloatTensor if predictions.is_cuda else torch.FloatTensor
        device = predictions.device
=======

        before_scale_anchors = before_scale_anchors.view(-1, 4)

        FloatTensor = torch.cuda.FloatTensor if predictions.is_cuda else torch.FloatTensor
        device = predictions.device
=======

        before_scale_anchors = before_scale_anchors.view(-1, 4)

        FloatTensor = torch.cuda.FloatTensor if predictions.is_cuda else torch.FloatTensor
        device = predictions.device
>>>>>>> Stashed changes

        #把coco的x1y1x2y2格式转换为xcycwh的格式
        bbox_annotations[..., 2] = bbox_annotations[..., 2] - bbox_annotations[..., 0]
        bbox_annotations[..., 3] = bbox_annotations[..., 3] - bbox_annotations[..., 1]
        bbox_annotations[..., 0] = bbox_annotations[..., 0] + bbox_annotations[..., 2] / 2
        bbox_annotations[..., 1] = bbox_annotations[..., 1] + bbox_annotations[..., 3] / 2
<<<<<<< Updated upstream
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes

        for i in range(batch_size):
            #anchor的放缩
            anchors = before_scale_anchors * scales[i]
            classification, regression = classifictions[i, :, :], regressions[i, :, :]
            confidence = confidences[i, :, :]
            # 为了计算iou方便
            # 把(num_anchors,grid_size,grid_size,num_classes)的classification向量view成(num_anchors*grid_size*grid_size,num_classes)
            # 下面的regression和anchor也同样view一下 view不会改变相对的位置信息
            # 将一个shape为(2,3,3)的张量view成(18,1)的张量再view回(2,3,3) 张量不会发生变化
            classification = classification.view(-1, num_classes)
            regression = regression.view(-1, 4)
            confidence = confidence.view(-1, 1)
            bbox_annotation = bbox_annotations[i]
            bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]

            if bbox_annotation.shape[0] == 0:
<<<<<<< Updated upstream
<<<<<<< Updated upstream
                regression_losses.append(torch.tensor(0.).to(device))
=======
=======
>>>>>>> Stashed changes
                # regression_losses.append(torch.tensor(0.).to(device))
                regression_losses_x.append(torch.tensor(0.).to(device))
                regression_losses_y.append(torch.tensor(0.).to(device))
                regression_losses_w.append(torch.tensor(0.).to(device))
                regression_losses_h.append(torch.tensor(0.).to(device))
<<<<<<< Updated upstream
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes
                classification_losses.append(torch.tensor(0.).to(device))
                confidence_losses.append(torch.tensor(0.).to(device))
                continue
            # yolov3损失分为置信度损失 边界框损失和分类损失
            # 分类损失和边界框损失只计算正样本损失项 置信度损失计算所有框的损失
            # 1 找到与每个网格的3个anchor框iou最大也就是最匹配的gt框 即为每个网格的每个anchor框分配一个gt框
            # 2 anchor框与gt框的iou>=iou_thresh的 认为该网格的该anchor框框中了物体 即有正样本
            # (num_anchors*grid_size*grid_size)
            ious = IOU(anchors, bbox_annotation, formatting='xcycwh')

            # ious_max是每个anchor框对应的一堆gt框中iou最大的 ious_argmax是每个anchor框对应的最匹配的gt框的id
            ious_max, ious_argmax = ious.max(dim=1)

            # 置信度损失 置信度损失作者在论文使用交叉熵在代码使用mse
            # https://zhuanlan.zhihu.com/p/142408168
            # 找到每个anchor对应的gt框的id后 得到每个anchor框对应的gt框预测向量

            gt_confidence = FloatTensor(confidence.shape).fill_(0)
            # iou大于阈值的认为有正样本
            positive_indices = ious_max.ge(iou_thresh)
            num_positive_anchors = positive_indices.sum()
            # 匹配到正样本的anchor框
            # 下面是一种神奇的广播用法
            # 假设ious_argmax[0]=2代表第0个anchor框匹配到了第2个gt框
            # assigned_annotations=bbox_annotation[ious_argmax[0],:]=bbox_annotation[2,:]就是第2个gt框的预测向量
            # (num_anchors*grid_size*grid_size,5)
            assigned_annotations = bbox_annotation[ious_argmax, :]

            gt_confidence[positive_indices, 0] = 1
            #降低负样本对置信度损失的影响
            # noobj = 0.5  原文0.5
<<<<<<< Updated upstream
<<<<<<< Updated upstream
            noobj = 0.5
=======
            noobj = 0.1
>>>>>>> Stashed changes
=======
            noobj = 0.1
>>>>>>> Stashed changes
            #平衡损失函数 回归的损失项比较小
            coord = 5.
            if positive_indices.sum() <= 0:
                mse_conf_obj = torch.tensor(0.).to(device)
            else:
                mse_conf_obj = mse_criterion(confidence[positive_indices, :], gt_confidence[positive_indices, :])
            mse_conf_noobj = mse_criterion(confidence[~positive_indices, :], gt_confidence[~positive_indices, :])
            conf_loss = mse_conf_obj.sum() + mse_conf_noobj.sum() * noobj
            confidence_losses.append(conf_loss)

            # if positive_indices.sum() > 1:
            #     print()
<<<<<<< Updated upstream
<<<<<<< Updated upstream
            # 计算分类损失和定位损失
            if positive_indices.sum() <= 0:
                regression_losses.append(torch.tensor(0.).to(device))
=======
            # 计算定位损失和分类损失
            if positive_indices.sum() <= 0:
=======
            # 计算定位损失和分类损失
            if positive_indices.sum() <= 0:
>>>>>>> Stashed changes
                # regression_losses.append(torch.tensor(0.).to(device))
                regression_losses_x.append(torch.tensor(0.).to(device))
                regression_losses_y.append(torch.tensor(0.).to(device))
                regression_losses_w.append(torch.tensor(0.).to(device))
                regression_losses_h.append(torch.tensor(0.).to(device))
<<<<<<< Updated upstream
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes
                classification_losses.append(torch.tensor(0.).to(device))
            else:
                # yolov3使用多个二分类 bce损失
                # 正样本的gt分类设置为1 正样本的gt其他分类设置0 负样本所有分类设置为0
                # 分类预测向量为(num_anchors*grid_size*grid*size,num_classes)
                # 如果使用ce loss 就相当于有num_anchors*grid_size*grid*size个多分类器
                # 如果使用bce loss 就相当于有num_anchors*grid_size*grid*size*num_classes个二分类器
                # 只将有正样本的网格的num_classes个二分类器中真实分类设置为1 每个正样本的预测分类相当于有num_classes-1个二分类器是0
                # 负样本的num_classes个二分类器对应的gt值全部设置为0
                gt_classification = FloatTensor(classification.shape).fill_(0)  #设置的gt框的向量不需要梯度
                gt_classification[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1
                # 对有正样本的二分类器计算bce
                # 手动实现 之后focal loss好改
                # bce_cls = -(gt_classification * torch.log(classification) +
                #             (1. - gt_classification) * torch.log(1. - classification))
<<<<<<< Updated upstream
<<<<<<< Updated upstream
                #接口实现 预测框在前 gt框在后 gt框不许有梯度
                bce_cls = bce_criterion(classification[positive_indices, :], gt_classification[positive_indices, :])
                cls_loss = bce_cls.sum()
                classification_losses.append(cls_loss)
=======
=======
>>>>>>> Stashed changes
                #但是像下面这么写会显存炸掉 如果positive_indices全是false torch.log(classification[positive_indices, :])和各种计算函数的结果会是nan导致炸显存
                # bce_cls = -(gt_classification[positive_indices, :] * torch.log(classification[positive_indices, :]) +
                #             (1. - gt_classification[positive_indices, :]) * torch.log(1. - classification[positive_indices, :]))
                #接口实现 预测框在前 gt框在后 gt框不许有梯度
                bce_cls = bce_criterion(classification[positive_indices, :], gt_classification[positive_indices, :])
                cls_loss = bce_cls.sum()
                classification_losses.append(cls_loss / torch.clamp(num_positive_anchors.to(device), min=1.0))
<<<<<<< Updated upstream
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes

                gt_ctr_x = assigned_annotations[positive_indices, 0] / stride
                gt_ctr_y = assigned_annotations[positive_indices, 1] / stride
                #scaled_anchor_w*e^(tw)*stride预测gt_w
                #log(gt_w/(stride*scaled_anchor_w))对应tw 括号里的a_w是cfg里写的anchor大小
                #在此处stride*scaled_anchor_w=anchor_w
<<<<<<< Updated upstream
<<<<<<< Updated upstream
                gt_w = torch.clamp(assigned_annotations[positive_indices, 2], min=1)
                gt_h = torch.clamp(assigned_annotations[positive_indices, 3], min=1)
=======
                gt_w = torch.clamp(assigned_annotations[positive_indices, 2], min=1) / stride
                gt_h = torch.clamp(assigned_annotations[positive_indices, 3], min=1) / stride
>>>>>>> Stashed changes
=======
                gt_w = torch.clamp(assigned_annotations[positive_indices, 2], min=1) / stride
                gt_h = torch.clamp(assigned_annotations[positive_indices, 3], min=1) / stride
>>>>>>> Stashed changes

                #https://www.jianshu.com/p/86b8208f634f
                tx = regression[positive_indices, 0]
                ty = regression[positive_indices, 1]
                tw = regression[positive_indices, 2]
                th = regression[positive_indices, 3]
<<<<<<< Updated upstream
<<<<<<< Updated upstream
                #https://www.jianshu.com/p/86b8208f634f
=======
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes
                sigmoid_tx = torch.sigmoid(tx)
                sigmoid_ty = torch.sigmoid(ty)

                anchor_ctr_x = anchors[positive_indices, 0] / stride
                anchor_ctr_y = anchors[positive_indices, 1] / stride
<<<<<<< Updated upstream
<<<<<<< Updated upstream
                anchor_w = anchors[positive_indices, 2]
                anchor_h = anchors[positive_indices, 3]

                #gt框的偏移量
                sigmoid_tx_gt = torch.sigmoid(gt_ctr_x - anchor_ctr_x)
                sigmoid_ty_gt = torch.sigmoid(gt_ctr_y - anchor_ctr_y)
=======
                anchor_w = anchors[positive_indices, 2] / stride
                anchor_h = anchors[positive_indices, 3] / stride

                #gt框的偏移量
                sigmoid_tx_gt = gt_ctr_x - anchor_ctr_x
                sigmoid_ty_gt = gt_ctr_y - anchor_ctr_y
>>>>>>> Stashed changes
=======
                anchor_w = anchors[positive_indices, 2] / stride
                anchor_h = anchors[positive_indices, 3] / stride

                #gt框的偏移量
                sigmoid_tx_gt = gt_ctr_x - anchor_ctr_x
                sigmoid_ty_gt = gt_ctr_y - anchor_ctr_y
>>>>>>> Stashed changes
                tw_gt = torch.log(gt_w / anchor_w + 1e-16)
                th_gt = torch.log(gt_h / anchor_h + 1e-16)
                #用下面的代码可以验证从(num_anchors, grid_size, grid_size,4/5)展
                #开到(num_anchors*grid_size*grid_size,4/5)的过程中元素还是正确的对应着
                # regression = regression.view(num_anchors, grid_size, grid_size, 5)
                # assigned_annotations = assigned_annotations.view(num_anchors, grid_size, grid_size, 5)
                # anchors = anchors.view(num_anchors, grid_size, grid_size, 4)
                # positive_indices = positive_indices.view(num_anchors, grid_size, grid_size)
<<<<<<< Updated upstream
<<<<<<< Updated upstream

                #为了使得框的大小对损失的影响减小 v3在回归损失前加上了(2-tw_gt*th_gt) v1用的是对tw_gt和th_gt开根号
                param = 2. - tw_gt.abs() * th_gt.abs()
                reg_x_loss = param * mse_criterion(sigmoid_tx, sigmoid_tx_gt)
                reg_y_loss = param * mse_criterion(sigmoid_ty, sigmoid_ty_gt)
                reg_w_loss = param * mse_criterion(tw, tw_gt)
                reg_h_loss = param * mse_criterion(th, th_gt)
                reg_loss = (reg_x_loss + reg_y_loss + reg_w_loss + reg_h_loss).sum() * coord

                regression_losses.append(reg_loss)
=======
=======
>>>>>>> Stashed changes

                #为了使得框的大小对损失的影响减小 v3在回归损失前加上了(2-tw_gt*th_gt) v1用的是对tw_gt和th_gt开根号
                param = 2. - tw_gt.abs() * th_gt.abs()
                # param = 1.
                reg_x_loss = (mse_criterion(sigmoid_tx, sigmoid_tx_gt)).sum()
                reg_y_loss = (mse_criterion(sigmoid_ty, sigmoid_ty_gt)).sum()
                reg_w_loss = (param * mse_criterion(tw, tw_gt)).sum()
                reg_h_loss = (param * mse_criterion(th, th_gt)).sum()

                # reg_loss = (reg_x_loss + reg_y_loss + reg_w_loss + reg_h_loss).sum() * coord

                # regression_losses.append(reg_loss)
                regression_losses_x.append(reg_x_loss / torch.clamp(num_positive_anchors.to(device), min=1.0))
                regression_losses_y.append(reg_y_loss / torch.clamp(num_positive_anchors.to(device), min=1.0))
                regression_losses_w.append(reg_w_loss / torch.clamp(num_positive_anchors.to(device), min=1.0))
                regression_losses_h.append(reg_h_loss / torch.clamp(num_positive_anchors.to(device), min=1.0))
<<<<<<< Updated upstream
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes
        #debug了两天 这里的梯度一直有问题 regression_losses这里 当正样本为0时是直接添加的tensor(0) 这里出了问题
        #debug的时候就一个个的试 比如y=regression_losses y.backward() 报错就说明是regression_losses的问题
        #然后顺着regression_losses往回试找到问题所在
        #判断有没有问题的依据是1不报错2对这个张量backward()后 网络的叶子节点的梯度被计算出来 比如model.module_list[0][0].weight.grad
        # y=torch.stack(regression_losses).mean(dim=0, keepdim=True).sum()
        # y.backward()
<<<<<<< Updated upstream
<<<<<<< Updated upstream
        return torch.stack(classification_losses).mean(dim=0, keepdim=True),\
               torch.stack(regression_losses).mean(dim=0, keepdim=True),\
               torch.stack(confidence_losses).mean(dim=0, keepdim=True)
=======
=======
>>>>>>> Stashed changes
        # return torch.stack(classification_losses).mean(dim=0, keepdim=True),\
        #        torch.stack(regression_losses).mean(dim=0, keepdim=True),\
        #        torch.stack(confidence_losses).mean(dim=0, keepdim=True)
        return torch.stack(classification_losses).mean(dim=0, keepdim=True),\
               torch.stack(regression_losses_x).mean(dim=0, keepdim=True),\
               torch.stack(regression_losses_y).mean(dim=0, keepdim=True),\
               torch.stack(regression_losses_w).mean(dim=0, keepdim=True),\
               torch.stack(regression_losses_h).mean(dim=0, keepdim=True),\
<<<<<<< Updated upstream
               torch.stack(confidence_losses).mean(dim=0, keepdim=True)
>>>>>>> Stashed changes
=======
               torch.stack(confidence_losses).mean(dim=0, keepdim=True)
>>>>>>> Stashed changes
