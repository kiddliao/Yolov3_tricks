import torch
import torch.nn as nn
import cv2
import numpy as np
from utils.utils import *


class YOLOV3Loss(nn.Module):
    def __init__(self):
        super(YOLOV3Loss, self).__init__()
    
    def forward(self, predictions, targets, input_dim, anchors, num_classes, iou_thresh=0.5):
        '''
        predictions是预测结果 predictions.shape=anchors.shape=(n,num_anchors,grid_size,grid_size,num_classes+5) 
        predictions[1,2,i,j,:4]是第2张图片的坐标为(i,j)的网格预测的第3个框向量
        anchors[1,2,i,j,:4]是第2张图片的坐标为(i,j)的网格的第3个预设框向量
        predictions放的是每张图片的每个网格的3个尺度所预测的结果
        '''
        batch_size = predictions.shape[0]
        cls_loss = []
        reg_loss = []
        classifictions, regressions = predictions[...,:4], predictions[...,4:]

        for i in range(batch_size):
            classifiction, regression = classifictions[i,:,:], regressions[i,:,:]
            bbox_annotation = targets[i]
            #看看那些网格里有目标 预设框和gt框的iou大于0.5就代表这个网格有东西
            
            
            
