import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

def parse_cfg(cfg_path):
    with open(cfg_path, 'r') as f:
        lines = f.read().split('\n')
        lines = list(filter(lambda x: (len(x) > 0 and x[0] != '#'), lines))
        lines = list(map(lambda x: x.strip(), lines))
    blocks = []
    block = {}
    for i in range(len(lines)):
        line = lines[i]
        if line[0] == '[' and line[-1] == ']':
            if len(block)!=0:
                blocks.append(block.copy())
                block = {}
            block['type'] = line[1:-1]
        else:
            key, value = line.split('=')
            block[key.strip()] = value.strip()
        if i == len(lines) - 1:
            blocks.append(block.copy())
    return blocks

def IOU(box1, box2, x1y1x2y2=False, iou_type='IOU'):
    '''
    box1.shape=(1,4) box2.shape=(n,4) 转换为左上角和右小角的坐标 方便求iou
    '''
    if x1y1x2y2: 
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  #coco格式 x1y1wh
        b1_x1, b1_x2 = box1[0], box1[0] + box1[2]
        b1_y1, b1_y2 = box1[1], box1[1] + box1[3]
        b2_x1, b2_x2 = box2[0], box2[0] + box2[2]
        b2_y1, b2_y2 = box2[1], box2[1] + box2[3]
    #相交面积 torch.clamp(input,min,max)
    inter_area = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(min=0) * \
                 (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(min=0)
    
    #并集面积
    area1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    unino_area = area1 + area2 - inter_area + 1e-16
    iou = inter_area / unino_area
    if iou_type is 'IOU':
        return iou
    else:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if iou_type=='GIoU':  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + 1e-16  # convex area
            return iou - (c_area - union_area) / c_area  # GIoU 能包含两个框的最小的矩形面积
        if iou_type in ['CIoU','DIoU']:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            # convex diagonal squared
            c2 = cw ** 2 + ch ** 2 + 1e-16
            # centerpoint distance squared
            rho2 = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2)) ** 2 / 4 + ((b2_y1 + b2_y2) - (b1_y1 + b1_y2)) ** 2 / 4
            if iou_type=='DIoU':
                return iou - rho2 / c2  # DIoU
            elif iou_type=='CIoU':  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (1 - iou + v)
                return iou - (rho2 / c2 + v * alpha)  # CIoU

    

    






            
