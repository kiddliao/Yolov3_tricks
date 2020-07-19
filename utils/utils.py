import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
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
            if len(block) != 0:
                blocks.append(block.copy())
                block = {}
            block['type'] = line[1:-1]
        else:
            key, value = line.split('=')
            block[key.strip()] = value.strip()
        if i == len(lines) - 1:
            blocks.append(block.copy())
    return blocks


def IOU(box1, box2, formatting='xcycwh', iou_type='IoU'):
    '''
    box1.shape=(m,4) box2.shape=(n,4) 转换为左上角和右小角的坐标 方便求iou
    '''
    # 进行unsqueeze后box1.shape=(n,4,1) b1_x1.shape=(n,1) b1_x1.shape=(i)才能进行广播
    box1 = torch.unsqueeze(box1, dim=-1)
    if formatting == 'x1y1x2y2':
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    elif formatting == 'x1y1wh':
        b1_x1, b1_x2 = box1[:, 0], box1[:, 0] + box1[:, 2]
        b1_y1, b1_y2 = box1[:, 1], box1[:, 1] + box1[:, 3]
        b2_x1, b2_x2 = box2[:, 0], box2[:, 0] + box2[:, 2]
        b2_y1, b2_y2 = box2[:, 1], box2[:, 1] + box2[:, 3]
    elif formatting == 'xcycwh':
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        raise NameError
    # 相交面积 torch.clamp(input,min,max)
    inter_area = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(min=0) * \
                 (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(min=0)

    # 并集面积
    area1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    unino_area = area1 + area2 - inter_area + 1e-16
    iou = inter_area / unino_area
    if iou_type is 'IoU':
        return iou
    else:
        # convex (smallest enclosing box) width
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if iou_type == 'GIoU':  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + 1e-16  # convex area
            return iou - (c_area - union_area) / c_area  # GIoU 能包含两个框的最小的矩形面积
        # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
        if iou_type in ['CIoU', 'DIoU']:
            # convex diagonal squared
            c2 = cw**2 + ch**2 + 1e-16
            # centerpoint distance squared
            rho2 = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2)) ** 2 / 4 + \
                ((b2_y1 + b2_y2) - (b1_y1 + b1_y2)) ** 2 / 4
            if iou_type == 'DIoU':
                return iou - rho2 / c2  # DIoU
            elif iou_type == 'CIoU':  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * \
                    torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (1 - iou + v)
                return iou - (rho2 / c2 + v * alpha)  # CIoU


def parse_yml(project_path):
    with open(project_path, 'r') as f:
        params = yaml.safe_load(f.read())
    return params


def NMS(img_ids, predictions, conf_thresh=0.5, iou_thresh=0.5, style='OR', type='IoU'):
    yolo_results = []
    coco_pred_sample = {"image_id": -1, "category_id": -1, "score": -1, "bbox": [-1, -1, -1, -1]}
    batch_size = predictions.shape[0]
    #NMS是对每张图片预测的每个分类的张量进行后处理
    for i in range(batch_size):
        image_id = img_ids[i]
        #得到分类id
        prediction = predictions[i]
        bbox_pred = prediction[:, :4]
        conf_pred = prediction[:, 4:5]
        cls_pred = prediction[:, 5:]
        cls_max, cls_argmax = cls_pred.max(dim=1, keepdim=True)
        prediction = torch.cat((bbox_pred, conf_pred, cls_argmax.float()), dim=-1)
        #对单张图片预测的每个分类进行切割
        cls_num = cls_pred.shape[-1]
        coco_pred_sample['image_id'] = image_id
        for j in range(cls_num):
            category_id = j
            cls_indice = (prediction[:, -1] == float(category_id))
            coco_pred_sample['category_id'] = category_id
            nms_results = NMS_core(prediction[cls_indice, :5], conf_thresh=0.5, iou_thresh=0.5, style='OR', type='IoU')
            for k in range(len(nms_results)):
                num_result = nms_results[k]
                coco_pred_sample['score'] = num_result[-1]
                coco_pred_sample['bbox'] = num_result[:4]
                yolo_results.append(coco_pred_sample.copy())
    return yolo_results


def NMS_core(prediction, conf_thresh=0.5, iou_thresh=0.5, style='OR', type='IoU'):
    '''
    prediction是预测结果
    confidence是objectness分数阈值,confidence大于此值才进行计算
    nms_conf是NMS的IoU阈值
    '''
    #coco是左上角坐标和宽高
    if torch.sum(prediction[:, 4] < conf_thresh) == prediction.size()[0]:
        return []

    conf_mask = (prediction[:, 4] >= conf_thresh)
    prediction = prediction[conf_mask, :]  #删除置信度小于阈值的预测框
    #宽高有问题可能会导致merge进入死循环
    w_true_ind = prediction[:, 2] >= 1
    prediction = prediction[w_true_ind, :]
    h_true_ind = prediction[:, 3] >= 1
    prediction = prediction[h_true_ind, :]

    #置信度排序
    indmax = torch.argsort(-1 * prediction[:, 4])
    prediction = prediction[indmax, :]

    if prediction.size()[0] == 1:
        return prediction.tolist()
    det_max = []

    # if style == 'OR':  #hard nms,直接删除相邻的同类别目标,密集目标的输出不友好
    #     for idx, bbox in enumerate(prediction):
    #         if prediction[idx][4] != 0:
    #             ious = lh_IOU(box_corner[idx,:4].unsqueeze(0), box_corner[idx + 1 :,:4])

    #             ind_ious_bool = (ious > iou_thresh)
    #             interplote = torch.zeros(idx + 1).bool()
    #             ind_ious_bool = torch.cat((interplote, ind_ious_bool))
    #             prediction[ind_ious_bool,:] *= 0
    #             box_corner[ind_ious_bool,:] *= 0
    dc = prediction.clone()
    if style == 'OR':  #fast nms(传统的NMS可以利用矩阵简化从而降低时间,但不得不牺牲一些精度,实验结果显示虽然降低了0.1mAP,但时间比基于Cython实现的NMS快11.8ms)
        # METHOD1
        # ind = list(range(len(dc)))
        # while len(ind):
        # j = ind[0]
        # det_max.append(dc[j:j + 1,:4].squeeze().tolist())  # save highest conf detection
        # reject = (IOU(dc[ind], dc[j].unsqueeze_(0),iou_type=type) > iou_thresh).nonzero()
        # [ind.pop(i) for i in reversed(reject)]
        #METHOD2
        while dc.size()[0]:
            det_max.append(dc[:1].squeeze().tolist())
            if len(dc) == 1:
                break
            iou = IOU(dc[1:], dc[0].unsqueeze_(0), formatting='xcycwh', iou_type=type)
            indice = iou.le(iou_thresh).squeeze(-1)
            dc = dc[1:][indice]

    elif style == 'AND':  #and nms,在hard-nms的逻辑基础上,增加是否为单独框的限制,删除没有重叠框的框(减少误检)
        while dc.size()[0] > 1:
            iou = IOU(dc[1:], dc[0].unsqueeze_(0), formatting='xcycwh', iou_type=type)
            indice = iou.le(iou_thresh).squeeze(-1)
            if iou.max() > iou_thresh:
                det_max.append(dc[:1].squeeze().tolist())
            dc = dc[1:][indice]

    elif style == 'MERGE':  #merge nms,在hard-nms的基础上,增加保留框位置平滑策略(重叠框位置信息求解平均值),使框的位置更加精确
        while dc.size()[0]:
            if len(dc) == 1:
                det_max.append(dc.squeeze().tolist())
                break
            iou = IOU(dc, dc[0].unsqueeze_(0), formatting='xcycwh', iou_type=type)
            i = iou.gt(iou_thresh).squeeze(-1)
            weights = dc[i, 4:5]
            # if torch.isnan(((weights * dc[i,:4]).sum(0) / weights.sum())[0]):
            #     print(1)
            dc[0, :4] = (weights * dc[i, :4]).sum(0) / weights.sum()
            det_max.append(dc[:1].squeeze().tolist())
            dc = dc[i == 0]

    elif style == 'SOFT':  #soft nms,改变其相邻同类别目标置信度,后期通过置信度阈值进行过滤,适用于目标密集的场景
        sigma = 0.5
        while dc.size()[0]:
            if len(dc) == 1:
                det_max.append(dc.squeeze().tolist())
                break
            det_max.append(dc[:1].squeeze().tolist())
            iou = IOU(dc[1:], dc[0].unsqueeze_(0), formatting='xcycwh', iou_type=type)
            dc = dc[1:]
            dc[:, 4:5] *= torch.exp(-iou**2 / sigma)  #decay confidences
            dc = dc[(dc[:, 4] >
                     conf_thresh).squeeze(-1)]  # new line per https://github.com/ultralytics/yolov3/issues/362
    return det_max



