import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from utils.utils import *
from datasets.datasets import *
from losses.yolov3_loss import *
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()


class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors
        self.criterion = YOLOV3Loss()
        self.grid_size = 0  #网格大小先初始化为0
        self.num_anchors = len(self.anchors)

    def compute_grid_offsets(self, grid_size, cuda=False):
        self.grid_size = grid_size
        g = self.grid_size
        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.stride = self.input_dim / self.grid_size  #对比输入图片 特征图的一个像素对应原图的几个像素
        self.grid_x = torch.arange(g).repeat(g, 1).t().view([1, 1, g, g]).type(FloatTensor)  #网格的序号 (1,1,13,13)
        self.grid_y = torch.arange(g).repeat(g, 1).view([1, 1, g, g]).type(FloatTensor)  #(1,1,13,13)
        self.scaled_anchors = FloatTensor([(a_w / self.stride, a_h / self.stride)
                                           for a_w, a_h in self.anchors])  #anchor框对应原图的像素/特征图对应原图几个像素=anchor框对应特征图几个像素
        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))  #(1,3,1,1)
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))  #(1,3,1,1)

    def forward(self, x, targets=None, input_dim=416, anchors=[], num_classes=80):
        # gpu
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor

        self.input_dim = input_dim
        self.num_classes = num_classes
        num_samples = x.size(0)  #图片数
        grid_size = x.size(2)  #网格大小就是特征图的宽高 每个像素都是一个网格

        prediction = x.view(num_samples, self.num_anchors, self.num_classes + 5, grid_size,
                            grid_size).permute(0, 1, 3, 4, 2).contiguous()  #如果不是连续内存就划分新内存存储
        #(n,num_anchors,grid_size,grid_size,num_classes+5)
        #将n张图的每张图划分为grid_size*grid_size个网格 每个网格预测num_anchors次 预测得到的向量是num_classes+5

        #获得输出
        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])
        w = torch.exp(prediction[..., 2])
        h = torch.exp(prediction[..., 3])
        pred_conf = torch.sigmoid(prediction[..., 4])
        pred_cls = torch.sigmoid(prediction[..., 5:])

        if grid_size != self.grid_size:
            self.compute_grid_offsets(grid_size, cuda=x.is_cuda)

        pred_boxes = FloatTensor(prediction[..., :4].shape)
        #x.shape=(n,num_anchors,grid_size,grid_size) self.grid_x.shape=(1,1,grid_size,grid_size) 这里的加法有广播机制
        #x预测的是相对于网格左上角的偏移 现在加上每个网格左上角的绝对坐标 就得到x的绝对坐标(这里的坐标还是特征图的坐标不是映射到原图的坐标)
        #pred_boxes.shape=(n,num_anchors,grid_size,grid_size,4) pred_boxes[0,1,i,j,:]是第一张图片的坐标为(i,j)的网格预测的第2个框向量
        pred_boxes[..., 0] = x.data + self.grid_x
        pred_boxes[..., 1] = y.data + self.grid_y
        pred_boxes[..., 2] = w.data * self.anchor_w
        pred_boxes[..., 3] = h.data * self.anchor_h

        no_loss_prediction = torch.cat(
            (
                pred_boxes.view(num_samples, -1, 4)*self.stride,
                #从特征图的坐标乘上stride得到原图的坐标 (n,3,grid_size,grid_size,4) to (n,3*grid_size*grid_size,4)
                pred_conf.view(num_samples, -1, 1),
                pred_cls.view(num_samples, -1, self.num_classes)),
            2)  #输出(n,3*grid_size*grid_size,(4+1+80)*3=255)

        if type(targets) != torch.Tensor:  # 没传标签 就是训练
            return no_loss_prediction, 0, 0
        else:
            final_predictions = torch.cat(  #保留位置信息计算损失 计算损失需要未转换的框向量
                (prediction[..., :4].view(num_samples, self.num_anchors, grid_size, grid_size,
                                          4), pred_conf.view(num_samples, self.num_anchors, grid_size, grid_size, 1),
                 pred_cls.view(num_samples, self.num_anchors, grid_size, grid_size, self.num_classes)),
                -1).requires_grad_()
            #生成预设框
            #anchor_boxes.shape=(n,num_anchors,grid_size,grid_size,4) anchor_boxes[0,1,i,j,:]是第一张图片的坐标为(i,j)的网格的第2个预设框向量
            anchor_boxes = FloatTensor(prediction[..., :4].shape[1:]).requires_grad_()
            #anchor_boxes是xcycwh的格式
            anchor_boxes[..., 0] = FloatTensor(prediction[..., 0].shape[1:]).fill_(0.5) + self.grid_x
            anchor_boxes[..., 1] = FloatTensor(prediction[..., 1].shape[1:]).fill_(0.5) + self.grid_y
            #(3,1) 2 (n,3,grid_size,grid_size)
            anchor_boxes[..., 2] = self.scaled_anchors[:, 0:1].view(self.num_anchors, 1,
                                                                    1).repeat(1, grid_size, grid_size)
            anchor_boxes[..., 3] = self.scaled_anchors[:, 1:2].view(self.num_anchors, 1,
                                                                    1).repeat(1, grid_size, grid_size)
            final_anchors = anchor_boxes * self.stride
            cls_loss, reg_loss, conf_loss = self.criterion(final_predictions, targets, input_dim, final_anchors,
                                                           self.num_classes, self.num_anchors, grid_size, self.stride)
            return no_loss_prediction, cls_loss, reg_loss, conf_loss


class Darknet(nn.Module):
    def __init__(self, cfg_path):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfg_path)
        self.hyperparameters, self.module_list = self.create_modules(self.blocks)
        self.header_info = np.array([0, 2, 5, 1823744, 0])
        self.seen = self.header_info[3]

    def create_modules(self, blocks):
        hyperparameters = blocks.pop(0)
        module_list = nn.ModuleList()
        prev_filters = 3  #输入通道数默认是rgb的3通道
        output_filters = []  #存放每层的输出通道数

        for i, x in enumerate(blocks):
            module = nn.Sequential()
            if x['type'] == 'convolutional':
                if 'batch_normalize' in x:
                    batch_normalize = int(x['batch_normalize'])
                    bias = False
                else:
                    batch_normalize = 0
                    bias = True
                kernel_size = int(x['size'])
                stride = int(x['stride'])
                padding = int(x['pad'])
                filters = int(x['filters'])
                activation = x['activation']
                #yolov3中是'SAME'卷积 卷积前后特征图大小不变
                pad = (kernel_size - 1) // 2 if padding != 0 else 0
                #卷积块包括卷积层,BN层和激活函数
                conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad,
                                 bias=bias)  #(输入通道数,输出通道数,卷积核大小,步长,pad,偏移)
                module.add_module('conv_{}'.format(i), conv)

                if batch_normalize:
                    bn = nn.BatchNorm2d(filters)
                    module.add_module('batch_norm_{}'.format(i), bn)

                if activation == 'leaky':
                    activn = nn.LeakyReLU(0.1, inplace=True)  #利用in-place计算可以节省内(显)存,同时还可以省去反复申请和释放内存的时间
                    module.add_module("leaky_ReLU_{}".format(i), activn)

            elif x['type'] == 'upsample':
                stride = int(x['stride'])
                upsample = nn.Upsample(scale_factor=stride, mode='nearest')  #yolov3默认是放大2倍 最近插值法
                module.add_module('upsample_{}'.format(i), upsample)

            elif x['type'] == 'shortcut':
                #跳远连接很简单 直接在前向传播中进行实现 这里拿EmptyLayer占个位置
                shortcut = EmptyLayer()
                module.add_module('shortcut_{}'.format(i), shortcut)

            elif x['type'] == 'route':
                #yolov3的route层就是concat操作 把上采样的深层特征图和浅层特征图连接起来
                #当route只有一个值-4 意思是回退4层到深层的待融合特征图
                #当route有两个值-1,61 意思是将回退1层到已经upsample过的深层特征图和第61层浅层特征图进行融合
                if len(x['layers'].split(',')) == 1:
                    start, end = i + int(x['layers']), 0
                else:
                    layers = x['layers'].split(',')
                    start, end = i + int(layers[0]), int(layers[1])
                route = EmptyLayer()
                module.add_module("route_{}".format(i), route)
                if end > 0:
                    filters = output_filters[start] + output_filters[end]
                else:
                    filters = output_filters[start]

            elif x['type'] == 'yolo':
                mask = list(map(int, x['mask'].split(',')))
                anchors = list(map(int, x['anchors'].split(',')))
                anchors = [[anchors[i], anchors[i + 1]] for i in range(0, len(anchors), 2)]
                anchors = [anchors[i] for i in mask]  #len(anchors)=3

                detection = DetectionLayer(anchors)
                module.add_module('Detection_{}'.format(i), detection)

            module_list.append(module)
            prev_filters = filters  #只有卷积层和route的连接操作会改变特征图的通道数
            output_filters.append(filters)

        return (hyperparameters, module_list)

    def forward(self, x, targets=None):  #targets是标签 代表需不需要计算误差
        modules = self.blocks
        outputs = {}  #存储每层输出的特征图
        yolo_outputs = []
        cls_losses = []
        reg_losses = []
        conf_losses = []
        for i, module in enumerate(modules):
            module_type = module['type']
            if module_type == 'convolutional' or module_type == 'upsample':  #这两种层torch有实现
                x = self.module_list[i](x)

            elif module_type == 'route':
                layers = list(map(int, module['layers'].split(',')))
                if len(layers) == 1:
                    x = outputs[i + layers[0]]
                else:
                    start, end = i + layers[0], layers[1]
                    map1, map2 = outputs[start], outputs[end]
                    x = torch.cat((map1, map2), dim=1)

            elif module_type == 'shortcut':
                start, end = i + int(module['from']), i - 1
                x = outputs[start] + outputs[end]

            elif module_type == 'yolo':
                anchors = self.module_list[i][0].anchors
                input_dim = int(self.hyperparameters['height'])
                num_classes = int(module['classes'])
                x, cls_loss, reg_loss,conf_loss = self.module_list[i][0](x, targets, input_dim, anchors, num_classes)
                total_loss = cls_loss + reg_loss + conf_loss
                cls_losses.append(cls_loss)
                reg_losses.append(reg_loss)
                conf_losses.append(conf_loss)
                yolo_outputs.append(x)
            outputs[i] = x
        yolo_outputs = torch.cat(yolo_outputs, 1)
        return yolo_outputs if targets is None else(yolo_outputs,\
                                                    torch.stack(cls_losses).sum(dim=0,keepdim=True),\
                                                    torch.stack(reg_losses).sum(dim=0,keepdim=True),\
                                                    torch.stack(conf_losses).sum(dim=0,keepdim=True))

    def load_darknet_weights(self, weights_path):
        """用预训练模型初始化网络参数'"""
        with open(weights_path, "rb") as f:
            header = np.fromfile(f, dtype=np.int32, count=5)  #前五个参数不是网络参数
            self.header_info = header
            self.seen = header[3]
            weights = np.fromfile(f, dtype=np.float32)  #除了前五个参数,其他的是网络权重
        cutoff = None
        if ".weights" in weights_path:
            cutoff = 75
        ptr = 0
        for i, (block, module) in enumerate(zip(self.blocks, self.module_list)):
            if i == cutoff:
                break
            if block["type"] == "convolutional":
                conv_layer = module[0]
                if "batch_normalize" in block:  #bn层没有bias
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel()  #这层mean和std的数量
                    #获取bn的mean
                    bn_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.bias)
                    #初始化bn的mean
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    #获取bn的std
                    bn_w = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.weight)
                    #初始化bn的std
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    #获取bn的累计mean
                    bn_rm = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_mean)
                    #初始化bn的累计mean
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    #获取bn的累计std
                    bn_rv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_var)
                    #初始化bn的累计std
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    #没有bn的卷积层是yolo层的前一个卷积层
                    num_b = conv_layer.bias.numel()
                    #获取卷积层的bias
                    conv_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(conv_layer.bias)
                    #初始化卷积层的bias
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                #获得卷积层的卷积权重的数量
                num_w = conv_layer.weight.numel()
                #获取卷积层的卷积权重
                conv_w = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(conv_layer.weight)
                #初始化卷积层的卷积权重
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w

    def save_darknet_weights(self, path, cutoff=-1):
        """
        保存训练好的权重
            @:param path    - path of the new weights file
            @:param cutoff  - save layers between 0 and cutoff (cutoff = -1 -> all are saved)
        """
        fp = open(path, "wb")
        self.header_info[3] = self.seen
        self.header_info.tofile(fp)

        #迭代保存每层的权重
        for i, (block, module) in enumerate(zip(self.blocks[:cutoff], self.module_list[:cutoff])):
            if block["type"] == "convolutional":
                conv_layer = module[0]
                #如果有bn就把bn的参数保存在前面
                if "batch_normalize" in block:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(fp)
                    bn_layer.weight.data.cpu().numpy().tofile(fp)
                    bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                    bn_layer.running_var.data.cpu().numpy().tofile(fp)
                #保存bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(fp)
                #保存卷积权重
                conv_layer.weight.data.cpu().numpy().tofile(fp)
        fp.close()


if __name__ == '__main__':
    a = Darknet('cfg/flir_yolov3.cfg')
    a.load_darknet_weights('weights/flir_yolov3_65_18.weights')
    # a.save_darknet_weights('weights\\test.weights')
    training_set = DIYDataset('../datasets/coco_flir/coco',
                              'train',
                              mean_std_path=None,
                              cal_mean_std=False,
                              transform=transforms.Compose([Normalizer(), Augmenter(),
                                                            Resizer(416)]))
    training_params = {'batch_size': 4, 'shuffle': True, 'drop_last': True, 'collate_fn': collater, 'num_workers': 0}
    training_generator = DataLoader(training_set, **training_params)
    for i, data in enumerate(training_generator):
        imgs = data['img']
        img_ids = data['img_id']
        targets = data['annot']
        a.forward(imgs, targets)
