import torch
import torch.nn as nn

# init_weights(model) #为方便 统一为下面的写法
# def init_weights(model):
#     modules = model.module_list
#     for name, m in modules.named_modules():
#         if isinstance(m, nn.Conv2d):
#             torch.nn.init.kaiming_uniform_(m.weight.data, 0.01, nonlinearity='leaky_relu')  #0.01参数为leaky_relu激活函数负
#             #只有yolo前一层的卷积层有bias
#             if isinstance(m.bias, nn.parameter.Parameter):
#                 torch.nn.init.constant_(m.bias.data, 0.0)
#         elif isinstance(m, nn.BatchNorm2d):
#             torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
#             torch.nn.init.constant_(m.bias.data, 0.0)


#yolov3模型用kaiming初始化的话 输出会爆炸 亲测
def weights_init_kaiming_uniform(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight.data, 0.01, nonlinearity='leaky_relu')  #0.01参数为leaky_relu激活函数负半轴的斜率
        #只有yolo前一层的卷积层有bias
        if isinstance(m.bias, nn.parameter.Parameter):
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:  #默认不管bias 置0
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)