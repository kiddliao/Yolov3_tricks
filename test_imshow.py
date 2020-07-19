import json
import os
import cv2

import argparse
import torch
import yaml
import numpy as np

from torchvision import transforms

from utils.utils import *
from darknet import Darknet
from datasets.datasets import *
from utils.weight_init import *

def get_args():
    parser = argparse.ArgumentParser('yolov3 detector test')
    parser.add_argument('-p', '--project', type=str, default='shape', help='config file in /project/*yml')
    parser.add_argument('--batch_size', type=int, default=16, help='the batch_size of dataloader')
    parser.add_argument('--load_weights',
                        type=str,
                        default=os.path.join('backup', 'shape_yolov3_final_49_2800.weights'),
                        help='pretrained models or recover training')
    parser.add_argument('--stat_path', type=str, default='cfg', help='mean val and std val of dataset')
    parser.add_argument('--conf_thresh', type=float, default=0.5)
    parser.add_argument('--iou_thresh', type=float, default=0.5)
    parser.add_argument('--float16', type=bool, default=False)
    args = parser.parse_args()
    return args


def test_imshow(model, img_path, stat_txt_path, opt, category, cuda=False):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    else:
        torch.manual_seed(42)
    img = cv2.imread(path)
    tensor_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor_img = tensor_img.astype(np.float32) / 255.

    img_id = [-1]
    img_ids = [[-1]]
    annot = np.array([[-1., -1., -1., -1., -1.]])
    sample = {'img': tensor_img, 'img_id': img_id, 'annot': annot}

    transform = transforms.Compose([Resizer(416)])
    # transform = transforms.Compose([Normalizer(mean_std_path=stat_txt_path), Resizer(416)])
    sample = transform(sample)
    tensor_imgs = sample['img'].unsqueeze(0).permute(0, 3, 1, 2)
    scale = sample['scale']
    if cuda:
        tensor_imgs = tensor_imgs.cuda()

    yolo_outputs = model(tensor_imgs)
    pred_batch_imgs = NMS(img_ids,
                          yolo_outputs,
                          conf_thresh=opt.conf_thresh,
                          iou_thresh=opt.iou_thresh,
                          style='OR',
                          type='IoU')
    for pred_batch_img in pred_batch_imgs:
        bbox = pred_batch_img['bbox']
        xc, yc, w, h = bbox
        x1, y1, x2, y2 = xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2
        x1, y1, x2, y2 = list(map(int, list(map(lambda x: x / scale, [x1, y1, x2, y2]))))
        cid = pred_batch_img['category_id']
        confidence = pred_batch_img['score']
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  #2是线的宽度
        cv2.putText(img, '{}, {:.3f}'.format(category[cid], confidence), (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (255, 255, 0), 2)
    # cv2.namedWindow('prediction', 0)
    # cv2.resizeWindow(img_name, 416, 416)
    # cv2.imshow('prediction', img)
    # cv2.waitKey(0)
    cv2.imwrite(os.path.join('images', 'prediction.jpg'), img)
    # cv2.destroyAllWindows()
    print('预测成功! 图片保存在 images/prediction.jpg')


if __name__ == '__main__':
    opt = get_args()
    params = parse_yml(f'projects/{opt.project}.yml')
    model = Darknet(cfg_path=params['cfg_path'])
    # model.load_darknet_weights(opt.load_weights)
    # print('读入模型成功!')

    model.apply(weights_init_normal)
    if params['num_gpus'] == 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        cuda = False
    else:
        cuda = True
        model = model.cuda()
    model.eval()
    model.requires_grad_(False)

    stat_txt_path = os.path.join(opt.stat_path, '{}_stat.txt'.format(opt.project))
    while True:
        # try:
        path = '225.jpg'
        # path = input()
        test_imshow(model, path, stat_txt_path, opt, params['obj_list'], cuda)
        # except Exception as e:
        #     print(e)
        break
