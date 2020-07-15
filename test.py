import json
import os

import argparse
import torch
import yaml
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.autonotebook import tqdm

from utils.utils import *
from darknet import Darknet
from datasets.datasets import *


def get_args():
    parser = argparse.ArgumentParser('yolov3 detector test')
    parser.add_argument('-p', '--project', type=str, default='flir', help='config file in /project/*yml')
    parser.add_argument('-n', '--num_workers', type=int, default=0, help='the num_workers of dataloader')
    parser.add_argument('--batch_size', type=int, default=2, help='the batch_size of dataloader')
    parser.add_argument('--data_path', type=str, default=os.path.join('..', '..', 'REMOTE', 'datasets', 'coco_flir'))
    parser.add_argument('--load_weights',
                        type=str,
                        default=os.path.join('weights', 'flir_yolov3_65_18.weights'),
                        help='pretrained models or recover training')
    parser.add_argument('--stat_path', type=str, default='cfg', help='mean val and std val of dataset')
    parser.add_argument('--conf_thresh', type=float, default=0.5)
    parser.add_argument('--iou_thresh', type=float, default=0.5)
    parser.add_argument('--float16', type=bool, default=False)
    parser.add_argument('--override', type=bool, default=True, help='override previous bbox results file if exists')
    args = parser.parse_args()
    return args


def test(opt, params):
    if opt.override is True and not os.path.exists('{}2017_bbox_results.json'.format(params['test_set'])):
        model = Darknet(cfg_path=params['cfg_path'])
        model.load_darknet_weights(opt.load_weights)
        print('读入模型成功!')
        model.requires_grad_(False)
        model.eval()
        if params['num_gpus'] > 0:
            model.cuda()
            if opt.float16 is True:
                model.half()
    else:
        return '{}2017_bbox_results.json'.format(params['test_set'])

    if params['num_gpus'] == 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
        else:
            torch.manual_seed(42)

    stat_txt_path = os.path.join(opt.stat_path, '{}_stat.txt'.format(opt.project))

    test_set = DIYDataset(
        path=os.path.join(opt.data_path, params['project_name']),
        set_name=params['test_set'],
        mean_std_path=stat_txt_path,  #计算训练集的均值和方差
        cal_mean_std=False,
        transform=transforms.Compose([Normalizer(mean_std_path=stat_txt_path),
                                      Augmenter(), Resizer(416)]))
    test_params = {
        'batch_size': opt.batch_size,
        'shuffle': False,
        'drop_last': False,
        'collate_fn': collater,
        'num_workers': opt.num_workers
    }
    test_generator = DataLoader(test_set, **test_params)
    progress_bar = tqdm(test_generator)
    yolo_results = []
    for iter, data in enumerate(progress_bar):
        imgs, img_ids = data['img'], data['img_id']
        if params['num_gpus'] > 0:
            imgs = imgs.cuda()
        yolo_outputs = model(imgs)
        pred_batch_imgs = NMS(img_ids,
                              yolo_outputs,
                              conf_thresh=opt.conf_thresh,
                              iou_thresh=opt.iou_thresh,
                              style='OR',
                              type='DIoU')
        yolo_results += pred_batch_imgs

    return '{}2017_bbox_results.json'.format(params['test_set'])


def _eval(coco_gt, image_ids, pred_json_path):  #image_ids是所有测试集有标注的图 pred_json_path是预测文件的路径
    coco_pred = coco_gt.loadRes(pred_json_path)
    print('所有类的BBox')
    coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    for cat in coco_pred.cats.values():
        print("{cat['name']}类的BBOX")
        coco_eval.params.catIds = [cat['id']]
        coco_eval.params.imgIds = image_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()


if __name__ == '__main__':
    opt = get_args()
    params = parse_yml(f'projects/{opt.project}.yml')

    pred_path = test(opt, params)

    gt_json = os.path.join(opt.data_path, params['project_name'], 'annotations',
                           'instances_{}2017.json'.format(params['test_set']))
    MAX_IMAGES = 10000
    coco_gt = COCO(gt_json)
    image_ids = coco_gt.getImgIds()[:MAX_IMAGES]

    _eval(coco_gt, image_ids, pred_path)
