import datetime
import os
import argparse
import traceback
import time
import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.autonotebook import tqdm
from tensorboardX import SummaryWriter

from utils.utils import *
from utils.lr_decay import *
from datasets.datasets import *
from darknet import *


def get_args():
    parser = argparse.ArgumentParser('yolov3 detector train')
    parser.add_argument('-p', '--project', type=str, default='flir', help='config file in /project/*yml')
    parser.add_argument('-n', '--num_workers', type=int, default=0, help='the num_workers of dataloader')
    parser.add_argument('--batch_size', type=int, default=6, help='the batch_size of dataloader')
    parser.add_argument('--head_only', type=bool, default=False, help='freeze all layers except classification layer')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--optim',
                        type=str,
                        default='adamw',
                        help='suggest using adamw until the very final stage then switch to sgd')
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--val_interval', type=int, default=1, help="'intervals of calculate val_datasets' mAP")
    parser.add_argument('--save_interval', type=int, default=1477, help='number of steps between saving')
    parser.add_argument('--data_path', type=str, default=os.path.join('..', 'datasets', 'coco_flir'))
    parser.add_argument('--log_path', type=str, default='logs', help='tensorboardX log')
    parser.add_argument('--load_weights', type=str, default='weights/flir_yolov3_65_18.weights', help='pretrained models or recover training')
    parser.add_argument('--train_from_scratch',
                        type=bool,
                        default=True,
                        help='recover from train or train from scratch')
    parser.add_argument('--saved_path', type=str, default='backup', help="'saved models'path")
    parser.add_argument('--stat_path', type=str, default='cfg', help='mean val and std val of dataset')

    args = parser.parse_args()
    return args


def train(opt):
    params = parse_yml(f'projects/{opt.project}.yml')

    if params['num_gpus'] == 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    else:
        torch.manual_seed(42)

    os.makedirs(opt.log_path, exist_ok=True)
    os.makedirs(opt.saved_path, exist_ok=True)

    stat_txt_path = os.path.join(opt.stat_path, '{}_stat.txt'.format(opt.project))

    train_params = {
        'batch_size': opt.batch_size,
        'shuffle': True,
        'drop_last': True,
        'collate_fn': collater,
        'num_workers': opt.num_workers
    }
    val_params = {
        'batch_size': opt.batch_size,
        'shuffle': False,
        'drop_last': True,
        'collate_fn': collater,
        'num_workers': opt.num_workers
    }

    # train_set = DIYDataset(
    #     path=os.path.join(opt.data_path, params['project_name']),
    #     set_name=params['train_set'],
    #     mean_std_path=stat_txt_path,  #计算训练集的均值和方差
    #     cal_mean_std=True,
    #     transform=transforms.Compose([Normalizer(mean_std_path=stat_txt_path),
    #                                   Augmenter(), Resizer(416)]))
    train_set = DIYDataset(
        path=os.path.join(opt.data_path, params['project_name']),
        set_name=params['train_set'],
        mean_std_path=stat_txt_path,  #计算训练集的均值和方差
        cal_mean_std=False,
        transform=transforms.Compose([Normalizer(mean_std_path=stat_txt_path),
                                      Augmenter(), Resizer(416)]))

    train_generator = DataLoader(train_set, **train_params)

    val_set = DIYDataset(path=os.path.join(opt.data_path, params['project_name']),
                         set_name=params['val_set'],
                         cal_mean_std=False,
                         transform=transforms.Compose(
                             [Normalizer(mean_std_path=stat_txt_path),
                              Augmenter(), Resizer(416)]))
    val_generator = DataLoader(val_set, **val_params)

    model = Darknet(cfg_path=params['cfg_path'])

    #预训练或者恢复训练
    if opt.load_weights:
        weights_path = opt.load_weights
        if opt.train_from_scratch is True:  #重新训练
            last_step = 0
            model.load_darknet_weights(weights_path)
            print('新的训练,加载预训练模型')
        else:  #恢复训练
            try:
                last_step = int(os.path.basename(weights_path).split('.')[0].split('_')[-1])
            except Exception as e:
                print('恢复训练失败')
                return
            model.load_darknet_weights(weights_path)
            print(f'恢复训练成功,恢复步数为{last_step}')
    else:
        last_step = 0
        print('新的训练,随机初始化网络')
        init_weights(model)#还没写

    if opt.head_only: #有点问题 没有解决怎么不冻结分类层和定位层
        def freeze_backbone(m):
            for param in m.parameters():
                param.requires_grad = False

        model.apply(freeze_backbone)
        print('冻结主干网络')

    if params['num_gpus'] > 0:
        model = model.cuda()
        if params['num_gpus'] > 1:
            print('分布式训练还没写')
            return

    if opt.optim == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), opt.lr)
    elif opt.optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), opt.lr, momentum=0.9, nesterov=True)
    else:
        print('暂时只写了adamw和sgd')
        return

    #学习率衰减 warmup接step decay
    scheduler_step = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[400, 450], gamma=0.1)
    scheduler_warmup_step = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=10, after_scheduler=scheduler_step)

    best_epoch = 0
    step = max(0, last_step)
    model.train()

    writer = SummaryWriter(os.path.join(opt.log_path,f'{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'))

    num_iter_per_epoch = len(train_generator)
    try:
        for epoch in range(opt.num_epochs):
            last_epoch = step // num_iter_per_epoch
            if epoch < last_epoch:
                continue
            epoch_loss = []
            progress_bar = tqdm(train_generator)
            for iter, data in enumerate(progress_bar):
                if iter < step - last_epoch * num_iter_per_epoch:
                    progress_bar.update()
                    continue
                imgs, annots = data['img'].requires_grad_(), data['annot'].requires_grad_()
                if params['num_gpus'] > 0:
                    imgs = imgs.cuda()
                    annots = annots.cuda()

                optimizer.zero_grad()
                x, cls_loss, reg_loss = model(imgs, annots)
                loss = cls_loss + reg_loss
                if loss.item == 0 or not torch.isfinite(loss):
                    continue
                loss.backward()
                optimizer.step()
                epoch_loss.append(float(loss))
                progress_bar.set_description(
                            'Step: {}. Epoch: {}/{}. Iteration: {}/{}. Cls loss: {:.5f}. Reg loss: {:.5f}. Total loss: {:.5f}'.format(
                                step, epoch, opt.num_epochs, iter + 1, num_iter_per_epoch, cls_loss.item(),
                                reg_loss.item(), loss.item()))
                writer.add_scalars('Loss', {'train': loss}, step)
                writer.add_scalars('Regression_loss', {'train': reg_loss}, step)
                writer.add_scalars('Classfication_loss', {'train': cls_loss}, step)

                current_lr = optimizer.param_groups[0]['lr']
                writer.add_scalar('learning_rate', current_lr, step)
                step += 1
                torch.cuda.empty_cache()
                if step % opt.save_interval == 0 and step > 0:
                    model.save_darknet_weights(os.path.join(opt.saved_path, '{}_yolov3_{}_{}.weights'.format(opt.project,epoch,step)))
                    print('保存模型'+'{}_yolov3_{}_{}.weights'.format(opt.project,epoch,step))
            scheduler_warmup_step.step()
            #训练集的损失每个batch更新一次 验证集的损失每个epoch更新一次
            if epoch % opt.val_interval == 0:
                model.eval()
                val_cls_losses = []
                val_reg_losses = []
                print('开始评估验证集')
                val_bar = tqdm(val_generator)
                for iter, data in enumerate(val_bar):
                    with torch.no_grad():
                        imgs, annots = data['img'], data['annot']
                        if params['num_gpus'] > 0:
                            imgs = imgs.cuda()
                            annots = annots.cuda()
                        x, cls_loss, reg_loss = model(imgs, annots)

                        loss = cls_loss + reg_loss
                        if loss == 0 or not torch.isfinite(loss):
                            continue
                        val_cls_losses.append(cls_loss.item())
                        val_reg_losses.append(reg_loss.item())
                cls_loss = torch.stack(val_cls_losses).mean()
                reg_loss = torch.stack(val_reg_losses).mean()
                loss = cls_loss + reg_loss
                print('Val. Epoch: {}/{}. Classification loss: {:1.5f}. Regression loss: {:1.5f}. Total loss: {:1.5f}'.format(
                            epoch, opt.num_epochs, cls_loss, reg_loss, loss))
                writer.add_scalars('Loss', {'val': loss}, step)
                writer.add_scalars('Regression_loss', {'val': reg_loss}, step)
                writer.add_scalars('Classfication_loss', {'val': cls_loss}, step)
                torch.cuda.empty_cache()
                model.train()
        model.save_darknet_weights(os.path.join(opt.saved_path, f'{opt.project}_yolov3_final_{epoch}_{step}.weights'))
    except KeyboardInterrupt as e:
        print('暂停训练')
        model.save_darknet_weights(os.path.join(opt.saved_path, f'{opt.project}_yolov3_backup_{epoch}_{step}.weights'))
        print('模型保存成功')
    writer.close()






if __name__ == '__main__':
    opt = get_args()
    start = time.time()
    train(opt)
    end = time.time()
    print(f'训练用时{round((end-start)/3600,1)}h')
