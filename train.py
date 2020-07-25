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
from test import _eval

from utils.utils import *
from utils.lr_decay import *
from datasets.datasets import *
from darknet import *
from utils.weight_init import *


def get_args():
    parser = argparse.ArgumentParser('yolov3 detector train')
    parser.add_argument('-p', '--project', type=str, default='shape', help='config file in /project/*yml')
    parser.add_argument('-n', '--num_workers', type=int, default=0, help='the num_workers of dataloader')
    parser.add_argument('--batch_size', type=int, default=16, help='the batch_size of dataloader')
    parser.add_argument('--head_only', type=bool, default=False, help='freeze all layers except classification layer')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--optim',
                        type=str,
                        default='adamw',
                        help='suggest using adamw until the very final stage then switch to sgd')
<<<<<<< Updated upstream
<<<<<<< Updated upstream
    parser.add_argument('--num_epochs', type=int, default=50)
=======
    parser.add_argument('--num_epochs', type=int, default=200)
>>>>>>> Stashed changes
=======
    parser.add_argument('--num_epochs', type=int, default=200)
>>>>>>> Stashed changes
    parser.add_argument('--val_interval', type=int, default=1, help="'intervals of calculate val_datasets' mAP")
    parser.add_argument('--save_interval', type=int, default=200, help='number of steps between saving')
    parser.add_argument('--data_path', type=str, default=os.path.join('..', 'datasets', 'coco_shape'))
    parser.add_argument('--log_path', type=str, default='logs', help='tensorboardX log')
    parser.add_argument('--load_weights', type=str, default='', help='pretrained models or recover training')
    parser.add_argument('--train_from_last',
                        type=bool,
                        default=True,
                        help='recover from the last training or training with pretrained models')
    parser.add_argument('--saved_path', type=str, default='backup', help="'saved models'path")
    parser.add_argument('--stat_path', type=str, default='cfg', help='mean val and std val of dataset')
    parser.add_argument('--conf_thresh', type=float, default=0.5)
    parser.add_argument('--iou_thresh', type=float, default=0.5)

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
<<<<<<< Updated upstream
<<<<<<< Updated upstream
        transform=transforms.Compose([Augmenter(), Resizer(416)]))
    # transform=transforms.Compose([Normalizer(mean_std_path=stat_txt_path),
    #                               Augmenter(), Resizer(416)]))
=======
        # transform=transforms.Compose([Normalizer(mean_std_path=stat_txt_path),
        #                               Augmenter(), Resizer(416)]))
        transform=transforms.Compose([Augmenter(), Resizer(416)]))
>>>>>>> Stashed changes
=======
        # transform=transforms.Compose([Normalizer(mean_std_path=stat_txt_path),
        #                               Augmenter(), Resizer(416)]))
        transform=transforms.Compose([Augmenter(), Resizer(416)]))
>>>>>>> Stashed changes

    train_generator = DataLoader(train_set, **train_params)

    val_set = DIYDataset(path=os.path.join(opt.data_path, params['project_name']),
                         set_name=params['val_set'],
                         cal_mean_std=False,
<<<<<<< Updated upstream
                         transform=transforms.Compose([Augmenter(), Resizer(416)]))
    # transform=transforms.Compose(
    #      [Normalizer(mean_std_path=stat_txt_path),
    #       Augmenter(), Resizer(416)]))
=======
                         transform=transforms.Compose(
                             [Normalizer(mean_std_path=stat_txt_path),
                              Augmenter(), Resizer(416)]))

<<<<<<< Updated upstream
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes
    val_generator = DataLoader(val_set, **val_params)

    model = Darknet(cfg_path=params['cfg_path'])

    #预训练或者恢复训练
    if opt.load_weights:
        weights_path = opt.load_weights
        if opt.train_from_last is False:  #重新训练 用预训练模型初始化
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
<<<<<<< Updated upstream
<<<<<<< Updated upstream
        model.apply(weights_init_normal)
        print('新的训练,高斯初始化网络')
=======
=======
>>>>>>> Stashed changes
        # model.apply(weights_init_normal)
        # print('新的训练,高斯初始化网络')
        model.apply(weights_init_kaiming_uniform)
        print('新的训练,kaiming初始化网络')
<<<<<<< Updated upstream
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes

    if opt.head_only:  #有点问题 没有解决怎么不冻结分类层和定位层

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
<<<<<<< Updated upstream
<<<<<<< Updated upstream
    scheduler_step = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[35, 45], gamma=0.1)
    scheduler_warmup_step = GradualWarmupScheduler(optimizer,
                                                   multiplier=1,
                                                   total_epoch=10,
                                                   after_scheduler=scheduler_step)
=======
=======
>>>>>>> Stashed changes
    scheduler_step = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[140, 180], gamma=0.1)
    scheduler_warmup_step = GradualWarmupScheduler(optimizer,
                                                   multiplier=1,
                                                   total_epoch=5,
                                                   after_scheduler=scheduler_step)
    # scheduler_warmup_step = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[35, 45], gamma=0.1)
<<<<<<< Updated upstream
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes

    best_epoch = 0
    step = max(0, last_step)
    model.train()

    writer = SummaryWriter(os.path.join(opt.log_path, f'{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'))

    num_iter_per_epoch = len(train_generator)
    try:
        for epoch in range(opt.num_epochs):
            last_epoch = step // num_iter_per_epoch
            if epoch < last_epoch:
                continue
            progress_bar = tqdm(train_generator)
            for iter, data in enumerate(progress_bar):
                if iter < step - last_epoch * num_iter_per_epoch:
                    progress_bar.update()
                    continue
                imgs, annots = data['img'], data['annot']
<<<<<<< Updated upstream
<<<<<<< Updated upstream
=======
=======
>>>>>>> Stashed changes
                scales = data['scale']
                img_ids = data['img_id']

                #输出图片和标注框看看有没得问题
                debug_imshow(imgs, annots, img_ids)

<<<<<<< Updated upstream
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes
                if params['num_gpus'] > 0:
                    imgs = imgs.cuda()
                    annots = annots.cuda()

                optimizer.zero_grad()
<<<<<<< Updated upstream
<<<<<<< Updated upstream
                x, cls_loss, reg_loss, conf_loss = model(imgs, annots)
                loss = torch.cat((cls_loss, reg_loss, conf_loss)).sum()
=======
=======
>>>>>>> Stashed changes
                # x, cls_loss, reg_loss, conf_loss = model(imgs, annots)
                # loss = torch.cat((cls_loss, reg_loss, conf_loss)).sum()
                x, cls_loss, reg_loss_x, reg_loss_y, reg_loss_w, reg_loss_h, conf_loss = model(imgs, annots, scales)
                # loss = torch.cat((reg_loss_x,reg_loss_y)).sum()
                loss = torch.cat((cls_loss, reg_loss_x, reg_loss_y, reg_loss_w, reg_loss_h, conf_loss)).sum()
<<<<<<< Updated upstream
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes
                if loss.item == 0 or not torch.isfinite(loss):
                    continue
                loss.backward()
                optimizer.step()
<<<<<<< Updated upstream
<<<<<<< Updated upstream
                cls_loss = cls_loss.sum(dim=0, keepdim=True)
                reg_loss = reg_loss.sum(dim=0, keepdim=True)
                conf_loss = conf_loss.sum(dim=0, keepdim=True)
                progress_bar.set_description(
                    'Epoch: {}/{}. Iteration: {}/{}. Cls loss: {:.5f}. Reg loss: {:.5f}. Conf loss: {:.5f}. Total loss: {:.5f}'
                    .format(epoch, opt.num_epochs, iter + 1, num_iter_per_epoch, cls_loss.item(), reg_loss.item(),
                            conf_loss.item(), loss.item()))
=======
                # cls_loss = cls_loss.sum(dim=0, keepdim=True)
                # reg_loss = reg_loss.sum(dim=0, keepdim=True)
                # conf_loss = conf_loss.sum(dim=0, keepdim=True)
                # progress_bar.set_description(
                #     'Epoch: {}/{}. Iteration: {}/{}. Cls loss: {:.5f}. Reg loss: {:.5f}. Conf loss: {:.5f}. Total loss: {:.5f}'
                #     .format(epoch, opt.num_epochs, iter + 1, num_iter_per_epoch, cls_loss.item(), reg_loss.item(),
                #             conf_loss.item(), loss.item()))
                # writer.add_scalars('Loss', {'train': loss}, step)
                # writer.add_scalars('Regression_loss', {'train': reg_loss}, step)
                # writer.add_scalars('Classfication_loss', {'train': cls_loss}, step)
                # writer.add_scalars('Confidence_loss', {'train': conf_loss}, step)

                cls_loss = cls_loss.sum(dim=0, keepdim=True)
                reg_loss_x = reg_loss_x.sum(dim=0, keepdim=True)
                reg_loss_y = reg_loss_y.sum(dim=0, keepdim=True)
                reg_loss_w = reg_loss_w.sum(dim=0, keepdim=True)
                reg_loss_h = reg_loss_h.sum(dim=0, keepdim=True)
                conf_loss = conf_loss.sum(dim=0, keepdim=True)
                progress_bar.set_description(
                    'Epoch: {}/{}. Iteration: {}/{}. Cls loss: {:.5f}. Reg_x loss: {:.5f}. Reg_y loss: {:.5f}. Reg_w loss: {:.5f}. Reg_h loss: {:.5f}. Conf loss: {:.5f}. Total loss: {:.5f}'
                    .format(epoch, opt.num_epochs, iter + 1, num_iter_per_epoch, cls_loss.item(), reg_loss_x.item(),
                            reg_loss_y.item(), reg_loss_w.item(), reg_loss_h.item(), conf_loss.item(), loss.item()))
>>>>>>> Stashed changes
=======
                # cls_loss = cls_loss.sum(dim=0, keepdim=True)
                # reg_loss = reg_loss.sum(dim=0, keepdim=True)
                # conf_loss = conf_loss.sum(dim=0, keepdim=True)
                # progress_bar.set_description(
                #     'Epoch: {}/{}. Iteration: {}/{}. Cls loss: {:.5f}. Reg loss: {:.5f}. Conf loss: {:.5f}. Total loss: {:.5f}'
                #     .format(epoch, opt.num_epochs, iter + 1, num_iter_per_epoch, cls_loss.item(), reg_loss.item(),
                #             conf_loss.item(), loss.item()))
                # writer.add_scalars('Loss', {'train': loss}, step)
                # writer.add_scalars('Regression_loss', {'train': reg_loss}, step)
                # writer.add_scalars('Classfication_loss', {'train': cls_loss}, step)
                # writer.add_scalars('Confidence_loss', {'train': conf_loss}, step)

                cls_loss = cls_loss.sum(dim=0, keepdim=True)
                reg_loss_x = reg_loss_x.sum(dim=0, keepdim=True)
                reg_loss_y = reg_loss_y.sum(dim=0, keepdim=True)
                reg_loss_w = reg_loss_w.sum(dim=0, keepdim=True)
                reg_loss_h = reg_loss_h.sum(dim=0, keepdim=True)
                conf_loss = conf_loss.sum(dim=0, keepdim=True)
                progress_bar.set_description(
                    'Epoch: {}/{}. Iteration: {}/{}. Cls loss: {:.5f}. Reg_x loss: {:.5f}. Reg_y loss: {:.5f}. Reg_w loss: {:.5f}. Reg_h loss: {:.5f}. Conf loss: {:.5f}. Total loss: {:.5f}'
                    .format(epoch, opt.num_epochs, iter + 1, num_iter_per_epoch, cls_loss.item(), reg_loss_x.item(),
                            reg_loss_y.item(), reg_loss_w.item(), reg_loss_h.item(), conf_loss.item(), loss.item()))
>>>>>>> Stashed changes
                writer.add_scalars('Loss', {'train': loss}, step)
                writer.add_scalars('Regression_loss_x', {'train': reg_loss_x}, step)
                writer.add_scalars('Regression_loss_y', {'train': reg_loss_y}, step)
                writer.add_scalars('Regression_loss_w', {'train': reg_loss_w}, step)
                writer.add_scalars('Regression_loss_h', {'train': reg_loss_h}, step)
                writer.add_scalars('Classfication_loss', {'train': cls_loss}, step)
                writer.add_scalars('Confidence_loss', {'train': conf_loss}, step)

                current_lr = optimizer.param_groups[0]['lr']
                writer.add_scalar('learning_rate', current_lr, step)
                step += 1
                if step % opt.save_interval == 0 and step > 0:
<<<<<<< Updated upstream
<<<<<<< Updated upstream
                    model.save_darknet_weights(
                        os.path.join(opt.saved_path, '{}_yolov3_{}_{}.weights'.format(opt.project, epoch, step)))
                    print('保存模型' + '{}_yolov3_{}_{}.weights'.format(opt.project, epoch, step))
=======
=======
>>>>>>> Stashed changes
                    # model.save_darknet_weights(
                    #     os.path.join(opt.saved_path, '{}_yolov3_{}_{}.weights'.format(opt.project, epoch, step)))
                    save_checkpoint(
                        model, os.path.join(opt.saved_path, '{}_yolov3_{}_{}.pth'.format(opt.project, epoch, step)))
                    print('保存模型' + '{}_yolov3_{}_{}.weights'.format(opt.project, epoch, step))
                torch.cuda.empty_cache()
<<<<<<< Updated upstream
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes
            scheduler_warmup_step.step(epoch=epoch)
            # 训练集的损失每个batch更新一次 验证集的损失每个epoch更新一次
            if epoch % opt.val_interval == 0:
                model.eval()
                # val_cls_losses = []
                # val_reg_losses = []
                # val_conf_losses = []

                val_cls_losses = []
<<<<<<< Updated upstream
<<<<<<< Updated upstream
                val_reg_losses = []
                val_conf_losses = []
                print('开始评估验证集')
                val_bar = tqdm(val_generator)
=======
=======
>>>>>>> Stashed changes
                val_reg_x_losses = []
                val_reg_y_losses = []
                val_reg_w_losses = []
                val_reg_h_losses = []
                val_conf_losses = []

                print('开始评估验证集')
                val_bar = tqdm(val_generator)
                predictions = []
                img_ids = []
<<<<<<< Updated upstream
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes
                for iter, data in enumerate(val_bar):
                    with torch.no_grad():
                        imgs, annots = data['img'], data['annot']
                        img_id = data['img_id']
                        if params['num_gpus'] > 0:
                            imgs = imgs.cuda()
                            annots = annots.cuda()
<<<<<<< Updated upstream
<<<<<<< Updated upstream
                        x, cls_loss, reg_loss, conf_loss = model(imgs, annots)
                        val_cls_losses.append(cls_loss)
                        val_reg_losses.append(reg_loss)
                        val_conf_losses.append(conf_loss)
                cls_loss = torch.stack(val_cls_losses)
                reg_loss = torch.stack(val_reg_losses)
                conf_loss = torch.stack(val_conf_losses)
                cls_loss = cls_loss.mean()
                reg_loss = reg_loss.mean()
                conf_loss = conf_loss.mean()
                loss = cls_loss + reg_loss + conf_loss
                print('Val.  Cls loss: {:1.5f}. Reg loss: {:1.5f}. Conf loss: {:1.5f}. Total loss: {:1.5f}'.format(
                    cls_loss, reg_loss, conf_loss, loss))
=======
=======
>>>>>>> Stashed changes

                        x, cls_loss, reg_loss_x, reg_loss_y, reg_loss_w, reg_loss_h, conf_loss
                        val_cls_losses.append(cls_loss)
                        val_reg_x_losses.append(reg_loss_x)
                        val_reg_y_losses.append(reg_loss_y)
                        val_reg_w_losses.append(reg_loss_w)
                        val_reg_h_losses.append(reg_loss_h)
                        val_conf_losses.append(conf_loss)

                        # x, cls_loss, reg_loss, conf_loss = model(imgs, annots)
                        # val_cls_losses.append(cls_loss)
                        # val_reg_losses.append(reg_loss)
                        # val_conf_losses.append(conf_loss)
                        predictions.append(x)
                        img_ids += img_id
                predictions = torch.cat(predictions)

                # cls_loss = torch.stack(val_cls_losses).mean()
                # reg_loss = torch.stack(val_reg_losses).mean()
                # conf_loss = torch.stack(val_conf_losses).mean()
                # loss = cls_loss + reg_loss + conf_loss
                # print('Val.  Cls loss: {:1.5f}. Reg loss: {:1.5f}. Conf loss: {:1.5f}. Total loss: {:1.5f}'.format(
                #     cls_loss, reg_loss, conf_loss, loss))
                # writer.add_scalars('Loss', {'val': loss}, step)
                # writer.add_scalars('Regression_loss', {'val': reg_loss}, step)
                # writer.add_scalars('Classfication_loss', {'val': cls_loss}, step)
                # writer.add_scalars('Confidence_loss', {'val': conf_loss}, step)

                cls_loss = torch.stack(val_cls_losses).mean()
                reg_loss_x = torch.stack(val_reg_x_losses).mean()
                reg_loss_y = torch.stack(val_reg_y_losses).mean()
                reg_loss_w = torch.stack(val_reg_w_losses).mean()
                reg_loss_h = torch.stack(val_reg_h_losses).mean()
                conf_loss = torch.stack(val_conf_losses).mean()
                loss = cls_loss + reg_loss_x + reg_loss_y + reg_loss_w + reg_loss_h + conf_loss
                print(
                    'Val.  Cls loss: {:1.5f}. Reg_x loss: {:1.5f}. Reg_y loss: {:1.5f}. Reg_w loss: {:1.5f}. Reg_h loss: {:1.5f}. Conf loss: {:1.5f}. Total loss: {:1.5f}'
                    .format(cls_loss, reg_loss_x.item(), reg_loss_y.item(), reg_loss_w.item(), reg_loss_h.item(),
                            conf_loss, loss))
<<<<<<< Updated upstream
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes
                writer.add_scalars('Loss', {'val': loss}, step)
                writer.add_scalars('Regression_loss_x', {'val': reg_loss_x}, step)
                writer.add_scalars('Regression_loss_y', {'val': reg_loss_y}, step)
                writer.add_scalars('Regression_loss_w', {'val': reg_loss_w}, step)
                writer.add_scalars('Regression_loss_h', {'val': reg_loss_h}, step)
                writer.add_scalars('Classfication_loss', {'val': cls_loss}, step)
                writer.add_scalars('Confidence_loss', {'val': conf_loss}, step)
<<<<<<< Updated upstream
<<<<<<< Updated upstream
=======
=======
>>>>>>> Stashed changes

                pred_batch_imgs = NMS(img_ids,
                                      predictions,
                                      conf_thresh=opt.conf_thresh,
                                      iou_thresh=opt.iou_thresh,
                                      style='OR',
                                      type='DIoU')
                if not pred_batch_imgs:
                    print('训练初期不计算mAP')
                    continue
                pred_path = 'training_val_2017_bbox_results.json'
                with open(pred_path, 'w') as f:
                    json.dump(pred_batch_imgs, f)
                gt_json = os.path.join(opt.data_path, params['project_name'], 'annotations',
                                       'instances_{}2017.json'.format(params['val_set']))
                MAX_IMAGES = 10000
                coco_gt = COCO(gt_json)
                image_ids = coco_gt.getImgIds()[:MAX_IMAGES]
                _eval(coco_gt, image_ids, pred_path)
<<<<<<< Updated upstream
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes
                torch.cuda.empty_cache()
                model.train()
        # model.save_darknet_weights(os.path.join(opt.saved_path, f'{opt.project}_yolov3_final_{epoch}_{step}.weights'))
        save_checkpoint(model, os.path.join(opt.saved_path, f'{opt.project}_yolov3_final_{epoch}_{step}.pth'))
        print('模型保存成功 训练结束')
    except KeyboardInterrupt as e:
        print('暂停训练')
        # model.save_darknet_weights(os.path.join(opt.saved_path, f'{opt.project}_yolov3_backup_{epoch}_{step}.weights'))
        save_checkpoint(model, os.path.join(opt.saved_path, f'{opt.project}_yolov3_backup_{epoch}_{step}.pth'))
        print('模型保存成功')
    writer.close()


if __name__ == '__main__':
    opt = get_args()
    start = time.time()
    train(opt)
    end = time.time()
    print(f'训练用时{round((end-start)/3600,1)}h')
