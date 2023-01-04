# author: Daniel-Ji (e-mail: gepengai.ji@gmail.com)
# data: 2021-01-16
import imp
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from torchvision.utils import make_grid
from lib.Network_Res2Net_GRA_NCD import Network
from utils.data_val import get_loader, test_dataset
from utils.utils import clip_gradient, adjust_lr
from tensorboardX import SummaryWriter
import logging
import torch.backends.cudnn as cudnn
import numpy as np
import argparse
import yaml
from utils.eval import *
from tqdm import tqdm 

def structure_loss(pred, mask):
    """
    loss function (ref: F3Net-AAAI-2020)
    """
    # print('structure_loss, shape of pred:', np.shape(pred))
    # print('structure_loss, shape of mask:', np.shape(mask))
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, inputs, targets):
        if inputs.dim() > 2:
            inputs = inputs.view(inputs.size(0), inputs.size(1), -1)  # N,C,H,W => N,C,H*W
            inputs = inputs.transpose(1, 2)  # N,C,H*W => N,H*W,C
            inputs = inputs.contiguous().view(-1, inputs.size(2))  # N,H*W,C => N*H*W,C
        targets = targets.view(-1, 1)

        logpt = F.log_softmax(inputs, dim=1)
        logpt = logpt.gather(1, targets)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        if self.alpha is not None:
            if self.alpha.type() != inputs.data.type():
                self.alpha = self.alpha.type_as(inputs.data)
            at = self.alpha.gather(0, targets.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1 - pt) ** self.gamma * logpt # Focal loss
        if self.size_average: return loss.mean()
        return loss.sum()

def DiceLoss(pred, target):
    """
    Dice loss function
    """
    
def train(train_loader, model, optimizer, epoch, save_path, writer):
    """
    train function
    """
    global step
    model.train()
    loss_all = 0
    epoch_step = 0
    try:
        for i, (images, gts) in enumerate(train_loader, start=1):
            optimizer.zero_grad()

            images = images.cuda()
            
            gts = gts.squeeze(1) # (bs, 1, H, W) -> (bs, H, W)
            gts = gts.type(torch.LongTensor)
            gts = gts.cuda()

            preds = model(images)
            # 交叉熵损失
            lossfunc = nn.CrossEntropyLoss()
            # lossfunc = FocalLoss()
            loss_init = lossfunc(preds[0], gts) + lossfunc(preds[1], gts) + lossfunc(preds[2], gts)
            loss_final = lossfunc(preds[3], gts)

            loss = loss_init + loss_final
            loss.backward()

            clip_gradient(optimizer, opt.clip)
            optimizer.step()

            step += 1
            epoch_step += 1
            loss_all += loss.data

            if i % 20 == 0 or i == total_step or i == 1:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f} Loss1: {:.4f} Loss2: {:0.4f}'.
                      format(datetime.now(), epoch, opt.epoch, i, total_step, loss.data, loss_init.data, loss_final.data))
                logging.info(
                    '[Train Info]:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f} Loss1: {:.4f} '
                    'Loss2: {:0.4f}'.
                    format(epoch, opt.epoch, i, total_step, loss.data, loss_init.data, loss_final.data))
                # TensorboardX-Loss
                writer.add_scalars('Loss_Statistics',
                                   {'Loss_init': loss_init.data, 'Loss_final': loss_final.data,
                                    'Loss_total': loss.data},
                                   global_step=step)
                # TensorboardX-Training Data
                grid_image = make_grid(images[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('RGB', grid_image, step)
                grid_image = make_grid(gts[0].type(torch.FloatTensor).clone().cpu().data, 1, normalize=True)
                writer.add_image('GT', grid_image, step)

                # # TensorboardX-Outputs
                # res = preds[0][0].clone()
                # res = res.sigmoid().data.cpu().numpy().squeeze()
                # res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                # writer.add_image('Pred_init', torch.tensor(res), step, dataformats='HW')
                # res = preds[3][0].clone()
                # res = res.sigmoid().data.cpu().numpy().squeeze()
                # res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                # writer.add_image('Pred_final', torch.tensor(res), step, dataformats='HW')

        loss_all /= epoch_step
        logging.info('[Train Info]: Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format(epoch, opt.epoch, loss_all))
        writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)
        if epoch % 50 == 0:
            torch.save(model.state_dict(), save_path + 'Net_epoch_{}.pth'.format(epoch))
    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path + 'Net_epoch_{}.pth'.format(epoch + 1))
        print('Save checkpoints successfully!')
        raise


def val(test_loader, model, epoch, save_path, writer, classes, is_last_epoch):
    """
    validation function
    每个epoch，先train，后val。
    计算miou，保存最好的模型
    """
    global best_miou, best_epoch
    model.eval()
    with torch.no_grad():
        confusion_matrix= torch.zeros(len(classes), len(classes)).cuda()
        for i in tqdm(range(test_loader.size)):
            image, gt, name, img_for_post = test_loader.load_data()
            image = image.cuda()
            res5, res4, res3, res2 = model(image)
            # res2使用argmax获得类别
            res = res2.squeeze()
            res = res.argmax(axis=0).type(torch.uint8)
            res = res.unsqueeze(0)
            res = res.unsqueeze(0)
            # resize到原大小，采用临近插值法，不会出现新的数值
            res = F.interpolate(res, size=np.array(gt).shape, mode='nearest')
            res = res.squeeze()
            # 计算混淆矩阵，使用gpu加速
            # res的类型是gpu的tensor
            # gt的类型是PIL的image，转换为gpu的tensor
            gt = torch.from_numpy(np.array(gt)).cuda()
            # 计算混淆矩阵，累加到confusion_matrix
            confusion_matrix += get_confusion_matrix(res, gt, len(classes))

        print('confusion_matrix')
        print(confusion_matrix)
        # 计算iou和miou
        # 打印confusion_matrix，是一个二元数组
        iou = cal_iou(confusion_matrix, len(classes))
        # 打印iou
        for index, i in enumerate(iou):
            print('iou of class {}: {:.4f}'.format(classes[index], i))
        miou = torch.mean(iou)
        if epoch == 1:
            best_miou = miou
        else:
            if miou > best_miou:
                best_miou = miou
                best_epoch = epoch
                torch.save(model.state_dict(), save_path + 'Net_epoch_best.pth')
                print('Save state_dict successfully! Best epoch:{}.'.format(epoch))
        print('Epoch: {}, miou: {}, bestmiou: {}, bestEpoch: {}.'.format(epoch, miou, best_miou, best_epoch))
        logging.info(
            '[Val Info]:Epoch: {}, miou: {}, bestmiou: {}, bestEpoch: {}'.format(epoch, miou, best_miou, best_epoch))
        if is_last_epoch:
            torch.save(model.state_dict(), save_path + 'Net_epoch_last.pth')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=100, help='epoch number')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=10, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=352, help='training dataset size')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=50, help='every n epochs decay learning rate')
    parser.add_argument('--load', type=str, default=None, help='train from checkpoints')
    parser.add_argument('--gpu_id', type=str, default='0', help='train use gpu')
    parser.add_argument('--train_root', type=str, default='/home/liuxiangyu/SINet-V2-multi-class/Dataset/TrainValDataset/',
                        help='the training rgb images root')
    parser.add_argument('--val_root', type=str, default='/home/liuxiangyu/SINet-V2-multi-class/Dataset/TestDataset/COD10K/',
                        help='the test rgb images root')
    parser.add_argument('--save_path', type=str,
                        default='/home/liuxiangyu/SINet-V2-multi-class/snapshot/SINet_V2/',
                        help='the path to save model and log')
    with open('/home/liuxiangyu/SINet-V2-multi-class/config.yaml', 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)
    opt = parser.parse_args()

    # set the device for training
    if opt.gpu_id == '0':
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print('USE GPU 0')
    elif opt.gpu_id == '1':
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        print('USE GPU 1')
    cudnn.benchmark = True

    # build the model
    model = Network(num_classes = len(opt.CLASSES), channel=32).cuda()
    pretrained_state_dict = torch.load('/home/liuxiangyu/SINet-V2-multi-class/snapshot/Net_binary.pth')
    # 取shape相同的参数
    pretrained_state_dict = {k:v for k,v in pretrained_state_dict.items() if v.shape == model.state_dict()[k].shape}
    model.load_state_dict(pretrained_state_dict, strict=False)
    print('load pretrained model from binary segmentation weights (original SINet_V2 Network)')

    if opt.load is not None:
        model.load_state_dict(torch.load(opt.load))
        print('load model from ', opt.load)

    optimizer = torch.optim.Adam(model.parameters(), opt.lr)

    save_path = opt.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # load data
    print('load data...')
    train_loader = get_loader(image_root=opt.train_root + 'Imgs/',
                              gt_root=opt.train_root + 'GT/',
                              batchsize=opt.batchsize,
                              trainsize=opt.trainsize,
                              classes=opt.CLASSES,
                              num_workers=8,)
    val_loader = test_dataset(image_root=opt.val_root + 'Imgs/',
                              gt_root=opt.val_root + 'GT/',
                              testsize=opt.trainsize,
                              classes=opt.CLASSES)
    total_step = len(train_loader)

    # logging
    logging.basicConfig(filename=save_path + 'log.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info("Network-Train")
    logging.info('Config: epoch: {}; lr: {}; batchsize: {}; trainsize: {}; clip: {}; decay_rate: {}; load: {}; '
                 'save_path: {}; decay_epoch: {}'.format(opt.epoch, opt.lr, opt.batchsize, opt.trainsize, opt.clip,
                                                         opt.decay_rate, opt.load, save_path, opt.decay_epoch))

    step = 0
    writer = SummaryWriter(save_path + 'summary')
    best_miou = 0
    best_epoch = 1

    print("Start train...")
    for epoch in range(1, opt.epoch+1):
        cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        writer.add_scalar('learning_rate', cur_lr, global_step=epoch)
        train(train_loader, model, optimizer, epoch, save_path, writer)
        val(val_loader, model, epoch, save_path, writer, opt.CLASSES, is_last_epoch = epoch == opt.epoch)
