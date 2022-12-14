import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse
from scipy import misc
import cv2
from lib.Network_Res2Net_GRA_NCD import Network
from utils.data_val import test_dataset
import yaml
from utils.eval import *


parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--pth_path', type=str,
                    default='./snapshot/SINet_V2/Net_epoch_best.pth')
with open('./config.yaml', 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)
    parser.set_defaults(**cfg)
opt = parser.parse_args()




for _data_name in ['COD10K']:
    data_path = './Dataset/TestDataset/{}/'.format(_data_name)
    save_path = './res/{}/{}/'.format(opt.pth_path.split('/')[-2], _data_name)
    model = Network(num_classes=len(opt.CLASSES), imagenet_pretrained=False)
    model.load_state_dict(torch.load(opt.pth_path))
    model.cuda()
    model.eval()

    os.makedirs(save_path, exist_ok=True)
    image_root = '{}/Imgs/'.format(data_path)
    gt_root = '{}/GT/'.format(data_path)
    test_loader = test_dataset(image_root, gt_root, opt.testsize, opt.CLASSES)
    confusion_matrix = np.zeros((len(opt.CLASSES), len(opt.CLASSES)))

    for i in range(test_loader.size):
        image, gt, name, _ = test_loader.load_data()
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
        # 计算混淆矩阵
        confusion_matrix += get_confusion_matrix(
            np.array(res.cpu()), np.array(gt), len(opt.CLASSES))
        # colorize 对输出图着色
        colormap = get_colormap(len(opt.CLASSES))
        rgb = decode_segmap(np.array(res.cpu()), colormap=colormap, classes=opt.CLASSES)
        print('> {} - {} '.format(_data_name, name))
        # 将GBR转为RGB
        rgb = rgb[:, :, ::-1]
        # 使用cv2.imwrite保存彩色输出图
        cv2.imwrite(save_path+name, rgb*255)

    # 计算iou
    iou = cal_iou(confusion_matrix, len(opt.CLASSES))
    # 打印每个类别的iou
    print('iou:')
    for i in range(len(opt.CLASSES)):
        print('class {}: {}'.format(opt.CLASSES[i], iou[i]))
    # 计算平均iou
    print('mean iou: {}'.format(np.mean(iou)))
