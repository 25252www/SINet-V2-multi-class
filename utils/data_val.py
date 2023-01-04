import os
from PIL import Image

import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
import random
import numpy as np
from PIL import ImageEnhance
import imageio
from tqdm import tqdm
import utils.eval as eval


# several data augumentation strategies
def cv_random_flip(img, label):
    # left right flip
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
    return img, label


def randomCrop(image, label):
    border = 30
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return image.crop(random_region), label.crop(random_region)


def randomRotation(image, label):
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        # rotate默认mode=PIL.Image.NEAREST
        image = image.rotate(random_angle)
        label = label.rotate(random_angle)
    return image, label


def colorEnhance(image):
    bright_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity = random.randint(0, 20) / 10.0
    image = ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity = random.randint(0, 30) / 10.0
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image


def randomGaussian(image, mean=0.1, sigma=0.35):
    def gaussianNoisy(im, mean=mean, sigma=sigma):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im

    img = np.asarray(image)
    width, height = img.shape
    img = gaussianNoisy(img[:].flatten(), mean, sigma)
    img = img.reshape([width, height])
    return Image.fromarray(np.uint8(img))


def randomPeper(img):
    img = np.array(img)
    noiseNum = int(0.0015 * img.shape[0] * img.shape[1])
    for i in range(noiseNum):

        randX = random.randint(0, img.shape[0] - 1)

        randY = random.randint(0, img.shape[1] - 1)

        if random.randint(0, 1) == 0:

            img[randX, randY] = 0

        else:

            img[randX, randY] = 255
    return Image.fromarray(img)


# dataset for training
class PolypObjDataset(data.Dataset):

    def __init__(self, image_root, gt_root, trainsize, classes):
        self.trainsize = trainsize
        self.classes = classes
        # get filenames
        # 只取COD10K，共3040张
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') and f.startswith('COD10K') ]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') and f.startswith('COD10K')
                    or f.endswith('.png') and f.startswith('COD10K')]
        # images和gts只取在classes中的类别
        self.images = [image for image in self.images if os.path.basename(image).split('-')[5].lower() in self.classes]
        self.gts = [gt for gt in self.gts if os.path.basename(gt).split('-')[5].lower() in self.classes]
        # sorted files
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        # self.grads = sorted(self.grads)
        # self.depths = sorted(self.depths)
        # filter mathcing degrees of files
        self.filter_files()
        # transforms
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        # self.gt_transform = transforms.Compose([
        #     transforms.Resize((self.trainsize, self.trainsize)),
        #     transforms.ToTensor()])
        # 转换为mask形式，nearest插值不增加新的像素，PILToTensor()将PIL Image转为tensor，不归一化
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize), interpolation=InterpolationMode.NEAREST),
            transforms.PILToTensor()])
        # get size of dataset
        self.size = len(self.images)
        # 生成label_mask形式的图片，保存在pre_encoded文件夹下
        self.setup_annotations()

    def __getitem__(self, index):
        # read imgs/gts/grads/depths
        image = self.rgb_loader(self.images[index])
        # 转为灰度图
        gt = self.binary_loader(self.gts[index])
        # data augumentation
        image, gt = cv_random_flip(image, gt)
        image, gt = randomCrop(image, gt)
        image, gt = randomRotation(image, gt)

        image = colorEnhance(image)
        # 椒盐噪声
        # gt = randomPeper(gt)

        image = self.img_transform(image)
        gt = self.gt_transform(gt)

        # print(self.images[index], "的类别包含", np.unique(gt))
        # 如果gt包含>2种像素，报错
        if len(np.unique(gt)) > 2:
            raise ValueError('gt包含>2种像素',np.unique(gt), self.gts[index])

        # print(type(gt))
        # print(np.shape(gt))
        # for i in range(np.shape(gt)[1]):
        #     for j in range(np.shape(gt)[2]):
        #         print(gt[0][i][j].item(),end='', file=open("/home/liuxiangyu/SINet-V2-multi-class/gt.txt", "a"))
        #     print('\n', file=open("/home/liuxiangyu/SINet-V2-multi-class/gt.txt", "a"))

        return image, gt

    def filter_files(self):
        assert len(self.images) == len(self.gts) and len(self.gts) == len(self.images)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size

    def encode_segmap(self, gt):
        mask = imageio.imread(gt)
        mask = mask.astype(np.uint8)
        cls = os.path.basename(gt).split('-')[5]
        cls_num = self.classes.index(cls.lower())
        mask = np.where(mask == 255, cls_num, mask)
        return mask


    def setup_annotations(self):
        """pre-encode all segmentation labels into the common label_mask format,
        Under this format, each mask is an (M,N) array of integer values from 0 
        to num_class+1, where 0 represents the background class.
        """
        pre_encoded_path = "/home/liuxiangyu/SINet-V2-multi-class/Dataset/TrainValDataset/pre_encoded"
        colored_gt_path = "/home/liuxiangyu/SINet-V2-multi-class/Dataset/TrainValDataset/colored_gt"
        if not os.path.exists(pre_encoded_path):
            os.makedirs(pre_encoded_path)
            print("trainvaldataset: Pre-encoding segmentation masks...")
            for i, gt in enumerate(tqdm(self.gts)):
                lbl = self.encode_segmap(gt)
                imageio.imsave(os.path.join(pre_encoded_path, os.path.basename(gt)), lbl)
        self.gts = [os.path.join(pre_encoded_path,os.path.basename(gt)) for gt in self.gts]
        if not os.path.exists(colored_gt_path):
            os.makedirs(colored_gt_path)
            print("trainvaldataset: Coloring segmentation gts...")
            print(self.classes)
            print(type(self.classes))
            colormap = eval.get_colormap(len(self.classes))
            for i, gt in enumerate(tqdm(self.gts)):
                gt_img = self.binary_loader(gt)
                gt_img = np.array(gt_img)
                rgb = eval.decode_segmap(gt_img, colormap, self.classes)
                imageio.imsave(os.path.join(colored_gt_path,os.path.basename(gt)), rgb)
                
                
    

# dataloader for training
def get_loader(image_root, gt_root, batchsize, trainsize, classes, 
               shuffle=True, num_workers=12, pin_memory=True):
    dataset = PolypObjDataset(image_root, gt_root, trainsize, classes)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


# test dataset and loader
# 2026 张 COD10K
class test_dataset:
    def __init__(self, image_root, gt_root, testsize, classes):
        self.testsize = testsize
        self.classes = classes
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.tif') or f.endswith('.png')]
        # images和gts只取在classes中的类别
        self.images = [image for image in self.images if os.path.basename(image).split('-')[5].lower() in self.classes]
        self.gts = [gt for gt in self.gts if os.path.basename(gt).split('-')[5].lower() in self.classes]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        # self.gt_transform = transforms.ToTensor()
        self.gt_transform = transforms.PILToTensor()
        self.size = len(self.images)
        self.index = 0
        # 生成label_mask形式的图片，保存在pre_encoded文件夹下
        self.setup_annotations()

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)

        gt = self.binary_loader(self.gts[self.index])

        name = self.images[self.index].split('/')[-1]

        image_for_post = self.rgb_loader(self.images[self.index])
        image_for_post = image_for_post.resize(gt.size)

        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'

        self.index += 1
        self.index = self.index % self.size

        # 测试gt是否正确
        # print(type(gt))
        # print(np.shape(gt))
        # gt_print = np.array(gt)
        # with open("/home/liuxiangyu/SINet-V2-multi-class/gt.txt", "a") as f:
        #     f.truncate(0)
        #     for i in range(np.shape(gt_print)[0]):
        #         for j in range(np.shape(gt_print)[1]):
        #             print(gt_print[i][j].item(),end='', file=open("/home/liuxiangyu/SINet-V2-multi-class/gt.txt", "a"))
        #         # 添加换行符
        #         print('', file=open("/home/liuxiangyu/SINet-V2-multi-class/gt.txt", "a"))
                

        return image, gt, name, np.array(image_for_post)

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size
    
    def encode_segmap(self, gt):
        mask = imageio.imread(gt)
        mask = mask.astype(np.uint8)
        cls = os.path.basename(gt).split('-')[5]
        cls_num = self.classes.index(cls.lower())
        mask = np.where(mask == 255, cls_num, mask)
        return mask


    def setup_annotations(self):
        """pre-encode all segmentation labels into the common label_mask format,
        Under this format, each mask is an (M,N) array of integer values from 0 
        to num_class+1, where 0 represents the background class.
        """
        pre_encoded_path = "/home/liuxiangyu/SINet-V2-multi-class/Dataset/TestDataset/COD10K/pre_encoded"
        colored_gt_path = "/home/liuxiangyu/SINet-V2-multi-class/Dataset/TestDataset/COD10K/colored_gt"
        if not os.path.exists(pre_encoded_path):
            os.makedirs(pre_encoded_path)
            print("testdataset: Pre-encoding segmentation masks...")
            for i, gt in enumerate(tqdm(self.gts)):
                lbl = self.encode_segmap(gt)
                imageio.imsave(os.path.join(pre_encoded_path, os.path.basename(gt)), lbl)
        self.gts = [os.path.join(pre_encoded_path,os.path.basename(gt)) for gt in self.gts]
        if not os.path.exists(colored_gt_path):
            os.makedirs(colored_gt_path)
            print("trainvaldataset: Coloring segmentation gts...")
            colormap = eval.get_colormap(len(self.classes))
            for i, gt in enumerate(tqdm(self.gts)):
                gt_img = self.binary_loader(gt)
                gt_img = np.array(gt_img)
                rgb = eval.decode_segmap(gt_img, colormap, self.classes)
                imageio.imsave(os.path.join(colored_gt_path, os.path.basename(gt)), rgb)

