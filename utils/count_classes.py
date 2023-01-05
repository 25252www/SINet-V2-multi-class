import os

if __name__ == '__main__':

    img_folder = '/home/liuxiangyu/SINet-V2-multi-class/Dataset/TrainValDataset/Imgs'
    imgs_paths = [img_folder + f for f in os.listdir(img_folder) if f.startswith('COD10K')]
    
    classes = [os.path.basename(img_path).split('-')[5].lower() for img_path in imgs_paths]
    # 统计classes中每个类别出现的次数，从大到小排序
    classes_count = sorted([(classes.count(c), c) for c in set(classes)], reverse=True)
    print(classes_count)