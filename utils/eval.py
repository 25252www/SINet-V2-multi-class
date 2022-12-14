import numpy as np

# 根据类别数生成colormap，范围0-255
def get_colormap(num_classes):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    cmap = np.zeros((num_classes, 3))
    for i in range(0, num_classes):
        r = g = b = 0
        c = i
        for j in range(0, 8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3

        cmap[i, 0] = r
        cmap[i, 1] = g
        cmap[i, 2] = b

    cmap = cmap.astype(np.uint8)
    # 打印cmap
    print('cmap: ', cmap)
    return cmap

# 根据label_mask和colormap生成彩色图，返回rgb整型数组，范围0-255
def decode_segmap(label_mask, colormap, classes):
    # dev: 打印label_mask包含的类别
    classes = [classes[i] for i in np.unique(label_mask)]
    print('pic contains classes: ', classes)
    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for l in range(0, len(colormap)):
        r[label_mask == l] = colormap[l, 0]
        g[label_mask == l] = colormap[l, 1]
        b[label_mask == l] = colormap[l, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b
    return rgb.astype(np.uint8)


def get_confusion_matrix(res, gt, num_classes):
    confusion_matrix = np.zeros((num_classes, num_classes))
    for i in range(num_classes):
        for j in range(num_classes):
            confusion_matrix[i, j] = np.sum((res == i) * (gt == j))
    return confusion_matrix

# 通过混淆矩阵计算iou
def cal_iou(confusion_matrix, num_classes):
    iou = np.zeros(num_classes)
    for i in range(num_classes):
        iou[i] = confusion_matrix[i, i] / (np.sum(confusion_matrix[i, :]) + np.sum(
            confusion_matrix[:, i]) - confusion_matrix[i, i])
    return iou