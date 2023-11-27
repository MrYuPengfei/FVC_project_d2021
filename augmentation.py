# 20210820
# 自定义的pytorch的数据增强函数库
# 加力（augmentation），发动机在短时间内推力超过最大工作状态的过程。发动机加力可缩短飞机起飞滑跑距离。军用机在作战时可借以增大飞行速度、爬升率和机动性。
import random
import torch
import torch.nn.functional as F


def random_crop_and_pad_image_and_labels(image, labels, size):
    # 随机裁切和padding图像和标签 20210820 pengfei备注：
    # 暂时不晓得这三个参数是干什么用的，论文里有没有体现。
    # print(image.size()) # tensor torch.Size([3, 256, 448])
    # print(labels.size())# tensor torch.Size([3, 256, 448])
    # print(size)  # [256,256]
    # exit()
    # image = image.
    combined = torch.Tensor() 
    # print([len(image),len(image[0]),len(image[0][0])],[len(labels),len(labels[0]),len(labels[0][0])])
    # RuntimeError: torch.cat(): Sizes of tensors must match except in dimension 0. Got 448 and 256 in dimension 2 (The offending index is 1)
    # [3, 256, 448] [3, 256, 448]
    # [3, 256, 448] [3, 256, 256]

    torch.cat(tensors = (image, labels), out = combined)
    # print(combined)
    # print(combined.size())
    last_image_dim = (image.size())[0]
    image_shape = image.size()
    combined_pad = F.pad(input=combined, pad= (0, max(size[1], image_shape[2]) - image_shape[2],
                                               0, max(size[0], image_shape[1]) - image_shape[1]))
    freesize0 = random.randint(0, max(size[0], image_shape[1]) - size[0])
    freesize1 = random.randint(0,  max(size[1], image_shape[2]) - size[1])
    combined_crop = combined_pad[:, freesize0:freesize0 + size[0], freesize1:freesize1 + size[1]]
    return combined_crop[:last_image_dim, :, :], combined_crop[last_image_dim:, :, :]
    #  return (combined_crop[:last_image_dim, :, :], combined_crop[last_image_dim:, :, :])


def random_flip(images, labels):
    # augmentation setting....
    horizontal_flip = 1
    vertical_flip = 1
    transforms = 1
    if transforms and vertical_flip and random.randint(0, 1) == 1:
        images = torch.flip(images, [1])
        labels = torch.flip(labels, [1])
    if transforms and horizontal_flip and random.randint(0, 1) == 1:
        images = torch.flip(images, [2])
        labels = torch.flip(labels, [2])
    return images, labels


def random_flip_1(i1, i2, i3, i4, i5):
    # augmentation setting....
    horizontal_flip = 1
    vertical_flip = 1
    transforms = 1
    if transforms and vertical_flip and random.randint(0, 1) == 1:
        i1 = torch.flip(i1, [1])
        i2 = torch.flip(i2, [1])
        i3 = torch.flip(i3, [1])
        i4 = torch.flip(i4, [1])
        i5 = torch.flip(i5, [1])
    if transforms and horizontal_flip and random.randint(0, 1) == 1:
        i1 = torch.flip(i1, [2])
        i2 = torch.flip(i2, [2])
        i3 = torch.flip(i3, [2])
        i4 = torch.flip(i4, [2])
        i5 = torch.flip(i5, [2])
    return i1, i2, i3, i4, i5
# end
