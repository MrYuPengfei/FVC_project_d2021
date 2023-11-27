# 20210820 changed by pengfei
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.utils.data as data
# import argparse
# import cv2
# import sys
# import math
# import json
# # from subnet.EDSC import Network
# from tensorboardX import SummaryWriter
# # from net import Work
# from drawuvg import uvgdrawplt
# # from deform_net1 import Net
# import numpy as np
# from torch.autograd import Variable
# import torch
from PIL import Image
import os
import logging
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import make_grid     # save_image,
from net import *
from deform_net import Net
from dataset import New_train   # DataSet, TestDataSet, DataSet_Train, Test, DataSet_Train1, DataSet_vimeo,
os.environ["CUDA_VISIBLE_DEVICES"] = " 0, 1, 2, 3 "
# win10上没有装cuda，所以下面这行代码的警告先略过
torch.backends.cudnn.enabled = True
# gpu_num = 4
gpu_num = torch.cuda.device_count()
cur_lr = base_lr = 1e-4   # * gpu_num
train_lambda = 2048
print_step = 100
cal_step = 10
# print_step = 10
warmup_step = 0    # // gpu_num
gpu_per_batch = gpu_num
test_step = 10000    # // gpu_num
tot_epoch = 80
tot_step = 2000000
decay_interval = 1800000
lr_decay = 0.1
logger = logging.getLogger("VideoCompression")
tb_logger = None
global_step = 0
ref_i_dir = geti(train_lambda)


def adjust_learning_rate(optimizer, epoch):
    global cur_lr
    global warmup_step
    if epoch < 60:
        lr = base_lr
    elif 60 <= epoch < 70:
        lr = base_lr * lr_decay
    else:
        lr = base_lr * lr_decay * lr_decay
    cur_lr = lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def Var(x):
    return Variable(x.cuda())


def transfer(tensor, nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0):
    grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, range=range, scale_each=scale_each)
            # Make a grid of images.
            # Params:
            # tensor – 4D mini-batch Tensor of shape (B x C x H x W) or a list of images all of the same size.
            # nrow – Number of images displayed in each row of the grid. The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
            # padding – amount of padding. Default: ``2``.
            # normalize – If True, shift the image to the range (0, 1), by the min and max values specified by :attr:`range`. Default: ``False``.
            # value_range – tuple (min, max) where min and max are numbers, then these numbers are used to normalize the image. By default, min and max are computed from the tensor.
            # scale_each – If ``True``, scale each image in the batch of images separately rather than the (min, max) over all images. Default: ``False``.
            # pad_value – Value for the padded pixels. Default: ``0``.
            # Returns:
            # the tensor containing grid of images.
            # 制作一个图像网格。
            # 参数:
            # 张量-四维迷你批张量的形状(B x C x H x W)或所有相同大小的图像列表。
            # nrow -网格中每行显示的图像数量。最终的网格大小是' ' (B / nrow, nrow) ' '。默认值:' ' 8 ' '。
            # 填充-填充的数量。默认值:' ' 2 ' '。
            # normalize -如果为True，将图像移动到范围(0,1)，由:attr: ' range '指定的最小和最大值。默认值:' '假' '。
            # Value_range -元组(min, max)，其中min和max是数字，然后使用这些数字来规范化图像。默认情况下，最小值和最大值都是从张量计算出来的。
            # scale_each -如果' ' True ' '，将批图像中的每个图像分别缩放，而不是在所有图像上缩放(min, max)。默认值:' '假' '。
            # pad_value -填充像素的值。默认值:' ' 0 ' '。
            # 返回:
            # 包含图像网格的张量。
    ndarr = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    return im


def train(epoch, global_step):
    print("epoch", epoch)
    global gpu_per_batch
    train_loader = DataLoader(dataset=train_dataset, shuffle=True, num_workers=gpu_num, batch_size=gpu_num*5, pin_memory=True, drop_last=True)
    net.train()
    global optimizer
    bat_cnt = 0
    cal_cnt = 0
    sumloss = 0
    sumpsnr = 0
    sumalignpsnr = 0
    sumbpp = 0
    sumrefine_psnr = 0
    sumrecon_psnr = 0
    sumpredict_psnr = 0
    sumbpp_offsets = 0
    sumbpp_res = 0
    t0 = datetime.datetime.now()
    for batch_idx, input in enumerate(train_loader):
        optimizer.zero_grad()
        global_step += 1
        bat_cnt += 1
        ref, input = Var(input[0]), Var(input[1])
        clipped_recon_frame, recon_frame, mse_loss, align_loss, bpp_offsets, total_bpp = net(input, ref)
        mse_loss, align_loss, bpp_offsets, bpp =\
            torch.mean(mse_loss), torch.mean(align_loss), torch.mean(bpp_offsets), torch.mean(total_bpp)
        distribution_loss = bpp
        if epoch < 50:
            a, b = 1, 0.1
        else:
            a, b = 1, 0
        distortion = a * mse_loss + b * align_loss
        rd_loss = distortion * train_lambda + distribution_loss
        rd_loss.backward()
        optimizer.step()
        if global_step % cal_step == 0:
            cal_cnt += 1
            if mse_loss > 0:
                psnr = 10 * (torch.log(1 * 1 / mse_loss) / np.log(10)).cpu().detach().numpy()
            else:
                psnr = 100
            if align_loss > 0:
                alignpsnr = 10 * (torch.log(1 * 1 / align_loss) / np.log(10)).cpu().detach().numpy()
            else:
                alignpsnr = 100
            loss_ = rd_loss.cpu().detach().numpy()
            sumloss += loss_
            sumpsnr += psnr
            sumalignpsnr += alignpsnr
            sumbpp += bpp.cpu().detach()
        if (batch_idx % print_step) == 0 and bat_cnt > 1:
            t1 = datetime.datetime.now()
            deltatime = t1 - t0
            log1 = 'Train Epoch : {:02} [{:4}/{:4} ({:3.0f}%)] Avgloss:{:.6f} lr:{} time:{} bpp:{}'
            log1 = log1.format(epoch,
                               batch_idx,
                               len(train_loader),
                               100. * batch_idx / len(train_loader),
                               sumloss / cal_cnt,
                               cur_lr,
                               (deltatime.seconds + 1e-6 * deltatime.microseconds) / bat_cnt,
                               sumbpp / cal_cnt)
            print(log1)
            log2 = 'details :  align_psnr : {:.2f}  psnr : {:.2f}'.format(
                sumalignpsnr / cal_cnt, sumpsnr / cal_cnt)
            print(log2)
            bat_cnt = 0
            cal_cnt = 0
            sumbpp = sumloss  = sumpsnr = sumalignpsnr = sumrefine_psnr = sumrecon_psnr = sumpredict_psnr = 0
            t0 = t1
    log = 'Train Epoch : {:02} Loss:\t {:.6f}\t lr:{}'.format(epoch, sumloss / bat_cnt, cur_lr)
    logger.info(log)
    return global_step


if __name__ == "__main__":

    model = Net()
    net = model.cuda()
    net = torch.nn.DataParallel(net.cuda(), list(range(gpu_num)))
    bp_parameters = net.parameters()
    # bp_parameters = filter(lambda p: p.requires_grad, net.parameters())
    optimizer = optim.Adam(bp_parameters, lr=base_lr, weight_decay=5e-7)
    data_transform = transforms.Compose([transforms.ToTensor()])
    global train_dataset, test_dataset
    train_dataset = New_train()
    # path_checkpoint = "./snapshot/ckpt_best_40.pth"  # 断点路径
    # checkpoint = torch.load(path_checkpoint)  # 加载断点
    # net.load_state_dict(checkpoint['net'])  # 加载模型可学习参数
    # optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
    # stepoch = checkpoint['epoch']  # 设置开始的epoch
    stepoch = global_step // (train_dataset.__len__() // (gpu_per_batch))
    for epoch in range(stepoch, tot_epoch):
        adjust_learning_rate(optimizer, epoch)
        if epoch % 10 == 0:
            checkpoint = {
                "net": net.state_dict(),
                'optimizer': optimizer.state_dict(),
                "epoch": epoch
            }
            torch.save(checkpoint, './snapshot/ckpt_best_%s.pth' % (str(epoch)))
        global_step = train(epoch, global_step)
    # torch.save(net, "Deformable_1.model", _use_new_zipfile_serialization=False)
    torch.save(net, "Deformable_New_train_vimeo_pengfei1.model", _use_new_zipfile_serialization=False)
# end !
