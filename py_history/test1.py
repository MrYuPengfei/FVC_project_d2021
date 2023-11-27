import os
import argparse
import torch
import cv2
import logging
import numpy as np
from net import *
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
import sys
import math
import json
from dataset import DataSet, TestDataSet, DataSet_Train, Test1, HEVCDataSet
from tensorboardX import SummaryWriter
from drawuvg import uvgdrawplt
from PIL import Image
os.environ["CUDA_VISIBLE_DEVICES"] = " 1 "
torch.backends.cudnn.enabled = True

gpu_num = torch.cuda.device_count()
cur_lr = base_lr = 1e-4#  * gpu_num
train_lambda = 2048
print_step = 100
cal_step = 10
# print_step = 10
warmup_step = 0#  // gpu_num
gpu_per_batch = gpu_num
test_step = 10000#  // gpu_num
tot_epoch = 300
tot_step = 2000000
decay_interval = 1800000
lr_decay = 0.1
logger = logging.getLogger("VideoCompression")
tb_logger = None
global_step = 0
ref_i_dir = geti(train_lambda)


def Var(x):
    return Variable(x.cuda())


def test(global_step, testfull=True):
    with torch.no_grad():
        test_loader = DataLoader(dataset=test_dataset, shuffle=False, num_workers=0, batch_size=1, pin_memory=True)
        net.eval()
        sumbpp = 0
        sumpsnr = 0
        summsssim = 0
        cnt = 0
        for batch_idx, input in enumerate(test_loader):
            if batch_idx % 10 == 0:
                print("testing : %d/%d" % (batch_idx, len(test_loader)))
            input_images = input[0]
            ref_image = input[1]
            ref_bpp = input[2]
            ref_psnr = input[3]
            ref_msssim = input[4]
            seqlen = input_images.size()[1]
            print(seqlen, ref_psnr, ref_msssim, ref_bpp)
            sumbpp += torch.mean(ref_bpp).detach().numpy()
            sumpsnr += torch.mean(ref_psnr).detach().numpy()
            summsssim += torch.mean(ref_msssim).detach().numpy()
            cnt += 1
            for i in range(seqlen):
                input_image = input_images[:, i, :, :, :]
                inputframe, refframe = Var(input_image), Var(ref_image)
                clipped_recon_frame, recon_frame, mse_loss, align_loss, bpp_offsets, total_bpp \
                    = net(inputframe, refframe)
                bpp = torch.mean(total_bpp).cpu().detach().numpy()
                psnr = 10 * (torch.log(1 * 1 / mse_loss) / np.log(10)).cpu().detach().numpy()

                mssim = ms_ssim(clipped_recon_frame.cpu().detach(), input_image, data_range=1.0, size_average=True).numpy()

                print(bpp,  psnr,  mssim)
                sumbpp += bpp
                sumpsnr += psnr
                summsssim += mssim
                cnt += 1
                ref_image = clipped_recon_frame
        log = "global step %d : " % (global_step) + "\n"
        logger.info(log)
        sumbpp /= cnt
        sumpsnr /= cnt
        summsssim /= cnt
        log = "HEVC_Class_D : average bpp : %.6lf, average psnr : %.6lf, average msssim: %.6lf\n" % (sumbpp, sumpsnr, summsssim)
        print(log)
        logger.info(log)
        uvgdrawplt([sumbpp], [sumpsnr], [summsssim], global_step, testfull=testfull)


if __name__ == "__main__":
    model = torch.load("./Deformable_New_train_vimeo_pengfei1.model")
    net = model.cuda()
    net = torch.nn.DataParallel(net.cuda(), device_ids=[0])
    data_transform = transforms.Compose([transforms.ToTensor()])
    global train_dataset, test_dataset
    test_dataset = HEVCDataSet()
    print('testing HEVC_Class_D:')
    test(0, testfull=True)
    exit(0)
