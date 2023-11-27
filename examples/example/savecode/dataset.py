# 20210820 version1 pengfei
# 20210823 version2 pengfei: 使用该脚本前，仔细检查脚本中的文件路径。
from subnet.basics import *
from subnet.ms_ssim_torch import ms_ssim
from augmentation import random_flip, random_crop_and_pad_image_and_labels    # , random_flip_1
import os
import torch
from PIL import Image
import imageio
import numpy as np
import torch.utils.data as data
# from torchvision.utils import save_image
# import logging
# import cv2
# from os.path import join, exists
# import math
# import random
# import sys
# import json
# import random


class TestDataSet(data.Dataset):
    def __init__(self, root="/data/UVG", refdir='H265Q20', testfull=False):
        # with open(filelist) as f:
        #     folders = f.readlines()
        self.ref = []
        self.refbpp = []
        self.input_p = []
        self.hevcclass = []
        AllIbpp = self.getbpp(refdir)
        print(AllIbpp)
        seqIbpp = AllIbpp[0]
        imlist = os.listdir(root+"/images/Beauty")
        imlist = sorted(imlist)
        cnt = 0
        for im in imlist:
            if im[-4:] == '.png':
                cnt += 1
        if testfull:
            framerange = cnt // 12
        else:
            framerange = 1
        for i in range(framerange):
            refpath = root + "/videos_crop/Beauty_1920x1024_120fps_420_8bit_YUV/H265L20/" \
                      + 'im' + str(i * 12 + 1).zfill(4)+'.png'
            inputpath_p = []
            for j in range(11):
                inputpath_p.append(root + "/images/Beauty/" + 'im' + str(i * 12 + j + 1).zfill(3)+'.png')
            self.ref.append(refpath)
            self.input_p.append(inputpath_p)
            self.refbpp.append(seqIbpp)

    def getbpp(self, ref_i_folder):
        Ibpp = None
        if ref_i_folder == 'H265Q20':
            print('use H265Q20')
            Ibpp = [0.44844653406234286]# you need to fill bpps after generating crf=20 basketball
        elif ref_i_folder == 'H265Q23':
            print('use H265Q23')
            Ibpp = [0.30846962983911513]# you need to fill bpps after generating crf=23 basketball
        elif ref_i_folder == 'H265Q26':
            print('use H265Q26')
            Ibpp = [0.21394662833081954]# you need to fill bpps after generating crf=26 basketball
        elif ref_i_folder == 'H265Q29':
            print('use H265Q29')
            Ibpp = [0.14701875942684767]# you need to fill bpps after generating crf=29 basketball
        else:
            print('cannot find ref : ', ref_i_folder)
            exit()
        if len(Ibpp) == 0:
            print('You need to generate I frames and fill the bpps above!')
            exit()
        return Ibpp

    def __len__(self):
        return len(self.ref)

    def __getitem__(self, index):
        print(index)
        ref_image = imageio.imread(self.ref[index]).transpose(2, 0, 1).astype(np.float32) / 255.0
        h = (ref_image.shape[1] // 64) * 64
        w = (ref_image.shape[2] // 64) * 64
        ref_image = np.array(ref_image[:, :h, :w])
        input_images_p = []
        refpsnr = None
        refmsssim = None
        for filename in self.input_p[index]:
            input_image_p = (imageio.imread(filename).transpose(2, 0, 1)[:, :h, :w]).astype(np.float32) / 255.0
            # print(input_image.shape)
            if refpsnr is None:
                refpsnr = CalcuPSNR(input_image_p, ref_image)
                refmsssim = ms_ssim(torch.from_numpy(input_image_p[np.newaxis, :]), torch.from_numpy(ref_image[np.newaxis, :]), data_range=1.0).numpy()
            else:
                input_images_p.append(input_image_p[:, :h, :w])

        input_images_p = np.array(input_images_p)
        return input_images_p, ref_image, self.refbpp[index], refpsnr, refmsssim


class HEVCDataSet(data.Dataset):
    # def __init__(self, root="/data1/Data/HEVC_dataset/Class_D", filelist="./filelists/D.txt", refdir='H265L20', testfull=True):
    def __init__(self, root="/data1/Data/HEVC_dataset_original",
                 filelist="/data1/Data/HEVC_dataset_original/ls.txt", refdir='H265L20', testfull=True):
        # os.system(r"ls " + root + r"/*yuv >" + filelist)    # 删除了这个用法，因为该方法导致文件名称里包含了路径
        # filelist 是手工导入的。在linux上使用 ls *yuv >ls.txt
        with open(filelist) as f:
            folders = f.readlines()
        self.ref = []
        self.refbpp = []
        self.input = []
        self.hevcclass = []
        AllIbpp = self.getbpp(refdir)
        print(AllIbpp)
        ii = 0
        for folder in folders:
            seq = folder.rstrip()
            seqIbpp = AllIbpp[ii]
            imlist = os.listdir(os.path.join(root, seq))
            cnt = 0
            for im in imlist:
                if im[-4:] == '.png':
                    cnt += 1
            if testfull:
                framerange = cnt // 10
            else:
                framerange = 1
            for i in range(framerange):
                refpath = os.path.join(root, str(seq) + '/' + str(refdir) + '/im' + str(i * 10 + 1).zfill(4) + '.png')
                inputpath = []
                for j in range(10):
                    inputpath.append(os.path.join(root, seq, 'im' + str(i * 10 + j + 1).zfill(3) + '.png'))
                self.ref.append(refpath)
                self.refbpp.append(seqIbpp)
                self.input.append(inputpath)
            ii += 1

    def getbpp(self, ref_i_folder):
        Ibpp = None
        if ref_i_folder == 'H265L20':
            print('use H265Q20')
            # Ibpp = [1.178423584, 1.561825019, 1.371733643, 0.851354472, 1.404278056]  # Class B
            # Ibpp = [1.334076666, 1.233124785, 1.996887019, 1.253125715]  # Class C, crf=20
            Ibpp = [1.215820513, 2.301389423, 2.102279647, 1.437550748]  # Class D,  crf=20
        elif ref_i_folder == 'H265L23':
            print('use H265Q23')
            Ibpp = [1.07225491, 0.660947184, 0.850405518, 0.508370633, 0.949806552]
            # you need to fill bpps after generating crf=23
        elif ref_i_folder == 'H265L26':
            print('use H265Q26')
            Ibpp = [0.635096096, 0.35697168, 0.505873861, 0.315825229, 0.628444926]
            # you need to fill bpps after generating crf=26
        elif ref_i_folder == 'H265L29':
            print('use H265Q29')
            Ibpp = [0.388928562, 0.217248291, 0.323592285, 0.214755249, 0.421503025]
            # you need to fill bpps after generating crf=29
        else:
            print('cannot find ref : ', ref_i_folder)
            exit()
        if len(Ibpp) == 0:
            print('You need to generate I frames and fill the bpps above!')
            exit()
        return Ibpp

    def __len__(self):
        return len(self.input)

    def __getitem__(self, index):
        ref_image = imageio.imread(self.ref[index]).transpose(2, 0, 1).astype(np.float32) / 255.0
        h = (ref_image.shape[1] // 64) * 64
        w = (ref_image.shape[2] // 64) * 64
        ref_image = np.array(ref_image[:, :h, :w])
        input_images = []
        refpsnr = None
        refmsssim = None
        for filename in self.input[index]:
            # print(filename)
            input_image = (imageio.imread(filename).transpose(2, 0, 1)[:, :h, :w]).astype(np.float32) / 255.0
            if refpsnr is None:
                refpsnr = CalcuPSNR(input_image, ref_image)
                refmsssim = ms_ssim(torch.from_numpy(input_image[np.newaxis, :]),
                                    torch.from_numpy(ref_image[np.newaxis, :]), data_range=1.0).numpy()
            else:
                input_images.append(input_image[:, :h, :w])
        input_images = np.array(input_images)
        return input_images, ref_image, self.refbpp[index], refpsnr, refmsssim


class DataSet(data.Dataset):
    def __init__(self, path="/data1/vimeo_septuplet/video-90k.txt", im_height=256, im_width=256):
        self.image_input_b_list, self.image_input_p_list, self.image_ref_list = self.get_vimeo(filefolderlist=path)
        self.im_height = im_height
        self.im_width = im_width
        self.featurenoise = torch.zeros([out_channel_M, self.im_height // 16, self.im_width // 16])
        self.znoise = torch.zeros([out_channel_N, self.im_height // 64, self.im_width // 64])
        self.mvnois = torch.zeros([out_channel_mv, self.im_height // 16, self.im_width // 16])
        # print("dataset find image: ", len(self.image_input_list))
        print("dataset find image: ", len(self.image_input_p_list))

    def get_vimeo(self, rootdir="/data1/Data/vimeo_septuplet/sequences/",
                  filefolderlist="/data1/Data/vimeo_septuplet/video-90k.txt"):
        with open(filefolderlist) as f:
            data = f.readlines()
        fns_train_input_b = []
        fns_train_input_p = []
        fns_train_ref = []
        for n, line in enumerate(data, 1):
            y = os.path.join(rootdir, line.rstrip())
            fns_train_input_b += [y]
            refnumber = int(y[-5:-4]) - 1
            refname = y[0:-5] + str(refnumber) + '.png'
            fns_train_ref += [refname]
            refnumber_p = int(y[-5:-4]) - 1
            refname_p = y[0:-5] + str(refnumber_p) + '.png'
            fns_train_input_p += [refname_p]
        return fns_train_input_b, fns_train_input_p, fns_train_ref

    def __len__(self):
        return len(self.image_input_p_list)

    def __getitem__(self, index):
        input_image_b = imageio.imread(self.image_input_b_list[index])
        input_image_p = imageio.imread(self.image_input_p_list[index])
        ref_image = imageio.imread(self.image_ref_list[index])
        input_image_b = input_image_b.astype(np.float32) / 255.0
        input_image_p = input_image_p.astype(np.float32) / 255.0
        ref_image = ref_image.astype(np.float32) / 255.0
        input_image_b = input_image_b.transpose(2, 0, 1)
        input_image_p = input_image_p.transpose(2, 0, 1)
        ref_image = ref_image.transpose(2, 0, 1)
        input_image_b = torch.from_numpy(input_image_b).float()
        input_image_p = torch.from_numpy(input_image_p).float()
        ref_image = torch.from_numpy(ref_image).float()
        # input_image_b, input_image_p, ref_image \
        #     = random_crop_and_pad_image_and_labels(input_image_b, input_image_p,
        #                                            ref_image, [self.im_height, self.im_width])
        # input_image_b, input_image_p, ref_image \
        #     = random_flip(input_image_b, input_image_p, ref_image)
        input_image_b, ref_image = random_crop_and_pad_image_and_labels(input_image_b, ref_image, [self.im_height, self.im_width])
        input_image_p, ref_image = random_crop_and_pad_image_and_labels(input_image_p, ref_image, [self.im_height, self.im_width])
        input_image_b, ref_image = random_flip(input_image_b, ref_image)
        input_image_p, ref_image = random_flip(input_image_p, ref_image)
        quant_noise_feature, quant_noise_z, quant_noise_mv = torch.nn.init.uniform_(torch.zeros_like(self.featurenoise), -0.5, 0.5), torch.nn.init.uniform_(torch.zeros_like(self.znoise), -0.5, 0.5), torch.nn.init.uniform_(torch.zeros_like(self.mvnois), -0.5, 0.5)
        return input_image_b, input_image_p, ref_image, quant_noise_feature, quant_noise_z, quant_noise_mv


class DataSet_Train(data.Dataset):
    def __init__(self, path="CUHK.txt", im_height=256, im_width=256, transforms=None):
        self.image_input_list_b, self.image_input_list_p, self.image_ref_list = self.get_vimeo(filefolderlist=path)
        self.im_height = im_height
        self.im_width = im_width
        self.transforms = transforms
        # self.featurenoise = torch.zeros([out_channel_M, self.im_height // 16, self.im_width // 16])
        # self.znoise = torch.zeros([out_channel_N, self.im_height // 64, self.im_width // 64])
        # self.mvnois = torch.zeros([out_channel_mv, self.im_height // 16, self.im_width // 16])
        print("dataset find image: ", len(self.image_input_list_b))

    def get_vimeo(self, filefolderlist="CUHK.txt"):
        with open(filefolderlist) as f:
            data = f.readlines()

        fns_train_input_b = []
        fns_train_input_p = []
        fns_train_ref = []

        for n, line in enumerate(data, 0):
            if n % 2 == 0:
                y = os.path.join(line.rstrip())
                fns_train_ref += [y]

                if n < 3000:
                    later = y[31:]
                    later_ = later[:-4]
                    refnumber_b = int(later_) + 1
                    if refnumber_b != 299:
                        refnumber_b %= 299
                    refname_b = y[0:31] + str(refnumber_b) + '.png'
                    fns_train_input_b += [refname_b]

                    refnumber_p = int(later_) + 2
                    if refnumber_p != 299:
                        refnumber_p %= 299
                    refname_p = y[0:31] + str(refnumber_p) + '.png'
                    fns_train_input_p += [refname_p]
                else:
                    later = y[32:]
                    later_ = later[:-4]
                    refnumber_b = int(later_) + 1
                    if refnumber_b != 299:
                        refnumber_b %= 299
                    refname_b = y[0:32] + str(refnumber_b) + '.png'
                    fns_train_input_b += [refname_b]

                    refnumber_p = int(later_) + 2
                    if refnumber_p != 299:
                        refnumber_p %= 299
                    refname_p = y[0:32] + str(refnumber_p) + '.png'
                    fns_train_input_p += [refname_p]

        # print(fns_train_input_b[1000], fns_train_input_p[1000], fns_train_ref[1000])
        # print(fns_train_input_b[1001], fns_train_input_p[1001], fns_train_ref[1001])
        # print(fns_train_input_b[1002], fns_train_input_p[1002], fns_train_ref[1002])
        # print(fns_train_input_b[1003], fns_train_input_p[1003], fns_train_ref[1003])
        # print(fns_train_input_b[4], fns_train_input_p[4], fns_train_ref[4])
        # print(fns_train_input_b[5], fns_train_input_p[5], fns_train_ref[5])
        # print(fns_train_input_b[6], fns_train_input_p[6], fns_train_ref[6])
        # print(fns_train_input_b[7], fns_train_input_p[7], fns_train_ref[7])
        return fns_train_input_b, fns_train_input_p, fns_train_ref

    def __len__(self):
        return len(self.image_ref_list)

    def __getitem__(self, index):
        input_image_b = Image.open(self.image_input_list_b[index]).crop((0, 56, 320, 184))
        input_image_p = np.array(Image.open(self.image_input_list_p[index]).crop((0, 56, 320, 184)))
        ref_image = np.array(Image.open(self.image_ref_list[index]).crop((0, 56, 320, 184)))

        # input_image_b = input_image_b.astype(np.float32) / 255.0
        input_image_p = input_image_p.astype(np.float32) / 255.0
        ref_image = ref_image.astype(np.float32) / 255.0

        # input_image_b = input_image_b.transpose(2, 0, 1)
        input_image_p = input_image_p.transpose(2, 0, 1)
        ref_image = ref_image.transpose(2, 0, 1)

        # input_image_b = torch.from_numpy(input_image_b).float()
        input_image_p = torch.from_numpy(input_image_p).float()
        ref_image = torch.from_numpy(ref_image).float()
        if self.transforms is not None:
            input_image_b = self.transforms(input_image_b)
        # print(ref_image.shape)
        # input_image_b, input_image_p, ref_image = random_crop_and_pad_image_and_labels(input_image_b, input_image_p, ref_image, [self.im_height, self.im_width])
        # input_image_b, input_image_p, ref_image = random_flip(input_image_b, input_image_p, ref_image)

        # quant_noise_feature, quant_noise_z, quant_noise_mv = torch.nn.init.uniform_(torch.zeros_like(self.featurenoise), -0.5, 0.5), torch.nn.init.uniform_(torch.zeros_like(self.znoise), -0.5, 0.5), torch.nn.init.uniform_(torch.zeros_like(self.mvnois), -0.5, 0.5)
        return input_image_b, input_image_p, ref_image


class DataSet_Train1(data.Dataset):
    def __init__(self, root="/home/zhaoyu/HEVC_dataset/Class_B", filelist="./filelists/B.txt", transforms=None):
        with open(filelist) as f:
            folders = f.readlines()
        self.transforms = transforms
        self.fns_train_input_b = []
        self.fns_train_input_p = []
        self.fns_train_ref = []
        for folder in folders:
            seq = folder.rstrip()
            imlist = os.listdir(os.path.join(root, seq))
            cnt = 0
            for im in imlist:
                if im[-4:] == '.png':
                    cnt += 1
            for i in range(1, cnt):
                if i % 2 == 0:
                    self.fns_train_ref.append(os.path.join(root, seq, 'im' + str(i).zfill(3) + '.png'))
                    self.fns_train_input_b.append(os.path.join(root, seq, 'im' + str(i+1).zfill(3) + '.png'))
                    self.fns_train_input_p.append(os.path.join(root, seq, 'im' + str(i+2).zfill(3) + '.png'))

    def __len__(self):
        return len(self.fns_train_ref)

    def __getitem__(self, index):
        input_image_b = np.array(Image.open(self.fns_train_input_b[index]))
        input_image_p = np.array(Image.open(self.fns_train_input_p[index]))
        ref_image = np.array(Image.open(self.fns_train_ref[index]))

        input_image_b = input_image_b.astype(np.float32) / 255.0
        input_image_p = input_image_p.astype(np.float32) / 255.0
        ref_image = ref_image.astype(np.float32) / 255.0

        input_image_b = input_image_b.transpose(2, 0, 1)
        input_image_p = input_image_p.transpose(2, 0, 1)
        ref_image = ref_image.transpose(2, 0, 1)

        input_image_b = torch.from_numpy(input_image_b).float()
        input_image_p = torch.from_numpy(input_image_p).float()
        ref_image = torch.from_numpy(ref_image).float()
        # if self.transforms is not None:
        #     input_image_b = self.transforms(input_image_b)

        return input_image_b, input_image_p, ref_image


class DataSet_vimeo(data.Dataset):
    def __init__(self, path="./filelists/video-90k_3.txt"):
        self.image_input_list_p, self.image_input_list_b, self.image_ref_list = self.get_vimeo(filefolderlist=path)
        print("dataset find image: ", len(self.image_input_list_b))

    def get_vimeo(self, rootdir="/home/zhaoyu/video-90k/sequences", filefolderlist=None):
        with open(filefolderlist) as f:
            data = f.readlines()
        fns_train_input_p = []
        fns_train_input_b = []
        fns_train_ref = []
        for n, line in enumerate(data, 1):
            y = os.path.join(rootdir, line.rstrip())
            fns_train_input_p += [y]
            refnumber = int(y[-5:-4]) - 2
            bnumber = int(y[-5:-4]) - 1
            refname = y[0:-5] + str(refnumber) + '.png'
            bname = y[0:-5] + str(bnumber) + '.png'
            fns_train_ref += [refname]
            fns_train_input_b += [bname]
        return fns_train_input_p, fns_train_input_b, fns_train_ref

    def __len__(self):
        return len(self.image_input_list_b)

    def __getitem__(self, index):
        input_image_p = imageio.imread(self.image_input_list_p[index])
        input_image_b = imageio.imread(self.image_input_list_b[index])
        ref_image = imageio.imread(self.image_ref_list[index])

        input_image_p = input_image_p.astype(np.float32) / 255.0
        input_image_b = input_image_b.astype(np.float32) / 255.0
        ref_image = ref_image.astype(np.float32) / 255.0
        input_image_p = input_image_p.transpose(2, 0, 1)
        input_image_b = input_image_b.transpose(2, 0, 1)
        ref_image = ref_image.transpose(2, 0, 1)
        input_image_p = torch.from_numpy(input_image_p).float()
        input_image_b = torch.from_numpy(input_image_b).float()
        ref_image = torch.from_numpy(ref_image).float()
        return input_image_p, input_image_b, ref_image


class Test(data.Dataset):
    def __init__(self, path="./filelists/HEVC_ClassE/KristenAndSara.txt", im_height=256, im_width=256, transforms=None):
        self.image_input_list_b, self.image_input_list_p, self.image_ref_list = self.get_vimeo(filefolderlist=path)
        self.im_height = im_height
        self.im_width = im_width
        self.transforms = transforms
        print("dataset find image: ", len(self.image_input_list_b))

    def get_vimeo(self, filefolderlist=None):
        with open(filefolderlist) as f:
            data = f.readlines()
        fns_train_input_b = []
        fns_train_input_p = []
        fns_train_ref = []
        for n, line in enumerate(data, 0):
            if n % 2 == 0:
                y = os.path.join(line.rstrip())
                fns_train_ref += [y]
                later = y[63:]
                later_ = later[:-4]
                print(later_)
                refnumber_b = int(later_) + 1
                refname_b = y[0:63] + str(refnumber_b).zfill(3) + '.png'
                fns_train_input_b += [refname_b]
                refnumber_p = int(later_) + 2
                refname_p = y[0:63] + str(refnumber_p).zfill(3) + '.png'
                fns_train_input_p += [refname_p]
        return fns_train_input_b, fns_train_input_p, fns_train_ref

    def __len__(self):
        return len(self.image_ref_list) - 2

    def __getitem__(self, index):
        input_image_b = np.array(Image.open(self.image_input_list_b[index]))
        input_image_p = np.array(Image.open(self.image_input_list_p[index]))
        ref_image = np.array(Image.open(self.image_ref_list[index]))
        input_image_b = input_image_b.astype(np.float32) / 255.0
        input_image_p = input_image_p.astype(np.float32) / 255.0
        ref_image = ref_image.astype(np.float32) / 255.0
        input_image_b = input_image_b.transpose(2, 0, 1)
        input_image_p = input_image_p.transpose(2, 0, 1)
        ref_image = ref_image.transpose(2, 0, 1)
        input_image_b = torch.from_numpy(input_image_b).float()
        input_image_p = torch.from_numpy(input_image_p).float()
        ref_image = torch.from_numpy(ref_image).float()
        # if self.transforms is not None:
        #     input_image_b = self.transforms(input_image_b)
        # print(ref_image.shape)
        # input_image_b, input_image_p, ref_image = random_crop_and_pad_image_and_labels(input_image_b, input_image_p, ref_image, [self.im_height, self.im_width])
        # input_image_b, input_image_p, ref_image = random_flip(input_image_b, input_image_p, ref_image)
        # quant_noise_feature, quant_noise_z, quant_noise_mv = torch.nn.init.uniform_(torch.zeros_like(self.featurenoise), -0.5, 0.5), torch.nn.init.uniform_(torch.zeros_like(self.znoise), -0.5, 0.5), torch.nn.init.uniform_(torch.zeros_like(self.mvnois), -0.5, 0.5)
        return input_image_b, input_image_p, ref_image


class Test1(data.Dataset):
    def __init__(self, path="./filelists/video-90k_4.txt", im_height=256, im_width=256, transforms=None):
        self.image_input_list_p, self.image_ref_list = self.get_vimeo(filefolderlist=path)
        self.im_height = im_height
        self.im_width = im_width
        self.transforms = transforms
        print("dataset find image: ", len(self.image_input_list_p))

    def get_vimeo(self, filefolderlist=None):
        with open(filefolderlist) as f:
            data = f.readlines()
        fns_train_input_p = []
        fns_train_ref = []
        for n, line in enumerate(data, 0):
            root = "/home/zhaoyu/video-90k/sequences/"
            y = os.path.join(root, line.rstrip())
            fns_train_input_p += [y]
            later = y[46:]
            later_ = later[:-4]
            refnumber_p = int(later_) - 1
            refname_p = y[0:46] + str(refnumber_p).zfill(1) + '.png'
            fns_train_ref += [refname_p]
        return fns_train_input_p, fns_train_ref

    def __len__(self):
        return len(self.image_ref_list) - 1

    def __getitem__(self, index):
        input_image_p = np.array(Image.open(self.image_input_list_p[index]))
        ref_image = np.array(Image.open(self.image_ref_list[index]))
        # input_image_b = input_image_b.astype(np.float32) / 255.0
        input_image_p = input_image_p.astype(np.float32) / 255.0
        ref_image = ref_image.astype(np.float32) / 255.0
        # input_image_b = input_image_b.transpose(2, 0, 1)
        input_image_p = input_image_p.transpose(2, 0, 1)
        ref_image = ref_image.transpose(2, 0, 1)
        # input_image_b = torch.from_numpy(input_image_b).float()
        input_image_p = torch.from_numpy(input_image_p).float()
        ref_image = torch.from_numpy(ref_image).float()
        return input_image_p, ref_image


class New_train(data.Dataset):
    # def __init__(self, path="./filelists/video-90k_4.txt"):
    def __init__(self, path="./filelists/video-90k_4.txt"):
        self.image_input_list_1, self.image_input_list_2 = self.get_vimeo(filefolderlist=path)
        print("dataset find image: ", len(self.image_input_list_1))

    def get_vimeo(self, rootdir="/data1/Data/vimeo_septuplet/sequences", filefolderlist=None):
        with open(filefolderlist) as f:
            data = f.readlines()
        fns_train_input_1 = []
        fns_train_input_2 = []
        # fns_train_input_3 = []
        # fns_train_input_4 = []
        # fns_train_input_5 = []
        for n, line in enumerate(data, 1):
            y = os.path.join(rootdir, line.rstrip())
            
            # print(y)
            # print(y[-5:-4])
            # 20210824
            # /data1/Data/vimeo_septuplet/sequences/00001/0001
            # /
            # exit()
            number = int(y[-5:-4]) # 此行代码有错误?，需要改正?
            # number = y[-4:]
            # if number == 1 | 2 | 3:
            #     fns_train_input_1 += [y[0:-5] + str(number) + '.png']
            #     fns_train_input_2 += [y[0:-5] + str(number + 1) + '.png']
            #     fns_train_input_3 += [y[0:-5] + str(number + 2) + '.png']
            #     fns_train_input_4 += [y[0:-5] + str(number + 3) + '.png']
            #     fns_train_input_5 += [y[0:-5] + str(number + 4) + '.png']
            # if number == 4:
            #     fns_train_input_1 += [y[0:-5] + str(number - 2) + '.png']
            #     fns_train_input_2 += [y[0:-5] + str(number - 1) + '.png']
            #     fns_train_input_3 += [y[0:-5] + str(number) + '.png']
            #     fns_train_input_4 += [y[0:-5] + str(number + 1) + '.png']
            #     fns_train_input_5 += [y[0:-5] + str(number + 2) + '.png']
            # if number == 5 | 6 | 7:
            #     fns_train_input_1 += [y[0:-5] + str(number - 4) + '.png']
            #     fns_train_input_2 += [y[0:-5] + str(number - 3) + '.png']
            #     fns_train_input_3 += [y[0:-5] + str(number - 2) + '.png']
            #     fns_train_input_4 += [y[0:-5] + str(number - 1) + '.png']
            #     fns_train_input_5 += [y[0:-5] + str(number) + '.png']
            fns_train_input_2 += [y[0:-5] + str(number) + '.png']
            fns_train_input_1 += [y[0:-5] + str(number - 1) + '.png']
        return fns_train_input_1, fns_train_input_2

    def __len__(self):
        return len(self.image_input_list_1)

    def __getitem__(self, index):
        input_image_1 = imageio.imread(self.image_input_list_1[index])
        input_image_2 = imageio.imread(self.image_input_list_2[index])
        # input_image_3 = imageio.imread(self.image_input_list_3[index])[:, 96:96+256, :]
        # input_image_4 = imageio.imread(self.image_input_list_4[index])[:, 96:96+256, :]
        # input_image_5 = imageio.imread(self.image_input_list_5[index])[:, 96:96+256, :]
        input_image_1 = input_image_1.astype(np.float32) / 255.0
        input_image_2 = input_image_2.astype(np.float32) / 255.0
        # input_image_3 = input_image_3.astype(np.float32) / 255.0
        # input_image_4 = input_image_4.astype(np.float32) / 255.0
        # input_image_5 = input_image_5.astype(np.float32) / 255.0
        input_image_1 = input_image_1.transpose(2, 0, 1)
        input_image_2 = input_image_2.transpose(2, 0, 1)
        # input_image_3 = input_image_3.transpose(2, 0, 1)
        # input_image_4 = input_image_4.transpose(2, 0, 1)
        # input_image_5 = input_image_5.transpose(2, 0, 1)
        input_image_1 = torch.from_numpy(input_image_1).float()
        input_image_2 = torch.from_numpy(input_image_2).float()
        # input_image_3 = torch.from_numpy(input_image_3).float()
        # input_image_4 = torch.from_numpy(input_image_4).float()
        # input_image_5 = torch.from_numpy(input_image_5).float()
        input_image_1, input_image_2 = random_crop_and_pad_image_and_labels(input_image_1, input_image_2, [256, 256])
        input_image_1, input_image_2 = random_flip(input_image_1, input_image_2)
        return input_image_1, input_image_2
# 20210820 end !
