# 20210821
# import numpy as np
# import os
# import torch
# import cv2
# from torch.autograd import Variable
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.utils.data import DataLoader
# import sys
# import math
# import torch.nn.init as init
# import logging
# from torch.nn.parameter import Parameter
# import torch
# from torch.autograd import Variable, Function
# from subnet.EDSC import Network
# from subnet import *
# from subnet.BiConvLSTM5 import BiConvLSTM
# from examples.example.savecode.subnet.synthesis import *

from subnet.analysis_mv import *
from subnet.analysis import *
from subnet.synthesis_mv import *
from subnet.synthesis import *
# from dcn.deform_conv import *
from torchvision.ops import DeformConv2d

class Res_Block(nn.Module):
    def __init__(self):
        super(Res_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        res = self.conv1(x)
        res = self.relu(res)
        res = self.conv2(res)
        return x + res


class Align_near(nn.Module):
    def __init__(self):
        super(Align_near, self).__init__()

        self.conv_first = nn.Conv2d(3, 64, 5, stride=2, padding=1)
        self.residual_layer = self.make_layer(Res_Block, 3)
        self.relu = nn.ReLU(inplace=True)

        # deformable
        self.cr1 = nn.Conv2d(128, 64, (3, 3), padding=1, bias=True)
        self.cr2 = nn.Conv2d(64, 64, (3, 3), padding=1, bias=True)
        self.conv1 = nn.Conv2d(128, 64, (3, 3), (1, 1), 1)
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), 1)
        self.off2d_1 = nn.Conv2d(64, 18 * 8, (3, 3), padding=1, bias=True)
        self.dconv_1 = DeformConv2d(64, 64, 3, padding=1, groups=8)
        self.off2d_2 = nn.Conv2d(64, 18 * 8, (3, 3), padding=1, bias=True)
        self.deconv_2 = DeformConv2d(64, 64, 3, padding=1, groups=8)
        self.off2d_3 = nn.Conv2d(64, 18 * 8, (3, 3), padding=1, bias=True)
        self.deconv_3 = DeformConv2d(64, 64, 3, padding=1, groups=8)
        self.off2d = nn.Conv2d(64, 18 * 8, (3, 3), padding=1, bias=True)
        self.dconv = DeformConv2d(64, 64, 3, padding=1, groups=8)
        self.recon_lr = nn.Conv2d(64, 3, (3, 3), padding=1, bias=True)

        # xavier initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def generator(self, ref_fea, input_fea):
        # print("---",input_fea.shape, ref_fea.shape)
        fea = torch.cat([input_fea, ref_fea], dim=1)
        offset = self.cr2(self.cr1(fea))
        return offset

    def align(self, offset, ref_fea):
        offset = self.off2d(offset)
        aligned_fea1 = self.dconv(ref_fea, offset)
        aligned_fea2 = torch.cat([aligned_fea1, ref_fea], 1)
        aligned_fea3 = self.conv2(self.conv1(aligned_fea2))
        aligned_fea = aligned_fea1 + aligned_fea3
        return aligned_fea

    def forward(self, ref_fea, input_fea):

        offset = self.generator(ref_fea, input_fea)
        aligned_fea = self.align(offset, ref_fea)

        return aligned_fea


class Non_local(nn.Module):
    def __init__(self, in_channels, inter_channels):
        super(Non_local, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                           padding=0)

        self.W = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1,
                           padding=0)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=3, stride=1,
                               padding=1)

        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                             padding=0)

        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, supp_feature, ref_feature):
        x = supp_feature  # b,c,h,w
        y = ref_feature

        batch_size = x.size(0)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)

        theta_x = theta_x.permute(0, 2, 1)

        phi_y = self.phi(y).view(batch_size, self.inter_channels, -1)

        f = torch.matmul(theta_x, phi_y)

        f_div_C = F.softmax(f, dim=1)

        x1 = torch.matmul(f_div_C, theta_x)

        x1 = x1.permute(0, 2, 1).contiguous()

        x1 = x1.view(batch_size, self.inter_channels, *supp_feature.size()[2:])
        z = self.W(x1)

        return z


# class NonLocalBlock(nn.Module):
#     def __init__(self, in_channels, inter_channels):
#         super(NonLocalBlock, self).__init__()
#
#         self.in_channels = in_channels
#         self.inter_channels = inter_channels
#
#         self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
#                            padding=0)
#
#         self.W = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1,
#                            padding=0)
#         nn.init.constant_(self.W.weight, 0)
#         nn.init.constant_(self.W.bias, 0)
#
#         self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
#                                padding=0)
#
#         self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
#                              padding=0)
#
#         self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
#
#     def forward(self, supp_feature, ref_feature):
#         x = supp_feature  # b,c,h,w
#         y = ref_feature
#
#         batch_size = x.size(0)
#
#         theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
#
#         theta_x = theta_x.permute(0, 2, 1)
#
#         phi_y = self.phi(y).view(batch_size, self.inter_channels, -1)
#
#         g_y = self.g(y).view(batch_size, self.inter_channels, -1)
#
#         g_y = g_y.permute(0, 2, 1)
#
#         f = torch.matmul(theta_x, phi_y)
#
#         f_div_C = F.softmax(f, dim=1)
#
#         x1 = torch.matmul(f_div_C, g_y)
#
#         x1 = x1.permute(0, 2, 1).contiguous()
#
#         x1 = x1.view(batch_size, self.inter_channels, *supp_feature.size()[2:])
#         W_x1 = self.W(x1)
#         z = x + W_x1
#
#         return z


class Fusion(nn.Module):
    def __init__(self):
        super(Fusion, self).__init__()
        self.align = Align_near()
        self.attention = Non_local(in_channels=64, inter_channels=32)
        self.conv = nn.Conv2d(256, 64, 1, 1)

    def forward(self, ini_recon_fea, ref_fea1, ref_fea2, ref_fea3):
        aligned_ref_fea1 = self.align(ref_fea1, ini_recon_fea)
        aligned_ref_fea2 = self.align(ref_fea2, ini_recon_fea)
        aligned_ref_fea3 = self.align(ref_fea3, ini_recon_fea)

        align_loss1 = torch.mean((ini_recon_fea - aligned_ref_fea1).pow(2))
        align_loss2 = torch.mean((ini_recon_fea - aligned_ref_fea2).pow(2))
        align_loss3 = torch.mean((ini_recon_fea - aligned_ref_fea3).pow(2))
        total_align_loss = align_loss1 + align_loss2 + align_loss3

        atten_ref_fea1 = self.attention(aligned_ref_fea1, ini_recon_fea)
        atten_ref_fea2 = self.attention(aligned_ref_fea2, ini_recon_fea)
        atten_ref_fea3 = self.attention(aligned_ref_fea3, ini_recon_fea)
        atten_self_fea = self.attention(ini_recon_fea, ini_recon_fea)

        fused_fea = torch.cat([atten_ref_fea1, atten_ref_fea2, atten_ref_fea3, atten_self_fea], dim=1)
        final_recon_fea = self.conv(fused_fea) + ini_recon_fea

        return final_recon_fea, total_align_loss


























