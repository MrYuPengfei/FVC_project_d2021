# Custom convolution kernel
# 在PyTorch中nn.Module类是用于定义网络中前向结构的父类，当要定义自己的网络结构时就要继承这个类。
# 现有的那些类式接口（如nn.Linear、nn.BatchNorm2d、nn.Conv2d等）也是继承这个类的。
# nn.Module类可以嵌套若干nn.Module的对象，来形成网络结构的嵌套组合，下面记录nn.Module的功能。
# 1.继承nn.Module类的模块
# 使用其初始化函数创建对象，然后调用forward函数就能使用里面的前向计算过程。
# 包括：Linear、ReLU、Sigmoid、Conv2d、ConvTransposed2d、Dropout...
# 2.容器nn.Sequential()
# nn.Sequential是一个Sequential容器，模块将按照构造函数中传递的顺序添加到模块中。
# 通俗的说，就是根据自己的需求，把不同的函数组合成一个（小的）模块使用或者把组合的模块添加到自己的网络中。
import torch.nn

from deform_net import *
from torch.autograd import Variable
# Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))]
# 在Compensation()进行dcn之前，对Δx贾一个mask，对offset进行pooling
# torch_mask=torch.zeros(1)
# torch_mask=torch.Tensor([[[[0]], [[0]]], [[[0]], [[0]]], [[[0]], [[0]]],
#                          [[[0]], [[0]]], [[[1]], [[1]]], [[[0]], [[0]]],
#                         [[[0]], [[0]]], [[[0]], [[0]]], [[[0]], [[0]]],
#                         [[[0]], [[0]]], [[[0]], [[0]]], [[[0]], [[0]]],
#                         [[[0]], [[0]]], [[[1]], [[1]]], [[[0]], [[0]]],
#                         [[[0]], [[0]]], [[[0]], [[0]]], [[[0]], [[0]]],
#                          ]
#                         )
#
# torch_mask=Variable(torch_mask,)
t1=torch.rand(1,18,256,256)
t2=torch.rand(1,18,256,256)
# cov1=torch.nn.Conv2d(64,64,(1,1),(1,1),padding = 0,groups = 8)
# print(cov1.shape())
# print(cov1.size())
# print(cov1.ndim())

# torch_mask.to('cuda')
# print(torch_mask)
# print(torch.rand(4,2,1,1))
class Mask1(nn.Module):
    """
    替代 nn.Conv2d(64, 18 * 8, (1, 1), (0, 0), bias = False)
    实质上是一个通道上的滤波器，对通道维度的特征向量进行剪枝。
    Network Slimming——有效的通道剪枝方法（Channel Pruning）
"Learning Efficient Convolutional Networks through Network Slimming"
这篇文章提出了一种有效的结构性剪枝方法，即规整的通道剪枝策略：
在训练期间，通过对网络BN层的gamma系数施加L1正则约束，使得模型朝着结构性稀疏的方向调整参数。
原文链接：https://blog.csdn.net/nature553863/article/details/80649182
    mask=
    [0, 0, 0],
     [0, 1, 0],
     [0, 0, 0],
     [0, 0, 0],
     [0, 1, 0],
     [0, 0, 0]]
     # [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
     # [[[[1][[1]]]]
    """
    def __init__(self):
        super(Mask1, self).__init__()
        # self.self_weight=torch.Tensor([0], [0])
        # self.self_bias=torch.Tensor([1], [1])
        self.cpu()
        self.torch_weight=torch.Tensor([[[[0]], [[0]]], [[[0]], [[0]]], [[[0]], [[0]]],
                                      [[[0]], [[0]]], [[[1]], [[1]]], [[[0]], [[0]]],
                                      [[[0]], [[0]]], [[[0]], [[0]]], [[[0]], [[0]]],
                                      [[[0]], [[0]]], [[[0]], [[0]]], [[[0]], [[0]]],
                                      [[[0]], [[0]]], [[[1]], [[1]]], [[[0]], [[0]]],
                                      [[[0]], [[0]]], [[[0]], [[0]]], [[[0]], [[0]]]])
        self.torch_bias=torch.Tensor([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        self.weight = nn.Parameter(self.torch_weight)  # 自定义的权值
        self.bias =  nn.Parameter(self.torch_bias)  # 自定义的偏置
    def forward(self,x):
        # x = x.view(x.size(0), -1)
        out = F.conv2d(x, self.weight, self.bias, stride = 1, padding = 0,groups = 1)
        return out
mask= Mask1()
t2 = mask(t1)
print(t2)


# class Compensation_withMask(nn.Module):
#     def __init__(self):
#         super(Compensation_withMask, self).__init__()
#         # add mask 20210905
#         self.mask = nn.Conv2d(64, 18 * 8, (1, 1), (0, 0), bias = False)
#         self.conv_first = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=1)
#         self.residual_layer = self.make_layer(Res_Block, 5)
#         self.relu = nn.ReLU(inplace=True)
#         # deformable
#         self.cr1 = nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=1)
#         self.cr2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=1)
#         self.off2d = nn.Conv2d(64, 18 * 8, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True)
#         self.dconv = torchvision.ops.DeformConv2d(64, 64, kernel_size=3, padding=1, groups=8)
#         # xavier initialization
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#
#     def make_layer(self, block, num_of_layer):
#         layers = []
#         for _ in range(num_of_layer):
#             layers.append(block())
#         return nn.Sequential(*layers)
#
#     def align(self, offset, ref_fea):
#         offset = self.off2d(offset)
#         # add mask 20210905
#         offset = self.mask(offset)
#         aligned_fea1 = self.dconv(ref_fea, offset)
#         # print(aligned_fea1.shape, ref_fea.shape)
#         aligned_fea2 = torch.cat([aligned_fea1, ref_fea], 1)
#         # print(aligned_fea2.shape)
#         aligned_fea3 = self.cr2(self.cr1(aligned_fea2))
#         aligned_fea = aligned_fea1 + aligned_fea3
#         return aligned_fea
#
#     def forward(self, offset, ref_fea):
#         aligned_fea = self.align(offset, ref_fea)
#         return aligned_fea

# Snet = Compensation_withMask()