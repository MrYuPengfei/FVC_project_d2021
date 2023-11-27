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
import torch.nn as nn


class Res_Block1(nn.Module):
    def __init__(self):
        super(Res_Block1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3),
                               stride=(1, 1), padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3),
                               stride=(1, 1),  padding=1, bias=True)

    def forward(self, x):
        res = self.conv1(x)
        res = self.relu(res)
        res = self.conv2(res)
        return x + res

res_block = Res_Block1()
print(list(res_block.children()))
