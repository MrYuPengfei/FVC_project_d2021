#!/Library/Frameworks/Python.framework/Versions/3.5/bin/python3.5
from .basics import *
import pickle
import os
import codecs
# gdn = tf.contrib.layers.gdn


class Res_Block(nn.Module):
    def __init__(self):
        super(Res_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        res = self.conv1(x)
        res = self.relu(res)
        res = self.conv2(res)
        return x + res


class Synthesis_mv_net(nn.Module):
    '''
    Compress motion
    '''
    def __init__(self):
        super(Synthesis_mv_net, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(out_channel_mv, out_channel_mv, 3, stride=2, padding=1, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv1.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv1.bias.data, 0.01)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1)
        self.deconv2 = nn.Conv2d(out_channel_mv, out_channel_mv, 3, stride=1, padding=1)
        torch.nn.init.xavier_normal_(self.deconv2.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv2.bias.data, 0.01)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1)
        self.deconv3 = nn.ConvTranspose2d(out_channel_mv, out_channel_mv, 3, stride=2, padding=1, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv3.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv3.bias.data, 0.01)
        self.relu3 = nn.LeakyReLU(negative_slope=0.1)
        self.deconv4 = nn.Conv2d(out_channel_mv, out_channel_mv, 3, stride=1, padding=1)
        torch.nn.init.xavier_normal_(self.deconv4.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv4.bias.data, 0.01)
        self.relu4 = nn.LeakyReLU(negative_slope=0.1)
        self.deconv5 = nn.ConvTranspose2d(out_channel_mv, out_channel_mv, 3, stride=2, padding=1, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv5.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv5.bias.data, 0.01)
        self.relu5 = nn.LeakyReLU(negative_slope=0.1)
        self.deconv6 = nn.Conv2d(out_channel_mv, out_channel_mv, 3, stride=1, padding=1)
        torch.nn.init.xavier_normal_(self.deconv6.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv6.bias.data, 0.01)
        self.relu6 = nn.LeakyReLU(negative_slope=0.1)
        self.deconv7 = nn.ConvTranspose2d(out_channel_mv, out_channel_mv, 3, stride=2, padding=1, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv7.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv7.bias.data, 0.01)
        self.relu7 = nn.LeakyReLU(negative_slope=0.1)
        self.deconv8 = nn.Conv2d(out_channel_mv, 64, 3, stride=1, padding=1)
        torch.nn.init.xavier_normal_(self.deconv8.weight.data, (math.sqrt(2 * 1 * (out_channel_mv + 2) / (out_channel_mv + out_channel_mv))))
        torch.nn.init.constant_(self.deconv8.bias.data, 0.01)
        # self.encoder = nn.Sequential(
        #     nn.ConvTranspose2d(out_channel_mv, out_channel_mv, 3, stride=2, padding=1, output_padding=1),
        #     nn.LeakyReLU(negative_slope=0.1),
        #     nn.Conv2d(out_channel_mv, out_channel_mv, 3, stride=1, padding=1),
        #     nn.LeakyReLU(negative_slope=0.1),
        #     nn.ConvTranspose2d(out_channel_mv, out_channel_mv, 3, stride=2, padding=1, output_padding=1),
        #     nn.LeakyReLU(negative_slope=0.1),
        #     nn.Conv2d(out_channel_mv, out_channel_mv, 3, stride=1, padding=1),
        #     nn.LeakyReLU(negative_slope=0.1),
        #     nn.ConvTranspose2d(out_channel_mv, out_channel_mv, 3, stride=2, padding=1, output_padding=1),
        #     nn.LeakyReLU(negative_slope=0.1),
        #     nn.Conv2d(out_channel_mv, out_channel_mv, 3, stride=1, padding=1),
        #     nn.LeakyReLU(negative_slope=0.1),
        #     nn.ConvTranspose2d(out_channel_mv, out_channel_mv, 3, stride=2, padding=1, output_padding=1),
        #     nn.LeakyReLU(negative_slope=0.1),
        #     nn.Conv2d(out_channel_mv, 2, 3, stride=1, padding=1),
        # )

        # self.down2 = nn.Conv2d(128, 64, 3, 1, 1)
        # self.down4 = nn.Conv2d(128, 64, 3, 1, 1)
        # self.down6 = nn.Conv2d(128, 64, 3, 1, 1)

    def forward(self, x):
        x1 = self.deconv1(x)
        x2 = self.deconv2(self.relu1(x1))
        x3 = self.deconv3(self.relu2(x2))
        x4 = self.deconv4(self.relu3(x3))
        x5 = self.deconv5(self.relu4(x4))
        x6 = self.deconv6(self.relu5(x5))
        x7 = self.deconv7(self.relu6(x6))
        x8 = self.deconv8(self.relu7(x7))

        return x8


# class Synthesis_mv_net1(nn.Module):
#     '''
#     Compress motion
#     '''
#     def __init__(self):
#         super(Synthesis_mv_net1, self).__init__()
#         self.residual_layer = self.make_layer(Res_Block, 3)
#
#         self.conv1 = nn.ConvTranspose2d(128, 128, 3, 2, output_padding=1)
#         # torch.nn.init.xavier_normal_(self.conv1.weight.data, math.sqrt(2))
#         torch.nn.init.constant_(self.conv1.bias.data, 0.01)
#         self.relu1 = nn.LeakyReLU(negative_slope=0.1)
#
#         self.conv2 = nn.ConvTranspose2d(128, 128, 3, 2, output_padding=1)
#         # torch.nn.init.xavier_normal_(self.conv2.weight.data, math.sqrt(2))
#         torch.nn.init.constant_(self.conv2.bias.data, 0.01)
#         self.relu2 = nn.LeakyReLU(negative_slope=0.1)
#
#         self.conv3 = nn.ConvTranspose2d(128, 64, 3, 2, output_padding=1)
#         # torch.nn.init.xavier_normal_(self.conv3.weight.data, math.sqrt(2))
#         torch.nn.init.constant_(self.conv3.bias.data, 0.01)
#         self.relu3 = nn.LeakyReLU(negative_slope=0.1)
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#
#     def unit1(self, x):
#         x = x + self.residual_layer(x)
#         x = self.conv1(self.relu1(x))
#         return x
#
#     def unit2(self, x):
#         x = x + self.residual_layer(x)
#         x = self.conv2(self.relu2(x))
#         return x
#
#     def unit3(self, x):
#         x = x + self.residual_layer(x)
#         x = self.conv3(self.relu2(x))
#         return x
#
#     def make_layer(self, block, num_of_layer):
#         layers = []
#         for _ in range(num_of_layer):
#             layers.append(block())
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         x = self.unit1(x)
#         # print(x.shape)
#         x = self.unit2(x)
#         # print(x.shape)
#         x = self.unit3(x)
#         # print(x.shape)
#
#         return x


class Synthesis_mv_net1(nn.Module):
    '''
    Compress motion
    '''
    def __init__(self):
        super(Synthesis_mv_net1, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(out_channel_mv, out_channel_mv, 3, stride=2, padding=1, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv1.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv1.bias.data, 0.01)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1)
        self.deconv2 = nn.Conv2d(out_channel_mv, out_channel_mv, 3, stride=1, padding=1)
        torch.nn.init.xavier_normal_(self.deconv2.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv2.bias.data, 0.01)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1)
        self.deconv3 = nn.ConvTranspose2d(out_channel_mv, out_channel_mv, 3, stride=2, padding=1, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv3.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv3.bias.data, 0.01)
        self.relu3 = nn.LeakyReLU(negative_slope=0.1)
        self.deconv4 = nn.Conv2d(out_channel_mv, out_channel_mv, 3, stride=1, padding=1)
        torch.nn.init.xavier_normal_(self.deconv4.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv4.bias.data, 0.01)
        self.relu4 = nn.LeakyReLU(negative_slope=0.1)
        self.deconv5 = nn.ConvTranspose2d(out_channel_mv, out_channel_mv, 3, stride=2, padding=1, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv5.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv5.bias.data, 0.01)
        self.relu5 = nn.LeakyReLU(negative_slope=0.1)
        self.deconv6 = nn.Conv2d(out_channel_mv, out_channel_mv, 3, stride=1, padding=1)
        torch.nn.init.xavier_normal_(self.deconv6.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv6.bias.data, 0.01)
        self.relu6 = nn.LeakyReLU(negative_slope=0.1)
        self.deconv7 = nn.ConvTranspose2d(out_channel_mv, out_channel_mv, 3, stride=2, padding=1, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv7.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv7.bias.data, 0.01)
        self.relu7 = nn.LeakyReLU(negative_slope=0.1)
        self.deconv8 = nn.Conv2d(out_channel_mv, 2, 3, stride=1, padding=1)
        torch.nn.init.xavier_normal_(self.deconv8.weight.data, (math.sqrt(2 * 1 * (out_channel_mv + 2) / (out_channel_mv + out_channel_mv))))
        torch.nn.init.constant_(self.deconv8.bias.data, 0.01)
        # self.encoder = nn.Sequential(
        #     nn.ConvTranspose2d(out_channel_mv, out_channel_mv, 3, stride=2, padding=1, output_padding=1),
        #     nn.LeakyReLU(negative_slope=0.1),
        #     nn.Conv2d(out_channel_mv, out_channel_mv, 3, stride=1, padding=1),
        #     nn.LeakyReLU(negative_slope=0.1),
        #     nn.ConvTranspose2d(out_channel_mv, out_channel_mv, 3, stride=2, padding=1, output_padding=1),
        #     nn.LeakyReLU(negative_slope=0.1),
        #     nn.Conv2d(out_channel_mv, out_channel_mv, 3, stride=1, padding=1),
        #     nn.LeakyReLU(negative_slope=0.1),
        #     nn.ConvTranspose2d(out_channel_mv, out_channel_mv, 3, stride=2, padding=1, output_padding=1),
        #     nn.LeakyReLU(negative_slope=0.1),
        #     nn.Conv2d(out_channel_mv, out_channel_mv, 3, stride=1, padding=1),
        #     nn.LeakyReLU(negative_slope=0.1),
        #     nn.ConvTranspose2d(out_channel_mv, out_channel_mv, 3, stride=2, padding=1, output_padding=1),
        #     nn.LeakyReLU(negative_slope=0.1),
        #     nn.Conv2d(out_channel_mv, 2, 3, stride=1, padding=1),
        # )

    def forward(self, x):
        x = self.relu1(self.deconv1(x))
        x = self.relu2(self.deconv2(x))
        x = self.relu3(self.deconv3(x))
        x = self.relu4(self.deconv4(x))
        x = self.relu5(self.deconv5(x))
        x = self.relu6(self.deconv6(x))
        x = self.relu7(self.deconv7(x))
        return self.deconv8(x)

