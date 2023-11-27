from .basics import *
# import pickle
# import os
# import codecs
from .analysis import Analysis_net


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


class Analysis_mv_net(nn.Module):
    '''
    Compress motion
    '''
    def __init__(self):
        super(Analysis_mv_net, self).__init__()
        self.conv1 = nn.Conv2d(64, out_channel_mv, 3, stride=2, padding=1)
        torch.nn.init.xavier_normal_(self.conv1.weight.data, (math.sqrt(2 * (2 + out_channel_mv) / (4))))
        torch.nn.init.constant_(self.conv1.bias.data, 0.01)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1)
        self.conv2 = nn.Conv2d(out_channel_mv, out_channel_mv, 3, stride=1, padding=1)
        torch.nn.init.xavier_normal_(self.conv2.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv2.bias.data, 0.01)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1)
        self.conv3 = nn.Conv2d(out_channel_mv, out_channel_mv, 3, stride=2, padding=1)
        torch.nn.init.xavier_normal_(self.conv3.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv3.bias.data, 0.01)
        self.relu3 = nn.LeakyReLU(negative_slope=0.1)
        self.conv4 = nn.Conv2d(out_channel_mv, out_channel_mv, 3, stride=1, padding=1)
        torch.nn.init.xavier_normal_(self.conv4.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv4.bias.data, 0.01)
        self.relu4 = nn.LeakyReLU(negative_slope=0.1)
        self.conv5 = nn.Conv2d(out_channel_mv, out_channel_mv, 3, stride=2, padding=1)
        torch.nn.init.xavier_normal_(self.conv5.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv5.bias.data, 0.01)
        self.relu5 = nn.LeakyReLU(negative_slope=0.1)
        self.conv6 = nn.Conv2d(out_channel_mv, out_channel_mv, 3, stride=1, padding=1)
        torch.nn.init.xavier_normal_(self.conv6.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv6.bias.data, 0.01)
        self.relu6 = nn.LeakyReLU(negative_slope=0.1)
        self.conv7 = nn.Conv2d(out_channel_mv, out_channel_mv, 3, stride=2, padding=1)
        torch.nn.init.xavier_normal_(self.conv7.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv7.bias.data, 0.01)
        self.relu7 = nn.LeakyReLU(negative_slope=0.1)
        self.conv8 = nn.Conv2d(out_channel_mv, out_channel_mv, 3, stride=1, padding=1)
        torch.nn.init.xavier_normal_(self.conv8.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv8.bias.data, 0.01)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.relu5(self.conv5(x))
        x = self.relu6(self.conv6(x))
        x = self.relu7(self.conv7(x))
        return self.conv8(x)


class Analysis_mv_net1(nn.Module):
    '''
    Compress motion
    '''
    def __init__(self):
        super(Analysis_mv_net1, self).__init__()
        self.conv1 = nn.Conv2d(2, out_channel_mv, 3, stride=2, padding=1)
        torch.nn.init.xavier_normal_(self.conv1.weight.data, (math.sqrt(2 * (2 + out_channel_mv) / (4))))
        torch.nn.init.constant_(self.conv1.bias.data, 0.01)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1)
        self.conv2 = nn.Conv2d(out_channel_mv, out_channel_mv, 3, stride=1, padding=1)
        torch.nn.init.xavier_normal_(self.conv2.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv2.bias.data, 0.01)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1)
        self.conv3 = nn.Conv2d(out_channel_mv, out_channel_mv, 3, stride=2, padding=1)
        torch.nn.init.xavier_normal_(self.conv3.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv3.bias.data, 0.01)
        self.relu3 = nn.LeakyReLU(negative_slope=0.1)
        self.conv4 = nn.Conv2d(out_channel_mv, out_channel_mv, 3, stride=1, padding=1)
        torch.nn.init.xavier_normal_(self.conv4.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv4.bias.data, 0.01)
        self.relu4 = nn.LeakyReLU(negative_slope=0.1)
        self.conv5 = nn.Conv2d(out_channel_mv, out_channel_mv, 3, stride=2, padding=1)
        torch.nn.init.xavier_normal_(self.conv5.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv5.bias.data, 0.01)
        self.relu5 = nn.LeakyReLU(negative_slope=0.1)
        self.conv6 = nn.Conv2d(out_channel_mv, out_channel_mv, 3, stride=1, padding=1)
        torch.nn.init.xavier_normal_(self.conv6.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv6.bias.data, 0.01)
        self.relu6 = nn.LeakyReLU(negative_slope=0.1)
        self.conv7 = nn.Conv2d(out_channel_mv, out_channel_mv, 3, stride=2, padding=1)
        torch.nn.init.xavier_normal_(self.conv7.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv7.bias.data, 0.01)
        self.relu7 = nn.LeakyReLU(negative_slope=0.1)
        self.conv8 = nn.Conv2d(out_channel_mv, out_channel_mv, 3, stride=1, padding=1)
        torch.nn.init.xavier_normal_(self.conv8.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv8.bias.data, 0.01)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.relu5(self.conv5(x))
        x = self.relu6(self.conv6(x))
        x = self.relu7(self.conv7(x))
        return self.conv8(x)


# class Analysis_mv_net1(nn.Module):
#     '''
#     Compress motion
#     '''
#     def __init__(self):
#         super(Analysis_mv_net1, self).__init__()
#
#         self.residual_layer = self.make_layer(Res_Block, 3)
#
#         self.conv1 = nn.Conv2d(64, 128, 3, 2)
#         # torch.nn.init.xavier_normal_(self.conv1.weight.data, math.sqrt(2))
#         torch.nn.init.constant_(self.conv1.bias.data, 0.01)
#         self.relu1 = nn.LeakyReLU(negative_slope=0.1)
#
#         self.conv2 = nn.Conv2d(128, 128, 3, 2)
#         # torch.nn.init.xavier_normal_(self.conv2.weight.data, math.sqrt(2))
#         torch.nn.init.constant_(self.conv2.bias.data, 0.01)
#         self.relu2 = nn.LeakyReLU(negative_slope=0.1)
#
#         self.conv3 = nn.Conv2d(128, 128, 3, 2)
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
#         x = self.conv1(x)
#         # print(x.shape)
#         map = self.residual_layer(self.relu1(x))
#         x = map + x
#         return x
#
#     def unit2(self, x):
#         x = self.conv2(x)
#         map = self.residual_layer(self.relu2(x))
#         x = map + x
#         return x
#
#     def unit3(self, x):
#         x = self.conv3(x)
#         map = self.residual_layer(self.relu3(x))
#         x = map + x
#         return x
#
#     def make_layer(self, block, num_of_layer):
#         layers = []
#         for _ in range(num_of_layer):
#             layers.append(block())
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         x = self.relu1(self.unit1(x))
#         # print(x.shape)
#         x = self.relu2(self.unit2(x))
#         # print(x.shape)
#         x = self.unit3(x)
#         # print(x.shape)
#
#         return x


class Motion_refinement(nn.Module):
    def __init__(self):
        super(Motion_refinement, self).__init__()
        self.basic = torch.nn.Sequential(
            nn.Conv2d(5, 64, 3, padding=1),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.Conv2d(64, 2, 3, padding=1)
        )

    def forward(self, mv, ref):
        x = torch.cat((mv, ref), dim=1)
        x = self.basic(x) + mv
        return x


class Motion_refinement_(nn.Module):
    def __init__(self):
        super(Motion_refinement_, self).__init__()
        self.basic = torch.nn.Sequential(
            nn.Conv2d(2, 64, 3, padding=1),
            # nn.Conv2d(64, 64, 3, padding=1),
            # nn.Conv2d(64, 64, 3, padding=1),
            # nn.Conv2d(64, 64, 3, padding=1),
            # nn.Conv2d(64, 64, 3, padding=1),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.Conv2d(64, 2, 3, padding=1)
        )

    def forward(self, mv):
        x = self.basic(mv) + mv
        return x


def build_model():
    analysis_net = Analysis_net()
    analysis_mv_net = Analysis_mv_net()
    
    feature = torch.zeros([3, 2, 256, 256])
    z = analysis_mv_net(feature)
    print("feature : ", feature.size())
    print("z : ", z.size())

if __name__ == '__main__':
    build_model()