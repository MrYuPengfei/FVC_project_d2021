#!/Library/Frameworks/Python.framework/Versions/3.5/bin/python3.5
from .basics import *
# import pickle
# import os
# import codecs

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


class Analysis_net(nn.Module):
    '''
    Compress residual
    '''
    def __init__(self):
        super(Analysis_net, self).__init__()
        self.conv1 = nn.Conv2d(64, out_channel_N, 5, stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.conv1.weight.data, (math.sqrt(2 * (3 + out_channel_N) / (6))))
        torch.nn.init.constant_(self.conv1.bias.data, 0.01)
        self.gdn1 = GDN(out_channel_N)
        self.conv2 = nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.conv2.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv2.bias.data, 0.01)
        self.gdn2 = GDN(out_channel_N)
        self.conv3 = nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.conv3.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv3.bias.data, 0.01)
        self.gdn3 = GDN(out_channel_N)
        self.conv4 = nn.Conv2d(out_channel_N, out_channel_M, 5, stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.conv4.weight.data, (math.sqrt(2 * (out_channel_M + out_channel_N) / (out_channel_N + out_channel_N))))
        torch.nn.init.constant_(self.conv4.bias.data, 0.01)
        # self.resEncoder = nn.Sequential(
        #     nn.Conv2d(3, out_channel_N, 5, stride=2, padding=2),# how to initialize ???
        #     GDN(out_channel_N),
        #     nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2),# how to initialize ???
        #     GDN(out_channel_N),
        #     nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2),# how to initialize ???
        #     GDN(out_channel_N),
        #     nn.Conv2d(out_channel_N, out_channel_M, 5, stride=2, padding=2),# how to initialize ???
        # )

    def forward(self, x):
        x = self.gdn1(self.conv1(x))
        x = self.gdn2(self.conv2(x))
        x = self.gdn3(self.conv3(x))
        return self.conv4(x)


class Analysis_net1(nn.Module):
    '''
    Compress residual
    '''
    def __init__(self):
        super(Analysis_net1, self).__init__()
        self.residual_layer = self.make_layer(Res_Block, 3)

        self.conv1 = nn.Conv2d(64, 128, 3, 2)
        # torch.nn.init.xavier_normal_(self.conv1.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv1.bias.data, 0.01)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1)

        self.conv2 = nn.Conv2d(128, 128, 3, 2)
        # torch.nn.init.xavier_normal_(self.conv2.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv2.bias.data, 0.01)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1)

        self.conv3 = nn.Conv2d(128, 128, 3, 2)
        # torch.nn.init.xavier_normal_(self.conv3.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv3.bias.data, 0.01)
        self.relu3 = nn.LeakyReLU(negative_slope=0.1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def unit1(self, x):
        x = self.conv1(x)
        map = self.residual_layer(self.relu1(x))
        x = map + x
        return x

    def unit2(self, x):
        x = self.conv2(x)
        map = self.residual_layer(self.relu2(x))
        x = map + x
        return x

    def unit3(self, x):
        x = self.conv3(x)
        map = self.residual_layer(self.relu3(x))
        x = map + x
        return x

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu1(self.unit1(x))
        # print(x.shape)
        x = self.relu2(self.unit2(x))
        # print(x.shape)
        x = self.unit3(x)
        # print(x.shape)

        return x


def build_model():
        input_image = Variable(torch.zeros([4, 3, 256, 256]))

        analysis_net = Analysis_net()
        feature = analysis_net(input_image)

        print(feature.size())
        # feature = sess.run(weights)

        # print(weights_val)

        # gamma_val = sess.run(gamma)

        # print(gamma_val)


if __name__ == '__main__':
    build_model()
