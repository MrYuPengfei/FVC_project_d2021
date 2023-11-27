#!/Library/Frameworks/Python.framework/Versions/3.5/bin/python3.5  
from .basics import *
import pickle
import os
import codecs
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


class Synthesis_net(nn.Module):
    '''
    Decode residual
    '''
    def __init__(self):
        super(Synthesis_net, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(out_channel_M, out_channel_N, 5, stride=2, padding=2, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv1.weight.data, (math.sqrt(2 * 1 * (out_channel_M + out_channel_N) / (out_channel_M + out_channel_M))))
        torch.nn.init.constant_(self.deconv1.bias.data, 0.01)
        self.igdn1 = GDN(out_channel_N, inverse=True)
        self.deconv2 = nn.ConvTranspose2d(out_channel_N, out_channel_N, 5, stride=2, padding=2, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv2.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv2.bias.data, 0.01)
        self.igdn2 = GDN(out_channel_N, inverse=True)
        self.deconv3 = nn.ConvTranspose2d(out_channel_N, out_channel_N, 5, stride=2, padding=2, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv3.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv3.bias.data, 0.01)
        self.igdn3 = GDN(out_channel_N, inverse=True)
        self.deconv4 = nn.ConvTranspose2d(out_channel_N, 64, 5, stride=2, padding=2, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv4.weight.data, (math.sqrt(2 * 1 * (out_channel_N + 3) / (out_channel_N + out_channel_N))))
        torch.nn.init.constant_(self.deconv4.bias.data, 0.01)
        # self.resDecoder = nn.Sequential(
        #     nn.ConvTranspose2d(out_channel_M, out_channel_N, 5, stride=2, padding=2, output_padding=1),# how to initialize ???
        #     GDN(out_channel_N, inverse=True),
        #     nn.ConvTranspose2d(out_channel_N, out_channel_N, 5, stride=2, padding=2, output_padding=1),# how to initialize ???
        #     GDN(out_channel_N, inverse=True),
        #     nn.ConvTranspose2d(out_channel_N, out_channel_N, 5, stride=2, padding=2, output_padding=1),# how to initialize ???
        #     GDN(out_channel_N, inverse=True),
        #     nn.ConvTranspose2d(out_channel_N, 3, 5, stride=2, padding=2, output_padding=1),# how to initialize ???
        # )
    def forward(self, x):
        x = self.igdn1(self.deconv1(x))
        x = self.igdn2(self.deconv2(x))
        x = self.igdn3(self.deconv3(x))
        # print(torch.std(x.view(-1)).cpu().detach().numpy())
        x = self.deconv4(x)
        # print(torch.std(x.view(-1)).cpu().detach().numpy())
        return x


class Synthesis_net1(nn.Module):
    '''
    Decode residual
    '''
    def __init__(self):
        super(Synthesis_net1, self).__init__()
        self.residual_layer = self.make_layer(Res_Block, 3)

        self.conv1 = nn.ConvTranspose2d(128, 128, 3, 2, output_padding=1)
        # torch.nn.init.xavier_normal_(self.conv1.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv1.bias.data, 0.01)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1)

        self.conv2 = nn.ConvTranspose2d(128, 128, 3, 2, output_padding=1)
        # torch.nn.init.xavier_normal_(self.conv2.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv2.bias.data, 0.01)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1)

        self.conv3 = nn.ConvTranspose2d(128, 64, 3, 2, output_padding=1)
        # torch.nn.init.xavier_normal_(self.conv3.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv3.bias.data, 0.01)
        self.relu3 = nn.LeakyReLU(negative_slope=0.1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def unit1(self, x):
        x = x + self.residual_layer(x)
        x = self.conv1(self.relu1(x))
        return x

    def unit2(self, x):
        x = x + self.residual_layer(x)
        x = self.conv2(self.relu2(x))
        return x

    def unit3(self, x):
        x = x + self.residual_layer(x)
        x = self.conv3(self.relu3(x))
        return x

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.unit1(x)
        # print(x.shape)
        x = self.unit2(x)
        # print(x.shape)
        x = self.unit3(x)
        # print(x.shape)

        return x


def build_model():
    input_image = torch.zeros([7,3,256,256])
    analysis_net = Analysis_net()
    synthesis_net = Synthesis_net()
    feature = analysis_net(input_image)
    recon_image = synthesis_net(feature)

    print("input_image : ", input_image.size())
    print("feature : ", feature.size())
    print("recon_image : ", recon_image.size())


if __name__ == '__main__':
  build_model()
