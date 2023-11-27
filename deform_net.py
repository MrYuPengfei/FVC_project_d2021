# d20210827
from subnet import *
from subnet.mul_frame_align import *

# from dcn.deform_conv import *
# pytorch1.7.0 之前的版本 torchvision 里边似乎没有实现dcn 所以需要导入这个包
# 在pytorch1.9.0 里不需要了，所以采用torchvision的里的ops.DeformConv2d
# from torchvision.ops import DeformConv2d  #前边已经导入过torchvision了，所以这里不需要再导入了！


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


class Res_Block2(nn.Module):
    def __init__(self):
        super(Res_Block2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3),
                               stride=(1, 1),  padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3),
                               stride=(1, 1),  padding=1, bias=True)

    def forward(self, x):
        res = self.conv1(x)
        res = self.relu(res)
        res = self.conv2(res)
        return x + res


class Res_Block3(nn.Module):
    def __init__(self):
        super(Res_Block3, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=(3, 3),
                               stride=(1, 1),  padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=(3, 3),
                               stride=(1, 1),  padding=1, bias=True)

    def forward(self, x):
        res = self.conv1(x)
        res = self.relu(res)
        res = self.conv2(res)
        return x + res


class Offset_Generator(nn.Module):
    def __init__(self):
        super(Offset_Generator, self).__init__()
        self.conv_first2 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.residual_layer2 = self.make_layer(Res_Block2, 3)
        self.relu = nn.ReLU(inplace=True)
        self.cr1 = nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.cr2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def generator(self, input, ref):
        ref1 = self.conv_first2(ref)
        ref2 = self.residual_layer2(self.relu(ref1))
        ref_fea = ref1 + ref2
        input1 = self.conv_first2(input)
        input2 = self.residual_layer2(self.relu(input1))
        input_fea = input1 + input2
        fea = torch.cat([input_fea, ref_fea], dim=1)
        offset = self.cr2(self.relu(self.cr1(fea)))
        return offset, input_fea, ref_fea

    def forward(self, input, ref):
        offset, input_fea, ref_fea = self.generator(input, ref)
        return offset, input_fea, ref_fea


class Compensation(nn.Module):
    def __init__(self):
        super(Compensation, self).__init__()
        self.conv_first = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.residual_layer = self.make_layer(Res_Block, 5)
        self.relu = nn.ReLU(inplace=True)
        # deformable
        self.cr1 = nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.cr2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.off2d = nn.Conv2d(64, 18 * 8, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True)
        self.dconv = torchvision.ops.DeformConv2d(64, 64, kernel_size=3, padding=1, groups=8)
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

    def align(self, offset, ref_fea):
        offset = self.off2d(offset)
        aligned_fea1 = self.dconv(ref_fea, offset)
        # print(aligned_fea1.shape, ref_fea.shape)
        aligned_fea2 = torch.cat([aligned_fea1, ref_fea], 1)
        # print(aligned_fea2.shape)
        aligned_fea3 = self.cr2(self.cr1(aligned_fea2))
        aligned_fea = aligned_fea1 + aligned_fea3
        return aligned_fea

    def forward(self, offset, ref_fea):
        aligned_fea = self.align(offset, ref_fea)
        return aligned_fea


class Quantization(nn.Module):
    def __init__(self):
        super(Quantization, self).__init__()
        pass

    def forward(self, yt):
        # Quantisation
        noise = torch.nn.init.uniform_(torch.zeros_like(yt), -0.5, 0.5)
        if self.training:
            y_t = yt + noise
        else:
            y_t = torch.round(yt)
        return y_t


class Frame_recon(nn.Module):
    def __init__(self):
        super(Frame_recon, self).__init__()
        # self.conv = nn.ConvTranspose2d(64, 3, 5, 2, output_padding=1)
        self.conv = nn.ConvTranspose2d(64, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.residual_layer = self.make_layer(Res_Block2, 3)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, final_recon_fea):
        map = self.residual_layer(final_recon_fea)
        recon_frame = self.conv(map + final_recon_fea)
        return recon_frame


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.generator = Offset_Generator()
        self.compensation = Compensation()  # 20210827，搞不定GPU，采用CPU吧？
        self.fea_fusion = Fusion()    # 最后面的多特征融合模块，占用现存过多且提升不大，故未加入
        self.frame_recon = Frame_recon()
        self.mv_encoder = Analysis_mv_net()
        self.mv_decoder = Synthesis_mv_net()
        self.Q = Quantization()
        self.bitEstimator = BitEstimator(channel=128)
        self.bitEstimator_z = BitEstimator(channel=96)
        self.res_encoder = Analysis_net()
        self.res_decoder = Synthesis_net()

    def forward(self, input_image, ref_image):
        offsets, input_fea, ref_fea = self.generator(ref_image, input_image)
        feature_mv = self.mv_encoder(offsets)
        quan_feature = self.Q(feature_mv)
        recon_offstes = self.mv_decoder(quan_feature)
        aligned_fea = self.compensation(recon_offstes, ref_fea)
        fea_residual = input_fea - aligned_fea
        feature = self.res_encoder(fea_residual)
        compressed_feature_renorm = self.Q(feature)
        recon_res = self.res_decoder(compressed_feature_renorm)
        recon_fea = aligned_fea + recon_res
        recon_frame = self.frame_recon(recon_fea)
        clipped_recon_frame = recon_frame.clamp(0., 1.)
        mse_loss = torch.mean((input_image - recon_frame).pow(2))
        align_loss = torch.mean((input_fea - aligned_fea).pow(2))

        def iclr18_estrate_bits(fea):
            prob = self.bitEstimator(fea + 0.5) - self.bitEstimator(fea - 0.5)
            total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-5) / math.log(2.0), 0, 50))
            return total_bits, prob

        def iclr18_estrate_bits_z(z):
            prob = self.bitEstimator_z(z + 0.5) - self.bitEstimator_z(z - 0.5)
            total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-5) / math.log(2.0), 0, 50))
            return total_bits, prob

        total_bits_z, _ = iclr18_estrate_bits_z(compressed_feature_renorm)
        im_shape = input_image.size()
        batch_size = input_image.size()[0]
        bpp_z = total_bits_z / (batch_size * im_shape[2] * im_shape[3])
        total_bits_offsets, _ = iclr18_estrate_bits(quan_feature)
        bpp_res = bpp_z
        fea_shape = input_fea.size()
        batch_size = input_fea.size()[0]
        bpp_offsets = total_bits_offsets / (batch_size * fea_shape[2] * fea_shape[3])
        total_bpp = bpp_offsets + bpp_res
        return clipped_recon_frame, recon_frame, mse_loss, align_loss, bpp_offsets, total_bpp





