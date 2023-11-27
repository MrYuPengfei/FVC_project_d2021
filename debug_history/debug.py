from util.ms_ssim_torch import *
def test_msssim():
    pass

if __name__ == '__main__':
    test_msssim()


# ---------------------- Please Modve this line to other place, if necessary !  ---------------------------------------------------------------------
# from Net import Net
# from util.FeatureCompressor import *
# from torchviz import make_dot
# from tensorboardX import SummaryWriter

# def build_model():
#     input_image = (torch.rand(8, 128, 16, 16)*256).to('cuda')
#     net= FeatureCompressor()
#     net.to('cuda')
#     print(net(input_image))
# def viz_model():
#     model = Net().cuda()
#     x1 = (torch.ones(8, 3, 256, 256).clamp(1,255)).to('cuda')
#     x2 = (torch.ones(8, 3, 256, 256).clamp(1,255)).to('cuda')
#     with SummaryWriter(comment = '_Net') as w:
#         w.add_graph(model, [x1,x2])
    
# def test_freezing():
#     net=Net()
#     pth_load_path = '/data/users/pengfei/GOP_project_d2021_9_11/FVC_P/log/snapshot/' + str(1024) + '_Vimeo_all_singleGPU/ckpt_best_' + str(0) +'.pth'
#     checkpoint = torch.load(pth_load_path)  # 加载断点
#     net.load_state_dict({k.replace('module.',''):v for [k,v] in (checkpoint['net']).items()})  # 加载模型可学习参数
#     optimizer = torch.optim.Adam( net.parameters(), lr = 1e-4, weight_decay = 5e-7)
#     for para in net.frame_recon.parameters():
# 	    para.requires_grad = False
#     for para in net.generator.parameters():
# 	    para.requires_grad = False
#     for para in net.generator.cr2.parameters():
# 	    para.requires_grad = True
#     # optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
#     image1 = torch.ones(8, 3, 256, 256)
#     image2 = torch.ones(8, 3, 256, 256)
#     image1 = image1.cuda()
#     image2 = image2.cuda()
#     net = net.cuda()

    # input_image = torch.ones(8, 3, 256, 256)
    # quant_noise_feature = torch.zeros(input_image.size(0), 128, input_image.size(2) // 16, input_image.size(3) // 16).cuda()
    # quant_noise_z = torch.zeros(input_image.size(0),128, input_image.size(2) // 64, input_image.size(3) // 64).cuda()
    # print(quant_noise_feature)
    # print(quant_noise_z)
    # quant_noise_feature = torch.nn.init.uniform_(torch.zeros_like(quant_noise_feature), -0.5, 0.5)
    # quant_noise_z = torch.nn.init.uniform_(torch.zeros_like(quant_noise_z), -0.5, 0.5)
