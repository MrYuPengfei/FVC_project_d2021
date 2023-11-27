# 20210911 changed by pengfei
import logging
import argparse
from util.basics import *
from util.VimeoDataset import Vimeo_all     # Vimeo_all为归一化后的数据集
from util.ms_ssim_torch import ms_ssim
from Net import Net
"""
Note: YuPengfei
Least debug: trainFreezingF2I
Last modified date: 20210915
If you need to modify Net, back up a new py file before modifying it,
Net with the same name only returns different parameters, network structure is not allowed to change!
如果需要修改Net,修改前备份一个新的py文件，同名的Net仅仅是返回参数不同，网络结构不许改变！
"""
parser = argparse.ArgumentParser(description='Pytorch reimplement for FVC')
parser.add_argument('--train_lambda', default= 256,type=int, required=True, help='lambda for RDO')
args = parser.parse_args()
train_lambda=args.train_lambda
name =  str(train_lambda) + '_Vimeo_all_singleGPU'
print('lambda: ' + str(train_lambda))
print(name)
train_dataset = Vimeo_all()
torch.backends.cudnn.enabled = True
gpu_num = torch.cuda.device_count()
cur_lr = base_lr = 1e-4   
pth_load_path = './log/snapshot/' + str(train_lambda) + '_Vimeo_all_singleGPU/ckpt_best_10.pth'
save_path = './log/snapshot/'
print_step = 32#30
warmup_step = 0    
gpu_per_batch = gpu_num
test_step = 10000   
tot_epoch = 40
tot_step = 2000000
decay_interval = 1800000
lr_decay = 0.1
logger = logging.getLogger("train")
tb_logger = None
global_step = 0


def train(epoch, global_step):
    print("epoch", epoch)
    print("learnRate",lr)
    return_name=['recon_frame', 'input_image', 'mse_loss_image', 'mse_loss_res', 'se_loss_mv', 'total_bits', 'bits_res', 'bits_mv', 'bpp']
    print(return_name)
    global gpu_per_batch
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, shuffle=True, num_workers=gpu_num, batch_size=gpu_num*5, pin_memory=True, drop_last=True)
    len_train_loader =len(train_loader)
    net.train()
    global optimizer
    for batch_idx, input in enumerate(train_loader):
        optimizer.zero_grad()
        global_step += 1
        ref_image, input_image = (input[0]).cuda(), (input[1]).cuda()
        # print(input_image) 0->255   # print(ms_ssim(input_image,input_image)) 1 #  print(ms_ssim(input_image,ref_image)) 0.8
        recon_image, input_image, mse_loss_image, mse_loss_res, mse_loss_mv, total_bits, bits_res, bits_mv, bpp= net(input_image, ref_image)
        # print(ms_ssim(input_image,input_image1)) 1 # print(ms_ssim(recon_image,input_image)) nan # exit()
        mse_loss_image, mse_loss_res, mse_loss_mv, total_bits, bits_res, bits_mv, bpp=\
            torch.mean(mse_loss_image), torch.mean(mse_loss_res), torch.mean(mse_loss_mv), torch.mean(total_bits),torch.mean(bits_res),torch.mean(bits_mv),torch.mean(bpp)
        # distribution_loss = bpp     # 此处有误，论文里是这样描述的：Ro和Rr表示用于编码的比特数 (不是bpp)
        #if epoch < 5:
        #    a, b, c, d, e, f = 1, 1, 1, 1, 10, 10
        #else:
        #    a, b, c, d, e, f = 1, 0, 0, 1, 0, 0
        #bits =   a * total_bits + b * bits_res + c * bits_mv
        #distortion = d * mse_loss_image + e * mse_loss_res + f * mse_loss_mv
        rd_loss = bpp + train_lambda * mse_loss_image 
        #print(rd_loss)
        #print(mse_loss_image)
        #    from torchvision import utils as vutils
        # input_tensor = input_image.clone().detach()
        # input_tensor = input_tensor.to(torch.device('cpu'))
        # vutils.save_image(input_tensor, './input.png')
        # input_tensor = ref_image.clone().detach()
        # input_tensor = input_tensor.to(torch.device('cpu'))
        # vutils.save_image(input_tensor, './ref.png')
        # input_tensor = recon_image.clone().detach()
        # input_tensor = input_tensor.to(torch.device('cpu'))
        # vutils.save_image(input_tensor, './rec.png')
        #vutils.save_image(ref_image, './ref_png')
        #vutils.save_image(recon_image, './rec.png')
        # exit()
        rd_loss.backward()
        optimizer.step()
        if (batch_idx % print_step) == 0:
            psnr = 10 * (torch.log(1 / mse_loss_image) / np.log(10))
            msssim=ms_ssim(recon_image,input_image)
            time_now = time.strftime("%Y-%m-%d:%H:%M:%S ", time.localtime())
            log0 = '{:02}[{:.2f}%]>>> rd_loss:{:.6f} bpp(R):{:.6f} mse(D):{:.6f}'
            log0 = log0.format(epoch, 100.*batch_idx/len_train_loader, rd_loss, bpp, mse_loss_image)
            log1 = 'bases  >>> msssim:{:.6f} psnr:{:.6f} bpp:{:.6f} '
            log1 = log1.format(msssim, psnr.cpu().detach().numpy(), bpp.cpu().detach().numpy(),)
            log2 = 'Details>>> mse_loss_re:{:.6f} mse_loss_mv:{:.6f} total_bits:{:.6} bits_res:{:.6} bits_mv:{:.6}'
            log2 = log2.format(mse_loss_res.cpu().detach().numpy(),
                               mse_loss_mv.cpu().detach().numpy(),
                               total_bits.cpu().detach().numpy(),
                               bits_res.cpu().detach().numpy(),
                               bits_mv.cpu().detach().numpy(),
                               )
            print(time_now)
            print(log0)
            print(log1)
            print(log2)
    return global_step
if __name__ == "__main__":
    net = Net()
    if pth_load_path != '':
        path_checkpoint = pth_load_path           # 断点路径
        print(path_checkpoint)
        checkpoint = torch.load(path_checkpoint)  # 加载断点
        net.load_state_dict({k.replace('module.',''):v for [k,v] in (checkpoint['net']).items()})  # 加载模型可学习参数
        bp_parameters = net.parameters()
        optimizer = torch.optim.Adam(bp_parameters, lr = base_lr, weight_decay = 5e-7)
        optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
        start_epoch= checkpoint['epoch']   # 设置开始的epoch
        global_step = start_epoch * (train_dataset.__len__() * (gpu_per_batch))
    net = net.cuda()
    net = torch.nn.DataParallel(net.cuda(), list(range(gpu_num)))
    bp_parameters = net.parameters()
    optimizer = torch.optim.Adam(bp_parameters, lr=base_lr, weight_decay=5e-7)
    data_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    stepoch = global_step // (train_dataset.__len__() // (gpu_per_batch))
    for epoch in range(stepoch, tot_epoch):
        if epoch < 12:
            lr = base_lr
        elif 12 <= epoch < 15:
            lr = base_lr * lr_decay
        else:
            lr = base_lr * lr_decay * lr_decay
        cur_lr = lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        checkpoint = {
            "net": net.state_dict(),
            'optimizer': optimizer.state_dict(),
            "epoch": epoch
        }
        torch.save(checkpoint, save_path+name+'/ckpt_best_%s.pth' % (str(epoch)))
        global_step = train(epoch, global_step)
    torch.save(net, save_path+name+"/best.model", _use_new_zipfile_serialization=False)
    # !
# end !
