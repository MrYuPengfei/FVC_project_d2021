# 20210911 changed by pengfei
import logging
import argparse
from torch.utils.tensorboard import SummaryWriter
from torch.functional import Tensor
from torch.nn.functional import mse_loss
from util.basics import *
from util.VimeoDataset_2 import Vimeo_all     # Vimeo_all为归一化后的数据集
from util.ms_ssim_torch import ms_ssim
from Net import Net
"""
Note: YuPengfei
Least debug: trainOnlyMse
Last modified date: 20210915
If you need to modify Net, back up a new py file before modifying it,
Net with the same name only returns different parameters, network structure is not allowed to change!
如果需要修改Net,修改前备份一个新的py文件，同名的Net仅仅是返回参数不同，网络结构不许改变！
"""
parser = argparse.ArgumentParser(description='Pytorch reimplement for FVC')
parser.add_argument('--train_lambda', default=256, required=True, help='lambda for Mse')
parser.add_argument('--start_epoch', default=11,type=int, required=True, help='epoch for Mse')
args = parser.parse_args()
train_lambda=args.train_lambda
train_lambda = int(train_lambda)
start_epoch = args.start_epoch  
tot_epoch = 80

name =  str(train_lambda) + 'trainMse'
print(name)
train_dataset = Vimeo_all()
torch.backends.cudnn.enabled = True
gpu_num = torch.cuda.device_count()
base_lr = 1e-3  
pth_load_path = './log/snapshot/1024train/ckpt_best_38.pth'#_Vimeo_all_singleGPU/best_19.model' # ckpt_best_%s.pth' % (str(start_epoch))#/trainOnlyF2I.model'
save_path = './log/snapshot/'
path_log = save_path+name+"/train.log"
print_step = 32#30
warmup_step = 0    
gpu_per_batch = 4  # 这个参数是干什么用的？
test_step = 10000 
tot_step = 2000000
decay_interval = 1800000
lr_decay = 0.1
#logger = logging.getLogger("train")
#tb_logger = None
global_step = 0

tb_writer = SummaryWriter(log_dir=save_path+name+"/tb_writer")  # from tensorboardX

def train(epoch, global_step):
    # global optimizer
    # global gpu_per_batch
    # global train_lambda
    print("epoch", epoch)
    return_name=['recon_frame', 'mse_loss_fea', 'mse_loss_image', 'mse_loss_res', 'se_loss_mv', 'total_bits', 'bits_res', 'bits_mv', 'bpp']
    print(return_name)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, shuffle=True, num_workers=gpu_num, batch_size=gpu_num*5, pin_memory=True, drop_last=True)
    len_train_loader =len(train_loader)
    net.cuda()
    for batch_idx, input in enumerate(train_loader):
        optimizer.zero_grad()
        global_step += 1
        ref_image, input_image = (input[0]).cuda(), (input[1]).cuda()
        msssim_fea,msssim_image,recon_image, mse_loss_fea, mse_loss_image, mse_loss_res, mse_loss_mv, total_bits, bits_res, bits_mv, bpp= net(input_image,ref_image)
        total_bits, bits_res, bits_mv, bpp=torch.mean(total_bits),torch.mean(bits_res),torch.mean(bits_mv),torch.mean(bpp)
        for name, para in net.named_parameters():
            para.requires_grad_(False)
            if "compressor_mv" in name:
                para.requires_grad_(True)
            elif "compressor_res" in name:
                para.requires_grad_(True)
        if epoch <= 40:
            lr = base_lr
            rd_loss= bpp + train_lambda * (mse_loss_fea + mse_loss_image + mse_loss_res + mse_loss_mv)
        elif epoch <= 60:
            lr = base_lr * lr_decay
            rd_loss= bpp + train_lambda * (mse_loss_fea + mse_loss_image)
        else :
            lr = base_lr * lr_decay * lr_decay
            rd_loss= bpp+ train_lambda * mse_loss_image
        # rd_loss = bpp + train_lambda * mse_loss_image
        for param_group in optimizer.param_groups:#在每次更新参数前迭代更改学习率 
             param_group["lr"] = lr 
        
        rd_loss.backward()
        optimizer.step()
        if (batch_idx % print_step) == 0:
            psnr = 10 * (torch.log(1 / mse_loss_image) / np.log(10))
            msssim=ms_ssim(X=recon_image,Y=input_image,data_range=1)
            time_now = time.strftime("%Y-%m-%d:%H:%M:%S ", time.localtime())
            log0 = '{:d}[{:d}/{:d}]>>> rd_loss:{:.6f} bpp(R):{:.6f} psnr(D):{:.6f} lr:{:.6f}'
            log0 = log0.format(epoch,batch_idx, len_train_loader, rd_loss, bpp, psnr,lr)
            log1 = 'bases  >>> total_bits{:.6f} bits_res{:.6f} bits_mv{:.6f} bpp:{:.6f}'
            log1 = log1.format(total_bits, bits_res, bits_mv, bpp)
            log2 = 'Details>>> mse_loss_fea:{:.6f} mse_loss_image:{:.6f} mse_loss_res:{:.6f} mse_loss_mv:{:.6f}'
            log2 = log2.format(mse_loss_fea, mse_loss_image, mse_loss_res, mse_loss_mv )
            log3 = 'Ms-ssim of ref with org: {:.6f}'.format(ms_ssim(X=ref_image,Y=input_image,data_range=1))
            log4 = 'Ms-ssim of ref with rec: {:.6f}'.format(ms_ssim(X=ref_image,Y=recon_image,data_range=1))
            log5 = 'Ms-ssim of rec with org: {:.6f}'.format(msssim)
            FILE_WRITE = open(path_log,"a")  #只追加到文件末尾
            FILE_WRITE.writelines(time_now+"\n"+log0+"\n"+log1+"\n"+log2+"\n"+log3+"\n"+log4+"\n"+log5+"\n")
            FILE_WRITE.close()

    tb_writer.add_scalar("rd_loss", rd_loss, epoch)
    tb_writer.add_scalar("psnr", psnr, epoch)
    tb_writer.add_scalar("msssim", msssim, epoch)
    tb_writer.add_scalar("bpp", bpp, epoch)
    tb_writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)
    
    return global_step
if __name__ == "__main__":
    print(pth_load_path)
    global net
    if pth_load_path[-6:]=='.model':
        net = torch.load(pth_load_path)
        net = net.cuda()
        net = torch.nn.DataParallel(net.cuda(), list(range(gpu_num)))
        bp_parameters = net.parameters()
        optimizer = torch.optim.Adam(bp_parameters, lr=base_lr, weight_decay=5e-7)
        # start_epoch= 10  # 设置开始的epoch
        global_step = start_epoch * (train_dataset.__len__() * (gpu_per_batch))   
    elif pth_load_path[-4:]=='.pth':  # 存在一些bug未修正
        net = Net()
        checkpoint = torch.load(pth_load_path)  # 加载断点
        net.load_state_dict({k.replace('module.',''):v for [k,v] in (checkpoint['net']).items()})  # 加载模型可学习参数
        net = net.cuda()  # 这行代码删掉，梯度无法回传
        bp_parameters = net.parameters()
        optimizer = torch.optim.Adam(bp_parameters, lr=base_lr, weight_decay=5e-7)
        optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
        # start_epoch= checkpoint['epoch']  # 设置开始的epoch
        global_step = start_epoch * (train_dataset.__len__() * (gpu_per_batch))        
    else:
        net = Net()
        net = net.cuda()  # 这行代码删掉，梯度无法回传
        bp_parameters = net.parameters()
        optimizer = torch.optim.Adam(bp_parameters, lr=base_lr, weight_decay=5e-7)
        global_step = start_epoch * (train_dataset.__len__() * (gpu_per_batch))     
    data_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    # stepoch = global_step // (train_dataset.__len__() // (gpu_per_batch))
    for epoch in range(start_epoch, tot_epoch):

        checkpoint = {
            "net": net.state_dict(),
            'optimizer': optimizer.state_dict(),
            "epoch": epoch
        }  # 这里设置所需要存储在.pth文件中的网络参数
        torch.save(checkpoint, save_path+name+'/ckpt_best_%s.pth' % (str(epoch))) # 保存的是训练前的.pth
        global_step = train(epoch, global_step)  # 训练结果存储在epoch+1 的.pth 或model
    torch.save(net, save_path+name+"/best_%s.model" % (str(epoch)), _use_new_zipfile_serialization=False)
    # !
# end !

