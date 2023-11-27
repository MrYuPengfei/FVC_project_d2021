# 20210830
import logging
import argparse
# import torchvision.transforms as transforms
from torch.utils.data import DataLoader
# import torch.optim as optim
from util.dataset import *
# from util.drawuvg import HEVCdrawplt
from Net import Net
from util.VimeoDataset import Vimeo_all

parser = argparse.ArgumentParser(description='Pytorch reimplement for FVC')
parser.add_argument('--train_lambda', default= 256,type=int, required=True, help='lambda for RDO')
parser.add_argument('--epoch', default= 0,type=int, required=True, help='epoch for RDO')
parser.add_argument('--magic_number', default= 0,type=int, required=True, help='lambda for RDO')
args = parser.parse_args()
train_lambda = args.train_lambda
epoch = args.epoch
magic_number = args.magic_number
name =  str(train_lambda) + '_Vimeo_all_singleGPU'
pth_load_path = './log/snapshot/' + str(train_lambda) + '_Vimeo_all_singleGPU/ckpt_best_' + str(epoch) +'.pth'
image_save_path = './log/validate/'+str(magic_number)+'/'
print('lambda: ' + str(train_lambda))
print(name)

validate_dataset = Vimeo_all()
torch.backends.cudnn.enabled = True
gpu_num = torch.cuda.device_count()
cur_lr = base_lr = 1e-4   
# print_step = 1
warmup_step = 0    
gpu_per_batch = gpu_num
test_step = 10000   
tot_epoch = 40
tot_step = 2000000
decay_interval = 1800000
lr_decay = 0.1
logger = logging.getLogger("validate")
tb_logger = None
global_step = 0



def validate(global_step, testfull=True):
    print("epoch", start_epoch)
    return_name=['recon_frame', 'input_image', 'mse_loss_image', 'mse_loss_res', 'se_loss_mv', 'total_bits', 'bits_res', 'bits_mv', 'bpp']
    print(return_name)
    # global gpu_per_batch
    train_loader = torch.utils.data.DataLoader(dataset=validate_dataset, shuffle=True, num_workers=gpu_num, batch_size=gpu_num*5, pin_memory=True, drop_last=True)
    len_train_loader =len(train_loader)
    with torch.no_grad():
        test_loader = DataLoader(dataset=validate_dataset, shuffle=False, num_workers=0, batch_size=1, pin_memory=True)
        net.eval()
        # sumbpp = 0
        # sumpsnr = 0
        # summsssim = 0
        # sumbpp_list=[]
        # sumpsnr_list=[]
        # summsssim_list=[]
        cnt = 0
        for batch_idx, input in enumerate(test_loader):
            print("validate : %d/%d" % (batch_idx, len(test_loader)))
            input_image = input[0]
            ref_image = input[0]    # [1]
            optimizer.zero_grad()
            global_step += 1
            ref_image, input_image = (input[0]).cuda(), (input[0]).cuda()

            # print(input_image) #0->1   # print(ms_ssim(input_image,input_image)) 1 #  print(ms_ssim(input_image,ref_image)) 0.8
            recon_image, input_image, mse_loss_image, mse_loss_res, mse_loss_mv, total_bits, bits_res, bits_mv, bpp= net(input_image, ref_image)
            # print(ms_ssim(input_image,input_image1)) 1 # print(ms_ssim(recon_image,input_image)) nan # exit()
            # print(recon_image)
            mse_loss_image, mse_loss_res, mse_loss_mv, total_bits, bits_res, bits_mv, bpp=\
                torch.mean(mse_loss_image), torch.mean(mse_loss_res), torch.mean(mse_loss_mv), torch.mean(total_bits),torch.mean(bits_res),torch.mean(bits_mv),torch.mean(bpp)
            rd_loss = train_lambda * mse_loss_image 
            psnr = 10 * (torch.log(1 / mse_loss_image) / np.log(10))
            msssim=ms_ssim(X=recon_image,Y=input_image,win_size=11,win_sigma=1.5,data_range=1)
            # print(msssim) # 0.9992
            # exit() 
            # msssim=ms_ssim(X=recon_image,Y=input_image,win_size=11,win_sigma=1.5,win=None,data_range=1,size_average=True,full=False,weights=False)
            time_now = time.strftime("%Y-%m-%d:%H:%M:%S ", time.localtime())
            log0 = '[{:.2f}%]  >>> rd_loss:{:.6f} bpp(R):{:.6f} mse(D):{:.6f}'
            log0 = log0.format(100.*batch_idx/len_train_loader, rd_loss, bpp, mse_loss_image)
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
            image = input_image.clone().detach()
            torchvision.utils.save_image(image, image_save_path+str(cnt)+'org.png')
            image = ref_image.clone().detach()
            torchvision.utils.save_image(image, image_save_path+str(cnt)+'ref.png')
            image = recon_image.clone().detach()
            torchvision.utils.save_image(image, image_save_path+str(cnt)+'rec.png')
            cnt+=1
if __name__ == "__main__":
    print('pth_load_path:'+pth_load_path)
    net = Net()
    if pth_load_path != '':
        path_checkpoint = pth_load_path           # 断点路径
        print(path_checkpoint)
        checkpoint = torch.load(path_checkpoint)  # 加载断点
        net.load_state_dict({k.replace('module.',''):v for [k,v] in (checkpoint['net']).items()})  # 加载模型可学习参数
        bp_parameters = net.parameters()
        optimizer = torch.optim.Adam(bp_parameters, lr = base_lr, weight_decay = 5e-7)
        optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
        start_epoch= checkpoint['epoch']  # 设置开始的epoch
        global_step = start_epoch * (validate_dataset.__len__() * (gpu_per_batch))
    for para in net.parameters():
        para.requires_grad = False
    net = net.cuda()
    net = torch.nn.DataParallel(net.cuda(), list(range(gpu_num)))
    bp_parameters = net.parameters()
    optimizer = torch.optim.Adam(bp_parameters, lr=base_lr, weight_decay=5e-7)
    data_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    stepoch = global_step // (validate_dataset.__len__() // (gpu_per_batch))
    validate(0, testfull=True)
