# 20210830
import logging
import argparse
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from util.dataset import *
from util.drawuvg import HEVCdrawplt
from Net import Net
from util.VimeoDataset import Vimeo_all

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
pth_load_path = ''#./log/snapshot/' + str(train_lambda) + '_Vimeo_all_singleGPU/ckpt_best_0.pth'
save_path = './log/snapshot/'
print_step = 32#30
warmup_step = 0    
gpu_per_batch = gpu_num
test_step = 10000   
tot_epoch = 40
tot_step = 2000000
decay_interval = 1800000
lr_decay = 0.1
logger = logging.getLogger("test")
tb_logger = None
global_step = 0



def test(global_step, testfull=True):
    with torch.no_grad():
        test_loader = DataLoader(dataset=test_dataset, shuffle=False, num_workers=0, batch_size=1, pin_memory=True)
        net.eval()
        sumbpp = 0
        sumpsnr = 0
        summsssim = 0
        sumbpp_list=[]
        sumpsnr_list=[]
        summsssim_list=[]
        cnt = 0
        for batch_idx, input in enumerate(test_loader):
            if batch_idx % 10 == 0:
                print("testing : %d/%d" % (batch_idx, len(test_loader)))
            input_image = input[0]
            ref_image = input[1]
            seqlen = input_image.size()[1]
            sumbpp += 0         # torch.mean(ref_bpp).detach().numpy()
            sumpsnr += 0        # torch.mean(ref_psnr).detach().numpy()
            summsssim += 0      # torch.mean(ref_msssim).detach().numpy()
            cnt += 1
            for i in range(seqlen):
                # input_image = input_images[:, i, :, :, :]
                inputframe, refframe = Var(input_image), Var(ref_image)
                recon_frame, mse_loss, align_loss, total_bits, bpp \
                    = net(inputframe, refframe)
                bpp = torch.mean(bpp).cpu().detach().numpy()
                psnr = 10 * (torch.log(1 * 1 / mse_loss) / np.log(10)).cpu().detach().numpy()
                mssim = ms_ssim(recon_frame.cpu().detach(), input_image, data_range=1.0, size_average=True).numpy()
                print(bpp,  psnr, mssim)
                sumbpp += bpp
                sumpsnr += psnr
                summsssim += mssim
                cnt += 1
                ref_image = recon_frame
                if(cnt%10==0):
                    sumbpp_list.append(sumbpp/10)
                    sumpsnr_list.append(sumpsnr/10)
                    summsssim_list.append(summsssim/10)
                    sumbpp = 0
                    sumpsnr = 0
                    summsssim = 0
        for i in range(len(sumbpp_list)):
            sumbpp += sumbpp_list[i]
            sumpsnr += sumpsnr_list[i]
        sumpsnr /= len(sumbpp_list)
        sumbpp /= len(sumbpp_list)
        log = r"step : %d  average bpp : %.6lf, average psnr : %.6lf,\n" % (global_step, sumbpp, sumpsnr)#, summsssim)
        print(log)
        logger.info(log)
        # HEVCdrawplt([sumbpp], [sumpsnr], [summsssim], global_step, testfull=testfull)


if __name__ == "__main__":
    print(pth_name)
    if(pth_name[-6:]=='.model'):
        net = torch.load(pth_path)
        net = net.cuda()
        net = torch.nn.DataParallel(net.cuda(), device_ids=[0])
    if(pth_name[-4:]=='.pth'):
        net = Net()
        net.cuda()
        path_checkpoint = pth_path+pth_name
        checkpoint = torch.load(path_checkpoint)  # 加载断点
        net.load_state_dict({k.replace('module.',''):v for [k,v] in (checkpoint['net']).items()})
        bp_parameters = net.parameters()
        optimizer = optim.Adam(bp_parameters, lr = base_lr, weight_decay = 5e-7)
        optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
    data_transform = transforms.Compose([transforms.ToTensor()])
    print('testing')
    test(0, testfull=True)
    exit(0)

