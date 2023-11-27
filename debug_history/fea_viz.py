# -*- coding: utf-8 -*-
import os
import shutil
import time
import torch
from torch.functional import Tensor
import torchvision
import torchvision.utils as vutil
from torch.utils.data import DataLoader
from Net import Net
from util.VimeoDataset_4 import Vimeo_all
from util import ms_ssim
import numpy as np
INPUT_CHANNEL = 3
INPUT_SIZE = 256
IMAGE_FOLDER = './log/fea_viz/save_image'
INSTANCE_FOLDER = None
train_lambda = 1024
epoch = 38
name =  str(train_lambda) + 'train'
pth_load_path = './log/snapshot/' + name + '/ckpt_best_' + str(epoch) +'.pth'

def hook_func(module, input, output):
    """
    Hook function of register_forward_hook

    Parameters:
    -----------
    module: module of neural network
    input: input of module
    output: output of module
    """
    image_name = get_image_name_for_hook(module)
    if type(output)!=tuple:
        data = output.clone().detach()
        data = data.permute(1, 0, 2, 3)
        vutil.save_image(data, image_name, pad_value=0.5)
    else:
        data = None
        for i in output:
            if i.dim!=4:
                break
            data= i.clone().detach().permute(1, 0, 2, 3)
            vutil.save_image(data, image_name, pad_value=0.5)
    


def get_image_name_for_hook(module):
    """
    Generate image filename for hook function

    Parameters:
    -----------
    module: module of neural network
    """
    os.makedirs(INSTANCE_FOLDER, exist_ok=True)
    base_name = str(module).split('(')[0]
    index = 0
    image_name = '.'  # '.' is surely exist, to make first loop condition True
    while os.path.exists(image_name):
        index += 1
        image_name = os.path.join(
            INSTANCE_FOLDER, '%s_%d.png' % (base_name, index))
    return image_name


if __name__ == '__main__':
    time_beg = time.time()
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
    tb_logger = None
    global_step = 0
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

    net = net.cuda()
    net = torch.nn.DataParallel(net.cuda(), list(range(gpu_num)))
    bp_parameters = net.parameters()
    optimizer = torch.optim.Adam(bp_parameters, lr=base_lr, weight_decay=5e-7)
    data_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    train_loss = []

    # ----------------- visualization -----------------

    with torch.no_grad():
        validate_loader = DataLoader(dataset=validate_dataset, shuffle=False, num_workers=0, batch_size=1, pin_memory=True)
        len_validate_loader=len(validate_loader)
        for batch_idx, input in enumerate(validate_loader):
            print("validate : %d/%d" % (batch_idx, len_validate_loader))
            INSTANCE_FOLDER = os.path.join(
                IMAGE_FOLDER, '%d' % (batch_idx))
            for name, module in net.named_modules():
                module.register_forward_hook(hook_func)
            optimizer.zero_grad()
            ref_image, input_image = (input[0]).cuda(), (input[1]).cuda()
            outputs= net(input_image, ref_image)
            if batch_idx > 10:
                break

