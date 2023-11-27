# pengfei20210903
import re   # 正则匹配的函数库
import numpy as np
from matplotlib import pyplot as plt
# config
train_log = './train256_Vimeo_all_singleGPU.log'                                 # 训练日志
save_path = './train_viz/'
train_name = train_log[:-4]
abs_log_file1 = save_path + train_name+'_1log.txt'         # 提取信息后的日记
abs_log_file2 = save_path + train_name+'_2log.txt'         # 提取信息后的日记
avgloss_png = save_path + train_name + '_avgloss.png'
bpp_png = save_path + train_name + '_bpp.png'
psnr_png = save_path + train_name + '_psnr.png'
alignpsnr_png = save_path + train_name + '_alignpsnr.png'

# ! config
# read_log
with open(train_log, 'r') as f1:      # 打开日记文件 train_file
    with open(abs_log_file1, 'w') as f2:    # 把日记文件的关键数据存到 log_file 中
        str1 = ''
        flag = False
        for line in f1:
            if '64600/64612 (100%)' in line:
                str1 = line[:-2]   # 不要换行符
                flag = True
                # print(line)
            elif(flag):
                f2.write(str1 + '\t' + line)
                flag = False
                # print(line)
end = 0     # 记录日记的长度
with open(abs_log_file1, 'r') as f3:
    with open(abs_log_file2, 'w') as f4:
        for line in f3:
            data = re.findall(r"\d+\.?\d*", str(line) )   # 正则表达式定位 [ ]
# Train Epoch : 22 [64600/64612 (100%)]
# Avgloss:0.002718 lr:0.0001 time:0.22351026000000002 bpp:0.001742350170388817
# details :  align_psnr : 53.94  psnr : 78.92
            epoch = str(float(data[0])+float(float(data[1])/float(data[2])))
            avgloss = data[-6]
            bpp = data[-3]
            align_psnr = data[-2]
            psnr = data[-1]
            f4.write(epoch + '\t' + avgloss + '\t' + bpp + '\t' + align_psnr + '\t' + psnr + '\n')           # 提取 [ ] 中的小数
            end += 1
# ! read_log
epoch = []
avgloss = []
bpp = []
align_psnr =[]
psnr = []
with open(abs_log_file2, 'r') as f5:
    for line in f5:
        data1 = float('%.2f'%float(line.split('\t')[0]))
        data2 = float('%.4f'%float(line.split('\t')[1]))
        data3 = float('%.6f'%float(line.split('\t')[2]))
        data4 = float('%.2f'%float(line.split('\t')[3]))
        data5 = float('%.2f'%float(line.split('\t')[4]))
        epoch.append(data1)
        avgloss.append(data2)
        bpp.append(data3)
        align_psnr.append(data4)
        psnr.append(data5)
# avgloss
plt1 =plt
plt1.title(train_name[7:-14])
plt1.xlabel('Epoch')
plt1.ylabel('Avgloss')
plt1.grid()      # 网格线
plt1.xticks(range(0,int(epoch[-1]),1))       # 横坐标的值和步长
plt1.xlim(left=0, right=epoch[-1])                  # 横坐标的最大长度
plt1.plot(epoch, avgloss)
plt1.savefig(avgloss_png)
plt1.close()  # 就是这里 一定要关闭
# !avgloss_plt
# bpp_plt
plt2 = plt
plt2.title(train_name[7:-14])
plt2.xlabel('Epoch')
plt2.ylabel('Bpp')
plt2.grid()      # 网格线
plt2.xticks(range(0,int(epoch[-1]),1))       # 横坐标的值和步长
plt2.xlim(left=0, right=epoch[-1])                  # 横坐标的最大长度
plt2.plot(epoch, bpp)
plt2.savefig(bpp_png)
plt2.close()  # 就是这里 一定要关闭
# !bpp_plt
# alignpsnr_plt
plt3 = plt
plt3.title(train_name[7:-14])
plt3.xlabel('Epoch')
plt3.ylabel('Align_Psnr')
plt3.grid()    # 网格线
plt3.xticks(range(0,int(epoch[-1]),1))       # 横坐标的值和步长
plt3.xlim(left=0, right=epoch[-1])                  # 横坐标的最大长度
plt3.plot(epoch, align_psnr)
plt3.savefig(alignpsnr_png)
plt3.close()  # 就是这里 一定要关闭
# !alignpsnr_plt
# psnr_plt
plt4 = plt
plt4.title(train_name[7:-14])
plt4.xlabel('Epoch')
plt4.ylabel('Psnr')
plt4.grid()      # 网格线
plt4.xticks(range(0,int(epoch[-1]),1))       # 横坐标的值和步长
plt4.xlim(left=0, right=epoch[-1])                  # 横坐标的最大长度
plt4.plot(epoch, psnr)
plt4.savefig(psnr_png)
plt4.close()  # 就是这里 一定要关闭
# !psnr_plt
# !plt

