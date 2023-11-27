import os
# data_root = os.path.abspath(os.path.join(os.getcwd()))
# root = data_root + "/clean"
# root = data_root + "/flow"
#
root = '/data1/Data/HEVC_dataset/Class_E/KristenAndSara_1280x704_60'
#构建所有文件名的列表
filename = []
dirs = os.listdir(root)
dirs = sorted(dirs, key=lambda x: int(x[2:-4]))
for dir in dirs:
    filename.append(root + "/" + dir)
    # dir_path = root + '/' + dir
    # names = os.listdir(dir_path)
    # names = sorted(names, key=lambda x: int(x[:-4]))
    # for n in names:
    #     filename.append(dir_path + '/' + n)
# train = filename[:int(len(filename))]
# # test = filename[int(len(filename)*0.8):]
with open('KristenAndSara.txt', 'w') as f1:
    for i in filename:
        f1.write(i + '\n')
    # for j in test:
    #     f2.write(j + '\n')
print('成功！')
# foldernames = os.listdir(root)
# filelist = []
# # filelist_recon = []
# for foldername in foldernames:
#     folderroot = os.path.join(root, foldername)
#     imgnames = os.listdir(folderroot)
#     imgnames = sorted(imgnames, key=lambda x: str(x[:]))
#     for imgname in imgnames:
#         y = os.path.join(root, foldername, imgname) + '\n'
#         filelist.append(y)
#         # y_recon = y.replace('truth', 'construction')
#         # filelist_recon.append(y_recon)
# filelist.sort()
# # filelist_recon.sort()
# # print(filelist)
# # print(filelist_recon)
# with open('./filelist.txt','w') as f:
#     f.writelines(filelist)