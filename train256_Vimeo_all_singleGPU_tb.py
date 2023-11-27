from deform_net import *
from torchviz import make_dot

x1 = torch.rand(8, 3, 256, 512)
x2 = torch.rand(8, 3, 256, 512)
model = Net()
y = model(x1, x2)
g = make_dot(y)
g.render('Net', view=False)
# 会自动保存为一个 Net.pdf，第二个参数为True,则会自动打开该PDF文件，为False则不打开

#
# # # 可视化
# import torch
# from torchvision.models import AlexNet
# # from net import *
# from deform_net import *
# from torchviz import make_dot
#
# x1 = torch.rand(8, 3, 256, 512)
# x2 = torch.rand(8, 3, 256, 512)
# # model = AlexNet()
# model = Net()
#
# y = model(x1, x2)
# # 这三种方式都可以
# g = make_dot(y)
# # g=make_dot(y, params=dict(model.named_parameters()))
# # g = make_dot(y, params=dict(list(model.named_parameters()) + [('x', x)]))
# # 这两种方法都可以
# # g.view() # 会生成一个 Digraph.gv.pdf 的PDF文件
# # g.render('espnet_model', view=False) # 会自动保存为一个 espnet.pdf，第二个参数为True,则会自动打开该PDF文件，为False则不打开
# g.render('Net', view=False) # 会自动保存为一个 espnet.pdf，第二个参数为True,则会自动打开该PDF文件，为False则不打开
#
#
# # import torch
# # from torchvision.models import AlexNet
# #
# # from tensorboardX import SummaryWriter
# #
# # x = torch.rand(8, 3, 256, 512)
# # model = AlexNet()
# #
# # with SummaryWriter(comment = 'AlexNet') as w:
# #     w.add_graph(model, x)  # 这其实和tensorflow里面的summarywriter是一样的。