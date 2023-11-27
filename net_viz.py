from deform_net import *
from torchviz import make_dot

x1 = torch.rand(8, 3, 256, 256)
x2 = torch.rand(8, 3, 256, 256)
# model = Net()
# y = model(x1, x2)
# g = make_dot(y)
# g.render('Net', view=False)

# model = Offset_Generator()
# y = model(x1, x2)
# g = make_dot(y)
# g.render('Offset_Generator', view=False)

from tensorboardX import SummaryWriter
#
# with SummaryWriter(comment = '_Net') as w:
#     w.add_graph(model, [x1,x2])

f1 = torch.rand(8, 64, 256, 256)
f2 = torch.rand(8, 64, 256, 256)
# f3 = torch.rand(1, 64, 256, 256)
# f4 = torch.rand(1, 64, 256, 256)
# model = Fusion()
# with SummaryWriter(comment = '_Fusion') as w:
    # w.add_graph(model, [f1,f2,f3,f4])
# y = model(f1,f2,f3,f4)
# g = make_dot(y)
# g.render('Fusion', view=False)
# # 画不出来的，用了200多个GB的内存！
# model = Analysis_mv_net()
offset = torch.rand(8, 64, 256, 256)
# y = model(offset)
# g = make_dot(y)
# g.render("Analysis_mv_net")

mv = torch.rand(8, 128, 16, 16)
# model = Synthesis_mv_net()
# y = model(mv)
# g = make_dot(y)
# g.render("Synthesis_mv_net")

# model = Compensation()
# y = model(offset,f2)
# g = make_dot(y)
# g.render("Compensation")
#
# fea_residual
# y = model(fea_residual)
# g = make_dot(y)
# g.render("Analysis_ne= torch.rand(8, 64, 256, 256)
# # model = Analysis_net()t")

fea_rec = torch.rand(8, 64, 256, 256)
model = Frame_recon()
y = model(fea_rec)
g = make_dot(y)
g.render("Frame_recon")