import torch
import torch.nn as nn


class model1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3,3,3)
        
    def forward(self, x):
        x = self.conv(x)
        return x

net = model1()

opr = torch.optim.SGD(net.parameters(), 0.1)
lrs = torch.optim.lr_scheduler.CosineAnnealingLR(opr, 10)

from xnas.runner.scheduler import adjust_learning_rate_per_batch

for cur_epoch in range(10):
    print("epoch:{} lr:{}".format(cur_epoch, lrs.get_last_lr()[0]))
    opr.zero_grad()
    opr.step()
    lrs.step()

print("*"*20)
del opr, lrs

opr = torch.optim.SGD(net.parameters(), 0.1)
lrs = torch.optim.lr_scheduler.CosineAnnealingLR(opr, 10)

for cur_epoch in range(10):
    opr.zero_grad()
    for cur_iter in range(5):
        new_lr = adjust_learning_rate_per_batch(
            init_lr=0.1,
            n_epochs=10,
            n_warmup_epochs=3,
            epoch=cur_epoch,
            n_iter=5,
            iter=cur_iter,
            warmup=(cur_epoch < 3),
            warmup_lr=0.1,     # use base_lr as warmup_lr
        )
        for param_group in opr.param_groups:
            param_group["lr"] = new_lr
        if cur_iter == 0:
            print("epoch:{} iter:{} lr:{}".format(cur_epoch, cur_iter, new_lr))
    opr.step()
