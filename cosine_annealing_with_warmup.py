import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

def lr_update(epoch, lr_warmup_epoch, optimizer, scheduler):
  if 0 <= epoch < lr_warmup_epoch:
    mul_rate = 10 ** (1/lr_warmup_epoch)

    for param_group in optimizer.param_groups:
        param_group['lr'] *= mul_rate
  
  else:
    scheduler.step()
