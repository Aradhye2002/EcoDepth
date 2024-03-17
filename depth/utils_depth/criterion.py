
import torch
import torch.nn as nn


class SiLogLoss(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.variance_focus = args.variance_focus
        self.args = args

    def forward(self, pred, target):
        valid_mask = (target > 0).detach()
        diff_log = torch.log(target[valid_mask]) - torch.log(pred[valid_mask])
        loss = torch.sqrt(torch.pow(diff_log, 2).mean() -
                          self.variance_focus * torch.pow(diff_log.mean(), 2)) * 10
        
        return loss

