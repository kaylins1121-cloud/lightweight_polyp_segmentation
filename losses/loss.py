import torch
import torch.nn as nn

class StructureLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        # BCE Loss
        bce_out = self.bce(pred, target)
        
        # Dice Loss
        pred_sig = torch.sigmoid(pred).view(-1)
        target_flat = target.view(-1)
        inter = (pred_sig * target_flat).sum()
        dice_out = 1. - (2. * inter + 1.0) / (pred_sig.sum() + target_flat.sum() + 1.0)
        
        return bce_out + dice_out
