import torch
from torch import nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=1,reduction='mean'): # 2
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
      
    def forward(self, input, target):
        target = target.float()
        beta = 1 - torch.mean(target)
        input = input[:, 0, :, :]
        weights = 1 - beta + (2 * beta - 1) * target
        bce_loss = F.binary_cross_entropy(input, target, weights, reduction=self.reduction )
        pt = torch.exp(-bce_loss)
        focal_loss = ((1 - pt) ** self.gamma * bce_loss).mean()
        return focal_loss

class FocalTverskyloss(nn.Module):
    def __init__(self, smooth=1, alpha=0.9, gamma=1): # 0.7, 0.75      #05_25: 0.9 0.75, 0.9 1 #05_24: 0.7 0.75 #05_23: 0.3 0.33
        super(FocalTverskyloss, self).__init__()
        self.smooth = smooth
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, input, target):
        target_pos = target.flatten()
        input_pos = input.flatten()
        true_pos = (target_pos * input_pos).sum()
        false_neg = (target_pos * (1 - input_pos)).sum()
        false_pos = ((1 - target_pos) * input_pos).sum()
        tversky = (true_pos + self.smooth) / (true_pos + self.alpha * false_neg + (1 - self.alpha) * false_pos + self.smooth)
        return pow((1 - tversky), self.gamma)

class WeightedBCELoss(nn.Module):
    def __init__(self):
        super(WeightedBCELoss, self).__init__()
    def forward(self, input, target):
        # _assert_no_grad(target)
        target = target.float()
        # input = input[0][0]
        beta = 1 - torch.mean(target)
        # input = F.softmax(input, dim=1)
        input = input[:, 0, :, :]
        # target pixel = 1 -> weight beta
        # target pixel = 0 -> weight 1-beta
        weights = 1 - beta + (2 * beta - 1) * target
        return F.binary_cross_entropy(input, target, weights, reduction='mean')
        # return torch.nn.functional.binary_cross_entropy_with_logits(input, target, weights, reduction='mean')

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        target = target.float()
        N = target.size(0)
        smooth = 1 # 1
        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)
        intersection = input_flat * target_flat
        loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth) # typo??
        # loss = (2 * intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss.sum() / N
        return loss

class BatchDiceLoss(nn.Module):
    def __init__(self):
        super(BatchDiceLoss, self).__init__()

    def forward(self, input, target):
        target = target.float()
        smooth = 1
        input_flat = input.view(1, -1)
        target_flat = target.view(1, -1)
        # N = input_flat.size(1)
        # N = target.size(0)
        intersection = input_flat * target_flat
        loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss
        return loss

class ComboLoss(nn.Module):
    def __init__(self):
        super(ComboLoss, self).__init__()
        self.dice = DiceLoss()
        self.bce = WeightedBCELoss()

    def forward(self, input, target):
        loss1 = self.bce(input, target)
        loss2 = self.dice(input, target)
        return 10*loss1 + loss2

class ComboLoss_focal(nn.Module):
    def __init__(self):
        super(ComboLoss_focal, self).__init__()
        self.bce = WeightedBCELoss()
        self.focal_tversky = FocalTverskyloss()

    def forward(self, input, target):
        loss1 = self.bce(input, target)
        loss2 = self.focal_tversky(input, target)
        return 10*loss1 + loss2

class ComboLoss_batch(nn.Module):
    def __init__(self):
        super(ComboLoss_batch, self).__init__()
        self.batch_dice = BatchDiceLoss()
        self.BCE = WeightedBCELoss()

    def forward(self, input, target):
        loss1 = self.BCE(input, target)
        loss2 = self.batch_dice(input, target)
        return 10*loss1 + loss2

myloss = {
    'weighted_bce': WeightedBCELoss,
    'dice': DiceLoss,
    'batch_dice': BatchDiceLoss,
    'combo': ComboLoss,
    'combo_focal': ComboLoss_focal,
    'combo_batch': ComboLoss_batch,
    'focal': FocalLoss,
    'focal_tversky': FocalTverskyloss
}