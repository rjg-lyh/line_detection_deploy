import torch.nn as nn
import torch.nn.functional as F
import torch 

class FocalLoss(nn.Module):
    def __init__(self, type, alpha = 0.75, gamma = 2.0):
        super(FocalLoss, self).__init__()
        self.type = type
        self.alpha = alpha
        self.gamma = gamma
        if self.type == 'bce':
            self.loss_func = nn.BCELoss(reduction='none')
        else:
            self.loss_func = nn.CrossEntropyLoss(reduction='none')

    def forward(self, preds, labels):
        alpha = self.alpha
        gamma = self.gamma
        if self.type == 'bce':
            weight = alpha * (1 - preds).pow(gamma) * labels + (1 - alpha) * preds.pow(gamma) * (1 - labels)
            loss = self.loss_func(preds, labels)
            focal_loss = loss*weight
        else:
            loss = self.loss_func(preds, labels)
            alpha = 1
            pt = torch.exp(-loss)
            focal_loss = alpha * (1-pt)**gamma * loss
        return focal_loss.mean()


class criterion(nn.Module):
    def __init__(self, type, need_focal=False, alpha=0.75, gamma=2.0):
        super(criterion, self).__init__()
        self.type = type
        if not need_focal:
            if type == 'bce':
                self.loss_func = nn.BCELoss(reduction='mean')
            else:
                self.loss_func = nn.CrossEntropyLoss(reduction='mean')
        else:
            self.loss_func = FocalLoss(type, alpha, gamma)
    
    def forward(self, preds, labels):
        if self.type == 'bce':
            labels = labels.to(torch.float32)
            preds = preds.sigmoid()
        return self.loss_func(preds, labels)

            
