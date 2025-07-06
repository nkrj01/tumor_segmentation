import torch
import torch.nn as nn
import torch.nn.functional as F

def reweight(cls_num_list, beta=0.9999): #beta=0.9999
    """
    Implement reweighting by effective numbers
    :param cls_num_list: a list containing # of samples of each class
    :param beta: hyper-parameter for reweighting, see paper for more details
    :return:
    """
    per_cls_weights = None

    per_cls_weights = [(1-beta)/(1-beta**n) for n in cls_num_list]

    per_cls_weights = torch.tensor(per_cls_weights)

    per_cls_weights =  per_cls_weights/torch.sum(per_cls_weights)*per_cls_weights.size(0)

    return per_cls_weights


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0):
        super().__init__()
        assert gamma >= 0
        self.gamma = gamma
        if weight is not None:
            self.register_buffer('weight_tensor', torch.tensor(weight))
        else:
            self.weight_tensor = None

    def forward(self, input, target):
        B, C, H, W = input.shape
        log_probs = F.log_softmax(input, dim=1)
        target_expanded = target.unsqueeze(1)
        log_p_t = torch.gather(log_probs, dim=1, index=target_expanded).squeeze(1)
        p_t = log_p_t.exp()

        focal_factor = (1 - p_t) ** self.gamma

        if self.weight_tensor is not None:
            pixel_weights = self.weight_tensor[target]
        else:
            pixel_weights = torch.ones_like(p_t)

        loss = -(focal_factor * pixel_weights * log_p_t).mean()
        return loss
