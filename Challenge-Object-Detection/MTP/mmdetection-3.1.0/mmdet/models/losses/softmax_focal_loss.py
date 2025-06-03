from mmcv.ops import SoftmaxFocalLoss
from mmdet.registry import MODELS
import torch


@MODELS.register_module()
class SoftmaxFocalLossMMDet(SoftmaxFocalLoss):
    def __init__(self, gamma=2.0, alpha=0.25, weight=None, reduction='mean', loss_weight=1.0):
        if weight is not None:
            weight = torch.tensor(weight, dtype=torch.float32)
        else:
            weight = None
        super().__init__(gamma=gamma, alpha=alpha, weight=weight, reduction=reduction)
        self.loss_weight = loss_weight

    def forward(self, input, target, weight=None, avg_factor=None, reduction_override=None):
        # Tu peux ignorer les arguments supplémentaires si tu ne les utilises pas
        loss = super().forward(input, target)
        return self.loss_weight * loss