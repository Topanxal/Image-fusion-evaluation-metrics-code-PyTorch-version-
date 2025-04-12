from .base_metric import Metric
from typing import List
import torch


class CCMetric(Metric):
    def compute(self, fused_img: torch.Tensor, visible_img: torch.Tensor, infrared_img: torch.Tensor) -> List[float]:
        A = visible_img.float()
        B = infrared_img.float()
        F = fused_img.float()

        def corrcoef(x, y):
            x = x.view(-1)
            y = y.view(-1)
            vx = x - torch.mean(x)
            vy = y - torch.mean(y)
            corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
            return corr

        W1 = corrcoef(A, F)
        W2 = corrcoef(B, F)
        out = (W1 + W2) / 2
        return [out.item()]

    @property
    def metric_names(self) -> List[str]:
        return ['cc']
    