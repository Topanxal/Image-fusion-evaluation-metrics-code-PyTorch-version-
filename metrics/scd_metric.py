from .base_metric import Metric
from typing import List
import torch


def corr2(x, y):
    x = x.flatten()
    y = y.flatten()
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)
    return torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))


class SCDMetric(Metric):
    def compute(self, fused_img: torch.Tensor, visible_img: torch.Tensor, infrared_img: torch.Tensor) -> List[float]:
        r = corr2(fused_img - infrared_img, visible_img) + corr2(fused_img - visible_img, infrared_img)
        return [r.item()]

    @property
    def metric_names(self) -> List[str]:
        return ['SCD']
    