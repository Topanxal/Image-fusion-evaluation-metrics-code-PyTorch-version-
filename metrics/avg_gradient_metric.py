from .base_metric import Metric
from typing import List
import torch


class AvgGradientMetric(Metric):
    def compute(self, fused_img: torch.Tensor, visible_img: torch.Tensor, infrared_img: torch.Tensor) -> List[float]:
        def avg_gradient(img):
            img = img.float()
            if img.dim() == 2:
                img = img.unsqueeze(2)
            r, c, b = img.shape
            g = []
            for k in range(b):
                band = img[:, :, k]
                dzdx, dzdy = torch.gradient(band)
                s = torch.sqrt((dzdx ** 2 + dzdy ** 2) / 2)
                g.append(torch.sum(s) / ((r - 1) * (c - 1)))
            return torch.mean(torch.tensor(g)).item()

        return [avg_gradient(fused_img)]

    @property
    def metric_names(self) -> List[str]:
        return ['avg_gradient']
    