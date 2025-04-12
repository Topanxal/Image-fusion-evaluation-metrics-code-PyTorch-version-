from .base_metric import Metric
from typing import List
import torch


def rgb_to_gray(img):
    if img.dim() == 3:
        return torch.mean(img, dim=0)
    return img


class RelativelyWarpMetric(Metric):
    def compute(self, fused_img: torch.Tensor, visible_img: torch.Tensor, infrared_img: torch.Tensor) -> List[float]:
        def relatively_warp(h1, h2):
            f1 = rgb_to_gray(h1)
            f2 = rgb_to_gray(h2)
            G1 = f1.float()
            G2 = f2.float()
            m1, n1 = G1.shape
            m2, n2 = G2.shape
            u1 = torch.sum(G1) / (m1 * n1)
            u2 = torch.sum(G2) / (m2 * n2)
            c1 = torch.sum((G1 - u1) ** 2)
            c2 = torch.sum((G2 - u2) ** 2)
            f1 = torch.sqrt(c1 / (m1 * n1))
            f2 = torch.sqrt(c2 / (m2 * n2))
            f = (f1 - f2) / f1
            return f.item()

        warp_visible = relatively_warp(visible_img, fused_img)
        warp_infrared = relatively_warp(infrared_img, fused_img)
        relatively_warp = 0.5 * warp_visible + 0.5 * warp_infrared
        return [relatively_warp]

    @property
    def metric_names(self) -> List[str]:
        return ['relatively_warp']
    