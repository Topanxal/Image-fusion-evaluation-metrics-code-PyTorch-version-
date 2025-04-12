from .base_metric import Metric
import torch
from typing import List


class ENMetric(Metric):
    """熵指标"""
    def compute(self, fused_img: torch.Tensor, visible_img: torch.Tensor, infrared_img: torch.Tensor) -> List[float]:
        grey_matrix = fused_img.squeeze()
        grey_level = 256
        # 确保像素值在 1 - 256 范围内
        grey_matrix = torch.clamp(grey_matrix, 1, 256).long()
        # 统计每个灰度级的出现次数
        counter = torch.bincount(grey_matrix.flatten(), minlength=grey_level).float()
        total = counter.sum()
        p = counter / total
        non_zero_mask = p > 0
        entropy = -torch.sum(p[non_zero_mask] * torch.log2(p[non_zero_mask]))
        return [entropy.item()]

    @property
    def metric_names(self) -> List[str]:
        return ['EN']
    