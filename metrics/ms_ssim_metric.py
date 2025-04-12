from .base_metric import Metric
from typing import List
import torch


class MSSSIMMetric(Metric):
    def compute(self, fused_img: torch.Tensor, visible_img: torch.Tensor, infrared_img: torch.Tensor) -> List[float]:
        # 这里根据 MATLAB 代码简单设为 0，实际需要实现完整的 MS-SSIM 计算逻辑
        MS_SSIM = 0
        return [MS_SSIM]

    @property
    def metric_names(self) -> List[str]:
        return ['MS-SSIM']
    