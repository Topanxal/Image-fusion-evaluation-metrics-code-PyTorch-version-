from .base_metric import Metric
from typing import List
import torch


class PSNRMetric(Metric):
    def compute(self, fused_img: torch.Tensor, visible_img: torch.Tensor, infrared_img: torch.Tensor) -> List[float]:
        def psnr(originalimg, restoredimg):
            md = (originalimg - restoredimg) ** 2
            summation = torch.sum(md)
            psnr = originalimg.numel() * torch.max(originalimg ** 2) / summation
            psnr = 10 * torch.log10(psnr)
            return psnr.item()

        psnr_visible = psnr(visible_img, fused_img)
        psnr_infrared = psnr(infrared_img, fused_img)
        return [(psnr_visible + psnr_infrared) / 2]

    @property
    def metric_names(self) -> List[str]:
        return ['psnr']
    