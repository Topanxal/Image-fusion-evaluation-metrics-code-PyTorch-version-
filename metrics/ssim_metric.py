from .base_metric import Metric
from typing import List
import torch


def ssim(img1, img2, window_size=11, K1=0.01, K2=0.03, L=255):
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    mu1 = torch.nn.functional.avg_pool2d(img1, window_size, stride=1, padding=window_size // 2)
    mu2 = torch.nn.functional.avg_pool2d(img2, window_size, stride=1, padding=window_size // 2)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = torch.nn.functional.avg_pool2d(img1 * img1, window_size, stride=1, padding=window_size // 2) - mu1_sq
    sigma2_sq = torch.nn.functional.avg_pool2d(img2 * img2, window_size, stride=1, padding=window_size // 2) - mu2_sq
    sigma12 = torch.nn.functional.avg_pool2d(img1 * img2, window_size, stride=1, padding=window_size // 2) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean().item()


class SSIMMetric(Metric):
    def compute(self, fused_img: torch.Tensor, visible_img: torch.Tensor, infrared_img: torch.Tensor) -> List[float]:
        fused_img = fused_img.unsqueeze(0).unsqueeze(0)
        visible_img = visible_img.unsqueeze(0).unsqueeze(0)
        infrared_img = infrared_img.unsqueeze(0).unsqueeze(0)

        I1 = ssim(visible_img, fused_img)
        I2 = ssim(infrared_img, fused_img)
        out = I1 + I2
        # 如果 out 已经是 float 类型，直接返回 [out]
        if isinstance(out, float):
            return [out]
        return [out.item()]

    @property
    def metric_names(self) -> List[str]:
        return ['SSIM']