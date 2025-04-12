from .base_metric import Metric
from typing import List
import torch
import torch.nn.functional as F


def sobel_kernel():
    w = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float64)
    return w


class EdgeIntensityMetric(Metric):
    def compute(self, fused_img: torch.Tensor, visible_img: torch.Tensor, infrared_img: torch.Tensor) -> List[float]:
        def edge_intensity(img):
            # 将输入图像转换为双精度浮点数，与 MATLAB 保持一致
            img = img.double()
            if img.dim() == 2:
                img = img.unsqueeze(0).unsqueeze(0)
            elif img.dim() == 3:
                img = img.permute(2, 0, 1).unsqueeze(0)
            elif img.dim() != 4:
                raise ValueError(f"Unsupported input tensor dimension: {img.dim()}")

            w = sobel_kernel().unsqueeze(0).unsqueeze(0).to(img.device)
            w = w.type_as(img)

            # 手动实现 'replicate' 边界填充
            padded_img = F.pad(img, (1, 1, 1, 1), mode='replicate')
            gx = F.conv2d(padded_img, w)
            gy = F.conv2d(padded_img, w.transpose(-1, -2))

            g = torch.sqrt(gx ** 2 + gy ** 2)
            outval = torch.mean(g)
            return outval.item()

        return [edge_intensity(fused_img)]

    @property
    def metric_names(self) -> List[str]:
        return ['edge_intensity']
    