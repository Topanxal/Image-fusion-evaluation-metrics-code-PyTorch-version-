from .base_metric import Metric
from typing import List
import torch
import torch.nn.functional as F


class Std2Metric(Metric):
    def compute(self, fused_img: torch.Tensor, visible_img: torch.Tensor, infrared_img: torch.Tensor) -> List[float]:
        def std2(a):
            fast_data_type = a.dtype in [torch.bool, torch.int8, torch.uint8, torch.int16, torch.uint16]
            big_enough = a.numel() > 2e5 if fast_data_type else False

            if fast_data_type and not torch.is_sparse(a) and big_enough:
                if a.dtype == torch.bool:
                    num_bins = 2
                else:
                    num_bins = torch.iinfo(a.dtype).max - torch.iinfo(a.dtype).min + 1

                bin_counts = F.histc(a.float(), bins=num_bins, min=torch.iinfo(a.dtype).min, max=torch.iinfo(a.dtype).max)
                bin_values = torch.arange(torch.iinfo(a.dtype).min, torch.iinfo(a.dtype).max + 1, dtype=torch.float32)

                total_pixels = a.numel()
                sum_of_pixels = torch.sum(bin_counts * bin_values)
                mean_pixel = sum_of_pixels / total_pixels

                bin_value_offsets = bin_values - mean_pixel
                bin_value_offsets_sqrd = bin_value_offsets ** 2

                offset_summation = torch.sum(bin_counts * bin_value_offsets_sqrd)
                s = torch.sqrt(offset_summation / total_pixels)
            else:
                if a.dtype != torch.float32:
                    a = a.float()
                s = torch.std(a.view(-1))
            return s.item()

        return [std2(fused_img)]

    @property
    def metric_names(self) -> List[str]:
        return ['std2']
    