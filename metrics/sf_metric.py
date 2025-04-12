from .base_metric import Metric
from typing import List
import torch


class SFMetric(Metric):
    def compute(self, fused_img: torch.Tensor, visible_img: torch.Tensor, infrared_img: torch.Tensor) -> List[float]:
        def sf(A):
            A = A.float()
            M, N = A.shape

            # Calculate RF
            RF = torch.sum((A[:, 1:] - A[:, :-1]) ** 2)
            RF = torch.sqrt(RF / (M * N))

            # Calculate CF
            CF = torch.sum((A[1:, :] - A[:-1, :]) ** 2)
            CF = torch.sqrt(CF / (M * N))

            # Calculate SF
            SF = torch.sqrt(RF ** 2 + CF ** 2)
            return SF.item()

        return [sf(fused_img)]

    @property
    def metric_names(self) -> List[str]:
        return ['SF']
    