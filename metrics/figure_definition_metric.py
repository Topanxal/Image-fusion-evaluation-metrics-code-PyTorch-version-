from .base_metric import Metric
from typing import List
import torch


class FigureDefinitionMetric(Metric):
    def compute(self, fused_img: torch.Tensor, visible_img: torch.Tensor, infrared_img: torch.Tensor) -> List[float]:
        def figure_definition(X):
            X = X.float()
            s, t = X.shape
            n = s * t
            x = X[:-1, :-1] - X[:-1, 1:]
            y = X[:-1, :-1] - X[1:, :-1]
            z = torch.sqrt((x ** 2 + y ** 2) / 2)
            G = torch.sum(z) / n
            return G.item()

        return [figure_definition(fused_img)]

    @property
    def metric_names(self) -> List[str]:
        return ['figure_definition']
    