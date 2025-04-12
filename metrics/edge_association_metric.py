from .base_metric import Metric
import torch
from typing import List


class EdgeAssociationMetric(Metric):
    """边缘关联度指标（GPU优化版）"""

    def __init__(self):
        super().__init__()
        # 预定义Sobel算子
        self.s1 = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float64).unsqueeze(0).unsqueeze(0)
        self.s2 = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float64).unsqueeze(0).unsqueeze(0)

    def compute(self, fused_img: torch.Tensor, visible_img: torch.Tensor, infrared_img: torch.Tensor) -> List[float]:
        device = fused_img.device
        A = visible_img.to(device).to(torch.float64).unsqueeze(0).unsqueeze(0)
        B = infrared_img.to(device).to(torch.float64).unsqueeze(0).unsqueeze(0)
        F = fused_img.to(device).to(torch.float64).unsqueeze(0).unsqueeze(0)

        gA, aA = self.get_g_a(A)
        gB, aB = self.get_g_a(B)
        gF, aF = self.get_g_a(F)

        GAF = self.get2G(gA, gF)
        GBF = self.get2G(gB, gF)
        AAF = self.get2A(aA, aF)
        ABF = self.get2A(aB, aF)

        QgAF = self.getQg(GAF, 0.9994, -15, 0.5)
        QgBF = self.getQg(GBF, 0.9994, -15, 0.5)
        QaAF = self.getQa(AAF, 0.9879, -22, 0.8)
        QaBF = self.getQa(ABF, 0.9879, -22, 0.8)

        QAF = self.getQ(QgAF, QaAF)
        QBF = self.getQ(QgBF, QaBF)

        numerator = torch.sum(QAF * gA + QBF * gB)
        denominator = torch.sum(gA + gB)
        edge_association = numerator / (denominator + torch.finfo(torch.float64).eps)
        return [edge_association.item()]

    @property
    def metric_names(self) -> List[str]:
        return ['edge_association']

    def get_g_a(self, im: torch.Tensor):
        device = im.device
        s1 = self.s1.to(device)
        s2 = self.s2.to(device)
        sx = torch.nn.functional.conv2d(im, s1, padding='same')
        sy = torch.nn.functional.conv2d(im, s2, padding='same')
        g = torch.sqrt(sx.square() + sy.square())
        a = torch.atan2(sy, sx + torch.finfo(torch.float64).eps)
        return g.squeeze(), a.squeeze()

    def getQ(self, Qg: torch.Tensor, Qa: torch.Tensor):
        return Qg * Qa

    def getQa(self, A: torch.Tensor, Ta: float, Ka: float, Oa: float):
        return Ta / (1 + torch.exp(Ka * (A - Oa)))

    def getQg(self, G: torch.Tensor, Tg: float, Kg: float, Og: float):
        return Tg / (1 + torch.exp(Kg * (G - Og)))

    def get2A(self, aIm1: torch.Tensor, aIm2: torch.Tensor):
        return 1 - torch.abs((aIm1 - aIm2) * (2 / torch.pi))

    def get2G(self, gIm1: torch.Tensor, gIm2: torch.Tensor):
        min_g = torch.minimum(gIm1, gIm2)
        max_g = torch.maximum(gIm1, gIm2)
        return min_g / (max_g + torch.finfo(torch.float64).eps)
    