from .base_metric import Metric
from typing import List
import torch


class NabfMetric(Metric):
    def __init__(self):
        super().__init__()
        # 预定义Sobel算子（在第一次使用时移动到对应设备）
        s1_single = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float64)
        s2_single = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float64)
        # 扩展为单通道
        self.s1 = s1_single.unsqueeze(0).unsqueeze(0)
        self.s2 = s2_single.unsqueeze(0).unsqueeze(0)

    def compute(self, fused_img: torch.Tensor, visible_img: torch.Tensor, infrared_img: torch.Tensor) -> List[float]:
        Td = 2
        wt_min = 0.001
        P = 1
        Lg = 1.5
        Nrg = 0.9999
        kg = 19
        sigmag = 0.5
        Nra = 0.9995
        ka = 22
        sigmaa = 0.5

        # 确保输入为双精度
        xrcw = fused_img.double()
        x1 = visible_img.double()
        x2 = infrared_img.double()

        # 获取边缘强度和方向
        gA, aA = self.get_g_a(x1)
        gB, aB = self.get_g_a(x2)
        gF, aF = self.get_g_a(xrcw)

        p, q = xrcw.shape
        gAF = torch.zeros_like(xrcw)
        gBF = torch.zeros_like(xrcw)

        zero_mask_A = (gA == 0) | (gF == 0)
        gAF[zero_mask_A] = 0
        greater_mask_A = gA > gF
        gAF[greater_mask_A] = gF[greater_mask_A] / gA[greater_mask_A]
        gAF[~zero_mask_A & ~greater_mask_A] = gA[~zero_mask_A & ~greater_mask_A] / gF[~zero_mask_A & ~greater_mask_A]

        zero_mask_B = (gB == 0) | (gF == 0)
        gBF[zero_mask_B] = 0
        greater_mask_B = gB > gF
        gBF[greater_mask_B] = gF[greater_mask_B] / gB[greater_mask_B]
        gBF[~zero_mask_B & ~greater_mask_B] = gB[~zero_mask_B & ~greater_mask_B] / gF[~zero_mask_B & ~greater_mask_B]

        aAF = torch.abs(torch.abs(aA - aF) - torch.pi / 2) * 2 / torch.pi
        aBF = torch.abs(torch.abs(aB - aF) - torch.pi / 2) * 2 / torch.pi

        QgAF = Nrg / (1 + torch.exp(-kg * (gAF - sigmag)))
        QaAF = Nra / (1 + torch.exp(-ka * (aAF - sigmaa)))
        QAF = torch.sqrt(QgAF * QaAF)

        QgBF = Nrg / (1 + torch.exp(-kg * (gBF - sigmag)))
        QaBF = Nra / (1 + torch.exp(-ka * (aBF - sigmaa)))
        QBF = torch.sqrt(QgBF * QaBF)

        wtA = wt_min * torch.ones_like(xrcw)
        wtB = wt_min * torch.ones_like(xrcw)
        cA = torch.ones_like(xrcw)
        cB = torch.ones_like(xrcw)

        Td_mask_A = gA >= Td
        wtA[Td_mask_A] = cA[Td_mask_A] * gA[Td_mask_A] ** Lg

        Td_mask_B = gB >= Td
        wtB[Td_mask_B] = cB[Td_mask_B] * gB[Td_mask_B] ** Lg

        wt_sum = torch.sum(wtA + wtB)
        QAF_wtsum = torch.sum(QAF * wtA) / wt_sum
        QBF_wtsum = torch.sum(QBF * wtB) / wt_sum
        QABF = QAF_wtsum + QBF_wtsum

        Qdelta = torch.abs(QAF - QBF)
        QCinfo = (QAF + QBF - Qdelta) / 2
        QdeltaAF = QAF - QCinfo
        QdeltaBF = QBF - QCinfo
        QdeltaAF_wtsum = torch.sum(QdeltaAF * wtA) / wt_sum
        QdeltaBF_wtsum = torch.sum(QdeltaBF * wtB) / wt_sum
        QdeltaABF = QdeltaAF_wtsum + QdeltaBF_wtsum
        QCinfo_wtsum = torch.sum(QCinfo * (wtA + wtB)) / wt_sum
        QABF11 = QdeltaABF + QCinfo_wtsum

        rr = torch.zeros_like(xrcw)
        rr[(gF <= gA) | (gF <= gB)] = 1
        LABF = torch.sum(rr * ((1 - QAF) * wtA + (1 - QBF) * wtB)) / wt_sum

        na1 = torch.zeros_like(xrcw)
        na1[(gF > gA) & (gF > gB)] = 2 - QAF[(gF > gA) & (gF > gB)] - QBF[(gF > gA) & (gF > gB)]
        NABF1 = torch.sum(na1 * (wtA + wtB)) / wt_sum

        na = torch.zeros_like(xrcw)
        na[(gF > gA) & (gF > gB)] = 1
        NABF = torch.sum(na * ((1 - QAF) * wtA + (1 - QBF) * wtB)) / wt_sum

        return [NABF.item()]

    def get_g_a(self, im: torch.Tensor):
        # 确保输入张量形状为 (N, C, H, W)
        if im.dim() == 2:
            im = im.unsqueeze(0).unsqueeze(0)  # 单通道灰度图
        elif im.dim() == 3:
            im = im.permute(2, 0, 1).unsqueeze(0)  # 将 (H, W, C) 转换为 (N, C, H, W)
        elif im.dim() != 4:
            raise ValueError(f"Unsupported input tensor dimension: {im.dim()}")

        # 确保Sobel算子在正确的设备上
        device = im.device
        s1 = self.s1.to(device)
        s2 = self.s2.to(device)
        s1 = s1.type_as(im)
        s2 = s2.type_as(im)

        # 使用卷积计算梯度（向量化）
        sx = torch.nn.functional.conv2d(im, s1, padding='same')
        sy = torch.nn.functional.conv2d(im, s2, padding='same')

        # 计算梯度和角度（向量化）
        g = torch.sqrt(sx.square() + sy.square())
        a = torch.atan2(sy, sx + torch.finfo(torch.float64).eps)

        return g.squeeze(), a.squeeze()

    @property
    def metric_names(self) -> List[str]:
        return ['Nabf']

    