from .base_metric import Metric
from typing import List
import torch
import torch.nn.functional as F


def gaussian_kernel(size, sigma):
    x, y = torch.meshgrid(torch.arange(-size // 2 + 1, size // 2 + 1), torch.arange(-size // 2 + 1, size // 2 + 1))
    kernel = torch.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    return kernel / kernel.sum()


def ComVidVindG(ref, dist, sq):
    sigma_nsq = sq
    Num = []
    Den = []
    G = []

    for scale in range(1, 5):
        N = 2 ** (4 - scale + 1) + 1
        win = gaussian_kernel(N, N / 5).unsqueeze(0).unsqueeze(0).to(ref.device)

        if scale > 1:
            ref = F.conv2d(ref.unsqueeze(0).unsqueeze(0), win, padding='valid').squeeze()
            dist = F.conv2d(dist.unsqueeze(0).unsqueeze(0), win, padding='valid').squeeze()
            ref = ref[::2, ::2]
            dist = dist[::2, ::2]

        mu1 = F.conv2d(ref.unsqueeze(0).unsqueeze(0), win, padding='valid').squeeze()
        mu2 = F.conv2d(dist.unsqueeze(0).unsqueeze(0), win, padding='valid').squeeze()
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = F.conv2d((ref * ref).unsqueeze(0).unsqueeze(0), win, padding='valid').squeeze() - mu1_sq
        sigma2_sq = F.conv2d((dist * dist).unsqueeze(0).unsqueeze(0), win, padding='valid').squeeze() - mu2_sq
        sigma12 = F.conv2d((ref * dist).unsqueeze(0).unsqueeze(0), win, padding='valid').squeeze() - mu1_mu2

        sigma1_sq = torch.clamp(sigma1_sq, min=0)
        sigma2_sq = torch.clamp(sigma2_sq, min=0)

        g = sigma12 / (sigma1_sq + 1e-10)
        sv_sq = sigma2_sq - g * sigma12

        mask1 = sigma1_sq < 1e-10
        g[mask1] = 0
        sv_sq[mask1] = sigma2_sq[mask1]
        sigma1_sq[mask1] = 0

        mask2 = sigma2_sq < 1e-10
        g[mask2] = 0
        sv_sq[mask2] = 0

        mask3 = g < 0
        sv_sq[mask3] = sigma2_sq[mask3]
        g[mask3] = 0
        sv_sq = torch.clamp(sv_sq, min=1e-10)

        G.append(g)
        VID = torch.log10(1 + g ** 2 * sigma1_sq / (sv_sq + sigma_nsq))
        VIND = torch.log10(1 + sigma1_sq / sigma_nsq)
        Num.append(VID)
        Den.append(VIND)

    return Num, Den, G


class VIFFMetric(Metric):
    def compute(self, fused_img: torch.Tensor, visible_img: torch.Tensor, infrared_img: torch.Tensor) -> List[float]:
        sq = 0.005 * 255 * 255
        C = 1e-7
        p = torch.tensor([1, 0, 0.15, 1]) / 2.15

        if fused_img.dim() == 3:  # Assume RGB image
            # Here we only use the first channel for simplicity
            Ix1 = visible_img[0]
            Ix2 = infrared_img[0]
            IxF = fused_img[0]
        else:
            Ix1 = visible_img
            Ix2 = infrared_img
            IxF = fused_img

        T1p = Ix1.float()
        T2p = Ix2.float()
        Trp = IxF.float()

        T1N, T1D, T1G = ComVidVindG(T1p, Trp, sq)
        T2N, T2D, T2G = ComVidVindG(T2p, Trp, sq)

        F = []
        for i in range(4):
            M_Z1 = T1N[i]
            M_Z2 = T2N[i]
            M_M1 = T1D[i]
            M_M2 = T2D[i]
            M_G1 = T1G[i]
            M_G2 = T2G[i]
            L = M_G1 < M_G2
            M_G = M_G2.clone()
            M_G[L] = M_G1[L]
            M_Z12 = M_Z2.clone()
            M_Z12[L] = M_Z1[L]
            M_M12 = M_M2.clone()
            M_M12[L] = M_M1[L]

            VID = torch.sum(M_Z12 + C)
            VIND = torch.sum(M_M12 + C)
            F.append(VID / VIND)

        F = torch.tensor(F).to(fused_img.device)
        # 将 p 移动到和 F 相同的设备上
        p = p.to(F.device)
        F = torch.sum(F * p)
        return [F.item()]

    @property
    def metric_names(self) -> List[str]:
        return ['VIFF']