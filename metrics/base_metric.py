from typing import List
import torch

class Metric:
    """
    这是一个用于计算图像融合指标的基类。
    所有具体的指标类都应该继承这个基类，并实现 compute 方法和 metric_names 属性。
    """
    def compute(self, fused_img: torch.Tensor, visible_img: torch.Tensor, infrared_img: torch.Tensor) -> List[float]:
        """
        此方法用于计算具体的指标值。
        参数:
            fused_img (torch.Tensor): 融合后的图像张量。
            visible_img (torch.Tensor): 原始可见光图像张量。
            infrared_img (torch.Tensor): 原始红外图像张量。
        返回:
            List[float]: 计算得到的指标值列表。
        """
        raise NotImplementedError("子类必须实现 compute 方法。")

    @property
    def metric_names(self) -> List[str]:
        """
        此属性用于获取指标的名称列表。
        返回:
            List[str]: 指标的名称列表。
        """
        raise NotImplementedError("子类必须实现 metric_names 属性。")
    