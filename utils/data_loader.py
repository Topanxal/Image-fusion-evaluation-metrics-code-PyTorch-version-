import torch
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import numpy as np


class ImageFusionDataset(Dataset):
    def __init__(self, fused_dir, visible_dir, infrared_dir):
        self.fused_dir = fused_dir
        self.visible_dir = visible_dir
        self.infrared_dir = infrared_dir
        self.filenames = os.listdir(fused_dir)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        # 转换为灰度图
        fused_image = Image.open(os.path.join(self.fused_dir, filename)).convert('L')
        visible_image = Image.open(os.path.join(self.visible_dir, filename)).convert('L')
        infrared_image = Image.open(os.path.join(self.infrared_dir, filename)).convert('L')

        # 转换为 numpy 数组再转换为 PyTorch 张量
        fused_image = torch.from_numpy(np.array(fused_image)).float()
        visible_image = torch.from_numpy(np.array(visible_image)).float()
        infrared_image = torch.from_numpy(np.array(infrared_image)).float()

        return filename, fused_image, visible_image, infrared_image


def get_dataloader(fused_dir, visible_dir, infrared_dir, batch_size, num_workers):
    dataset = ImageFusionDataset(fused_dir, visible_dir, infrared_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    return dataloader
    