import yaml
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import pandas as pd
from metrics.en_metric import ENMetric
from metrics.edge_association_metric import EdgeAssociationMetric
from utils.data_loader import get_dataloader
from utils.gpu_utils import init_gpu
from tqdm import tqdm
import re
# 导入所有指标类
from metrics.std2_metric import Std2Metric
from metrics.ssim_metric import SSIMMetric
from metrics.cc_metric import CCMetric
from metrics.sf_metric import SFMetric
from metrics.viff_metric import VIFFMetric
from metrics.psnr_metric import PSNRMetric
from metrics.avg_gradient_metric import AvgGradientMetric
from metrics.edge_intensity_metric import EdgeIntensityMetric
from metrics.figure_definition_metric import FigureDefinitionMetric
from metrics.relatively_warp_metric import RelativelyWarpMetric
from metrics.scd_metric import SCDMetric
from metrics.nabf_metric import NabfMetric
from metrics.ms_ssim_metric import MSSSIMMetric


def main(fused_dir):
    # 读取配置文件
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # 初始化 GPU
    gpu_ids = config.get('gpu_ids', [])
    device = init_gpu(gpu_ids)

    version = fused_dir.split('/')[-1]
    # 加载数据
    dataloader = get_dataloader(fused_dir, config['visible_dir'], config['infrared_dir'],
                                config['batch_size'], config['num_workers'])

    # 初始化指标类
    metrics_list = [
        EdgeAssociationMetric(),
        ENMetric(),
        Std2Metric(),
        SSIMMetric(),
        CCMetric(),
        SFMetric(),
        VIFFMetric(),
        PSNRMetric(),
        AvgGradientMetric(),
        EdgeIntensityMetric(),
        FigureDefinitionMetric(),
        RelativelyWarpMetric(),
        SCDMetric(),
        NabfMetric(),
        MSSSIMMetric()
    ]

    # 初始化结果列表
    results = []

    # 使用 tqdm 显示进度条
    for filenames, fused_images, visible_images, infrared_images in tqdm(dataloader, desc="Processing batches"):
        # 将数据移动到 GPU
        fused_images = fused_images.to(device)
        visible_images = visible_images.to(device)
        infrared_images = infrared_images.to(device)

        # 并行计算指标
        for i in range(len(fused_images)):
            filename = filenames[i]
            fused_image = fused_images[i]
            visible_image = visible_images[i]
            infrared_image = infrared_images[i]
            metrics = [filename]
            for metric in metrics_list:
                metrics.extend(metric.compute(fused_image, visible_image, infrared_image))
            results.append(metrics)

    # 生成列名
    column_names = ['filename']
    for metric in metrics_list:
        column_names.extend(metric.metric_names)

    # 创建 DataFrame
    df = pd.DataFrame(results, columns=column_names)

    # 从文件名中提取序号
    def extract_number(filename):
        match = re.search(r'\d+', filename)
        return int(match.group()) if match else float('inf')

    df['number'] = df['filename'].apply(extract_number)
    df = df.sort_values(by='number')
    df = df.drop(columns='number')

    # 保存结果
    df.to_csv(f'results/{version}.csv', index=False)


def get_subfolder_paths(folder_path):
    """
    获取给定文件夹下所有子文件夹的路径
    :param folder_path: 目标文件夹路径
    :return: 子文件夹路径列表
    """
    subfolder_paths = []
    
    # 遍历文件夹
    for root, dirs, files in os.walk(folder_path):
        for dir_name in dirs:
            # 获取子文件夹的完整路径
            dir_path = os.path.join(root, dir_name)
            subfolder_paths.append(dir_path)
    
    return subfolder_paths


if __name__ == "__main__":
    target_folder = "/home/hcj/HCJprogram/FDDB/codes/results/Dehazing/4-9_2rd"  # 替换为你的目标文件夹路径
    subfolders = get_subfolder_paths(target_folder)
    
    print("找到的子文件夹：")
    for root in subfolders:
        main(root)

    