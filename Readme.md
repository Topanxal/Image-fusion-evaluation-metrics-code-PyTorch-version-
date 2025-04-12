# 图像融合指标计算项目(Torch 加速版)

## 项目概述

本项目旨在计算图像融合结果的多种指标，用于评估图像融合的质量。通过读取配置文件，加载融合图像、可见光图像和红外图像数据，并使用多个指标类对图像进行并行计算，最终将计算结果保存为 CSV 文件。主要核心在于使用 Torch 加速，运行 FMB 测试集的 280 张图片，使用单张 4090，按照默认的参数平均时间为 5s，相较于之前开源的 MATLAB 版本测试代码，速度大大提高。

### 项目并行策略介绍

本项目为提高计算速度，采用了三种维度的并行策略，下面将结合 `metrics/main.py` 文件及相关代码详细介绍这些策略。

#### 1. 数据批量并行

- **原理**：数据批量并行是指在一次计算中同时处理多个数据样本，通过将多个数据样本组合成一个批次（batch），可以充分利用 GPU 的并行计算能力，减少单次计算的开销。
- **代码体现**：
  - 在 `main.py` 中，通过 `get_dataloader` 函数创建数据加载器，利用 `torch.utils.data.DataLoader` 来实现数据批量处理。`config['batch_size']` 参数指定了每个批次包含的数据样本数量。

```python
dataloader = get_dataloader(fused_dir, config['visible_dir'], config['infrared_dir'],
                            config['batch_size'], config['num_workers'])
```

    - 在遍历数据加载器时，每次获取一个批次的数据，包括文件名、融合图像、可见光图像和红外图像。

```python
for filenames, fused_images, visible_images, infrared_images in tqdm(dataloader, desc="Processing batches"):
    # 将数据移动到 GPU
    fused_images = fused_images.to(device)
    visible_images = visible_images.to(device)
    infrared_images = infrared_images.to(device)
```

    - 这样，一次可以处理多个图像数据，提高了计算效率。

#### 2. 多指标并行计算

- **原理**：多指标并行计算是指同时对多个指标进行计算，而不是依次计算每个指标。在本项目中，需要计算多个图像融合指标，如边缘关联度、熵、结构相似性等。通过并行计算这些指标，可以充分利用 GPU 的多核心计算能力，提高整体计算速度。
- **代码体现**：
  - 在 `main.py` 中，初始化了一个指标类列表 `metrics_list`，包含了所有需要计算的指标类。

```python
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
```

    - 在处理每个图像数据时，通过循环遍历指标类列表，依次调用每个指标类的 `compute` 方法进行计算。

```python
for i in range(len(fused_images)):
    filename = filenames[i]
    fused_image = fused_images[i]
    visible_image = visible_images[i]
    infrared_image = infrared_images[i]
    metrics = [filename]
    for metric in metrics_list:
        metrics.extend(metric.compute(fused_image, visible_image, infrared_image))
    results.append(metrics)
```

    - 由于每个指标的计算是相互独立的，因此可以在 GPU 上并行执行这些计算，提高了计算效率。

#### 3. 基于 `torch` 张量的并行计算

- **原理**：`torch` 提供了丰富的张量运算函数，这些函数能够在 GPU 上并行执行，避免了使用显式的 Python 循环带来的性能开销。在指标计算过程中，充分利用这些函数进行张量运算，从而加速计算。
- **代码体现**：
  - **卷积操作**：在多个指标计算类中，如 `EdgeAssociationMetric` 的 `get_g_a` 方法，使用 `torch.nn.functional.conv2d` 函数进行卷积操作。

```python
def get_g_a(self, im: torch.Tensor):
    device = im.device
    s1 = self.s1.to(device)
    s2 = self.s2.to(device)
    sx = torch.nn.functional.conv2d(im, s1, padding='same')
    sy = torch.nn.functional.conv2d(im, s2, padding='same')
    g = torch.sqrt(sx.square() + sy.square())
    a = torch.atan2(sy, sx + torch.finfo(torch.float64).eps)
    return g.squeeze(), a.squeeze()
```

    - 该函数会并行处理输入图像的每个像素，避免了使用显式的 Python 循环，大大提高了计算效率。
    - **张量运算**：在 `ENMetric` 的 `compute` 方法中，使用 `torch.clamp`、`torch.bincount`、`torch.sum` 等函数进行张量运算。

```python
def compute(self, fused_img: torch.Tensor, visible_img: torch.Tensor, infrared_img: torch.Tensor) -> List[float]:
    grey_matrix = fused_img.squeeze()
    grey_level = 256
    # 确保像素值在 1 - 256 范围内
    grey_matrix = torch.clamp(grey_matrix, 1, 256).long()
    # 统计每个灰度级的出现次数
    counter = torch.bincount(grey_matrix.flatten(), minlength=grey_level).float()
    total = counter.sum()
    p = counter / total
    non_zero_mask = p > 0
    entropy = -torch.sum(p[non_zero_mask] * torch.log2(p[non_zero_mask]))
    return [entropy.item()]
```

    - 这些函数都是基于张量的并行运算，能够高效地处理大规模数据。

### 总结

本项目通过数据批量并行、多指标并行计算和基于 `torch` 张量的并行计算三种维度的并行策略，充分利用了 GPU 的并行计算能力，提高了图像融合指标计算的速度。这些策略的结合使用是本项目的一个特色，可以有效处理大规模的图像数据。

## 安装说明

### 依赖库

本项目依赖于以下 Python 库：

- `torch`：用于深度学习计算和张量操作。
- `yaml`：用于读取配置文件。
- `pandas`：用于数据处理和保存结果为 CSV 文件。
- `PIL`（Pillow）：用于图像读取和处理。
- `tqdm`：用于显示进度条。
- `re`：用于正则表达式匹配。

### 安装命令

可以使用以下命令安装所需依赖：

```bash
pip install torch pyyaml pandas pillow tqdm
```

## 使用方法

### 配置文件设置

在 `config/config.yaml` 文件中设置融合图像、可见光图像和红外图像的目录，以及批量大小、工作进程数量和要使用的 GPU 编号等参数：

```yaml
fused_dir: ~ # 替换为你的融合结果文件夹路径
visible_dir: ~ # 替换为你的可见光文件夹路径
infrared_dir: ~ # 替换为你的红外文件夹路径
batch_size: 8
num_workers: 4
gpu_ids: [0，1] # 指定要使用的 GPU 编号，若为空则使用所有可用 GPU
```

### 运行项目

在项目根目录下运行以下命令：

```bash
python metrics/main.py
```

如果需要处理多个子文件夹下的融合图像，可以修改 `main.py` 文件中的 `target_folder` 变量为目标文件夹路径，程序将自动处理该文件夹下的所有子文件夹：

```python
if __name__ == "__main__":
    target_folder = "~"  # 替换为你的目标文件夹路径
    subfolders = get_subfolder_paths(target_folder)

    print("找到的子文件夹：")
    for root in subfolders:
        main(root)
```

## 部分指标说明

### 边缘关联度（edge_association）

衡量融合图像与原始可见光和红外图像边缘信息的关联程度。值越高，表示融合图像在边缘信息上与原始图像的关联越好。

### 熵（EN）

反映融合图像的信息丰富程度。熵值越大，说明图像包含的信息越多。

### 结构相似性（SSIM）

评估融合图像与原始图像在结构上的相似性。值越接近 1，表示结构越相似。

### 其他指标

还包括 Std2、CC、SF、VIFF、PSNR、AvgGradient、EdgeIntensity、FigureDefinition、RelativelyWarp、SCD、Nabf 和 MS-SSIM 等指标，具体含义可参考相关文献。

## 项目结构

```
metrics/
├── config/           # 配置文件目录
│   └── config.yaml
├── metrics/          # 指标计算类目录
│   ├── base_metric.py
│   ├── en_metric.py
│   ├── edge_association_metric.py
│   ├── ssim_metric.py
│   ...
├── utils/            # 工具函数目录
│   ├── data_loader.py
│   ├── gpu_utils.py
│   └── __init__.py
└── main.py           # 主程序文件
```

## 代码解释

### `main.py`

- 读取配置文件 `config/config.yaml`，获取图像目录、批量大小、工作进程数量和 GPU 编号等参数。
- 初始化 GPU。
- 调用 `get_dataloader` 函数加载数据。
- 初始化所有指标类。
- 遍历数据加载器，对每个批次的图像数据进行处理，计算各项指标。
- 将计算结果保存为 CSV 文件，文件名根据融合图像文件夹的名称生成。

### `utils/data_loader.py`

- `ImageFusionDataset` 类：用于加载融合图像、可见光图像和红外图像数据，并将图像转换为灰度图和 PyTorch 张量。
- `get_dataloader` 函数：根据给定的图像目录、批量大小和工作进程数量，创建数据加载器。

### `metrics/` 目录下的指标类

每个指标类都继承自 `Metric` 基类，实现了 `compute` 方法，用于计算相应的指标值。

### 与 MATLAB 版本的对比测试结果

为了验证本项目计算结果的准确性，我们将本项目的计算结果与 MATLAB 版本进行了对比测试，针对同一份图像融合结果计算了各项指标的差值。以下是测试得到的对比数据：

| 指标              | 指标平均值 | 差值平均值 | 绝对差值平均值 | 相对差异（差值原始）（%） | 相对绝对差异（差值/原始）（%） | 差值方差 | 绝对差值方差 |
| ----------------- | ---------- | ---------- | -------------- | ------------------------- | ------------------------------ | -------- | ------------ |
| Nabf              | 0.03       | 0.00       | 0.00           | 7.10                      | 9.78                           | 0.00     | 0.00         |
| SSIM              | 1.41       | -0.03      | 0.03           | -2.20                     | 2.20                           | 0.00     | 0.00         |
| edge_association  | 0.63       | -0.01      | 0.01           | -0.84                     | 1.03                           | 0.00     | 0.00         |
| EN                | 6.57       | 0.00       | 0.00           | -0.01                     | 0.01                           | 0.00     | 0.00         |
| VIFF              | 0.41       | 0.00       | 0.00           | 0.00                      | 0.00                           | 0.00     | 0.00         |
| relatively_warp   | 0.15       | 0.00       | 0.00           | 0.00                      | 0.00                           | 0.00     | 0.00         |
| figure_definition | 5.93       | 0.00       | 0.00           | 0.00                      | 0.00                           | 0.00     | 0.00         |
| SCD               | 1.44       | 0.00       | 0.00           | 0.00                      | 0.00                           | 0.00     | 0.00         |
| std2              | 28.64      | 0.00       | 0.00           | 0.00                      | 0.00                           | 0.00     | 0.00         |
| avg_gradient      | 4.76       | 0.00       | 0.00           | 0.00                      | 0.00                           | 0.00     | 0.00         |
| cc                | 0.63       | 0.00       | 0.00           | 0.00                      | 0.00                           | 0.00     | 0.00         |
| edge_intensity    | 48.70      | 0.00       | 0.00           | 0.00                      | 0.00                           | 0.00     | 0.00         |
| SF                | 15.73      | 0.00       | 0.00           | 0.00                      | 0.00                           | 0.00     | 0.00         |
| psnr              | 17.16      | 0.00       | 0.00           | 0.00                      | 0.00                           | 0.00     | 0.00         |
| MS - SSIM         | 0          | 0          | 0              |                           |                                | 0        | 0            |

#### 结果分析

从上述对比数据可以看出，大部分指标的差值平均值、绝对差值平均值、差值方差和绝对差值方差都接近 0，相对差异和相对绝对差异也极小，这表明本项目在这些指标的计算结果与 MATLAB 版本基本一致，具有较高的准确性和可靠性。

然而，有三个指标（Nabf、SSIM 和 edge_association）的相对差异和相对绝对差异相对较大。分别是 9.78%、2.2%、0.84%。可能是由于以下原因导致的：

- **计算方法的细微差异**：虽然本项目和 MATLAB 版本的核心计算逻辑可能相同，但在具体实现过程中，可能存在一些细微的差异，例如边界处理、数据类型转换等，这些差异可能会对最终的计算结果产生一定的影响。
- **数值精度的差异**：Python 和 MATLAB 在数值计算方面的精度可能存在一定的差异，特别是在处理浮点数运算时，这种差异可能会随着计算的累积而逐渐放大，从而导致最终结果的差异。

针对这些差异，后续可以进一步深入分析，对计算方法进行优化和调整，以提高本项目与 MATLAB 版本计算结果的一致性。同时，也可以增加更多的测试用例，对项目的准确性进行更全面的验证。

## 注意事项

- 确保配置文件中的图像目录路径正确，且图像文件存在。
- 确保融合结果、可将光、红外三个文件夹中对应图像的文件名一致
- 如果使用 GPU 计算，确保系统中安装了正确的 CUDA 驱动和 PyTorch 的 CUDA 版本。
- `MS-SSIM` 指标目前在代码中简单设为 0，实际需要实现完整的计算逻辑。

## 贡献指南

如果你希望为项目做出贡献，可以按照以下步骤进行：

1. Fork 本项目到你的 GitHub 仓库。
2. 创建一个新的分支，进行代码修改和功能添加。
3. 提交代码到你的分支，并发起 Pull Request。
4. 等待项目维护者审核和合并。

## 许可证信息

本项目采用 [MIT 许可证](LICENSE)。
