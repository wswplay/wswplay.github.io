---
title: 计算机视觉、Computer Vision
outline: deep
---

# CV：Computer Vision

计算机视觉、机器视觉。无论是医疗诊断、**自动驾驶**，还是智能滤波器、摄像头监控，许多计算机视觉领域的应用都与我们当前和未来的生活密切相关。

## 微调(fine-tuning)

**迁移学习**将从源数据集中学到的知识迁移到目标数据集，**微调**是迁移学习的常见技巧。

**微调**包括以下四个步骤：

1. 在源数据集（例如 ImageNet 数据集）上预训练神经网络模型，即**源模型**。
2. 创建一个新的神经网络模型，即**目标模型**。这将复制源模型上的所有模型设计及其参数（输出层除外）。我们假定这些模型参数包含从源数据集中学到的知识，这些知识也将适用于目标数据集。我们还假设源模型的输出层与源数据集的标签密切相关；因此不在目标模型中使用该层。
3. 向目标模型添加**输出层**，其输出数是目标数据集中的类别数。然后随机初始化该层的模型参数。
4. 在目标数据集（如椅子数据集）上训练目标模型。**输出层将从头开始**进行训练，而**其他层**参数将根据源模型参数进行**微调**。

通常，**微调**参数使用**较小学习率**，而**从头开始**训练输出层可以使用**更大学习率**。

**示意代码**：

```py
import os
import torch
import torchvision
from torch import nn
from d2l import torch as d2l

finetune_net = torchvision.models.resnet18(pretrained=True)
finetune_net.fc = nn.Linear(finetune_net.fc.in_features, 2)
nn.init.xavier_uniform_(finetune_net.fc.weight)

# 如果param_group=True，输出层中的模型参数将使用十倍的学习率
def train_fine_tuning(net, learning_rate, batch_size=128, num_epochs=5, param_group=True):
  train_iter = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
    os.path.join(data_dir, 'train'), transform=train_augs),
    batch_size=batch_size, shuffle=True)

  test_iter = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
    os.path.join(data_dir, 'test'), transform=test_augs),
    batch_size=batch_size)

  devices = d2l.try_all_gpus()
  loss = nn.CrossEntropyLoss(reduction="none")

  if param_group:
    params_1x = [
      param for name, param in net.named_parameters() if name not in ["fc.weight", "fc.bias"]
    ]
    trainer = torch.optim.SGD(
      [{'params': params_1x}, {'params': net.fc.parameters(), 'lr': learning_rate * 10}],
      lr=learning_rate, weight_decay=0.001
    )
  else:
    trainer = torch.optim.SGD(net.parameters(), lr=learning_rate, weight_decay=0.001)

  d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)

# 使用较小学习率，通过微调预训练获得的模型参数
train_fine_tuning(finetune_net, 5e-5)
```

## 目标检测和边界框

**目标检测/识别**：不仅可以检测<sup>object detection</sup>识别<sup>object recognition</sup>图像中所有感兴趣的物体，还能识别它们的位置，该位置通常由矩形边界框表示。

**边界框**：在目标检测中，通常使用边界框<sup>bounding box</sup>来描述对象的空间位置。

边界框通常是**矩形**，两种常用边界框表示「中心 $(x,y)$，宽度，高度」和「左上 $x$，右下 $y$」。

## 锚框与交并比(IOU)

**锚框**：目标检测算法通常会采样大量区域，判断其中是否包含目标，并调整边界更准确地预测目标真实边界框<sup>ground-truth bounding box</sup>，这些边界框被称为锚框<sup>anchor box</sup>。

不同模型采样各异。比如以每个像素为中心，生成多个缩放比和宽高比<sup>aspect ratio</sup>的不同边界框。

那么如何衡量锚框**准确性**呢？换言之，若已知目标真实边界框，如何衡量锚框和**真实边界框**之间**相似性**？杰卡德系数<sup>Jaccard</sup>可以衡量两者之间相似性。

**IOU**: Intersection Over Union **交并比**，两个边界框**交集除以并集**，也被称为杰卡德系数。

$$
J(A, B) = \frac{|A \cap B|}{|A \cup B|}
$$

交并比的取值范围在 0 和 1 之间：0 表示两个边界框无重合像素，1 表示两个边界框完全重合。
![An Image](./img/iou.svg)

## 语义分割

语义分割<sup>semantic segmentation</sup>重点关注于如何将图像分割成属于不同语义类别的区域。与目标检测不同，语义分割标注的**像素级**边框显然更加**精细**。

## 转置卷积(上采样)

卷积神经网络<sup>CNN</sup>的卷积层和汇聚层，通常会减少**下采样**输入图像空间维度（高和宽）。

![An Image](./img/trans_conv.svg)
**转置卷积**<sup>transposed convolution</sup>通过卷积核**广播**输入元素，增加**上采样**中间层特征图空间维度，实现**输出大于输入**，用于逆转下采样导致的空间尺寸减小。

**填充**：转置卷积中，填充被应**用于输出**（常规卷积将填充应用于输入）。例如，当将高和宽两侧填充数指定为 1 时，转置卷积输出中将**删除第一和最后的行与列**。

![An Image](./img/trans_conv_stride2.svg)
**步幅**：被指定为中间结果（输出），而不是输入。

**多输入和输出通道**：转置卷积与常规卷积以**相同**方式运作。

**矩阵变换**：转置卷积层能够**交换**卷积层的**正向传播**函数和**反向传播**函数。

## 全卷积网络(FCN)

**FCN**：Fully Convolutional Network。

通过**转置卷积**，将中间层特征图的高和宽变换回输入图像的尺寸，输出类别预测与输入图像在像素级别上具有一一对应关系：**通道维输出**即该位置对应像素的**类别预测**。

### 构造模型

![An Image](./img/fcn.svg)
全卷积网络先使用**卷积神经网络**抽取图像特征，然后通过 $1 \times 1$ 卷积层**将通道数变换为类别个数**，最后通过**转置卷积层**将特征图高和宽**变换为输入图像尺寸**。因此，模型输出与输入图像的高和宽相同，且最终输出通道包含了该空间位置像素的类别预测。

### 初始化转置卷积层

在图像处理中，我们有时需要**将图像放大**，即**上采样**<sup>upsampling</sup>。

**双线性插值**<sup>bilinear interpolation</sup> 是常用上采样方法之一，它也经常用于**初始化**转置卷积层。

1. 将输出图像的坐标 $(x, y)$ 映射到输入图像的坐标 $(x', y')$ 上。例如，根据输入与输出的尺寸之比来映射。请注意，映射后的 $x'$ 和 $y'$ 是实数。
2. 在输入图像上找到离坐标 $(x', y')$ 最近的 4 个像素。
3. 输出图像在坐标 $(x, y)$ 的像素依据输入图像这 4 个像素及其与 $(x', y')$ 相对距离来计算。

```py
def bilinear_kernel(in_channels, out_channels, kernel_size):
  factor = (kernel_size + 1) // 2
  if kernel_size % 2 == 1:
    center = factor - 1
  else:
    center = factor - 0.5
  og = (torch.arange(kernel_size).reshape(-1, 1),
        torch.arange(kernel_size).reshape(1, -1))
  filt = (1 - torch.abs(og[0] - center) / factor) * \
          (1 - torch.abs(og[1] - center) / factor)
  weight = torch.zeros((in_channels, out_channels, kernel_size, kernel_size))
  weight[range(in_channels), range(out_channels), :, :] = filt
  return weight
```

## 风格迁移(style transfer)

把一张图的“内容”和另一张图的“风格”结合，生成一张“内容不变但风格变化”的图。

**典型例子：**
你拍了一张猫，用梵高《星夜》风格迁移它，就能得到一张“星夜风格猫”。

**底层原理（简化）：**

- 内容图像保留结构和形状信息（比如人脸、物体轮廓）。
- 风格图像提供纹理、颜色、笔触风格等。
- 通过神经网络（如卷积神经网络 CNN）提取两者的特征，并组合输出。

**常见技术：**

- Gatys 等人提出的经典神经风格迁移（基于 VGG 网络）。
- 更快的实时风格迁移（Fast Style Transfer）用于移动端 App（如 Prisma）。
