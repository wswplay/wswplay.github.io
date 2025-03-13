---
title: ComfyUI
# outline: deep
---

# Your creativity, We Visualed！

## VAE(变分自编码器)

VAE(`Variational Autoencoder`) 是一种生成模型，包含两个主要部分：编码器（Encoder）和解码器（Decoder）。

### 总结

- **编码器**：将数据映射到潜在空间。
- **解码器**：从潜在空间生成数据。
- **目标**：学习数据的低维表示并生成新数据。

## CLIP-(对比式语言-图像预训练)

CLIP（`Contrastive Language–Image Pretraining`）是 OpenAI 提出的一种多模态模型，用于将图像和文本映射到同一个语义空间。与 VAE 不同，CLIP 没有传统意义上的“编码器”和“解码器”，而是由两个核心组件组成：**图像编码器**和**文本编码器**。

### 总结

CLIP 的“编码器”包括图像编码器和文本编码器，它们分别将图像和文本映射到同一个语义空间。CLIP 没有传统意义上的“解码器”，而是通过对比学习实现图像和文本的语义对齐。CLIP 的核心优势在于其强大的多模态理解能力，能够广泛应用于图像-文本检索、零样本分类等任务。
