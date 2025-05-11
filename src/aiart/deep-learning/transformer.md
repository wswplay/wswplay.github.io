---
title: Transformer
outline: deep
---

# Transformer 架构

自注意力同时具有**并行计算**和**最短最大路径长度**这两个优势，因此使用自注意力来设计深度架构是很有吸引力的。

`Transformer` 模型**完全基于注意力机制**，没有任何卷积层或循环神经网络层。

尽管 Transformer 最初是应用于在**文本数据**上的序列到序列学习，但现在已经推广到各种现代的深度学习中，例如**语言、视觉、语音和强化学习**领域。

## 模型

![An Image](./img/transformer.svg)
Transformer 是**编码器－解码器**架构实例，基于**自注意力模块叠加**而成。

**源序列**（输入）嵌入（`embedding`），和**目标序列**（输出）嵌入加上**位置编码**（`positional encoding`），分别输入到编码器和解码器中。

**编码器**：由多个相同的层叠加而成的，每个层都有两个子层（子层表示为 `sublayer`）。

- 第一个子层：**多头自注意力**（`multi-head self-attention`）汇聚；
- 第二个子层：**基于位置的前馈网络**（`positionwise feed-forward network`）。
- 具体计算时，查询、键和值都**来自前一个编码器**层的输出，每个子层都采用了**残差连接**（`residual connection`），在残差连接的加法计算之后，紧接着应用**层规范化**（`layer normalization`）。

**解码器**：也是由多个相同的层叠加而成的，并且层中使用了残差连接和层规范化。
