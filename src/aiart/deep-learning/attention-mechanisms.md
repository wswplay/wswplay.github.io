---
title: 注意力机制
outline: deep
---

# 注意力机制(attention mechanisms)

## 查询、键和值

在注意力机制的背景下，**自主性提示**被称为**查询（query）**。

给定任何查询，注意力机制通过注意力汇聚（`attention pooling`） 将选择引导至感官输入（`sensory inputs`，例如中间特征表示）。这些感官输入被称为**值（value）**。

每个**值**都与一个**键**（`key`）配对，这可以想象为感官输入的**非自主提示**。

## 注意力机制

通过**注意力汇聚**将**查询**（自主性提示）和**键**（非自主性提示）结合在一起，实现对**值**（感官输入）的**选择**倾向（智能选择）。

“查询-键”对**越近**，注意力汇聚**注意力权重**就越高。

“是否包含自主性提示”将**注意力机制**与**全连接层或汇聚层**区别开来。

| 特性           | 注意力机制       | 全连接层       | 汇聚层         |
| -------------- | ---------------- | -------------- | -------------- |
| **参数化**     | 是（动态权重）   | 是（静态权重） | 否（固定规则） |
| **自主性提示** | ✅ 动态适应输入  | ❌ 静态处理    | ❌ 静态处理    |
| **输入依赖性** | 高度依赖         | 不依赖         | 不依赖         |
| **典型应用**   | Transformer, NLP | 传统分类模型   | CNN 的空间降维 |

## 多头注意力(multihead attention)

在实践中，当给定相同的查询、键和值的集合时，我们希望模型可以基于相同的注意力机制学习到不同的行为，然后将不同的行为作为知识组合起来，捕获序列内各种范围的依赖关系（例如，短距离依赖和长距离依赖关系）。

因此，允许注意力机制**组合查询**、键和值的不同**子空间**表示（representation subspaces）可能是有益的。

与其只使用单独一个注意力汇聚， 我们可以用独立学习得到的 $h$ 组不同的**线性投影**（linear projections）来变换**查询、键和值**。然后，这 $h$ 组变换后的查询、键和值将并行地送到注意力汇聚中。最后，将这 $h$ 个注意力汇聚的输出**拼接**在一起，并且通过另一个可以学习的线性投影进行变换，以产生最终输出。这种设计被称为**多头注意力（multihead attention）**。

对于 $h$ 个注意力汇聚输出，**每一个注意力汇聚**都被称作**一个头（head）**。

### 代码实现

在实现过程中通常选择缩放点积注意力作为每一个注意力头。

```py
import math
import torch
from torch import nn
from d2l import torch as d2l

class MultiHeadAttention(nn.Module):
  def __init__(self, key_size, query_size, value_size, num_hiddens,
                num_heads, dropout, bias=False, **kwargs):
    super(MultiHeadAttention, self).__init__(**kwargs)
    self.num_heads = num_heads
    self.attention = d2l.DotProductAttention(dropout)
    self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
    self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
    self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
    self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

  def forward(self, queries, keys, values, valid_lens):
    # queries，keys，values的形状: (batch_size，查询或者“键－值”对的个数，num_hiddens)
    # valid_lens的形状: (batch_size，)或(batch_size，查询的个数)
    # 经过变换后，输出的queries，keys，values　的形状:
    # (batch_size*num_heads，查询或者“键－值”对的个数，num_hiddens/num_heads)
    queries = transpose_qkv(self.W_q(queries), self.num_heads)
    keys = transpose_qkv(self.W_k(keys), self.num_heads)
    values = transpose_qkv(self.W_v(values), self.num_heads)

    if valid_lens is not None:
      # 在轴0，将第一项（标量或者矢量）复制num_heads次，
      # 然后如此复制第二项，然后诸如此类。
      valid_lens = torch.repeat_interleave(valid_lens, repeats=self.num_heads, dim=0)

    # output的形状:(batch_size*num_heads，查询的个数，num_hiddens/num_heads)
    output = self.attention(queries, keys, values, valid_lens)

    # output_concat的形状:(batch_size，查询的个数，num_hiddens)
    output_concat = transpose_output(output, self.num_heads)
    return self.W_o(output_concat)


# 为了多注意力头的并行计算而变换形状
def transpose_qkv(X, num_heads):
  # 输入X的形状:(batch_size，查询或者“键－值”对的个数，num_hiddens)
  # 输出X的形状:(batch_size，查询或者“键－值”对的个数，num_heads，num_hiddens/num_heads)
  X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)

  # 输出X的形状:(batch_size，num_heads，查询或者“键－值”对的个数，num_hiddens/num_heads)
  X = X.permute(0, 2, 1, 3)

  # 最终输出的形状:(batch_size*num_heads，查询或者“键－值”对的个数，num_hiddens/num_heads)
  return X.reshape(-1, X.shape[2], X.shape[3])

# 逆转transpose_qkv函数的操作
def transpose_output(X, num_heads):
  X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
  X = X.permute(0, 2, 1, 3)
  return X.reshape(X.shape[0], X.shape[1], -1)
```

## 自注意力和位置编码

### 自注意力

将词元序列输入**注意力池化**中，以便同一组词元同时充当查询、键和值。

即每个查询都会关注所有键－值对并生成一个注意力输出。由于**查询、键和值来自同一组输入**，被称为**自注意力**（`self-attention`），或**内部注意力**（`intra-attention`）。
