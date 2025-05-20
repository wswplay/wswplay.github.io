---
title: GPT
outline: deep
---

# GPT

**GPT**：Generative Pre-trained Transformer，**生成式·预训练·变换模型**。

**OpenAI 最新 GPT-4.1** 支持 100 万 token 的**超长上下文**（约 75 万字），适用于金融分析、法律文档处理、大型代码库分析等任务。

## 架构：Decoder-only

GPT 系列（GPT-1/2/3/4）采用 `Transformer` **解码器-only**<sup>Decoder-only</sup>架构，这不是“耍花样”，而是为了适应**自回归**语言建模目标而做的结构取舍。

没有**编码器-解码器交叉注意力**层（因无编码器）。

## 模型：MoE

**MoE**：Mixture of Experts，**多专家混合**。

多个子模型（专家网络），每次推理**只激活一部分**，大幅提升模型规模同时控制计算成本。

## 分词算法：BPE

GPT（如 GPT-2/3/4）使用 **BPE 分词算法**。

GPT-4 词表可能范围：

- **GPT-2 词表：50,257**（基于 BPE 分词）
- **GPT-3 词表：50,257**（与 GPT-2 相同）
- **GPT-4 词表：或仍在 50,257 左右**，或有调整。但通过更高效的 `token` 利用和模型架构优化（如 MoE）提升性能。

GPT 的 BPE 分词通过**字节级操作**和**动态合并策略**，在语言无关性、OOV 处理、计算效率上均领先。其开源实现（如[`tiktoken`](https://github.com/openai/tiktoken)）进一步推动了行业标准化。

以下是基于 Karpathy 的 **minBPE 项目**(简化版)（74 行 Python 实现）核心代码：

```python
import re
from collections import defaultdict

def get_stats(vocab):
  pairs = defaultdict(int)
  for word, freq in vocab.items():
    symbols = word.split()
    for i in range(len(symbols)-1):
      pairs[symbols[i], symbols[i+1]] += freq
  return pairs

def merge_vocab(pair, vocab_in):
  vocab_out = {}
  bigram = re.escape(' '.join(pair))
  p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
  for word in vocab_in:
    w_out = p.sub(''.join(pair), word)
    vocab_out[w_out] = vocab_in[word]
  return vocab_out

# 示例：训练BPE词表
vocab = {'l o w': 5, 'l o w e r': 2, 'n e w e s t': 6}
num_merges = 10
for i in range(num_merges):
  pairs = get_stats(vocab)
  if not pairs:
    break
  best = max(pairs, key=pairs.get)
  vocab = merge_vocab(best, vocab)
  print(f"Merge {i+1}: {best} -> {''.join(best)}")

# 输出示例
# Merge 1: ('e', 's') -> es
# Merge 2: ('es', 't') -> est
# ...
```

## 注意力机制：FlashAttention、稀疏注意力

## 位置编码：RoPE

**RoPE**：Rotary Position Embedding，**旋转位置编码**，广泛应用于现代大型语言模型（LLMs），包括 LLaMA、ChatGLM、Baichuan 等。

### 主要特点

**1. 绝对位置编码形式，相对位置编码效果**  
 RoPE 通过旋转矩阵对 Query 和 Key 向量进行变换，使得内积计算后能自动包含相对位置信息，而无需额外修改 Attention 结构。

**2. 外推性优势**  
 相比传统位置编码（如 Sinusoidal），RoPE 在长文本处理中表现更优，支持模型在推理时处理远超训练长度的序列（如从 1.6 万 tokens 扩展到 100 万 tokens）。

**3. 数学基础**  
 RoPE 复数旋转原理，通过欧拉公式将位置信息编码为旋转角度，确保位置关系线性可加性。

**4. 实现高效性**  
 RoPE 无需额外可学习参数，计算过程仅涉及固定三角函数变换，适合大规模模型部署。

- **更强的泛化能力**：RoPE 能更好地捕捉长距离依赖关系，适合 GPT 系列模型的生成式任务。
- **兼容性**：RoPE 可直接融入现有 Transformer 架构，无需修改 Attention 计算逻辑。
- **动态调整**：通过调整旋转角底数（如 NTK-aware 缩放），可进一步提升外推性能。

### 核心思想

通过**旋转矩阵**将**绝对位置信息**融入**相对位置计算**，使得注意力机制能够隐式捕获位置关系，同时保持**线性可加性**（即相对**位置可以通过旋转角度**差值表示）。

`RoPE` 灵感源于**复数旋转**。在复数空间中，一个向量 $z = x + iy$ 可通过乘以 $e^{i\theta}$ 进行旋转。

### 复数旋转

**1. 复数表示向量**

在复数空间中，一个向量 $\mathbf{v} \in \mathbb{R}^2$ 可以表示为复数：

$$
\mathbf{v} = a + ib
$$

$a$ 是实部（对应向量横坐标），$b$ 是虚部（对应向量纵坐标）

**2. 旋转矩阵**

旋转矩阵是表示**旋转操作**的方阵，**二维(2D)旋转**矩阵绕原点旋转角度 $\theta$（逆时针为正）：

$$
R(\theta) = \begin{pmatrix}
\cos\theta & -\sin\theta \\
\sin\theta & \cos\theta \\
\end{pmatrix}
$$

- **性质**：正交矩阵（$R^T = R^{-1}$），行列式为 1。
- **示例**：旋转 $90^\circ$ 时，矩阵为 $\begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix}$。

**3. 欧拉公式：连接复数、指数函数和三角函数**

$$
e^{i\theta} = \cos \theta + i \sin \theta
$$

[欧拉公式](/aiart/deep-learning/mathematics.html#欧拉公式)表示的是一个 **单位复数**（长度为 1），角度为 $\theta$。

$$
e^{i\theta} \cdot \mathbf{v} = (a + ib)(\cos \theta + i \sin \theta) = (a \cos \theta - b \sin \theta) + i(a \sin \theta + b \cos \theta)
$$

这结果正好对应旋转后的向量 $R(\theta) \cdot \begin{pmatrix} a \\ b \end{pmatrix}$，表示为：

$$
\begin{pmatrix}
a' \\
b'
\end{pmatrix}
=
\begin{pmatrix}
a \cos \theta - b \sin \theta \\
a \sin \theta + b \cos \theta
\end{pmatrix}
$$

<span style="color:#f00">**所以**</span>：**乘以 $e^{i\theta}$ 等价于 旋转角度 $\theta$**。

## PyTorch 实现简化版 GPT

```sh
# 安装torch和transformers（用于分词）
pip install torch transformers
```

```py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import GPT2Tokenizer

# 掩蔽自注意力机制
class SelfAttention(nn.Module):
  def __init__(self, embed_size, heads):
    super(SelfAttention, self).__init__()
    self.embed_size = embed_size
    self.heads = heads
    self.head_dim = embed_size // heads

    assert self.head_dim * heads == embed_size, "Embed size needs to be divisible by heads"

    self.keys = nn.Linear(embed_size, embed_size)
    self.queries = nn.Linear(embed_size, embed_size)
    self.values = nn.Linear(embed_size, embed_size)
    self.fc_out = nn.Linear(embed_size, embed_size)

  def forward(self, x, mask=None):
    N, seq_length, _ = x.shape
    # 分割为多头
    keys = self.keys(x).view(N, seq_length, self.heads, self.head_dim)
    queries = self.queries(x).view(N, seq_length, self.heads, self.head_dim)
    values = self.values(x).view(N, seq_length, self.heads, self.head_dim)

    # 计算注意力分数
    # 爱因斯坦求和约定(Einstein Summation)，表示复杂的张量乘法
    energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])  # (N, heads, seq_len, seq_len)
    if mask is not None:
      energy = energy.masked_fill(mask == 0, float("-1e20"))

    attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)
    out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
      N, seq_length, self.embed_size
    )
    return self.fc_out(out)

# Transformer解码器块
class TransformerBlock(nn.Module):
  def __init__(self, embed_size, heads, dropout, forward_expansion):
    super(TransformerBlock, self).__init__()
    self.attention = SelfAttention(embed_size, heads)
    self.norm1 = nn.LayerNorm(embed_size)
    self.norm2 = nn.LayerNorm(embed_size)

    self.feed_forward = nn.Sequential(
      nn.Linear(embed_size, forward_expansion * embed_size),
      # 高斯误差线性单元(Gaussian Error Linear Unit)比 ReLU 更平滑的非线性
      nn.GELU(),
      nn.Linear(forward_expansion * embed_size, embed_size),
    )
    self.dropout = nn.Dropout(dropout)

  def forward(self, x, mask):
    attention = self.attention(x, mask)
    x = self.norm1(attention + self.dropout(x))
    forward = self.feed_forward(x)
    out = self.norm2(forward + self.dropout(x))
    return out

# 位置编码
class PositionalEncoding(nn.Module):
  def __init__(self, embed_size, max_len=5000):
    super(PositionalEncoding, self).__init__()
    pe = torch.zeros(max_len, embed_size)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, embed_size, 2).float() * (-math.log(10000.0) / embed_size))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0)
    self.register_buffer("pe", pe)

  def forward(self, x):
    return x + self.pe[:, :x.shape[1], :]

# 简化版GPT模型
class GPT(nn.Module):
  def __init__(
    self,
    vocab_size,
    embed_size=256,
    num_layers=6,
    heads=8,
    forward_expansion=4,
    dropout=0.1,
    max_length=100,
  ):
    super(GPT, self).__init__()
    self.token_embedding = nn.Embedding(vocab_size, embed_size)
    self.position_encoding = PositionalEncoding(embed_size, max_length)
    self.layers = nn.ModuleList(
      [
        TransformerBlock(
          embed_size,
          heads,
          dropout=dropout,
          forward_expansion=forward_expansion,
        )
        for _ in range(num_layers)
      ]
    )
    self.fc_out = nn.Linear(embed_size, vocab_size)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x, mask=None):
    N, seq_length = x.shape
    out = self.dropout(self.token_embedding(x))
    out = self.position_encoding(out)
    for layer in self.layers:
      out = layer(out, mask)
    return self.fc_out(out)

  # 文本生成
  def generate(self, prompt, max_len=50, temperature=1.0):
    self.eval()
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokens = tokenizer.encode(prompt, return_tensors="pt")
    for _ in range(max_len):
      # 生成掩码（防止看到未来token）
      mask = torch.tril(torch.ones((1, tokens.size(1), tokens.size(1))).bool()
      outputs = self(tokens, mask)
      next_token_logits = outputs[:, -1, :] / temperature
      next_token = torch.argmax(torch.softmax(next_token_logits, dim=-1), dim=-1)
      tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)
    return tokenizer.decode(tokens[0].tolist())

# 示例用法
if __name__ == "__main__":
  vocab_size = 50257  # GPT-2的词汇表大小
  model = GPT(vocab_size)
  input_text = "The future of AI is"
  generated_text = model.generate(input_text, max_len=20)
  print(f"Generated text: {generated_text}")
```

关键点说明：

1. **自注意力掩码**：通过`torch.tril`生成下三角矩阵，确保生成时只能看到左侧上下文（自回归特性）。

2. **位置编码**：使用正弦/余弦函数生成位置信息（与原始 Transformer 相同），但实际应用中可替换为可学习的位置嵌入。

3. **文本生成**：`generate()`方法通过迭代预测下一个词实现生成，支持温度参数（`temperature`）控制随机性。

4. **简化处理**：未实现更复杂特性（如稀疏注意力、梯度检查点），但保留了 GPT 核心设计。
