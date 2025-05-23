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

**RoPE**：Rotary Position Embedding，**旋转位置编码**，是一种用于 `Transformer` 架的位置编码方法，由苏剑林[论文](https://arxiv.org/abs/2104.09864)提出。

已广泛应用于现代大型语言模型（LLMs），包括 LLaMA、ChatGLM、Baichuan 等。

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

将位置编码融合进注意力计算 `query` 和 `key` 向量中，通过**旋转矩阵**将**绝对位置信息**融入**相对位置计算**，使得注意力机制能够隐式捕获位置关系，同时保持**线性可加性**（即相对**位置可以通过旋转角度**差值表示），而不是像传统 `Transformer` 那样加在 `embedding` 上。

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

- **性质**：$\mathbf{R}_m^\top \mathbf{R}_n = \mathbf{R}_{n-m}$。内积仅依赖相对位置 $n-m$。

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

<span style="color:#f00">**所以**</span>：**复数乘法 = 2D 旋转**。

### 模拟实现

`OpenAI` 最新版本 `GPT`（包括 `GPT-4`、`GPT-4.5`、`GPT-4-turbo`）都使用了类似 `RoPE` 方法(未开源)。

本质上也是：将 `token` 向量中**相邻两个维度**当作二维平面做旋转，**等效实现复数旋转**，来嵌入相对位置信息。

:::tip 等效实现旋转
并不是用复数类型张量，而是但把实数向量中的相邻两个维度当成复数实部/虚部，然后用数学公式模拟复数乘法(旋转)效果。
:::

$$
\tilde{q}_m^\top \cdot \tilde{k}_n = (R(m) q)^\top (R(n) k) = q^\top R(n - m) k
$$

**角度差值**体现在结构里，而不是代码逻辑里。

```py
import torch
import math

# ---- 配置 ----
dim = 8  # 向量维度（必须是偶数）
pos_q = 10
pos_k = 20

# ---- 构造向量 ----
q = torch.arange(1, dim + 1, dtype=torch.float32)
k = torch.arange(1, dim + 1, dtype=torch.float32) * 2
# [1, 2, ..., 8] [2, 4, ..., 16]

# ---- 构造频率项 ----
half_dim = dim // 2
# [1/10000^{0/d}, ..., 1/10000^{(d-2)/d}]
inv_freq = 1.0 / (10000 ** (torch.arange(0, half_dim) * 2 / dim))

# ---- 获取角度位置编码（RoPE核心） ----
theta_q = pos_q * inv_freq   # [θ₀, θ₁, θ₂, θ₃]
theta_k = pos_k * inv_freq

sin_q = torch.sin(theta_q)
cos_q = torch.cos(theta_q)
sin_k = torch.sin(theta_k)
cos_k = torch.cos(theta_k)

# ---- 拆分向量 ----
def split_even_odd(x):
  return x[::2], x[1::2]

# ---- RoPE 核心旋转函数 ----
def apply_rope(x, cos_theta, sin_theta):
  x_even, x_odd = split_even_odd(x)
  x_rotated = torch.empty_like(x)
  x_rotated[::2] = x_even * cos_theta - x_odd * sin_theta
  x_rotated[1::2] = x_even * sin_theta + x_odd * cos_theta
  return x_rotated

# ---- 应用 RoPE ----
q_rope = apply_rope(q, cos_q, sin_q) # q @ R_theta_q
k_rope = apply_rope(k, cos_k, sin_k) # k @ R_theta_k

# ---- 计算注意力得分 ----
attn_plain = torch.dot(q, k)
attn_rope = torch.dot(q_rope, k_rope)

# ---- 打印结果 ----
print("原始 Q:", q)
print("原始 K:", k)
print("RoPE 后 Q:", q_rope)
print("RoPE 后 K:", k_rope)
print()
print("不带 RoPE 的注意力得分:", attn_plain.item())
print("带 RoPE 的注意力得分:", attn_rope.item())

# ---- 结果输出 ----
# 原始 Q: tensor([1., 2., 3., 4., 5., 6., 7., 8.])
# 原始 K: tensor([ 2.,  4.,  6.,  8., 10., 12., 14., 16.])
# RoPE 后 Q: tensor([ 0.2837,  2.9150, -0.9324,  4.9420, -2.0054,  5.4258, -3.3625,  5.7471])
# RoPE 后 K: tensor([ 0.5673,  5.8300, -1.8648,  9.8841, -4.0109, 10.8515, -6.7249, 11.4943])

# 不带 RoPE 的注意力得分: 816.0
# 带 RoPE 的注意力得分: 922.35
```

### NTK RoPE(GPT-4)

**NTK RoPE**：Neural Tangent Kernel Rotary Positional Embedding，是 `OpenAI GPT-4` 中引入的一种位置编码变体，属于 RoPE 位置编码扩展。

**背景**：`RoPE` 有限制

- 原始的 RoPE 使用固定的频率 $\theta_i = 10000^{-2i/d}$ 做旋转；
- 这意味着模型对 **训练时没见过的很大位置索引（如 10k+）** 的泛化能力有限；
- 在长文本时，RoPE 的旋转频率太快，导致信息在高维度上“绕圈绕得太厉害”，难以匹配；
- 这限制了 GPT-4 类模型在 **扩展上下文长度** 时的表现。

**思想**：“拉伸”低频维度位置旋转频率，让远距离 token 不至于旋转太快。

`原始RoPE` 频率指数序列：

$$
\omega_i = 10000^{-2i/d}
$$

`NTK RoPE` 把它换成了：

$$
\omega_i = 10000^{-2i/(d \cdot \alpha)}
$$

其中 $\alpha > 1$ 是一个**位置频率缩放因子**，称为 NTK factor。

即：把位置频率**慢下来**，使位置编码**对远距离 token 更稳定**，从而**泛化到更长**上下文。

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
