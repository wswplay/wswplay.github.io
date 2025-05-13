---
title: Transformer 架构
outline: deep
---

# Transformer 架构

自注意力同时具有**并行计算**和**最短最大路径长度**这两个优势，因此使用自注意力来设计深度架构是很有吸引力的。

`Transformer` 模型**完全基于注意力机制**，没有任何卷积层或循环神经网络层。

尽管 `Transformer` 最初是应用于在**文本数据**上的序列到序列学习，但现在已经推广到各种现代的深度学习中，例如**语言、视觉、语音和强化学习**领域。

## 模型

![An Image](./img/transformer.svg)
Transformer 是**编码器－解码器**架构实例，基于**自注意力模块叠加**而成。

**源序列**(输入)嵌入<sup>embedding</sup>，和**目标序列**(输出)嵌入加上**位置编码**<sup>positional encoding</sup>，分别输入到编码器和解码器中。

**编码器**：由多个相同层叠加而成的，**每个层都有两个子层**。

- 第一个子层：**多头自注意力**<sup>multi-head self-attention</sup>汇聚。
- 第二个子层：**逐位前馈网络**<sup>positionwise feed-forward network：FFN</sup>。
- 计算时，查询、键和值都**来自前一个编码器**层输出，每个子层都采用了**残差连接**<sup>residual connection</sup>，在残差连接加法计算后，应用**层规范化**<sup>layer normalization</sup>。

**解码器**：也是由多个相同层叠加而成，同样使用了**残差连接**和**层规范化**。

- 第三个子层：插入在这两个子层之间，称为**编码器－解码器注意力**<sup>encoder-decoder attention</sup>层：**查询**来自**前一个解码器**层输出，而**键和值**来自**整个编码器**输出。
- **解码器自注意力**中，查询、键和值都来**自上一个解码器**层输出。但解码器中每个位置只能考虑该位置之前的所有位置。这种**掩蔽**<sup>masked</sup>注意力保留了[**自回归**<sup>auto-regressive</sup>](/aiart/deep-learning/rnn.html#自回归模型)属性，确保预测仅依赖于已生成的输出词元。

## 基于位置的前馈网络(FFN)

`Transformer` 模型中基于位置的前馈网络使用同一个**多层感知机**，作用是对所有序列**位置表示进行转换**。

```py
class PositionWiseFFN(nn.Module):
  def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs, **kwargs):
    super(PositionWiseFFN, self).__init__(**kwargs)
    self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
    self.relu = nn.ReLU()
    self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

  def forward(self, X):
    return self.dense2(self.relu(self.dense1(X)))

# 实例化
ffn = PositionWiseFFN(4, 4, 8)
```

- 在标准 `Transformer` 中，`ffn_num_outputs` 通常与模型主维度（如输入维度）一致，而隐藏层是主维度 4 倍（如输入 512 → 隐藏层 2048）。
- 本例输出维度（8）与输入（4）不同，或破坏残差连接条件（需 `输入维度 == 输出维度`）。

### 基于位置

`nn.Linear` 默认行为：**位置独立**计算，不混合不同位置信息。

- 当输入是 (`B`=batch_size, `T`=sequence_length, `D_in`=输入特征维度) 时，对 `T` 的每个位置独立计算，即 "位置相关"。

| 层类型               | 位置独立 | 原因                                  |
| -------------------- | -------- | ------------------------------------- |
| `nn.Linear`          | ✅ 是    | 默认独立处理 `(B, T, D)` 中每个 `T`。 |
| `nn.Conv1d`          | ❌ 否    | 滑动窗口混合相邻位置的信息。          |
| `nn.LSTM`            | ❌ 否    | 隐状态依赖前序位置的计算结果。        |
| `MultiHeadAttention` | ❌ 否    | 显式计算所有位置间的注意力权重。      |
| `PositionWiseFFN`    | ✅ 是    | `nn.Linear` 堆叠，独立处理每个位置。  |

### 前馈网络

**前馈网络**<sup>Feedforward Neural Network, FNN</sup> 是一种最基本神经网络结构。其核心特点是：**数据单向流动（从输入层 → 隐藏层 → 输出层），没有循环或反馈连接**。

- **“Feed”**：数据被“喂入”网络。
- **“Forward”**：数据只向前流动，不反向或循环。

**数学表达**

给定输入 $\mathbf{X} \in \mathbb{R}^{B \times T \times D}$（B=批大小，T=序列长度，D=特征维度）：

$$
\text{FFN}(\mathbf{X}) = \mathbf{W}_2 \cdot \text{ReLU}(\mathbf{W}_1 \mathbf{X} + \mathbf{b}_1) + \mathbf{b}_2
$$

- $\mathbf{W}_1 \in \mathbb{R}^{D \times D_{hidden}}$, $\mathbf{W}_2 \in \mathbb{R}^{D_{hidden} \times D}$
- 每个位置的输出仅依赖该位置的输入，不依赖其他位置。

**前馈网络 vs 其他网络**

| 网络类型        | 典型用途               | 数据流动方向        | 示例                 |
| --------------- | ---------------------- | ------------------- | -------------------- |
| **前馈网络**    | 图像分类、特征提取     | 单向（输入 → 输出） | MLP, PositionWiseFFN |
| **循环网络**    | 时序数据（文本、语音） | 双向（含时间反馈）  | LSTM, GRU            |
| **卷积网络**    | 图像、空间数据         | 局部连接+权重共享   | ResNet, VGG          |
| **Transformer** | 序列建模（如机器翻译） | 自注意力+前馈       | BERT, GPT            |

## 残差连接和层规范化(add&norm)

`Transformer` 中的残差连接和层规范化，是训练非常**深度模型**的重要工具。

**层规范化**和**批量规范化**的目标相同，但层规范化是基于**特征维度进行规范化**。尽管批量规范化在计算机视觉中被广泛应用，但在**自然语言处理**任务中（输入通常是变长序列）批量规范化通常不如层规范化的效果好。

**残差连接**要求**两个输入形状相同**，以便**加法**操作后输出张量形状相同。

```py
# 残差连接后进行层规
class AddNorm(nn.Module):
  def __init__(self, normalized_shape, dropout, **kwargs):
    super(AddNorm, self).__init__(**kwargs)
    self.dropout = nn.Dropout(dropout)
    self.ln = nn.LayerNorm(normalized_shape)

  def forward(self, X, Y):
    return self.ln(self.dropout(Y) + X)

# 示例
add_norm = AddNorm([3, 4], 0.5)
add_norm.eval()
add_norm(torch.ones((2, 3, 4)), torch.ones((2, 3, 4))).shape
# torch.Size([2, 3, 4])
```

## 实现编码器

```py
# 编码器块
class EncoderBlock(nn.Module):
  def __init__(self, key_size, query_size, value_size, num_hiddens,
              norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
              dropout, use_bias=False, **kwargs):
    super(EncoderBlock, self).__init__(**kwargs)
    self.attention = d2l.MultiHeadAttention(
        key_size, query_size, value_size, num_hiddens, num_heads, dropout, use_bias)
    self.addnorm1 = AddNorm(norm_shape, dropout)
    self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
    self.addnorm2 = AddNorm(norm_shape, dropout)

  def forward(self, X, valid_lens):
    Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
    return self.addnorm2(Y, self.ffn(Y))

# Transformer编码器
class TransformerEncoder(d2l.Encoder):
  def __init__(self, vocab_size, key_size, query_size, value_size,
              num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
              num_heads, num_layers, dropout, use_bias=False, **kwargs):
    super(TransformerEncoder, self).__init__(**kwargs)
    self.num_hiddens = num_hiddens
    self.embedding = nn.Embedding(vocab_size, num_hiddens)
    self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
    self.blks = nn.Sequential()
    for i in range(num_layers):
      self.blks.add_module("block"+str(i), EncoderBlock(
        key_size, query_size, value_size, num_hiddens,
        norm_shape, ffn_num_input, ffn_num_hiddens,
        num_heads, dropout, use_bias))

  def forward(self, X, valid_lens, *args):
    # 因为位置编码值在-1和1之间，
    # 因此嵌入值乘以嵌入维度的平方根进行缩放，
    # 然后再与位置编码相加。
    X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
    self.attention_weights = [None] * len(self.blks)
    for i, blk in enumerate(self.blks):
      X = blk(X, valid_lens)
      self.attention_weights[i] = blk.attention.attention.attention_weights
    return X
```

## 解码器实现

```py
# 解码器中第i个块
class DecoderBlock(nn.Module):
  def __init__(self, key_size, query_size, value_size, num_hiddens,
                norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                dropout, i, **kwargs):
    super(DecoderBlock, self).__init__(**kwargs)
    self.i = i
    self.attention1 = d2l.MultiHeadAttention(
        key_size, query_size, value_size, num_hiddens, num_heads, dropout)
    self.addnorm1 = AddNorm(norm_shape, dropout)
    self.attention2 = d2l.MultiHeadAttention(
        key_size, query_size, value_size, num_hiddens, num_heads, dropout)
    self.addnorm2 = AddNorm(norm_shape, dropout)
    self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
    self.addnorm3 = AddNorm(norm_shape, dropout)

  def forward(self, X, state):
    enc_outputs, enc_valid_lens = state[0], state[1]
    # 训练阶段，输出序列的所有词元都在同一时间处理，
    # 因此state[2][self.i]初始化为None。
    # 预测阶段，输出序列是通过词元一个接着一个解码的，
    # 因此state[2][self.i]包含着直到当前时间步第i个块解码的输出表示
    if state[2][self.i] is None:
      key_values = X
    else:
      key_values = torch.cat((state[2][self.i], X), axis=1)
    state[2][self.i] = key_values
    if self.training:
      batch_size, num_steps, _ = X.shape
      # dec_valid_lens的开头:(batch_size,num_steps),
      # 其中每一行是[1,2,...,num_steps]
      dec_valid_lens = torch.arange(1, num_steps + 1, device=X.device).repeat(batch_size, 1)
    else:
      dec_valid_lens = None

    # 自注意力
    X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
    Y = self.addnorm1(X, X2)
    # 编码器－解码器注意力。
    # enc_outputs的开头:(batch_size,num_steps,num_hiddens)
    Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
    Z = self.addnorm2(Y, Y2)
    return self.addnorm3(Z, self.ffn(Z)), state

# 解码器
class TransformerDecoder(d2l.AttentionDecoder):
  def __init__(self, vocab_size, key_size, query_size, value_size,
                num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                num_heads, num_layers, dropout, **kwargs):
    super(TransformerDecoder, self).__init__(**kwargs)
    self.num_hiddens = num_hiddens
    self.num_layers = num_layers
    self.embedding = nn.Embedding(vocab_size, num_hiddens)
    self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
    self.blks = nn.Sequential()
    for i in range(num_layers):
      self.blks.add_module("block"+str(i),
          DecoderBlock(key_size, query_size, value_size, num_hiddens,
                        norm_shape, ffn_num_input, ffn_num_hiddens,
                        num_heads, dropout, i))
    self.dense = nn.Linear(num_hiddens, vocab_size)

  def init_state(self, enc_outputs, enc_valid_lens, *args):
    return [enc_outputs, enc_valid_lens, [None] * self.num_layers]

  def forward(self, X, state):
    X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
    self._attention_weights = [[None] * len(self.blks) for _ in range (2)]
    for i, blk in enumerate(self.blks):
      X, state = blk(X, state)
      # 解码器自注意力权重
      self._attention_weights[0][
          i] = blk.attention1.attention.attention_weights
      # “编码器－解码器”自注意力权重
      self._attention_weights[1][
          i] = blk.attention2.attention.attention_weights
    return self.dense(X), state

  @property
  def attention_weights(self):
    return self._attention_weights
```