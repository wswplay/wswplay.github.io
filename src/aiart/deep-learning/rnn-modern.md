---
title: 现代循环神经网络
outline: deep
---

# 现代循环神经网络

## 门控机制

### 长短期记忆网络(LSTM)

LSTM：long short-term memory。

可以认为 LSTM 的核心创新就是引入了`Cell State`（细胞状态）这个"全局记忆通道"，而 GRU 可以看作是对 LSTM 的简化版本。

![An Image](./img/lstm.svg)
核心组件：

- **遗忘门(Forget Gate, f)**：决定丢弃哪些信息。
- **输入门(Input Gate, i)**：决定更新哪些信息。
- **输出门(Output Gate, o)**：决定输出哪些信息。
- **细胞状态(Cell State, C)**：长期记忆通道。

长短期记忆网络的隐藏层输出包括“隐状态”和“记忆元”。只有隐状态会传递到输出层，而记忆元完全属于内部信息。

### 门控循环单元(GRU)

GRU：Gated Recurrent Unit。

普通循环神经网络之相比，门控循环单元与支持**隐状态门控**，是简化版的 LSTM。

这意味着模型有专门的机制来确定应该**何时更新隐状态，以及应该何时重置隐状态。这些机制是可学习的**。例如，如果第一个词元非常重要，模型将学会在第一次观测之后不更新隐状态。同样，模型也可以学会跳过不相关的临时观测。

它通过引入可学习的"门"来控制信息流动，决定哪些信息应该被保留、哪些应该被遗忘。

![An Image](./img/gru.svg)
核心组件：

- **重置门(Reset Gate, r)**：控制前一时刻隐藏状态有多少信息需要被"遗忘"。
- **更新门(Update Gate, z)**：控制新状态中有多少来自前一状态，有多少来自当前计算的新候选状态。

重置门打开时，门控循环单元包含基本循环神经网络；更新门打开时，门控循环单元可以跳过子序列。

### 对比与现代改进方向

| 机制       | GRU                        | LSTM                             |
| ---------- | -------------------------- | -------------------------------- |
| 门数量     | 2 个(更新门、重置门)       | 3 个(输入门、遗忘门、输出门)     |
| 状态变量   | 只有隐藏状态 h             | 隐藏状态 h + 细胞状态 C          |
| 参数数量   | 较少(约少 1/3)             | 较多                             |
| 计算复杂度 | 较低                       | 较高                             |
| 信息流     | 直接通过隐藏状态传递       | 通过细胞状态和隐藏状态双通道传递 |
| 性能表现   | 简单任务表现好，资源消耗低 | 复杂任务表现更稳定               |

现代最新架构（如 `Transformer`）实际上吸收了这两种思想的优点：

- LSTM 思想：通过残差连接实现"记忆高速公路"
- GRU 思想：简化门控机制（如 Transformer 中的 FFN 层）

## 编码器-解码器架构

![An Image](./img/encoder-decoder.svg)
编码器-解码器（`encoder-decoder`）架构两个主要组件：

1. **编码器**（encoder）：接受长度可变序列作为输入，并将其转换为具有固定形状编码状态。
2. **解码器**（decoder）：将固定形状编码状态映射到长度可变序列。

“编码器－解码器”架构可以将长度可变的序列作为输入和输出，因此适用于**机器翻译**等序列转换问题。

### 编码器

```py
from torch import nn
class Encoder(nn.Module):
  def __init__(self, **kwargs):
    super(Encoder, self).__init__(**kwargs)

  def forward(self, X, *args):
    raise NotImplementedError
```

### 解码器

```py
class Decoder(nn.Module):
  def __init__(self, **kwargs):
    super(Decoder, self).__init__(**kwargs)

  def init_state(self, enc_outputs, *args):
    raise NotImplementedError

  def forward(self, X, state):
    raise NotImplementedError
```

init_state 函数，用于将编码器的输出(`enc_outputs`)转换为编码后的状态。注意，此步骤可能需要额外的输入，例如：输入序列的有效长度。为了逐个地生成长度可变的词元序列，解码器在每个时间步都会将输入(例如：在前一时间步生成的词元)和编码后的状态映射成当前时间步的输出词元。

### 合并

```py
class EncoderDecoder(nn.Module):
  def __init__(self, encoder, decoder, **kwargs):
    super(EncoderDecoder, self).__init__(**kwargs)
    self.encoder = encoder
    self.decoder = decoder

  def forward(self, enc_X, dec_X, *args):
    enc_outputs = self.encoder(enc_X, *args)
    dec_state = self.decoder.init_state(enc_outputs, *args)
    return self.decoder(dec_X, dec_state)
```

“编码器-解码器”架构包含了一个编码器和一个解码器，并且还拥有可选的额外的参数。在前向传播中，编码器的输出用于生成编码状态，这个状态又被解码器作为其输入的一部分。

## 序列到序列学习(seq2seq)

seq2seq：sequence to sequence。

![An Image](./img/seq2seq.svg)
上图是机器翻译中使用两个循环神经网络进行序列到序列学习。特定的`“<eos>”`表示序列结束词元。一旦输出序列生成此词元，模型就会停止预测。

在循环神经网络解码器的初始化时间步，有两个特定的设计决定：首先，特定的`“<bos>”`表示序列开始词元，它是解码器的输入序列的第一个词元。其次，使用循环神经网络编码器最终的隐状态来初始化解码器的隐状态。

### 用 RNN 实现编码器

```py
import collections
import math
import torch
from torch import nn
from d2l import torch as d2l

# 用于序列到序列学习的循环神经网络编码器
class Seq2SeqEncoder(d2l.Encoder):
  def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs):
    super(Seq2SeqEncoder, self).__init__(**kwargs)
    # 嵌入层
    self.embedding = nn.Embedding(vocab_size, embed_size)
    self.rnn = nn.GRU(embed_size, num_hiddens, num_layers, dropout=dropout)

  def forward(self, X, *args):
    # 输出'X'的形状：(batch_size,num_steps,embed_size)
    X = self.embedding(X)
    # 在循环神经网络模型中，第一个轴对应于时间步
    X = X.permute(1, 0, 2)
    # 如果未提及状态，则默认为0
    output, state = self.rnn(X)
    # output的形状:(num_steps,batch_size,num_hiddens)
    # state的形状:(num_layers,batch_size,num_hiddens)
    return output, state
```

**嵌入层**(`embedding layer`，`nn.Embedding`)，用以获得输入序列中每个词元的特征向量。

嵌入层的权重是一个矩阵，其行数是输入词表大小(`vocab_size`)，其列数是特征向量维度(`embed_size`)。对于任意输入词元索引 $i$，嵌入层获取权重矩阵第 $i$ 行(`从0开始`)以返回其特征向量。

```py
encoder = Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2)
encoder.eval()
X = torch.zeros((4, 7), dtype=torch.long)
output, state = encoder(X)
output.shape, state.shape

# (torch.Size([7, 4, 16]), torch.Size([2, 4, 16]))
```

:::tip
`embed_size << vocab_size`，如大小接近，或导致参数量爆炸、过拟合。比率：`1/10 到 1/100`。  
`num_hiddens >= embed_size`，确保隐藏层有足够容量融合时序信息和输入特征。比率：`num_hiddens = 2~4 × embed_size`。
:::

最后一层的隐状态的输出是一个张量（output 由编码器的循环层返回），其形状为（时间步数，批量大小，隐藏单元数）。最后一个时间步的多层隐状态的形状是（隐藏层的数量，批量大小，隐藏单元的数量）。如果使用长短期记忆网络，`state`中还将包含记忆单元信息。
