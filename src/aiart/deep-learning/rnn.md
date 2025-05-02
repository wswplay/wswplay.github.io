---
title: 循环神经网络
outline: deep
---

# 循环神经网络(RNN)

**RNN**：Recurrent Neural Network。

到目前为止，我们遇到过两种类型的数据：表格**数据和图像**数据。对于图像数据，我们设计了专门的卷积神经网络架构来为这类特殊的数据结构建模。但若我们对图像中的像素位置进行重排，就会对图像中内容的推断造成极大的困难。

**卷积**神经网络处理**空间信息**，那么**循环神经网络**处理**序列信息**。

循环神经网络通过引入状态变量存储过去的信息和当前的输入，从而可以确定当前的输出。

预测明天的股价要比过去的股价更困难，尽管两者都只是估计一个数字。在统计学中，前者（对超出已知观测范围进行预测）称为**外推法**（`extrapolation`），而后者（在现有观测值之间进行估计）称为**内插法**（`interpolation`）。

:::tip 时间动力学
在深度学习和人工智能（AI）领域，时间动力学（Temporal Dynamics）主要研究**数据或系统状态随时间变化的规律**，以及如何建模、学习和预测这些动态行为。
:::

## 序列模型

### 自回归模型

$$
x_{t} \sim P(x_{t} \mid x_{t-1}, \ldots, x_{1})
$$

自回归模型（`Autoregressive Model`，简称 AR 模型）是一种统计方法，用于分析和预测时间序列数据。

核心思想：**用同一变量过去值来预测未来值**，即认为当前数据点与之前数据点存在**线性关系**。

自回归，是对自己执行回归。相对概念是**多变量时间序列回归**，如向量自回归（VAR, Vector Autoregression）：多个时间序列互相预测，例如用昨天的“气温”和“湿度”联合预测今天的“降水量”。

#### 隐变量自回归模型（latent autoregressive models）

![An Image](./img/sequence-model.svg)
保留一些过去观测总结 $h_t$，同时更新预测 $\hat{x}_t$ 和总结 $h_t$，产生基于 $\hat{x}_t = P(x_t \mid h_t)$ 估计 $x_t$，以及公式 $h_t = g(h_{t-1}, x_{t-1})$ 更新的模型。由于 $h_t$ 从未被观测到，这类模型也被称为**隐变量自回归模型**。

#### 马尔可夫模型

近似法中用 $x_{t-1}, \ldots, x_{t-\tau}$ 而不是 $x_{t-1}, \ldots, x_1$ 来估计 $x_t$ 。只要这种是近似精确的，我们就说序列满足**马尔可夫条件**（Markov condition）。

特别是，如果 $\tau=1$，得到一个一阶马尔可夫模型（first-order Markov model），$P(x)$ 由下式给出：

$$
P(x_1, \ldots, x_T) = \prod_{t=1}^T P(x_t \mid x_{t-1}) \quad \text{当} \quad P(x_1 \mid x_0) = P(x_1)
$$

:::tip 初始条件说明
当 $t=1$，$P(x_1 \mid x_0)$ 中 $x_0$ 不存在(或为初始状态)，故约定 $P(x_1 \mid x_0) = P(x_1)$，即 $x_1$ 的边缘概率。这确保了公式从 $t=1$ 开始时的合法性。
:::

当假设 $x_t$ 仅是离散值时，使用[动态规划](/aiart/deep-learning/basic-concept.html#动态规划)可以沿着[马尔可夫链](/aiart/deep-learning/basic-concept.html#马尔可夫链)精确地计算结果。可以高效地计算 $P(x_{t+1} \mid x_{t-1})$ :

$$
\begin{align*}
P(x_{t+1} \mid x_{t-1}) &= \frac{\sum_{x_t} P(x_{t+1}, x_t, x_{t-1})}{P(x_{t-1})} \\
&= \frac{\sum_{x_t} P(x_{t+1} \mid x_t, x_{t-1}) P(x_t, x_{t-1})}{P(x_{t-1})} \\
&= \sum_{x_t} P(x_{t+1} \mid x_t) P(x_t \mid x_{t-1})
\end{align*}
$$

由此，只需考虑一个非常短的历史：$P(x_{t+1} \mid x_t, x_{t-1}) = P(x_{t+1} \mid x_t)$。

## 文本预处理

### 常见步骤：将字符串顺序转化为数字索引(string order -> number index)

- 1、将文本作为字符串加载到内存中。
- 2、将字符串拆分为**词元**（如单词和字符）。
- 3、建立一个**词表**，将拆分的词元映射到数字索引。
- 4、将文本转换为数字索引序列，方便模型操作。

### 词元化(token)

每个文本序列被拆分成一个词元列表，**词元**是文本的基本单位，每个词元都是一个**字符串**（`string`）。

```py
['the', 'time', 'machine', 'by', 'h', 'g', 'wells']
```

### 词表(vocabulary)

词元的类型是字符串，而模型需要的输入是数字，因此构建一个字典，即**词表**：用来将**字符串**类型的词元**映射**到从
$0$ 开始的**数字索引**中。

将训练集所有文档合并，进行**唯一词元统计**，得到的结果称为**语料(corpus)**。

然后根据每个唯一词元的出现频率，为其分配一个数字索引。很少出现的词元通常被移除，这可以降低复杂性。另外，语料库中不存在或已删除的任何词元都将映射到一个特定的未知词元“`<unk>`”。我们可以选择增加一个列表，用于保存那些被保留的词元，例如：填充词元（“`<pad>`”）、序列开始词元（“`<bos>`”）、序列结束词元（“`<eos>`”）。

```py
[('<unk>', 0), ('the', 1), ('i', 2), ('and', 3), ('of', 4), ('a', 5)]
```

给每座山每条河流取一个温暖的名字，把每一条文本行转换成一个数字索引列表:

```py
文本: ['the', 'time', 'machine', 'by', 'h', 'g', 'wells']
索引: [1, 19, 50, 40, 2183, 2184, 400]
```

## 语言模型

比如 `deep learning niubi` 文本序列的概率是：

$$
P(deep, learning, niubi) = P(deep)P(learning \mid deep)P(niubi \mid deep, learning)
$$

为了训练语言模型，我们需要计算单词的概率，以及给定前面几个单词后出现某个单词的条件概率。这些概率本质上就是语言模型的参数。

## 循环神经网络

![An Image](./img/rnn.svg)
循环神经网络（recurrent neural networks，RNNs） 是具有隐状态的神经网络。循环神经网络模型的**参数数量**不会随着时间步的增加而增加。我们可以使用**困惑度**来评价语言模型的质量。

无隐状态的神经网络：

$$
H = \phi(\mathbf{X} \mathbf{W}_{xh} + \mathbf{b}_h)
$$

$$
O = \mathbf{H} \mathbf{W}_{hq} + \mathbf{b}_q
$$

有隐状态的**循环**神经网络：

$$
H_t = \phi(\mathbf{X}_t \mathbf{W}_{xh} + H_{t-1} \mathbf{W}_{hh} + \mathbf{b}_h)
$$

$$
O_t = \mathbf{H}_t \mathbf{W}_{hq} + \mathbf{b}_q
$$

`有隐藏状态`比`无隐藏状态`多了 $H_{t-1} \mathbf{W}_{hh}$，从相邻时间步的隐藏变量 $H_{t}$ 和 $H_{t-1}$ 之间关系可知，这些变量捕获并保留了序列直到其当前时间步的**历史信息**，就如当前时间步下神经网络的状态或**记忆**，因此这样的隐藏变量被称为**隐状态**（`hidden state`）。

由于在当前时间步中，**隐状态使用的定义与前一个时间步中使用的定义相同**，因此**计算是循环**的（`recurrent`）。 于是基于循环计算的隐状态神经网络被命名为**循环神经网络**，执行计算的层 称为**循环层**（`recurrent layer`）。

在任意时间步 $t$，隐状态的计算可以被视为：

- 1、拼接当前时间步 $t$ 的输入 $\mathbf{X}_t$ 和前一时间步 $t-1$ 的隐状态 $\mathbf{H}_{t-1}$；
- 2、将拼接结果送入带有激活函数 $\phi$ 的全连接层，全连接层输出当前时间步 $t$ 的隐状态 $\mathbf{H}_t$。

## 从零实现 RNN

```py
# 1、引入工具包和、加载数据
import math
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

# 2、参数初始化
def get_params(vocab_size, num_hiddens, device):
  num_inputs = num_outputs = vocab_size

  def normal(shape):
    return torch.randn(size=shape, device=device) * 0.01

  # 隐藏层参数
  W_xh = normal((num_inputs, num_hiddens))
  W_hh = normal((num_hiddens, num_hiddens))
  b_h = torch.zeros(num_hiddens, device=device)
  # 输出层参数
  W_hq = normal((num_hiddens, num_outputs))
  b_q = torch.zeros(num_outputs, device=device)
  # 附加梯度
  params = [W_xh, W_hh, b_h, W_hq, b_q]
  for param in params:
    param.requires_grad_(True)
  return params

# 3、建模
def init_rnn_state(batch_size, num_hiddens, device):
  return (torch.zeros((batch_size, num_hiddens), device=device), )

def rnn(inputs, state, params):
  # inputs的形状：(时间步数量，批量大小，词表大小)
  W_xh, W_hh, b_h, W_hq, b_q = params
  H, = state
  outputs = []
  # X的形状：(批量大小，词表大小)
  for X in inputs:
    H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
    Y = torch.mm(H, W_hq) + b_q
    outputs.append(Y)
  return torch.cat(outputs, dim=0), (H,)

class RNNModelScratch:
  def __init__(self, vocab_size, num_hiddens, device, get_params, init_state, forward_fn):
    self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
    self.params = get_params(vocab_size, num_hiddens, device)
    self.init_state, self.forward_fn = init_state, forward_fn

  def __call__(self, X, state):
    X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
    return self.forward_fn(X, state, self.params)

  def begin_state(self, batch_size, device):
    return self.init_state(batch_size, self.num_hiddens, device)

# 4、模型初始化
num_hiddens = 512
net = RNNModelScratch(len(vocab), num_hiddens, d2l.try_gpu(), get_params, init_rnn_state, rnn)
state = net.begin_state(X.shape[0], d2l.try_gpu())
Y, new_state = net(X.to(d2l.try_gpu()), state)
Y.shape, len(new_state), new_state[0].shape

# 5、预测：在prefix后面生成新字符
def predict_ch8(prefix, num_preds, net, vocab, device):
  state = net.begin_state(batch_size=1, device=device)
  outputs = [vocab[prefix[0]]]
  get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))
  for y in prefix[1:]:  # 预热期
    _, state = net(get_input(), state)
    outputs.append(vocab[y])
  for _ in range(num_preds):  # 预测num_preds步
    y, state = net(get_input(), state)
    outputs.append(int(y.argmax(dim=1).reshape(1)))
  return ''.join([vocab.idx_to_token[i] for i in outputs])

# 6、训练
def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
  state, timer = None, d2l.Timer()
  metric = d2l.Accumulator(2)  # 训练损失之和,词元数量
  for X, Y in train_iter:
    if state is None or use_random_iter:
      # 在第一次迭代或使用随机抽样时初始化state
      state = net.begin_state(batch_size=X.shape[0], device=device)
    else:
      if isinstance(net, nn.Module) and not isinstance(state, tuple):
        # state对于nn.GRU是个张量
        state.detach_()
      else:
        # state对于nn.LSTM或对于我们从零开始实现的模型是个张量
        for s in state:
          s.detach_()
    y = Y.T.reshape(-1)
    X, y = X.to(device), y.to(device)
    y_hat, state = net(X, state)
    l = loss(y_hat, y.long()).mean()
    if isinstance(updater, torch.optim.Optimizer):
      updater.zero_grad()
      l.backward()
      grad_clipping(net, 1)
      updater.step()
    else:
      l.backward()
      grad_clipping(net, 1)
      # 因为已经调用了mean函数
      updater(batch_size=1)
    metric.add(l * y.numel(), y.numel())
  return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()
```

### 梯度剪裁

对于长度为 $T$ 的序列，我们在迭代中计算这 $T$个时间步上的梯度，将会在反向传播过程中产生长度为 $O(T)$ 的矩阵乘法链。当 $T$ 较大时，它可能导致数值不稳定，例如可能导致梯度爆炸或梯度消失。有时梯度可能很大，从而优化算法可能无法收敛。一个流行的替代方案是通过将梯度 $\mathbf{g}$ 投影回给定半径(如 $\theta$)的球来裁剪梯度。如下式：

$$
\mathbf{g} \gets \min\left(1, \frac{\theta}{\|\mathbf{g}\|}\right) \mathbf{g}
$$

```py
def grad_clipping(net, theta):
  if isinstance(net, nn.Module):
    params = [p for p in net.parameters() if p.requires_grad]
  else:
    params = net.params
  norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
  if norm > theta:
    for param in params:
      param.grad[:] *= theta / norm
```

### 小结

- 循环神经网络模型在训练以前需要初始化状态，不过**随机抽样和顺序划分**使用初始化方法不同。当使用顺序划分时，我们需要分离梯度以减少计算量。
- 在进行任何预测之前，模型通过**预热期**进行自我更新（例如，获得比初始值更好的隐状态）。
- 梯度裁剪可以防止梯度爆炸，但不能应对**梯度消失**。
