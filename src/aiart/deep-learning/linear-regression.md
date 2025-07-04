---
title: 线性回归
outline: deep
---

# 线性回归

回归（`regression`）是能为一个或多个自变量与因变量之间关系建模的一类方法。

## 线性模型

使用 $n$ 来表示数据集中的样本数。对索引为 $i$ 的样本，其输入表示为 $x^{(i)} = [x_1^{(i)}, x_2^{(i)}]^T$，其对应的标签是 $y^{(i)}$。

线性假设是指目标（房屋价格）可以表示为特征（面积和房龄）的加权和，如下面的式子：

$$
price = w_{area} · area + w_{age} · age + b
$$

- $w_{area}$ 和 $w_{age}$ 称为**权重**（`weight`），权重决定了每个特征对预测值的影响。
- $b$ 称为**偏置**（`bias`）、偏移量（`offset`）或截距（`intercept`）。偏置是指当所有特征都取值为 0 时，预测值应该为多少。

即使现实中，不会有任何房子的面积是 0 或房龄正好是 0 年，仍需要偏置项。如果**没有偏置**项，模型的**表达能力将受到限制**。

给定一个数据集，目标是**寻找模型权重 $w$ 和偏置 $b$**，使得根据模型做出的预测大体符合数据里的真实价格。输出的预测值由输入特征通过线性模型的仿射变换决定，仿射变换**由所选权重和偏置确定**。

**标量**：当输入包含 $d$ 个特征，预测结果 $\hat{y}$（“尖角”$y$ 通常代表估计值）表示为：

$$
\hat{y} = w_1x_1 + ... +w_dx_d + b
$$

**向量**：将所有特征放到向量 $\mathbf{x} ∈ R^d$ 中， 并将所有权重放到向量 $\mathbf{w} ∈ R^d$ 中， 可用**点积**简洁表达模型：

$$
\hat{y} = \mathbf{w}^T\mathbf{x} + b
$$

**矩阵**：矩阵 $\mathbf{X} ∈ R^{n×d}$ 表示整个数据集 $n$ 个样本，预测值 $\hat{y} ∈ R^n$ 可通过**矩阵-向量乘法**表示为：

$$
\hat{y} = \mathbf{X}\mathbf{w} + b
$$

## 损失函数

损失函数（`loss function`）能够量化目标的**实际值与预测值**之间的**差距**。通常我们会选择**非负数**作为损失，且数值越小表示损失越小，完美预测时的损失为 0。

回归问题中最常用的损失函数是：**平方误差函数**。

单个样本 $i$ 预测值为 $\hat{y}^i$，真实标签为 $y^{i}$ 时，平方误差可以定义为以下公式：

$$
l^i(\mathbf{w}, b) = \frac{1}{2}(\hat{y}^i - y^i)^2
$$

常数 $\frac{1}{2}$ 不会带来本质的差别，但这样在形式上稍微简单一些 （因为当我们对损失函数求导后常数系数为 1）

$n$ 个样本上的**损失均值**（也等价于求和）：

$$
L(\mathbf{w}, b) = \frac{1}{n}\sum_{i=1}^n l^i(\mathbf{w}, b) = \frac{1}{n}\sum_{i=1}^n \frac{1}{2}(\mathbf{w}^T\mathbf{x}^i + b - y^i)^2
$$

训练模型时，寻找一组参数（$\mathbf{w}^*$, $b^*$），能最小化在所有训练样本上总损失。如下式：

$$
\mathbf{w}^*, b^* = \underset{\mathbf{w},b}{\mathrm{argmin}}\, L(\mathbf{w}, b)
$$

## 解析解

线性回归，刚好是一个很简单的优化问题。与其他大部分模型不同，线性回归的解可以用一个公式简单地表达出来，这类解叫作解析解（`analytical solution`）。

首先，我们将偏置 $b$ 合并到参数 $\mathbf{w}$ 中，合并方法是在包含所有参数的矩阵中附加一列，预测问题是最小化 $\Vert\mathbf{y} - \mathbf{X}\mathbf{w}\Vert^2$。这在损失平面上只有一个临界点，这个临界点对应于整个区域的损失极小点。将损失关于 $\mathbf{w}$ 的导数设为 0，得到解析解：

$$
\mathbf{w}^* = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}
$$

**并不是所有问题都存在解析解**。解析解可以进行很好的数学分析，但解析解对问题的限制很严格，导致它无法广泛应用在深度学习里。

## 随机梯度下降

梯度下降（`gradient descent`）， 几乎可以优化所有深度学习模型。它通过不断地在损失函数递减的方向上更新参数来降低误差。

可以调整但不在训练过程中更新的参数称为**超参数**（`hyperparameter`）。 调参（`hyperparameter tuning`）是选择超参数的过程。超参数通常是我们根据训练迭代结果来调整的，而训练迭代结果是在独立的验证数据集（validation dataset）上评估得到的。

## 正态分布与平方损失

正态分布（normal distribution），也称为**高斯分布**（Gaussian distribution），最早由德国数学家高斯（Gauss）应用于天文学研究。

若随机变量 $x$ 具有均值 $\mu$ 和方差 $\sigma^2$（标准差 $\sigma$ ），其正态分布概率密度函数如下：

$$
p(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{1}{2\sigma^2} (x - \mu)^2\right)
$$

在高斯噪声的假设下，最小化均方误差等价于对线性模型的极大似然估计。

## softmax 回归

统计学家很早就发明了一种表示分类的方法：**独热编码**（`one-hot encoding`）。独热编码是一个向量，它的分量和类别一样多。

类别对应的分量设置为`1`，其他所有分量设置为`0`。在我们的例子中，标签将是一个三维向量，其中(1,0,0)对应于“猫”、(0,1,0)对应于“鸡”、(0,0,1)对应于“狗”。

社会科学家邓肯·卢斯于 1959 年在选择模型（choice model）理论上发明[softmax 函数](/aiart/deep-learning/basic-concept.html#softmax-函数)：

> softmax 函数，能够将未规范化预测变换为非负数、且总和为 1，并让模型保持可导。

为了完成这一目标，我们首先对每个未规范化的预测求幂，这样可以确保输出非负。为了确保最终输出的概率值总和为 1，我们再让每个求幂后的结果除以它们的总和。

尽管 softmax 是一个非线性函数，但 softmax 回归的输出仍然由输入特征的仿射变换决定。因此，softmax 回归是一个线性模型（linear model）。

## 信息论与熵

信息论（information theory）涉及编码、解码、发送以及尽可能简洁地处理信息或数据。

信息论核心思想是**量化数据的信息内容**，即**信息量**。该数值被称为分布 $P$ 的**熵**（`entropy`）。

压缩与预测有什么关系呢？想象一下，我们有一个要压缩的数据流。如果我们很容易预测下一个数据，那么这个数据就很容易压缩。为什么呢？ 举一个极端的例子，假如数据流中的每个数据完全相同，这会是一个非常无聊的数据流。由于它们总是相同的，我们总是知道下一个数据是什么。所以，为了传递数据流的内容，我们不必传输任何信息。也就是说，“下一个数据是 xx”这个事件毫无信息量。

但是，如果我们不能完全预测每一个事件，那么我们有时可能会感到**惊异**。当我们赋予一个事件较低的概率时，我们的**惊异会更大，该事件的信息量也就更大**。

**熵**，是当分配的概率真正匹配数据生成过程时的**信息量的期望**。

## 从零实现线性回归

```python
# 引入包资源
import torch
import random
from d2l import torch as d2l

# 生成数据：y = Xw + b + noise
def synthetic_data(w, b, num_samples):
  X = torch.normal(0, 1, (num_samples, len(w)))
  y = torch.matmul(X, w) + b
  y += torch.normal(0, 0.01, y.shape)

  return X, y.reshape(-1, 1)

# 定义源参数
origin_w = torch.tensor([2.6, -3.4])
origin_b = 4.2

# 生成数据
features, labels = synthetic_data(origin_w, origin_b, 1000)

# 定义数据迭代器
def data_iter(batch_size, features, labels):
  num_samples = len(features)
  indexes = list(range(num_samples))
  random.shuffle(indexes)

  for i in range(0, num_samples, batch_size):
    batch_indexes = torch.tensor(indexes[i: min(i + batch_size, num_samples)])
    yield features[batch_indexes], labels[batch_indexes]

# 创建迭代数据
batch_size = 10
for X, y in data_iter(batch_size, features, labels):
  print(X, '\n', y)
  break

# 初始化模型参数
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
w, b

# 定义模型
def linereg(X, w, b):
  return torch.matmul(X, w) + b

# 定义损失函数
def squared_loss(y_hat, y):
  return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

# 定义优化器
def sgd(params, lr, batch_size):
  with torch.no_grad():
    for pitem in params:
      pitem -= lr * pitem.grad / batch_size
      pitem.grad.zero_()

# 定义超参
lr = 0.03
num_epochs = 3
net = linereg
loss = squared_loss

# 开启迭代训练
for epo in range(num_epochs):
  for X, y in data_iter(batch_size, features, labels):
    l = loss(net(X, w, b), y)
    l.sum().backward()
    # 更新参数
    sgd([w, b], lr, batch_size)

  with torch.no_grad():
    print('w:', w, '\n', 'b:', b)
    train_l = loss(net(features, w, b), labels)
    print(f'epoch {epo + 1}: loss {train_l.mean(): f} \n')

# epoch 1: loss  0.051860
# epoch 2: loss  0.000201
# epoch 3: loss  0.000048

# 训练完成，获取学习到的参数。与源参数对比，相差无几
w, b
# (tensor([[ 2.5996],
#          [-3.3998]], requires_grad=True),
#  tensor([4.1998], requires_grad=True))
```
