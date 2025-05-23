---
title: 概览
outline: deep
---

# ML：Machine Learning

根据业务逻辑设计自动化系统，驱动正常运行的产品和系统，是人类认知上的**非凡壮举**。

但如果它一直执行相同业务逻辑，无论积累多少经验，都不会自动提高，除非开发人员认识到问题并更新软件。如是，这**毫无智能**可言。

人类长期以来就有分析数据和预测未来结果的愿望，而**自然科学**大部分都植根于此。

幸运的是，对日益壮大的机器学习科学家群体来说，实现很多任务的自动化并**不再屈从于人类**所能考虑到的逻辑。

## 何为「机器」

机器学习中的「机器」既不是单纯的代码，也不是玄学的“智能体”，是一个由数学函数构成、通过数据动态优化参数、以泛化未知输入为目标的可编程系统。

**即：数据驱动的、参数化、可编程、数学函数系统优化器**。

本质是**数学**，代码是载体，数学是灵魂。

它的“智能”本质是**统计规律的可计算性表达**，而非人类的理解或意识。

机器学习模型在训练前后完全不同——初始代码是静态的，但训练后的模型是数据动态塑造的结果。

| 术语     | 定义                          | 与“机器”的关系       |
| -------- | ----------------------------- | -------------------- |
| **算法** | 解决问题的步骤（如梯度下降）  | “机器”的优化引擎     |
| **模型** | 参数化的数学函数（如 Y=WX+b） | “机器”的本体         |
| **代码** | 实现算法的编程语言表达        | “机器”的物理存在形式 |

## 特点

> **机器学习，是研究计算机系统如何利用经验（通常是数据）来提高特定任务的性能。**

我们只需要定义一个灵活的程序算法，其输出由许多参数（`parameter`）决定，然后使用数据集来确定当下的“最佳参数集”，这些参数通过某种性能度量方式来达到完成任务的最佳性能。

那么到底什么是参数呢？ 参数可以被看作旋钮，旋钮的转动可以调整程序的行为。 任一调整参数后的程序被称为模型（`model`）。 通过操作参数而生成的所有不同程序（输入-输出映射）的集合称为“模型族”。 使用数据集来选择参数的元程序被称为学习算法（`learning algorithm`）。

但如果模型所有的按钮（模型参数）都被随机设置，就不太可能识别出“Alexa”“Hey Siri”或任何其他单词。 在机器学习中，学习（learning）是一个训练模型的过程。 通过这个过程，我们可以发现正确的参数集，从而使模型强制执行所需的行为。 换句话说，我们用数据训练（train）模型。训练过程通常包含如下步骤：
:::tip

- 1、从一个随机初始化参数的模型开始，这个模型基本没有“智能”；
- 2、获取一些数据样本（例如，音频片段以及对应的是或否标签）；
- 3、调整参数，使模型在这些样本中表现得更好；
- 4、重复第（2）步和第（3）步，直到模型在任务中的表现令人满意。
  :::

## 要件

无论什么类型的机器学习问题，都会遇到这些组件：
:::tip

- 1、可以用来学习的数据（data）；
- 2、如何转换数据的模型（model）；
- 3、一个目标函数（objective function），用来量化模型的有效性；
- 4、调整模型参数以优化目标函数的算法（algorithm）。

:::

### 数据

每个数据集由一个个样本（example, sample）组成，大多时候，它们遵循独立同分布(independently and identically distributed, i.i.d.)。 样本有时也叫做数据点（data point）或者数据实例（data instance），通常每个样本由一组称为特征（features，或协变量（covariates））的属性组成。 机器学习模型会根据这些属性进行预测。 在上面的监督学习问题中，要预测的是一个特殊的属性，它被称为标签（label，或目标（target））。

当每个样本的特征类别数量都是相同的时候，其特征向量是固定长度的，这个长度被称为数据的**维数（dimensionality）**。 固定长度的特征向量是一个方便的属性，它可以用来量化学习大量样本。

再比如，如果用“过去的招聘决策数据”来训练一个筛选简历的模型，那么机器学习模型可能会无意中捕捉到**历史残留的不公正，并将其自动化**。
