---
title: 自然语言处理、NLP
outline: deep
---

# 自然语言处理(NLP)

**NLP**：Natural Language Processing，是指研究使用自然语言的**计算机和人类**之间的**交互**。

## 预训练(pre-training)

**自监督学习**<sup>self-supervised learning</sup>已被广泛用于**预训练**文本表示，例如通过使用周围文本的其它部分来预测文本的隐藏部分。

### 子词嵌入

```py
# 代码示例（Hugging Face 库）
from transformers import AutoTokenizer

# 加载BERT的WordPiece分词器
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokens = tokenizer.tokenize("unhappiness")  # 输出：['un', '##happy', '##ness']

# 加载BPE分词器（GPT-2）
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokens = tokenizer.tokenize("unhappiness")  # 输出：['un', 'happiness']
```

自然语言是用来表达人脑思维的复杂系统，**词**是意义的基本单元。

- **词向量**：用于表示**单词意义**的向量，并且还可以被认为是单词**特征向量或表示**。
- **词嵌入**：将**单词映射到实向量**的技术。

**子词嵌入**<sup>Subword Embedding</sup>通过**将单词拆分为更小的单元**（子词）来解决传统词嵌入局限性。

- 未登录词（OOV）问题：传统词嵌入（如 Word2Vec）无法处理词汇表外的单词。
- 形态学相似性：子词能捕捉单词间的结构关系（如"running" → "run" + "ing"）。
- 多语言支持：共享子词可跨语言建模（如拉丁语系的共同词根）。

**子词嵌入** = **子词分词**(`BPE`等) + **向量化**<sup>Embedding</sup>

### 字节对编码(BPE)

**BPE**：Byte Pair Encoding，是一种**子词分词算法**<sup>Subword Tokenization</sup>。

**核心思想**：通过迭代**合并高频符号对**(字节对)构建词汇表，有效平衡词表大小与语义覆盖能力。

**基本步骤**：

1. **初始化词汇表**：将所有基本字符（如字母、标点）加入词表。
2. **统计符号对频率**：在训练语料中统计所有相邻符号对的共现频率。
3. **合并最高频对**：将出现频率最高的符号对合并为一个新符号，加入词表。
4. **重复迭代**：持续合并直到达到预设的词表大小或迭代次数。

**示例演示**：

假设语料为 `"low low low lower newest widest"`：

- 初始词表：`{l, o, w, e, r, n, s, t, d, i}`
- 第 1 步：最高频对 `"lo"`（出现 6 次）→ 合并为 `"lo"`，词表新增 `"lo"`。
- 第 2 步：最高频对 `"low"`（出现 4 次）→ 合并为 `"low"`，词表新增 `"low"`。
- 后续可能合并 `"er"`、`"est"`等。
- 最终词表可能包含子词如：`low, er, est, newer, wid`。

**优点优势**：

- **解决未登录词（OOV）**：通过子词组合表示罕见词（如 `"unhappiness"` → `"un" + "happy" + "ness"`）。
- **压缩词表大小**：避免维护超大词表（如英语常用词约 20 万，BPE 词表可压缩到 1 万~3 万）。
- **保留语义**：相似词共享子词（如 `"playing"` 和 `"played"` 共享 `"play"`）。

**代码示例**：

```py
from transformers import GPT2Tokenizer

# GPT-2使用BPE分词
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokens = tokenizer.tokenize("unhappiness")  # 输出：['un', 'happiness']
```

## GPT

**GPT**：Generative Pre-trained Transformer，生成式·预训练·变换模型。

最新的 **GPT-4.1** 支持 100 万 token 的超长**上下文**（约 75 万字），远超 GPT-4o 的 12.8 万 token，适用于金融分析、法律文档处理、大型代码库分析等任务。

### 分词算法-BPE

GPT（如 GPT-2/3/4）**使用 BPE 分词算法**。

GPT-4 词表可能范围：

- **GPT-2 的词表：50,257**（基于 BPE 分词）
- **GPT-3 的词表：50,257**（与 GPT-2 相同）
- **GPT-4 的词表：可能仍保持在 50,257 左右**，或略有调整。但通过更高效的 `token` 利用和模型架构优化（如 MoE）提升性能。

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
```

输出示例：

```
Merge 1: ('e', 's') -> es
Merge 2: ('es', 't') -> est
...
```
