---
title: GPT
outline: deep
---

# GPT

**GPT**：Generative Pre-trained Transformer，**生成式·预训练·变换模型**。

**OpenAI 最新 GPT-4.1** 支持 100 万 token 的**超长上下文**（约 75 万字），适用于金融分析、法律文档处理、大型代码库分析等任务。

## 分词算法-BPE

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
