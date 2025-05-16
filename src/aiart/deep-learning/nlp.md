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
# Hugging Face 库调用
from transformers import GPT2Tokenizer

# GPT-2使用BPE分词
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokens = tokenizer.tokenize("unhappiness")  # 输出：['un', 'happiness']
```

**代码实现**：

```py
import collections

symbols = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
           'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
           '_', '[UNK]']

# 频率字典
raw_token_freqs = {'fast_': 4, 'faster_': 3, 'tall_': 5, 'taller_': 4}
token_freqs = {}
for token, freq in raw_token_freqs.items():
  token_freqs[' '.join(list(token))] = raw_token_freqs[token]
token_freqs
# {'f a s t _': 4, 'f a s t e r _': 3, 't a l l _': 5, 't a l l e r _': 4}

# 返回词内最频繁的连续符号对，其中词来自输入词典token_freqs的键
def get_max_freq_pair(token_freqs):
  pairs = collections.defaultdict(int)
  for token, freq in token_freqs.items():
    symbols = token.split()
    for i in range(len(symbols) - 1):
      # “pairs”的键是两个连续符号的元组
      pairs[symbols[i], symbols[i + 1]] += freq
  return max(pairs, key=pairs.get)  # 具有最大值的“pairs”键

# 基于连续符号频率的贪心方法，合并最频繁的连续符号对以产生新符号
def merge_symbols(max_freq_pair, token_freqs, symbols):
  symbols.append(''.join(max_freq_pair))
  new_token_freqs = dict()
  for token, freq in token_freqs.items():
    new_token = token.replace(' '.join(max_freq_pair), ''.join(max_freq_pair))
    new_token_freqs[new_token] = token_freqs[token]
  return new_token_freqs

# 迭代
num_merges = 10
for i in range(num_merges):
  max_freq_pair = get_max_freq_pair(token_freqs)
  token_freqs = merge_symbols(max_freq_pair, token_freqs, symbols)
  print(f'合并# {i+1}:',max_freq_pair)

# 合并# 1: ('t', 'a')
# 合并# 2: ('ta', 'l')
# 合并# 3: ('tal', 'l')
# 合并# 4: ('f', 'a')
# 合并# 5: ('fa', 's')
# 合并# 6: ('fas', 't')
# 合并# 7: ('e', 'r')
# 合并# 8: ('er', '_')
# 合并# 9: ('tall', '_')
# 合并# 10: ('fast', '_')

print(symbols)
# ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '_', '[UNK]', 'ta', 'tal', 'tall', 'fa', 'fas', 'fast', 'er', 'er_', 'tall_', 'fast_']

print(list(token_freqs.keys()))
# ['fast_', 'fast er_', 'tall_', 'tall er_']

# 用从一个数据集学习的子词来切分另一个数据集的单词
# 尝试将单词从输入参数symbols分成可能最长的子词
def segment_BPE(tokens, symbols):
  outputs = []
  for token in tokens:
    start, end = 0, len(token)
    cur_output = []
    # 具有符号中可能最长子字的词元段
    while start < len(token) and start < end:
      if token[start: end] in symbols:
        cur_output.append(token[start: end])
        start = end
        end = len(token)
      else:
        end -= 1
    if start < len(token):
      cur_output.append('[UNK]')
    outputs.append(' '.join(cur_output))
  return outputs

tokens = ['tallest_', 'fatter_']
print(segment_BPE(tokens, symbols))
# ['tall e s t _', 'fa t t er_']
```

字节对编码执行训练数据集的统计分析，以发现词内的公共符号。作为一种**贪心方法**，字节对编码迭代地**合并最频繁的连续符号对**。
