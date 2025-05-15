---
title: 自然语言处理、NLP
outline: deep
---

# 自然语言处理(NLP)

**NLP**：Natural Language Processing，是指研究使用自然语言的**计算机和人类**之间的**交互**。

## 预训练(pre-training)

**自监督学习**<sup>self-supervised learning</sup>已被广泛用于**预训练**文本表示，例如通过使用周围文本的其它部分来预测文本的隐藏部分。

### 子词嵌入

自然语言是用来表达人脑思维的复杂系统，**词**是意义的基本单元。

- **词向量**：用于表示**单词意义**的向量，并且还可以被认为是单词**特征向量或表示**。
- **词嵌入**：将**单词映射到实向量**的技术。

**子词嵌入**<sup>Subword Embedding</sup>是一种在自然语言处理（NLP）中用于表示单词的技术，它通过**将单词拆分为更小的单元**（子词）来解决传统词嵌入的局限性（如未登录词问题）。

- **未登录词（OOV）问题**：传统词嵌入（如 Word2Vec）无法处理词汇表外的单词。
- **形态学相似性**：子词能捕捉单词间的结构关系（如"running" → "run" + "ing"）。
- **多语言支持**：共享子词可跨语言建模（如拉丁语系的共同词根）。

**代码示例（Hugging Face 库）**

```python
from transformers import AutoTokenizer

# 加载BERT的WordPiece分词器
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokens = tokenizer.tokenize("unhappiness")  # 输出：['un', '##happy', '##ness']

# 加载BPE分词器（GPT-2）
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokens = tokenizer.tokenize("unhappiness")  # 输出：['un', 'happiness']
```

### 字节对编码(BPE)

**BPE**：Byte Pair Encoding。
