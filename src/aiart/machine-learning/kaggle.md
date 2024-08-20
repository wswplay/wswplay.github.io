---
title: kaggle
---

# Kaggle

### 登录 Hiuggngface

- 在 `notebook` 中加载 `Huggingface` 模型，有时需要 `token` 登录。
- 在 `Google colab` 中也是如此登录。

```python
from huggingface_hub import notebook_login

notebook_login()
```
