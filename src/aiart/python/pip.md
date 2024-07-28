---
title: pip命令合集
outline: deep
---

# pip

**pip** is the official tool for installing and using Python packages from various indexes.

## 命令与解释

### pip intall .

请检查项目的根目录中是否有 `pyproject.toml`、`requirements.txt`、`Pipfile` 或其他类似的文件，并按照相应的工具来安装依赖。如果这些文件都没有，建议你再仔细阅读项目的文档，看看是否有其他的安装指引或要求。

```bash
pip install .
```

- 1、项目使用 `pyproject.toml`：新的 Python 项目有时会使用 `pyproject.toml` 来替代 `setup.py`。这个文件通常会包含项目的配置信息。
- 2、项目使用 `requirements.txt`：有些项目直接在根目录下包含一个 requirements.txt 文件，该文件列出了所有依赖项。
- 3、项目使用 `Pipfile`：有些项目使用 Pipenv 来管理依赖项，并且包含一个 Pipfile。
- 4、项目使用`其他构建`工具：有些项目可能使用其他的构建工具或依赖管理系统，例如 poetry。这时，你需要根据项目文档中的指引来安装依赖。
