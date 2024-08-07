---
title: conda、anaconda
---

# Anaconda-数据科学工具包

Anaconda：Unleash AI innovation and value。(释放人工智能的创新和价值)

## 下载与安装

[官网下载地址](https://www.anaconda.com/download)

## 问题与方案

- **notebook 忽然打不开**，浏览器无法显示内容，好像是 js 错误。  
  **方案**：重新下载包，重新安装。get！(conda update --all 跟新所有之后就不兼容，打不开了。只能重新安装)

## 命令与开发

```bash
# 查看所有包版本
conda list
# 计算行数，得到包的总数量-1，包含了标题
conda list | wc -l
# 查看某个包版本
conda list xxx
```

### 包安装

有些工具包名字，在 `anaconda` 导航搜索不到。例如 `eli5`。那就如下安装：

```bash
conda install -c conda-forge eli5
# 或pip安装
pip install eli5
```

### 本地训练模型相关操作

```bash
# 列出所有虚拟环境
conda env list

# 创建和激活虚拟环境
conda create -n lora-env python=3.9
conda activate lora-env

# 包更新
pip list --outdated
pip install --upgrade package_name
```
