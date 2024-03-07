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
