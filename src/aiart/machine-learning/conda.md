---
title: conda、anaconda
---

# Anaconda-数据科学工具包

Anaconda：Unleash AI innovation and value。(释放人工智能的创新和价值)

## 下载与安装

[官网下载地址](https://www.anaconda.com/download)

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

## 常遇问题与方案

- **notebook 忽然打不开**，浏览器无法显示内容，好像是 js 错误。  
  **方案**：重新下载包，重新安装。get！(conda update --all 跟新所有之后就不兼容，打不开了。只能重新安装)

- **navigator** 这个傻逼玩意儿**自动下载更新**后，**老是崩**，启动不了。只能重新、彻底安装导航。

```bash
conda remove anaconda-navigator anaconda-client anaconda-auth navigator-updater
conda clean --all
conda install anaconda-navigator
# 完成后。命令行启动，大概就可以了
anaconda-navigator
```

- **误删库包**。有时候，你一怒之删了 base 环境某些东西，于是 navigator 就挂了，甚至 conda 命令都失效了。只能定向安装和卸载、重新安装。

```bash
# 定向安装：conda命令报错，缺少文件库
# ModuleNotFoundError: No module named 'psutil'
/opt/anaconda3/bin/python -m pip install psutil

# 修复后，conda命令可用，但 navigator 或 快捷方式无法启动
# 完全卸载
conda remove anaconda-navigator -y
/opt/anaconda3/bin/pip uninstall anaconda-navigator -y

# 清理
conda clean -a -y

# 重新安装
conda install anaconda-navigator -c anaconda -y

# 验证
which anaconda-navigator
anaconda-navigator --version
```
