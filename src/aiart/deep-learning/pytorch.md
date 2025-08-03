---
title: Pytorch
outline: deep
---

# Pytorch

Pytorch is an optimized tensor library for **deep learning** using GPUs and CPUs.

Summary: Tensors and Dynamic neural networks in Python with strong GPU acceleration.

## 计算图与自动求导

计算图（`Computational Graphs`）是一种描述运算的「语言」，它由`节点(Node)`和`边(Edge)`构成。记录所有节点和边的信息，可以方便地完成**自动求导**。

- **节点**：表示数据和计算操作。
- **边**：表示数据流向。

![An image](./img/compt-graph.png)

## VSCode Debugger 调试源码

- 1、在目标代码处，设置断点 breakpoint。
- 2、点击 VSCode 左边栏的「甲壳虫+播放」按钮，并创建调试配置文件(create a launch.json)。
- 3、**添加、添加、添加一行 `"justMyCode": false`**。

`VSCode`个坑爹的玩意儿，**默认没有`justMyCode`这个`key`，那就是 `True`**。老子调了好久都进不去 `torch` 源码，靠，满头大汗。微软是个大傻逼。

```json {13}
{
  // Use IntelliSense to learn about possible attributes.
  // Hover to view descriptions of existing attributes.
  // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python Debugger: Current File",
      "type": "debugpy",
      "request": "launch",
      "program": "${file}",
      "console": "integratedTerminal",
      "justMyCode": false
    }
  ]
}
```

## 方法简介

### unsqueeze

PyTorch 中用于在张量指定位置**添加一个新维度**的方法。

- unsqueeze：非原地操作，返回新张量，不修改原张量。
- unsqueeze\_：原地操作，修改原张量本身。
