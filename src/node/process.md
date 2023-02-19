---
title: Nodejs process模块及方法介绍与使用
---

# Process：进程

process 模块用来与当前进程互动，可以通过全局变量 process 访问，不必使用 require 命令加载。它是一个 EventEmitter 对象的实例。

## process.cwd()

`Current Work Directory` 的缩写。返回运行当前脚本的工作目录的路径。
