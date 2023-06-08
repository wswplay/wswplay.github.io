---
title: Web Worker简介与用法
---

# Web Worker

`Web Worker`为`Web内容`在`后台线程`中运行脚本提供了一种简单的方法。  
线程可以执行任务(`如I/O`)，而`不干扰用户界面`。

一旦创建，一个 worker 可以将消息发送到`创建它的JavaScript`代码，通过将消息发布到该代码指定的事件处理器(反之亦然)。

## Web Worker API
