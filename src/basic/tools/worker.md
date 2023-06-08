---
title: Web Worker简介与用法
---

# Web Worker

`Web Worker`为`Web内容`在`后台线程`中运行脚本提供了一种简单的方法。  
线程可以执行任务(`如I/O`)，而`不干扰用户界面`。

一旦创建，一个 worker 可以将消息发送到`创建它的JavaScript`代码，通过将消息发布到该代码指定的事件处理器(反之亦然)。

[参考 MDN](https://developer.mozilla.org/zh-CN/docs/Web/API/Web_Workers_API/Using_web_workers)

## Web Worker API

workers 和`主线程`之间，都使用 `postMessage()` 方法发送各自消息，使用 `onmessage` 事件响应消息（消息被包含在 message 事件的 `data` 属性中）。

数据并不是被共享，而是`被复制`。

:::tip
在主线程中使用时，onmessage 和 postMessage() 必须挂在 worker 对象上。  
而在 worker 中使用时，不用这样做。  
原因是，在 worker 内部，worker 是有效的全局作用域。
:::

```ts
// 主线程 main.js
const first = document.querySelector("#number1");
const second = document.querySelector("#number2");

const result = document.querySelector(".result");

if (window.Worker) {
  const myWorker = new Worker("worker.js");

  first.onchange = function () {
    myWorker.postMessage([first.value, second.value]);
    console.log("Message posted to worker");
  };

  second.onchange = function () {
    myWorker.postMessage([first.value, second.value]);
    console.log("Message posted to worker");
  };

  myWorker.onmessage = function (e) {
    result.textContent = e.data;
    console.log("Message received from worker");
  };
} else {
  console.log("Your browser doesn't support web workers.");
}
// worker.js
onmessage = function (e) {
  console.log("Worker: Message received from main script");
  const result = e.data[0] * e.data[1];
  if (isNaN(result)) {
    postMessage("Please write two numbers");
  } else {
    const workerResult = "Result: " + result;
    console.log("Worker: Posting message back to main script");
    postMessage(workerResult);
  }
};
```
