---
title: 原理
---
## webpack事件流
webpack的事件流是通过 ```Tapable``` 实现的，它就和我们的EventEmit一样，通过发布者-订阅者模式实现，是这一系列的事件的生成和管理工具，它的部分核心代码就像下面这样：
```js
class SyncHook{
  constructor(){
    this.hooks = [];
  }
  // 订阅事件
  tap(name, fn){
    this.hooks.push(fn);
  }
  // 发布
  call(){
    this.hooks.forEach(hook => hook(...arguments));
  }
}
```
在 webpack hook 上的所有钩子都是 Tapable 的实例，所以我们可以通过 tap 方法监听事件，使用 call 方法广播事件，就像官方文档介绍的这样：
```js
compiler.hooks.someHook.tap(/* ... */);
```