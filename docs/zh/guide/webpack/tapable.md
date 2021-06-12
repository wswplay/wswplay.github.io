---
title: 插件系统Tapable
---
## tap：注册钩子
```js
compiler.hooks.compilation.tap(pluginName, callback)
```

## call：使用钩子
```js
compiler.hooks.compilation.call(pluginName, callback)
```

## 用法
```js
const { SyncHook } = require('tapable');
const xiao = new SyncHook(['name', 'time']);
// 注册
xiao.tap('one', (name, time) => {
  console.log('one', name, time) // one wowo 2021-5-30
})
xiao.tap('two', (...arg) => {
  console.log('two', arg) // two [ 'wowo', '2021-5-30' ]
})
xiao.tap('three', (...arg) => {
  console.log('three', arg[1]) // three 2021-5-30
})
// 使用
xiao.call('wowo', '2021-5-30')
```

## 源码
tapable包暴露出很多钩子类，这些类可以用来为插件创建钩子函数。    
钩子类的构造函数都接收一个可选的参数，这个参数是一个由字符串参数组成的数组。(见用法)
```js
// tapable/index.js
exports.__esModule = true;
exports.Tapable = require("./Tapable");
exports.SyncHook = require("./SyncHook");
exports.SyncBailHook = require("./SyncBailHook");
exports.SyncWaterfallHook = require("./SyncWaterfallHook");
exports.SyncLoopHook = require("./SyncLoopHook");
exports.AsyncParallelHook = require("./AsyncParallelHook");
exports.AsyncParallelBailHook = require("./AsyncParallelBailHook");
exports.AsyncSeriesHook = require("./AsyncSeriesHook");
exports.AsyncSeriesBailHook = require("./AsyncSeriesBailHook");
exports.AsyncSeriesWaterfallHook = require("./AsyncSeriesWaterfallHook");
exports.HookMap = require("./HookMap");
exports.MultiHook = require("./MultiHook");
```
**钩子分为同步和异步，异步又分为并发执行和串行执行**。