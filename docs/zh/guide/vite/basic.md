---
title: 基础知识
---

## import.meta.url

`import.meta.url`在模块内部使用，返回当前模块的路径。
```js
// xxx.js
console.log(import.meta.url) // src/xxx/xxx.js
```
如果模块里面还有一个数据文件`data.txt`，那么就可以用下面的代码，获取这个数据文件的路径。

```js
new URL("data.txt", import.meta.url);
```
