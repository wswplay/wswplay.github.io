---
title: 工具类
---
## 判断两个值是否有改变(是否相等)
```js
const hasChanged = (value, oldValue) => value !== oldValue && (value === value || oldValue === oldValue);
```
## void 0判断是否undefined
如下，void 0 和 undefined是相等的。 为什么要这么用？
1. 为了防止undefined被改写，造成判断失效。
2. void 0代替undefined省3个字节。
3. 一看就是老司机的手法。装。。。
```js
if (key !== void 0) {}
// void 0 === undefined // true
```