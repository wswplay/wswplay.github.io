---
title: exports关键字
---
## module.exports和exports
exports和module.exports，其实是相等的。
**exports是module.exports的引用**。

::: tip
exports只能使用语法来向外暴露内部变量：如 exports.xxx = xxx;  
module.exports既可以通过语法，也可以直接赋值一个对象。
:::
```js
// ok
module.exports = {
  add,
}
exports.add = add // ok

// TypeError: add is not a function
exports = {
  add,
}
```