---
title: funciton 函数类型及扩展
---

# 函数类型及扩展

## Function.prototype.apply()

`apply()` 方法调用一个具有给定 `this` 值的函数，以及以一个数组（或一个类数组对象）的形式提供的参数。

```ts
const numbers = [5, 6, 2, 3, 7];
const max = Math.max.apply(null, numbers);
console.log(max);
// 7
```

## Function.prototype.bind()

bind() 方法创建一个新的函数，在 bind() 被调用时，这个新函数的 this 被指定为 bind() 的第一个参数，而其余参数将作为新函数的参数，供调用时使用。

```ts
const moduleObj = {
  x: 42,
  getX: function () {
    return this.x;
  },
};
const unboundGetX = moduleObj.getX;
console.log(unboundGetX()); // The function gets invoked at the global scope
// Expected output: undefined
const boundGetX = unboundGetX.bind(moduleObj);
console.log(boundGetX());
// Expected output: 42
```
