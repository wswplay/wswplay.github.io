---
title: funciton 函数类型及扩展
---

# 函数类型及扩展

## Function() 构造函数

Function() 构造函数创建了一个新的 Function 对象。直接调用构造函数可以动态创建函数，但可能会经受一些安全和类似于 eval()（但远不重要）的性能问题。然而，不像 eval（可能访问到本地作用域），Function 构造函数只创建全局执行的函数。

```ts
const sum = new Function("a", "b", "return a + b");
console.log(sum(2, 6));
// Expected output: 8
```

调用 Function() 时，可以使用或不使用 new。两者都会创建一个新的 Function 实例。

```ts
new Function(functionBody);
new Function(arg0, functionBody);
new Function(arg0, arg1, functionBody);
new Function(arg0, arg1, /* … ,*/ argN, functionBody);

Function(functionBody);
Function(arg0, functionBody);
Function(arg0, arg1, functionBody);
Function(arg0, arg1, /* … ,*/ argN, functionBody);
```

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
