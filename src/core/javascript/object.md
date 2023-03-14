---
title：Object 对象
---

# Object 对象

## Object.entries()

`Object.entries()` 方法返回一个给定对象自身可枚举属性的键值对数组，其排列与使用 `for...in` 循环遍历该对象时返回的顺序一致。

区别在于 `for-in` 循环还会枚举**原型链中属性**。

```ts
Object.entries([11, 66, 99]);
// [ [ '0', 11 ], [ '1', 66 ], [ '2', 99 ] ]
Object.entries({ id: 666, name: "三体" });
// [ [ 'id', 666 ], [ 'name', '三体' ] ]
```
