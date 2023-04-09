---
title: 数组特点,定义,方法介绍与使用
---

# Array 数组

## Array.prototype.flatMap()

[Array.prototype.flatMap()](https://developer.mozilla.org/zh-CN/docs/Web/JavaScript/Reference/Global_Objects/Array/flatMap)方法首先使用映射函数映射每个元素，然后将结果压缩成一个新数组。它与 map 连着深度值为 1 的 flat 几乎相同，但 flatMap 通常在合并成一种方法的效率稍微高一些。

```js
const arr1 = [1, 2, [3], [4, 5], 6, []];
const flattened = arr1.flatMap((num) => num);
console.log(flattened);
// [ 1, 2, 3, 4, 5, 6 ]
```

## Array.prototype.flat()

`flat()` 方法会按照一个可指定的深度递归遍历数组，并将所有元素与遍历到的子数组中的元素合并为一个新数组返回。

```ts
const arr1 = [0, 1, 2, [3, 4]];
console.log(arr1.flat());
// [0, 1, 2, 3, 4]

const arr2 = [0, 1, 2, [[[3, 4]]]];
// 深度为 2
console.log(arr2.flat(2));
// [0, 1, 2, Array [3, 4]]
```
