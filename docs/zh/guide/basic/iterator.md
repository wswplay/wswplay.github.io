---
title: 迭代器
---

可迭代协议允许 JavaScript 对象定义或定制它们的迭代行为。要成为可迭代对象，一个对象必须实现 `@@iterator` 方法。这意味着对象（或者它原型链上的某个对象）必须有一个键为 `@@iterator` 的属性，可通过常量 `Symbol.iterator` 访问该属性。[MDN](https://developer.mozilla.org/zh-CN/docs/Web/JavaScript/Reference/Iteration_protocols)

通常情况下，对象是不可迭代的。但如果一个对象实现了`Symbol.iterator`方法，那就可迭代。如下：

```js
const bianCheng = {
  val: 0,
  length: 6,
  [Symbol.iterator]() {
    return {
      next() {
        return {
          value: bianCheng.val++,
          done: bianCheng.val > bianCheng.length,
        };
      },
    };
  },
};
for (const item of bianCheng) {
  console.log(item); // 0,1,2,3,4,5
}
```

数组之所以可以迭代，是因为内建了`Symbol.iterator`方法。

```js
const bianList = [11, 33, 66];
const bIter = bianList[Symbol.iterator]();
console.log(bIter.next()); // { value: 11, done: false }
console.log(bIter.next()); // { value: 33, done: false }
console.log(bIter.next()); // { value: 66, done: false }
console.log(bIter.next()); // { value: undefined, done: true }
```
