---
title: 表达式与运算符
---

# Expressions and operators

## 可选链运算符（?.）

- 可选链运算符 `?.` 允许读取位于连接对象链深处的属性的值，而不必明确验证链中的每个引用是否有效。   
- `?.` 运算符的功能类似于 `.` 链式运算符，不同之处在于，在引用为空 `(nullish` ) (`null 或者 undefined`) 的情况下不会引起错误，该表达式短路返回值是 `undefined`。   
- 与函数调用一起使用时，如果给定的函数不存在，则返回 `undefined`。

当尝试访问可能不存在的对象属性时，可选链运算符将会使表达式更短、更简明。在探索一个对象的内容时，如果不能确定哪些属性必定存在，可选链运算符也是很有帮助的。

```ts
const adventurer = {
  name: "Alice",
  cat: {
    name: "Dinah",
  },
};

const dogName = adventurer.dog?.name;
console.log(dogName);
// Expected output: undefined

console.log(adventurer.someNonExistentMethod?.());
// Expected output: undefined
```

## 空值合并运算符（??）

- 空值合并运算符 `??` 是一个逻辑运算符，当左侧的操作数为 `null` 或者 `undefined` 时，返回其右侧操作数，否则返回左侧操作数。   
- 与逻辑或运算符 `||` 不同，逻辑或运算符会在左侧操作数为假值时返回右侧操作数。也就是说，如果使用 `||` 来为某些变量设置默认值，可能会遇到意料之外的行为。比如为假值（例如，`'' 或 0`）时。见下面的例子。

```ts
const foo = null ?? "default string";
console.log(foo);
// Expected output: "default string"

const baz = 0 ?? 42;
console.log(baz);
// Expected output: 0
```
