---
title: 基础概念
---

## 代理：Proxy

Proxy 对象用于创建一个对象的代理，从而实现基本操作的拦截和自定义（如属性查找、赋值、枚举、函数调用等）。[MDN 定义](https://developer.mozilla.org/zh-CN/docs/Web/JavaScript/Reference/Global_Objects/Proxy)  
:::tip
`Proxy` 是一个对象，它包装了另一个对象，并允许你拦截对该对象的任何交互。

:::
## 拦截器：Reflect
Reflect 是一个内置的对象，它提供拦截 JavaScript 操作的方法。这些方法与Proxy handlers的方法相同。Reflect的所有属性和方法都是静态的（就像Math对象）。[MDN](https://developer.mozilla.org/zh-CN/docs/Web/JavaScript/Reference/Global_Objects/Reflect)

```js {8,11,15}
const nanZhi = {
  id: "边城",
  address: "深圳",
};

const handler = {
  get(target, property) {
    return Reflect.get(...arguments);
  },
  set(target, property, value) {
    return Reflect.set(...arguments);
  }
};

const bianCheng = new Proxy(nanZhi, handler);

console.log(bianCheng.id); // 边城
console.log(bianCheng.nid); // undefined
bianCheng.id = "沈从文";
bianCheng.nid = "看过许多地方的云";
console.log(bianCheng.id); // 沈从文
console.log(bianCheng.nid); // 看过许多地方的云
```
