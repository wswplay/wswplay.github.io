---
title: 介绍Proxy基本使用及Vue中使用
---

# Proxy 代理

`Proxy` 对象用于创建一个对象的代理，从而实现基本操作的拦截和自定义。  
如属性**转发、查找、赋值、枚举、函数调用**等。

[Vue3 中 Proxy 用法](/vue/vue3/reactive.html#reactive-函数)

## 转发

```ts
let target = {};
let p = new Proxy(target, {});

p.a = 37; // 操作转发到目标
console.log(target.a); // 37. 操作已经被正确地转发
```
