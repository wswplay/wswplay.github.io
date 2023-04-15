---
title: 冷知识
---

# 工具函数-冷知识-酷代码

Vue 工具函数、特殊用法、鲜为人知的用法、hacker 用法、魔改用法。

## Vue 工具函数集锦

```ts
const extend = Object.assign;

const hasOwnProperty = Object.prototype.hasOwnProperty;
const hasOwn = (val: object, key: string | symbol): key is keyof typeof val =>
  hasOwnProperty.call(val, key);
const hasChanged = (value: any, oldValue: any): boolean =>
  !Object.is(value, oldValue);

const isArray = Array.isArray;
const isMap = (val: unknown): val is Map<any, any> =>
  toTypeString(val) === "[object Map]";
const isSet = (val: unknown): val is Set<any> =>
  toTypeString(val) === "[object Set]";
const isDate = (val: unknown): val is Date =>
  toTypeString(val) === "[object Date]";
const isRegExp = (val: unknown): val is RegExp =>
  toTypeString(val) === "[object RegExp]";
const isFunction = (val: unknown): val is Function => typeof val === "function";
const isString = (val: unknown): val is string => typeof val === "string";
const isSymbol = (val: unknown): val is symbol => typeof val === "symbol";
const isObject = (val: unknown): val is Record<any, any> =>
  val !== null && typeof val === "object";

const isPromise = <T = any>(val: unknown): val is Promise<T> => {
  return isObject(val) && isFunction(val.then) && isFunction(val.catch);
};
const isPlainObject = (val: unknown): val is object =>
  toTypeString(val) === "[object Object]";

const objectToString = Object.prototype.toString;
const toTypeString = (value: unknown): string => objectToString.call(value);

const looseToNumber = (val: any): any => {
  const n = parseFloat(val);
  return isNaN(n) ? val : n;
};
const toNumber = (val: any): any => {
  const n = isString(val) ? Number(val) : NaN;
  return isNaN(n) ? val : n;
};
```

## 是否相同 `Object.is()`

`Object.is()` 方法判断两个值是否为同一个值。

`Object.is()` 与 == 不同。== 运算符在判断相等前对两边的变量（如果它们不是同一类型）进行强制转换（这种行为将 "" == false 判断为 true），而 `Object.is` 不会强制转换两边的值。

`Object.is()` 与 === 也不相同。差别是它们对待有符号的零和 NaN 不同，例如，=== 运算符（也包括 == 运算符）将数字 -0 和 +0 视为相等，而将 Number.NaN 与 NaN 视为不相等。

```ts
Object.is(0, -0); // false
Object.is(+0, -0); // false
```
