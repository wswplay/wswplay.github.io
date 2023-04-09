---
title: Symbol特点,定义,方法介绍与使用
---

# Symbol 标识符

## Symbol.for()

`Symbol.for(key)` 根据给定 `key`，在运行时 `symbol` 注册表中寻找对应 `symbol`。  
如找到则返回，否则新建一个与该键关联 `symbol`，并放入`全局 symbol` 注册表中。

```ts
Symbol.for("foo"); // 创建一个 symbol 并放入 symbol 注册表中，键为 "foo"
Symbol.for("foo"); // 从 symbol 注册表中读取键为"foo"的 symbol

Symbol.for("bar") === Symbol.for("bar"); // true
Symbol("bar") === Symbol("bar"); // false：Symbol()函数每次都会返回一个新的symbol

var sym = Symbol.for("biancheng");
sym.toString();
// "Symbol(biancheng)"，biancheng 既是该 symbol 在 symbol 注册表中的键名，又是该 symbol 自身的描述字符串
```

为防止冲突，最好将放入 `symbol注册表` 中的 `symbol `带上键前缀。

```ts
Symbol.for("mdn.foo");
Symbol.for("mdn.bar");
```
