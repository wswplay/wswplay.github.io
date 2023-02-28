---
title: 基本认知
---

# 看清区别，提高认知

## interface 和 type 区别

`interface`：接口。  
`type`：类型别名。

1、`type`可以用于其它类型(联合类型、交叉类型、元组类型、基本类型(原始值))。  
 `interface`不支持。

```ts
type PointX = { x: number };
type PointY = { y: number };

// union联合类型
export type PiontInfo = PointX | PointY;

// intersection交叉类型
export type BigPoint = PiontInfo & { area: object; size: number };

// tuple元祖
export type PointData = [PointX, PointY];

// primitive原始值
export type Nanzhi = String;

// typeof的返回值
const div = document.createElement("div");
export type ElType = typeof div;
```

2、`type`能使用`in`关键字生成映射类型。`interface`不支持。

```ts
type Keys = "firstName" | "lastName" | "fullName";
type Jude = {
  [key in Keys]: string;
};
export const wowo: Jude = {
  firstName: "fist",
  lastName: "last",
  fullName: "full",
};
```

3、`interface`可以多次定义，并被视为合并所有声明成员。`type`不支持。

```ts
interface Point {
  x: number;
}
interface Point {
  y: number;
}
export const point: Point = { x: 123, y: 456 };
```

4、默认导出方式不同

```ts
// interface支持声明同时default默认导出
export default interface Config {
  feng: string;
  ling: object;
}
// type只能先声明，再default默认导出
type Setting = {
  option: object;
  cb: (msg: string) => object;
};
export default Setting;
```

## extends 与 implements 区别

### extends 继承

一个新的接口或者类，从父类或者接口继承所有的属性和方法。  
不可以重写属性，但可以重写方法。

> 类只能继承类，不能继承接口。接口可以继承接口或类。可多继承或多实现。

### implements 实现

一个新的类，从父类或者接口实现所有的属性和方法。  
同时可以重写属性和方法，包含一些新的功能。

> 只能用于**类**。类可以实现接口或类，接口不能实现接口或者类。可多继承或多实现。

### class 类

对类：即可实现，也可继承。  
对接口：只能实现，不能继承。

### interface 接口

对类：只能继承， 不能实现。  
对接口：只能继承， 不能实现。

## Pick 和 Omit 区别

### Pick

取**子集**，返回新类型。从一个已知的类型中，取出子集，作为一个新的类型返回。

### Omit
**剔除**属性，返回新类型。以一个类型为基础支持剔除某些属性，然后返回一个新类型。
