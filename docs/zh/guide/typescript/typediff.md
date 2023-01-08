---
title: 基本认知
---

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
