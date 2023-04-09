---
title: declare及声明文件
---

# Declare：有声明更智能，更高效省心

## 声明文件是什么？

### 有什么用？

声明文件，就是批量声明`变量类型`的文件。必须以`.d.ts`结尾。  
**作用**：就是给 js 代码补充类型标注。这样在 ts 编译环境下，就不会提示 js 文件"缺少类型"。

### 应该放在哪里？

网友断案说，任意路径/文件名，ts 编译器都可以识别。  
但为了避免后期一些可能的奇怪问题，推荐**放在根目录**下。

### @types/xxx

一般情况下，著名的 js 库，都已经有大佬在[npm 包库](https://www.npmjs.com/)的`@types`包下，写好了声明文件的。我们安装一下，拿来就用。比如 jQuery。安装完成后，可以在`node_modules/@types/jquery`看到声明文件。

```bash
npm i @types/jquery
```

如果`@types`包下没有相关的声明文件，那就得我们自己下手了。

## 怎么写全局声明文件？

### declare

`declare`声明`全局变量`类型。

```ts
// global.d.ts
declare var count: number;
declare let name: string;
declare const info: object;
declare function nanZhi(msg: string): number;
declare enum boxer {
  top,
  right,
  bottom,
  left,
}
```

### declare namespace

`namespace`后面的全局变量是一个自定义对象。

```ts
declare namespace bianCheng {
  var id: number;
  var city: string;
  var codeDay: (msg: string) => object;
}
```

### 修改已有的全局声明

安装`typescript`时, 会自带一些系统变量的声明文件, 在`node_modules/typescript/lib`下。  
例：为`node`下的`global` `String`添加属性声明。

```ts
declare global {
  interface String {
    nanzhi(input: string): string;
  }
}
```

## 怎么写模块声明文件？

### declare module 扩展模块声明

已有的库包已经存在声明文件，那怎么扩展声明？
如下为`Vue`添加`$fanyi`新属性的类型声明：

```ts
declare module "vue" {
  interface ComponentCustomProperties {
    $fanyi: (key: string) => string;
  }
}
```

### 对非 ts/js 文件模块进行类型扩充

`ts`只支持模块的导入导出。那`css/html`等文件怎么办呢？这时候就需要用通配符，让`ts`把他们当做模块。
如下为`Vue`官方对`.vue`文件的支持：

```ts
// global.d.ts
declare module "*.vue" {
  import { DefineComponent } from "vue";
  const component: DefineComponent<{}, {}, any>;
  export default component;
}
```

声明把`vue`文件当做模块, 同时标注模块的默认导出是`component`类型。这样在`vue`的`components`字段中注册模块才可以正确识别类型。

【[参考资料](https://juejin.cn/post/7008710181769084964)】
