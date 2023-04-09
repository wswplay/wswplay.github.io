---
title: package.json生成,设置,各字段介绍与作用
---

# package.json 字段介绍与作用

## main / browser

`main`字段用来指定包的入口文件。如果不设置,则默认使用包的根目录中的`index.js`文件作为入口。  
`main`字段可以在`browser`环境和`Nodejs`环境都可以使用，如果使用`browser`字段，则表明只能在浏览器环境下使，不能用于服务端。

```json
{
  "main": "index.js",
  "module": "dist/vue.runtime.esm-bundler.js"
}
```

## module

指定 ES6 模块化(esm)引入的入口文件。如`import xxx from xxx`；  
**模块化主流标准**：CommonJS、AMD、CMD、UMD、ES6 【[模块化有哪些规范](/basic/javascript/#模块化及规范)】

| 字段    | 规范 |
| ------- | :--: |
| main    | cjs  |
| module  | esm  |
| browser | umd  |

::: danger
不要使用 browser 字段，永远使用 module 字段支持 Tree-shaking。如必须支持 umd ，可以添加至 umd:main 字段（[Specifying builds in package.json](https://github.com/developit/microbundle#specifying-builds-in-packagejson)）
:::

## type

用于定义 package.json 文件和该文件所在目录根目录中 `.js` 文件和无扩展名文件的模块化处理规范，默认值为`'commonjs'`。

> 在早期的 nodejs 版本中，node 仅支持 Commonjs 模块化方案，不过在 nodejs 版本 13.2.0 中，node 正式支持 ES Modules 模块化，在 package.json 中的 type 字段声明。

```json
{
  "type": "module"
}
```

:::tip

- type 字段省略，则默认采用 commonjs 规范，当指定值为 module 时采用 ESModule 规范
- node 官方建议包的开发者明确指定 package.json 中的 type 字段值
- 不论 package.json 中的 type 字段为何值，.mjs 的文件都按照 es 模块来处理，.cjs 的文件都按照 commonjs 模块来处理  
  :::

## types / typings

指定类型文件位置。`js`项目配置用`types`，`ts`项目用`typings`。

```json
{
  "typings": "dist/types/index.d.ts"
}
```

## files

`files`字段是一个数组，指定哪些文件需要发布到`registry`。如果指定的是文件夹，则整个文件夹都会被提交。如下`Vue`的`files`字段：

```json
{
  "files": [
    "index.js",
    "index.mjs",
    "dist",
    "compiler-sfc",
    "server-renderer",
    "macros.d.ts",
    "macros-global.d.ts",
    "ref-macros.d.ts"
  ]
}
```

## bin

脚本配置。一些包，可以作为命令行工具使用。通过`npm install package_name -g`命令可以将脚本添加到执行路径中，之后可以在命令行中直接执行。

## exports

配置不同环境对应的模块入口文件，优先级最高。`exports`存在时，`main`字段会失效。如下为`Vue`的`exports`字段：

```json
{
  "exports": {
    ".": {
      "import": {
        "node": "./index.mjs",
        "default": "./dist/vue.runtime.esm-bundler.js"
      },
      "require": "./index.js",
      "types": "./dist/vue.d.ts"
    },
    "./server-renderer": {
      "import": "./server-renderer/index.mjs",
      "require": "./server-renderer/index.js",
      "types": "./server-renderer/index.d.ts"
    },
    "./compiler-sfc": {
      "import": "./compiler-sfc/index.mjs",
      "require": "./compiler-sfc/index.js",
      "types": "./compiler-sfc/index.d.ts"
    },
    "./dist/*": "./dist/*",
    "./package.json": "./package.json",
    "./macros": "./macros.d.ts",
    "./macros-global": "./macros-global.d.ts",
    "./ref-macros": "./ref-macros.d.ts"
  }
}
```

## repository

用于指定代码所在的位置，通常使用`Github`库地址。

## packageManager

定义处理当前项目时预期使用的包管理器，该领域目前处于`试验阶段`。  
它可以设置为任何[支持的包管理器](http://nodejs.cn/api-v18/corepack.html#supported-package-managers)，并确保您的团队使用完全相同的包管理器版本，而无需安装 Node.js 以外的任何其他东西。

```json
{
  "packageManager": "pnpm@7.27.0"
}
```
