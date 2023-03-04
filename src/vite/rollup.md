---
title: Rollup.js介绍使用方法教程
---

# Rollup Give U More Free & Feel！

- [Rollup](https://github.com/rollup/rollup) 是一个 `JavaScript` 模块打包工具，使用 `ES6模块标准`。让你更自由，想用就用。
- `Rollup` 还可以对代码进行`静态分析`，使你最终的代码没有冗余。

## 命令安装及使用

```bash
npm install -g rollup
```

命令行打包示例：

```bash
# 用于浏览器
rollup main.js --file bundle.js --format iife
# 用于Node.js compile to a CommonJS module ('cjs')
rollup main.js --file bundle.js --format cjs
# 同时用于浏览器和Node.js UMD format requires a bundle name
rollup main.js --file bundle.js --format umd --name "myBundle"
```

--config 或 -c 指向使用配置文件：

```bash
rollup --config xxx.config.js
# Debug模式
rollup --config --configDebug
```

为了实现互通性，Rollup 也支持从安装在 node_modules 目录下的包中加载配置文件。从 Node 包中加载配置：

```bash
# Rollup 首先会尝试加载 "rollup-config-my-special-config";
# 如果失败，Rollup 则会尝试加载 "my-special-config"
rollup --config node:my-special-config
```

## 配置文件

### 1、默认类型

配置文件是一个 ES 模块，对外导出一个对象。位于项目根目录，命名为 `rollup.config.js` 或 `rollup.config.mjs`。

```js
// rollup.config.js
export default {
  input: "src/main.js",
  output: {
    file: "bundle.js",
    format: "cjs",
  },
};
```

### 2、CommonJS

如果想用 `CommonJS` 的 `require` 和 `module.exports`，配置文件后缀应为 `.cjs`。

### 3、Typescript

如果配置文件后缀为 `.ts`，则应安装 `@rollup/plugin-typescript` 插件，并传入 `--configPlugin` 配置项。还需单独安装它的[依赖包](https://github.com/rollup/plugins/tree/master/packages/typescript#install)：`typescript` 和 `tslib`。  
[官网文档](https://rollupjs.org/command-line-interface/#configplugin-plugin)

```json
// package.json
{
  "scripts": {
    "build": "rollup -c rollup.config.ts --configPlugin typescript"
  },
  "devDependencies": {
    "@rollup/plugin-typescript": "^11.0.0",
    "tslib": "^2.5.0",
    "typescript": "^4.9.5"
  }
}
```

然后，配置文件中 `plugins` 字段需加入这个插件：

```ts
// my.config.ts
import typescript from "@rollup/plugin-typescript";
import { defineConfig } from "rollup";

export default defineConfig({
  input: "src/xiao.ts",
  output: {
    dir: "nanzhi",
  },
  plugins: [typescript()],
});
```
