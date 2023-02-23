---
title: 怎么写一个Vite插件
---

# 怎么写一个 Vite 插件

【[参考：Vite 官网](https://cn.vitejs.dev/guide/api-plugin.html)】  
Vite 插件扩展了设计出色的 Rollup 接口，带有一些 Vite 独有的配置项。  
**插件的本质就是**：合适的时间，调用合适的钩子，改点东西(增删改)。

## 举个例子

```js
// vite-plugin-demo.ts
import { Plugin } from "vite";

export default function viteDemo(): Plugin {
  return {
    name: "vite:demo",
    enforce: "pre",
    config(config, envConfig) {
      console.log("vite-demo-config: ", config, envConfig);
    },
    configResolved(refConfig) {
      console.log("vite-demo-resolved: ", refConfig);
    },
  };
}
// 用法 vite.config.ts
import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";
import viteDemoPlugin from "./src/plugins/vite-plugin-demo";

export default defineConfig({
  plugins: [vue(), viteDemoPlugin()],
  server: {
    port: 3062,
  },
});
```

## 命名约定

::: warning
对于 Vite 专属插件：

- Vite 插件应该有一个带 vite-plugin- 前缀、语义清晰的名称。
- 在 package.json 中包含 vite-plugin 关键字。
- 在插件文档增加一部分关于为什么本插件是一个 Vite 专属插件的详细说明（如，本插件使用了 Vite 特有的插件钩子）。

如只适用于特定框架，应该：

- `vite-plugin-vue-` 前缀作为 Vue 插件
- `vite-plugin-react-` 前缀作为 React 插件
- `vite-plugin-svelte-` 前缀作为 Svelte 插件
  :::

## 那些钩子

### Vite 独有钩子

::: tip

- `config` 在解析 Vite 配置前调用。
- `configResolved` 在解析 Vite 配置后调用。
- `configureServer` 用于配置开发服务器。
- `transformIndexHtml` 转换 index.html 的专用钩子。
- `handleHotUpdate` 执行自定义 HMR 更新处理。
  :::

### 源于 Rollup 的通用钩子

::: tip
服务器启动时被调用：

- options
- buildStart

在每个传入模块请求时被调用：

- resolveId
- load
- transform

服务器关闭时被调用：

- buildEnd
- closeBundle
  :::
