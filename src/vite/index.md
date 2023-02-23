---
title: Vite基本使用
---

# Vite：为开发提供极速响应

[Vite](https://github.com/vitejs/vite)，是一种新型前端构建工具，主要由两部分组成：    
1、**一个开发服务器**。基于 `原生 ES 模块` ，提供模块热更新（HMR）。  
2、**一套构建指令**。基于 [Rollup.js](https://rollupjs.org/) ，可输出用于生产环境的高度优化过的静态资源。

## 用 Vite 快速构建 Vue 项目

```bash
# npm 6.x
npm init vite@latest <project-name> --template vue

# npm 7+，需要加上额外的双短横线
npm init vite@latest <project-name> -- --template vue

cd <project-name>
npm install
npm run dev
```

`Vite`强缓存依赖包，所以本地调试`依赖源码`(如 Vue)，成为一个棘手的问题。[Vite 缓存](https://cn.vitejs.dev/guide/dep-pre-bundling.html#file-system-cache)

::: tip 这样根本没有解决问题啊？！！
1、通过浏览器调试工具的 Network 选项卡暂时禁用缓存；  
2、重启 Vite dev server，并添加 --force 命令以重新构建依赖；  
3、重新载入页面。
:::

## import.meta.url

`import.meta.url`在模块内部使用，返回当前模块的路径。

```js
// xxx.js
console.log(import.meta.url); // src/xxx/xxx.js
```

如果模块里面还有一个数据文件`data.txt`，那么就可以用下面的代码，获取这个数据文件的路径。

```js
new URL("data.txt", import.meta.url);
```
