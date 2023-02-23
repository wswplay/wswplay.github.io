---
title: Vite是怎么搭建生成一个项目的
---

# Vite 是怎么搭建生成一个项目的

Vite 官方文档中，[搭建第一个 Vite 项目](https://cn.vitejs.dev/guide/#scaffolding-your-first-vite-project)，运行如下命令能创建一个项目，什么原理？

```bash
npm create vite@latest my-vue-app -- --template vue
```

## `npm create x` 等于 `npx create-x`

【[参考 npm 文档](http://nodejs.cn/npm/cli/v8/commands/npm-init/#forwarding-additional-options)】  
【[npx 即 npm exec](/node/npx.html)】

```bash
npm init <package-spec> (same as `npx <package-spec>)
npm init <@scope> (same as `npx <@scope>/create`)
# aliases: create, innit

# 举例子。注意后面是 npx(npm exec)
npm init foo -> npx create-foo
npm init @usr -> npx @usr/create
npm init @usr/foo -> npx @usr/create-foo
# 附加选项都将直接传递给命令
npm init foo -- --hello -> npx -- create-foo --hello
```

因此，`npm create vite@latest my-vue-app -- --template vue` 实际上等于：

```bash
npx create-vite@latest my-vue-app -- --template vue
```

**说人话就是**：临时安装 [create-vite](https://github.com/vitejs/vite/tree/main/packages/create-vite) 包，并执行 `create-vite` 命令，用后即删。

## create-vite命令源码分析

