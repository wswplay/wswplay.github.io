---
title: Markdown介绍语法方法使用
---

# Markdown：开启你的记录

## 新窗口打开内页

Markdown 默认：外链-新窗口打开，内页-直接跳转。

```html
<a href="/zh/guide/basic/operators.html" target="_blank" rel="noreferrer">基础知识-位运算</a>
```

## 目录结构生成

看上去是一个小问题，可能是问题太小了，居然没有找到特别大众而且牛逼的包。  
试了几个类似的包，觉得`tree-node-cli`功能简约且够用，那就先用着吧。[Github](https://github.com/yangshun/tree-node-cli)

```bash
# 全局安装
npm install -g tree-node-cli
# 进入到目标项目，运行如下命令。
# 释：-L 2 最深层级为2；-I node_modules 忽略node_modules文件夹
tree -L 2 -I node_modules > cate.md
# cate.md
read-doc-next
├── README.md
├── cate.md
├── env.d.ts
├── index.html
├── package.json
├── pnpm-lock.yaml
├── public
│   └── favicon.ico
├── src
│   ├── App copy.vue
│   ├── App.vue
│   ├── assets
│   ├── components
│   ├── main.ts
│   ├── pages
│   ├── plugins
│   ├── router
│   ├── stores
│   ├── types
│   └── utils
├── tsconfig.config.json
├── tsconfig.json
└── vite.config.ts
```
