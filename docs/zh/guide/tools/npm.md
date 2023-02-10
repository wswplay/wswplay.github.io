---
title: npm
---

## 创建`package.json`文件

```bash
# 交互式创建
npm init
# 快速创建默认文件
npm init -y
```

## 设置源地址

> 有些项目，有些时候，我们一定要切到官方的源上，才能安装。如果不，会提示一些莫名其妙的错误。

```bash
# 查看源地址
npm get registry
# 将源地址改成taobao源
npm config set registry https://registry.npm.taobao.org
# 恢复成官方源地址
npm config set registry https://registry.npmjs.org
```

## 安装指定版本

```bash
# 安装Vue3.0及以上版本
npm i vue@^3.0.0 -S
# 安装最新版本
npm i xxx@latest -D
```
