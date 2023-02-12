---
title: npm
---

## 发布 npm 包

```bash
# 查看源地址
npm get registry
# 如不是npm官方源
# 则切换到官方源地址
npm config set registry https://registry.npmjs.org
# 查看是否已登录npm，如已登录则会显示用户名
npm who am i
# 如果没有登录，先登录，然后输入用户名、密码
npm login
# 发布
npm publish
# 如果包名称带有scope(如@wswplay)，则可能会发布失败
# You must sign up for private packages 意思是必须付费
# 但其实并不必须，发布为公共访问就行
npm publish --access public
# 发布成功。登录https://www.npmjs.com/ 个人中心packages就可以看到
```

## 创建`package.json`文件

```bash
# 交互式创建
npm init
# 快速创建默认文件
npm init -y
```

## nrm 管理 npm 源地址

```bash
# 全局安装nrm
npm install -g nrm
# 查看可选的源地址
nrm ls
# npm ---------- https://registry.npmjs.org/
# yarn --------- https://registry.yarnpkg.com/
# tencent ------ https://mirrors.cloud.tencent.com/npm/
# cnpm --------- https://r.cnpmjs.org/
# taobao ------- https://registry.npmmirror.com/
# npmMirror ---- https://skimdb.npmjs.com/registry/

# 切换到淘宝源
nrm use taobao
# 测试速度
nrm test npm
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
