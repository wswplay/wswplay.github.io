---
title: nvm:Nodejs版本管理工具介绍与使用
---

# Node.js Version Management

## Mac 安装 nvm

[nvm](https://github.com/nvm-sh/nvm)是一个`node`版本管理工具。运行如下命令会报错：

```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.2/install.sh | bash
# curl: (7) Failed to connect to raw.githubusercontent.com port 443 after 8 ms: Connection refused
```

**怎么办？** 把网络 DNS 改成`114.114.114.114`或者`8.8.8.8`就好了。

## nvm 常用命令

```bash
# 查看哪些命令
nvm
# 查看远程的所有版本
nvm ls-remote
# 安装node
nvm install v16.12.2

```