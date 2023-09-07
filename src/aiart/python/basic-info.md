---
title: Python最新安装方法
---

# Python——简单、强大

早前 MacOS 自带 Python 2.x 版本。但 node 新版本(如 16.5.0)需要 3.x 的 Python。

## Mac 怎么安装 3.x 的 Python

```bash
# 会自动安装，按提示操作即可
brew install python3
# 查看python的位置
which python3 # 比如我的路径是: /usr/local/bin/python3
# 打开并编辑zsh配置文件
vim ~/.zshrc
# 配置别名路径，覆盖默认版本。esc -> :wq
alias python='/usr/local/bin/python3'
# 刷新文件，立即生效
source ~/.zshrc
# 查看默认版本，O了
python --version 或者 python -V # Python 3.9.2
```

## 基本命令

```bash
# 退出python环境
control + z;
# or
exit();
# or
quit();
```

## IDE编辑器
[PyCharm Mac](https://www.jetbrains.com/pycharm/download/?section=mac)
