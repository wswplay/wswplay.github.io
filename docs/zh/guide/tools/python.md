---
title: Python
---
Mac自带的Python是2.x的版本。但是node的新版本(如16.5.0)，是需要3.x的Python。

## Mac怎么安装3.x的Python
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

