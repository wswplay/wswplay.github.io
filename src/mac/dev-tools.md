---
title: 开放工具的安装
outline: deep
---

# MacOS Dev-Tools-Set

## brew 安装

Homebrew——**The Missing Package Manager for macOS (or Linux)**。   
牛逼哄哄 inux 包管理器。[官网](https://brew.sh/)告诉我们，用这个：

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

> 试了 N 次终于安装成功。但不知是，在 GitHub 下的包、还是命令行，真安装成功了。

```bash
==> Checking for `sudo` access (which may request your password)...
Password:
==> This script will install:
/opt/homebrew/bin/brew
/opt/homebrew/share/doc/homebrew
/opt/homebrew/share/man/man1/brew.1
/opt/homebrew/share/zsh/site-functions/_brew
/opt/homebrew/etc/bash_completion.d/brew
/opt/homebrew

Press RETURN/ENTER to continue or any other key to abort:
==> /usr/bin/sudo /usr/sbin/chown -R youraccount:admin /opt/homebrew
==> Downloading and installing Homebrew...
HEAD is now at 12c8778af9 Merge pull request #15975 from iMichka/tf
Warning: /opt/homebrew/bin is not in your PATH.
  Instructions on how to configure your shell for Homebrew
  can be found in the 'Next steps' section below.
==> Installation successful!

==> Homebrew has enabled anonymous aggregate formulae and cask analytics.
Read the analytics documentation (and how to opt-out) here:
  https://docs.brew.sh/Analytics
No analytics data has been sent yet (nor will any be during this install run).

==> Homebrew is run entirely by unpaid volunteers. Please consider donating:
  https://github.com/Homebrew/brew#donations

==> Next steps:
- Run these two commands in your terminal to add Homebrew to your PATH:
    (echo; echo 'eval "$(/opt/homebrew/bin/brew shellenv)"') >> /Users/youraccount/.zprofile
    eval "$(/opt/homebrew/bin/brew shellenv)"
- Run brew help to get started
- Further documentation:
    https://docs.brew.sh
```

所以，我们伟大的墙内，推荐国内源：

```bash
/bin/zsh -c "$(curl -fsSL https://gitee.com/cunkai/HomebrewCN/raw/master/Homebrew.sh)"
```

### 环境变量设置

```bash
Warning: /opt/homebrew/bin is not in your PATH.
```

默认情况下，并没有把路径进到 bash 中，蛇精病的设计啊。

```bash
# 编辑文件
vim .zshrc
# 添加路径
export PATH="/opt/homebrew/bin:$PATH"
# 激活更新文件
source .zshrc
# 搞定
brew -v
# Homebrew 4.1.9
```

### 术语解释

- Formulae：**软件包**。包括了这个软件的依赖、源码位置及编译方法等。
- Casks：**应用包**。已经编译好的应用包，如图形界面程序等。
