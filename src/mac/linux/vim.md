---
title: vim Linux编辑器
outline: deep
---

# Vim：每个人的倚天剑

**[Vim](https://www.vim.org)——The power tool for everyone!**

## 基本命令

基本上 vi/vim 共分为三种模式，命令模式（`Command Mode`）、输入模式（`Insert Mode`）和命令行模式（`Command-Line Mode`）。

- i -- 切换到输入模式，在光标当前位置开始输入文本。
- : -- 切换到底线命令模式，以在最底一行输入命令。
- :w -- 保存文件。
- :q -- 退出 Vim 编辑器。
- :q! -- 强制退出 Vim 编辑器，不保存修改。

## 配置文件

### 显示行号：set number

在配置文件中，加入 `set number`。

```bash
vim ~/.vimrc
cat .vimrc
# set number
```
