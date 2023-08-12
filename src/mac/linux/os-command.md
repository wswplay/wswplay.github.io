---
title: 基础命令
outline: deep
---

# 基础命令

man xxx 查看帮助手册(如查看 du：man du)。

## du：disk usage

用于显示目录或文件的大小。

### -d 迭代深度

`-d` 表示迭代深度，当前目录深度是 `0`，`1` 表示最深迭代到当前目录的下一个深度，等价的命令是 `du --max-depth 1`。

### -h 人类可读

`-h` 或 `--human-readable` 以 `K，M，G` 为单位，提高信息的可读性。

### -s 文件大小总计

`-s` 或 `--summarize` 仅显示总计，包含子目录。

### |：`pipe 管道命令`

- 选取命令：`cut、grep`
- 排序命令：`sort、wc、uniq`  
  `-r 将排序结果逆序`
