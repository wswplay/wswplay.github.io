---
title: 基础命令
outline: deep
---

# 基础命令

man xxx 查看帮助手册(如查看 du：man du)。

## find 文件名查找

```bash
find . -iname '*google*' -d 1
# . 当前位置；/ 根目录
# i 不区分大小写；name 要搜索的文件名，后面紧跟文件名
# d 查找深度设置为 1，默认好像是无限递归查询？
```

## which 命令查找

```bash
which brew
# /opt/homebrew/bin/brew
```

## whereis 命令查找

查找结果除了显示命令所在的命令以外，还会列出帮助文档所在的目录。

```bash
where brew
# /opt/homebrew/bin/brew
whereis brew
# brew: /opt/homebrew/bin/brew /opt/homebrew/share/man/man1/brew.1
```

## grep 内容查找

global regular expression

```bash
grep -i -n 'python' .zshrc
# -i:忽略大小写 -n:显示行号 目标内容 目标文件
# 104:alias python="/usr/local/bin/python3"
```

## locate 文件名查找

`find` 遍历磁盘查找文件，占多资源，相对较慢。而 `locate` 命令在 Linux 文件数据库中查找，速度快。MacOs 貌似默认没有这个。

```bash
locate -i '*google*'
```

## du：disk usage 目录/文件大小

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

## echo 打印信息

```bash
echo $SHELL # 查看当前shell
# /bin/zsh
```
