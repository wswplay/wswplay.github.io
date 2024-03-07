---
title: 日常操作
outline: deep
---

# Day Day Up!

## sudo

遇到`Permission denied`时，用`sudo(Superuser Do)`超级管理员角色运行相关命令。

## 快捷键系列

### OS 快捷键

- 刷新网页：`Command+R`
- 切换窗口：`command + tab` (类似 Windows 窗口的切换方式)
- 显示/隐藏 `点文件(夹)`：`Command + Shift + .`，再按一次恢复隐藏

### Chrome 快捷键

- 开发者工具：option ＋ command ＋ i
- javascript 控制台：option ＋ command ＋ j
- 或者按 option ＋ command ＋ c 也可以打开

### VSCode 快捷键

- 唤起终端：^`(即：control + esc 下面 那个键)

## 删除/usr/bin 中的文件

其实是自从 OS X 10.11 的 El Captain 开始引入的一个系统安全功能，叫做系统完整性保护，英文是 System Integrity Protection，简称 SIP。  
有个这个保护之后，你就无法更改系统文件，即使你 sudo 获取 root 权限，也没有权限。

```bash
# 查看SIP是否启用，默认为启用状态。
csrutil status # System Integrity Protection status: enabled.
```

怎么办？那就禁止 SIP 先。
:::tip

1. 开机时，按住 Command + R 键，会进入到 Mac 系统恢复界面。
2. 点击顶部 实用工具 -> 终端 输入命令`csrutil disable` -> enter 执行命令，会返回 Successfully disabled System Integrity Protection. Please restart the machine for the changes to take effect，重启电脑。即设置成功，此时你可以随心所欲的删除/usr/bin 中的文件了。
3. 如何恢复保护？执行第 1 步，第 2 步输入`csrutil enable` -> enter 命令，重启后恢复保护。

:::

## mac 系统占 80G，怎么办？

通常安装 Xcode 之后，会占用很多空间。【[Linux 命令参考](/mac/linux/os-command.html)】

```bash
# 打开终端，输入如下命令，查看根目录所有文件的大小
du -sh *
# 例如果Library很大，那就看看为啥那么大
cd ~/Library
# 查看当前目录各个文件大小
du -d 1 -h
# 或逆序排列
du -d 1 -h | sort -hr
# 该删就删，Over！
```

## 使用 Finder 访问 Opt 文件夹

- 1、打开 `Finder`。
- 2、按 `Command+Shift+G` 打开对话框。
- 3、搜索：`opt`，即会有下拉提示。
