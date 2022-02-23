---
title: 快捷键
---
## 打开chrome开发者工具
1. 开发者工具：option＋command＋i
2. javascript控制台：option＋command＋j
3. 或者按option＋command＋c也可以打开

## 刷新网页
Command+R

## 删除/usr/bin中的文件
其实是自从OS X 10.11的El Captain开始引入的一个系统安全功能，叫做系统完整性保护，英文是System Integrity Protection，简称SIP。    
有个这个保护之后，你就无法更改系统文件，即使你sudo获取root权限，也没有权限。    
```bash
# 查看SIP是否启用，默认为启用状态。
csrutil status # System Integrity Protection status: enabled.
```
怎么办？那就禁止SIP先。
:::tip
1. 开机时，按住Command + R键，会进入到Mac系统恢复界面。
2. 点击顶部 实用工具 -> 终端 输入命令``csrutil disable`` -> enter执行命令，会返回Successfully disabled System Integrity Protection. Please restart the machine for the changes to take effect，重启电脑。即设置成功，此时你可以随心所欲的删除/usr/bin中的文件了。
3. 如何恢复保护？执行第1步，第2步输入```csrutil enable``` -> enter命令，重启后恢复保护。
:::

## 显示隐藏文件夹
```Command + Shift + .``` 可以显示隐藏文件、文件夹，再按一次，恢复隐藏。

## mac系统占80G，怎么办？
通常安装Xcode之后，会占用很多空间。
```bash
# 打开终端，输入如下命令，查看根目录所有文件的大小
du -sh *
# 例如果Library很大，那就看看为啥那么大
cd ~/Library
# 查看当前目录各个文件大小
du -d 1 -h
# 该删就删，Over！
```