---
title: git命令(如Mac中)的安装和使用
---

# Git：分布式版本控制系统

## Mac 安装 Git

[官网](https://git-scm.com/downloads)。又是 brew，又是 Xcode，巴拉巴拉。。。  
[直接文件安装](https://sourceforge.net/projects/git-osx-installer/)，不香吗？  
安装完成，git 一下，你就知道：会报错。

```bash
xcrun: error: invalid active developer path (/Library/Developer/CommandLineTools),
missing xcrun at: /Library/Developer/CommandLineTools/usr/bin/xcrun
```

Mac 傻逼，git 还要装 CommandLineTools，苹果官网下载[Command Line Tools](https://developer.apple.com/download/all/?q=Command)。安装完成，就大功告成！:dart:

> 查看 Command Line Tools 版本：`llvm-gcc --version`

```bash
llvm-gcc --version
# Apple clang version 15.0.0 (clang-1500.0.40.1)
# Target: arm64-apple-darwin23.0.0
# Thread model: posix
# InstalledDir: /Library/Developer/CommandLineTools/usr/bin
git -v
# git version 2.39.3 (Apple Git-145)
```

## 初始化

```bash
git init
```

## 分支操作

```bash
# 查看所有分支
git branch -a
# 切换分支
git checkout branchName
# 创建分支
git branch branchName
# 创建并切换到新分支
git checkout -b branchName
```

## 将 master 分支修改成 main 分支

通常初始化会是 master 分支，但现在主流已经是 main 了。

```bash
git branch -m main
```

## 删除 master 分支

```bash
# 删除本地
git branch -d master
# 删除远程master分支。先取消master为默认分支，才会生效，否则报错
git push origin -d master
```

## 查看配置信息

```bash
git config --list
```

## 设置用户名和邮箱

```bash
git config --global user.name xiao
git config --global user.email biancheng@xiao.com
```

## 查看用户名和邮箱

```bash
git config user.name
git config user.email
```

## 只 clone 某个文件夹

```bash
git init test && cd test     #新建仓库并进入文件夹
git config core.sparsecheckout true #设置允许克隆子目录
echo 'tt*' >> .git/info/sparse-checkout #设置要克隆的仓库的子目录路径，空格别漏
git remote add origin git@github.com:mygithub/test.git  #这里换成你要克隆的项目和库
git pull origin master    #下载
```

## 设置多个远程 push 地址

```bash
# 查看本项目远程提交地址
git remote -v
# git remote set-url --add <name> <url> 即可
git remote set-url --add origin https://github.com/xxx/xxx
```

## 删除远程提交地址

```bash
# 删除所有地址
git remote remove origin
# 删除一个地址
git remote set-url --delete origin https://github.com/xxx/xxx.git
```

## 本地项目与远程项目关联

```bash
# 本地项目内运行如下命令
# 查看远端信息
git remote -v
# 添加关联远程仓库
git remote add origin https://gitee.com/xxx/xx-xxx-xxx.git
# 再次查看，就能看到相关信息了
git remote -v
# 拉取代码
git pull origin master
# 如果出现如下错误(拒绝合并没有关联的历史记录)
fatal: refusing to merge unrelated histories
# 运行下面命令即可(即：允许合并没有关联的历史记录)
git pull origin master --allow-unrelated-histories
```

## 撤销 commit 提交

```bash
# 整体回到上次一次操作，绿字变红字(撤销add)
git reset HEAD
# 红字变无 (撤销没add修改)
git reset HEAD 路径文件名
# 撤销最近一次commit，git status 显示绿字
git reset --soft HEAD^
# 慎用！慎用！慎用！撤销commit，且会删除所有改动的代码。一夜回到解放前！
git reset --hard HEAD~1
```

:::tip
HEAD^ 表示上一个版本，即上一次的 commit，也可以写成 HEAD~1  
如果进行两次的 commit，想要都撤回，可以使用 HEAD~2
:::

## 修改 commit 注释

```bash
# 输入以下命令，会进入vim编辑器，修改完成保存即可
git commit --amend
```

## 项目内设置 git 对文件名 大小写敏感

默认为不敏感，当文件名只是更改大小写时，git 无法识别更新，导致报错。

```bash
# 查看忽略的设置
git config --get core.ignorecase
# 设置大小写敏感
git config core.ignorecase false
```

## 只提交修改的内容

```bash
git add -u
```

## 创建没有父节点的分支

```bash
# 把文件全部删除，就变成空白分支了啊
git checkout --orphan branchName
```

## .gitignore 文件改动，无效果

```bash
# 清空缓存
git rm -r --cached .
# 重新添加
git add .
#就能看到效果了
```
