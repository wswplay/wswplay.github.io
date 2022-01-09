---
title: 命令
---
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

## 只clone某个文件夹
```bash
git init test && cd test     #新建仓库并进入文件夹
git config core.sparsecheckout true #设置允许克隆子目录
echo 'tt*' >> .git/info/sparse-checkout #设置要克隆的仓库的子目录路径，空格别漏
git remote add origin git@github.com:mygithub/test.git  #这里换成你要克隆的项目和库
git pull origin master    #下载
```

## 设置多个远程push地址
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

## 撤销commit提交
```bash
# 撤销最近一次commit
git reset --soft HEAD^
```
:::tip
HEAD^ 表示上一个版本，即上一次的commit，也可以写成HEAD~1    
如果进行两次的commit，想要都撤回，可以使用HEAD~2
:::

## 修改commit注释
```bash
# 输入以下命令，会进入vim编辑器，修改完成保存即可
git commit --amend
```

## 项目内设置 git对文件名 大小写敏感
默认为不敏感，当文件名只是更改大小写时，git无法识别更新，导致报错。
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
## .gitignore文件改动，无效果
```bash
# 清空缓存
git rm -r --cached .
# 重新添加
git add .
#就能看到效果了
```
