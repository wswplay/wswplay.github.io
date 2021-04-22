---
title: 命令
---
## 只clone某个文件夹
```bash
git init test && cd test     //新建仓库并进入文件夹
git config core.sparsecheckout true //设置允许克隆子目录
echo 'tt*' >> .git/info/sparse-checkout //设置要克隆的仓库的子目录路径   //空格别漏
git remote add origin git@github.com:mygithub/test.git  //这里换成你要克隆的项目和库
git pull origin master    //下载
```