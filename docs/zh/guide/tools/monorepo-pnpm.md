---
title: Monorepo和pnpm
---
## 安装pnpm
```bash
# 安装
npm install -g pnpm
# 查看版本
pnpm -v
```

## pnpm命令
### -w, --workspace-root
将 workspace 的根目录作为 pnpm 的运行目录，而不是 当前目录
### --recursive, -r
当在 workspace 下使用时，将从 workspace 下的每个软件包中删除指定的一个或多个依赖包。   
当不在 workspace 下使用时，将在 子目录下寻找所有软件包并删除指定的一个或多个依赖包。
### --save-dev, -D
devDependencies 中列出的依赖包
### --save-prod, -P
dependencies 中列出的依赖包
### --filter <package_selector>
过滤和选择

```bash
# 项目全局安装依赖
pnpm install typescript -D -W
# 安装局部依赖
pnpm install axios -r --filter @xiao/vue
```

## 创建monorepo项目
```bash
# 创建package.json
pnpm init
# 设置 workspace
# 创建pnpm-workspace.yaml文件，并写入以下内容
# 意思就是通过 glob 语法将packages下的所有文件夹都当做一个package，
# 添加到 monorepo 中进行管理
packages:
  - "packages/**" # 注意中间有空格
# 创建packages目录，并在里面创建你的库文件夹
mkdir packages
...
# 用pnpm init给packages中的每个子库创建package.json文件
# 注意子库package.json name字段名字，是通过这个来安装局部依赖的
# Done!
```