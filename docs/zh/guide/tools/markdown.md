---
title: Markdown
---
## 新窗口打开内页
Markdown默认：外链-新窗口打开，内页-直接跳转。
```html
<a href="/zh/guide/basic/operators.html" target="_blank">基础知识-位运算</a>
```

## 生成README.md项目目录
看上去是一个小问题，可能是问题太小了，居然没有找到特别大众而且牛逼的包。   
试了几个类似的包，觉得```tree-node-cli```功能简约且够用，那就先用着吧。[Github](https://github.com/yangshun/tree-node-cli)
```bash
# 全局安装
npm install -g tree-node-cli
# 进入到目标项目，运行如下命令。
# 释：-L 2 最深层级为2；-I node_modules 忽略node_modules文件夹
tree -L 2 -I node_modules > cate.md
