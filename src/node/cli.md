---
title: 怎么写一个node命令行工具
outline: deep
---

# CLI：从 0 写一个 node 命令行工具

CLI：Command Line Interface。

## 创建一个项目及文件

```bash
# 创建一个文件夹
mkdir first-node-cli
# 进入文件夹
cd first-node-cli
# 快速创建package.json文件
npm init -y
# 创建index.js文件
touch index.js
# 编辑器打开项目
code .
```

## 初级：打印个 Hello 先

### index.js

在 index.js 文件中，写入以下代码：

```js
// index.js
#!/usr/bin/env node
console.log('Hello, Cli');
```

`#!/usr/bin/env node`：**该行必不可少**。意思是，让系统自动去环境设置寻找 node 目录，再调用对应路径下的解释器程序。实际上，就是告诉这是一个可以用 node **直接执行**的文件。

所以你可以直接这样运行 index.js

```bash
./index.js
# zsh: permission denied: ./index.js
```

于是你会得到` zsh: permission denied: ./index.js`，没有权限。那就添加权限

```bash
chmod 744 index.js
```

再次运行，你可以看到`Hello, Cli`了。

```bash
./index.js
# Hello, Cli
```

### package.json

在 bin 字段声明一个命令。想多个，就声明多个。

```json
{
  "bin": {
    "xiao-cli": "index.js"
  }
}
```

运行 `xiao-cli` 命令

```bash
xiao-cli
# zsh: command not found: xiao-cli
```

命令不存在？因为，我们还没有安装嘛，所以终端不认识这个。那就安起

> 本地测试一个 npm 包，可使用：npm link 本地安装这个包就行。一般是安装到全局。

```bash
npm link
```

再次运行命令 `xiao-cli` 命令，就成功了。

```bash
xiao-cli
# Hello, Cli
```

### 初级目录结构

```bash
first-node-cli
├── package.json
└── index.js
```

## 进阶：参数处理

### process.argv

`process.argv` 返回一个数组，前两位固定为 `node 程序的路径`和`脚本存放的位置`。  
从第 3 位开始才是我们输入的内容。

```js
// index.js
#!/usr/bin/env node

console.log(process.argv);
```

执行：`./xiao.js --template vue`

```bash
./xiao.js --template vue
```

```bash
[
  '/Users/youraccount/.nvm/versions/node/v18.14.0/bin/node',
  '/Users/youraccount/study/node-lab/first-node-cli/index.js',
  '--template',
  'vue'
]
```

执行：`./xiao.js --template=vue`

```bash
./xiao.js --template=vue
```

```bash
[
  '/Users/youraccount/.nvm/versions/node/v18.14.0/bin/node',
  '/Users/youraccount/study/node-lab/first-node-cli/index.js',
  '--template=vue'
]
```

## PS：命令安装的位置

```bash
where xiao-cli
# /Users/youraccount/.nvm/versions/node/v18.14.0/bin/xiao-cli
```
