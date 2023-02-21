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

### 目录结构

```md
first-node-cli
├── package.json
└── index.js
```

## 进阶：整个高级点的

### process.argv

`process.argv` 返回一个数组，前两位固定为 `node 程序的路径`和`脚本存放的位置`。  
从第 3 位开始才是我们输入的内容。

```js
// index.js
#!/usr/bin/env node

console.log(process.argv);
```

执行：`./index.js --template vue`

```bash
./index.js --template vue
```

```bash
[
  '/Users/youraccount/.nvm/versions/node/v18.14.0/bin/node',
  '/Users/youraccount/study/node-lab/first-node-cli/index.js',
  '--template',
  'vue'
]
```

执行：`./index.js --template=vue`

```bash
./index.js --template=vue
```

```bash
[
  '/Users/youraccount/.nvm/versions/node/v18.14.0/bin/node',
  '/Users/youraccount/study/node-lab/first-node-cli/index.js',
  '--template=vue'
]
```

### commander 处理参数

当参数很复杂时，需要借助第三方工具处理。【[Github 地址](https://github.com/tj/commander.js)】

**在安装前，先调整目录结构：**  
1、新建 bin 目录专门放置命令文件，将 index.js 移入其中。修改内容为：

```js
// index.js
#!/usr/bin/env node
const { Command } = require("commander");
const programer = new Command();

programer.version(require("../package.json").version);
programer.parse(process.argv);
```

2、在 package.json 中添加如下配置：

```json {2-4}
{
  "bin": {
    "xiao-cli": "bin/index.js"
  }
}
```

3、安装 commander

```bash
pnpm i commander -S
```

运行命令看看效果

```bash
xiao-cli -v
# error: unknown option '-v'
# V 要大写才行
xiao-cli -V
# 1.0.0
xiao-cli -h
# Usage: xiao-cli [options]
# Options:
#   -V, --version  output the version number
#   -h, --help     display help for command
```

### inquirer 添加问答

### 目录结构

```md
first-node-cli
├── bin
│ └── index.js
├── package.json
└── pnpm-lock.yaml
```

## PS：命令安装的位置

```bash
where xiao-cli
# /Users/youraccount/.nvm/versions/node/v18.14.0/bin/xiao-cli
```
