---
title: 本地调试
---
```js
// 1.初始化项目，生成package.json
npm init
// 2.安装webpack，并加入包依赖配置
npm install webpack@4.46.0 -D 
// 3.创建配置文件(webpack.config.js)
const path = require('path')
module.exports = {
  entry: './index.js',
  output: {
    path: path.resolve(__dirname, 'dist'),
    filename: 'bundle.js'
  }
}
// 4.定义调试scripts(package.json)
"scripts": {
  "build": "webpack --config webpack.config.js",
  "debugger": "node --inspect-brk ./node_modules/webpack/bin/webpack.js --config webpack.config.js",
}
// 5.执行debugger命令，开始调试
npm run debugger
```
### 参考
[node + chrome 调试](http://0.0.0.0:8080/zh/guide/node/init.html#%E6%9C%AC%E5%9C%B0%E8%B0%83%E8%AF%95)