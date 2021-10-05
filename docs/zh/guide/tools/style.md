---
title: css样式
---
## sass转成css
```bash
# 安装sass
npm install -g sass
# 转换文件
sass demo.scss demo.css
# 生成环境
sass --style compressed style.scss style.css
# 监听
sass --watch style.scss style.css
```

## Vue中使用less全局变量
1、定义变量、安装相关loader
```js
  // variable.less
  @fontSize18: 18px;
```
```bash
npm i style-resources-loader vue-cli-plugin-style-resources-loader -D
```
2、添加vue.config.js文件相关配置
```js
const path = require('path');

pluginOptions: {
  'style-resources-loader': {
    preProcessor: 'less',
    patterns: [path.resolve(__dirname, "src/style/variable.less")] // 引入全局样式变量
  }
},
```
3、.vue文件中适用
```js
  // style
  font-size: @fontSize18;
```
