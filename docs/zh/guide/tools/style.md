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

## 解决margin-top塌陷
:::tip   
一个盒子如果没有上补白(padding-top)和上边框(border-top)，那么这个盒子的上边距会和其内部文档流中的第一个子元素的上边距重叠。
[MDN](https://developer.mozilla.org/zh-CN/docs/Web/CSS/CSS_Box_Model/Mastering_margin_collapsing)
:::
即子元素的margin-top值，影响了父元素的位置。[Demo](https://wswplay.github.io/vdemo/#/css/margin)
```js
// 给目标元素加上这个class
.MarginCollapse {
  &::before {
    content: "";
    display: inline-block; // 这里至关重要
  }
}
```

## 行内样式怎么写伪元素(如:hover)
待解答