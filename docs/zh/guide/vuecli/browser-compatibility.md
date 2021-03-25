---
title: 浏览器兼容性
---
## browserslist
你会发现有 ```package.json``` 文件里的 ```browserslist``` 字段 (或一个单独的 ```.browserslistrc``` 文件)，指定了项目的目标浏览器的范围。这个值会被 ```@babel/preset-env``` 和 ```Autoprefixer``` 用来确定需要转译的 JavaScript 特性和需要添加的 CSS 浏览器前缀。

## Polyfill

## 现代模式