---
title: CLI服务与命令
---
## vue-cli-service serve
```--open``` 自动打开浏览器
```json {3}
// package.json
"scripts": {
  "dev": "vue-cli-service serve --open",
}
```

## vue-cli-service build
```--report```和```--report-json``` 根据构建统计生成报告
```json {4}
// package.json
"scripts": {
  "dev": "vue-cli-service serve --open",
  "build": "vue-cli-service build --report --report-json",
}
```

## vue-cli-service inspect
查看webpack配置详情，并将其写到```webpackDefaultConfig.js```文件中
```json {5}
// package.json
"scripts": {
  "dev": "vue-cli-service serve --open",
  "build": "vue-cli-service build --report --report-json",
  "in": "vue-cli-service inspect > webpackDefaultConfig.js"
}
```

## 缓存和并行处理
```cache-loader``` 会默认为 Vue/Babel/TypeScript 编译开启。文件会缓存在 ```node_modules/.cache``` 中——如果你遇到了编译方面的问题，记得先删掉缓存目录之后再试试看。

```thread-loader``` 会在多核 CPU 的机器上为 Babel/TypeScript 转译开启。