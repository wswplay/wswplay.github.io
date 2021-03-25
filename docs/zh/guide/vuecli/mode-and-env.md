---
title: 模式和环境变量
---
## 模式
模式是 Vue CLI 项目中一个重要的概念。默认情况下，一个 Vue CLI 项目有三个模式：
>1. ```development``` 模式用于 ```vue-cli-service serve```
>2. ```test``` 模式用于 ```vue-cli-service test:unit```
>3. ```production``` 模式用于 ```vue-cli-service build``` 和 ```vue-cli-service test:e2e```

你可以通过传递 --mode 选项参数为命令行覆写默认的模式。例如，如果你想要在构建命令中使用开发环境变量：
```bash
vue-cli-service build --mode development
```

## 代码中使用环境变量
只有以 ```VUE_APP_``` 开头的变量会被 ```webpack.DefinePlugin``` 静态嵌入到客户端侧的包中。你可以在应用的代码中这样访问它们：
```js
console.log(process.env.VUE_APP_SECRET)
```
在构建过程中，process.env.VUE_APP_SECRET 将会被相应的值所取代。在 VUE_APP_SECRET=secret 的情况下，它会被替换为 "secret"。

除了 ```VUE_APP_*``` 变量之外，在你的应用代码中始终可用的还有两个特殊的变量：
1. ```NODE_ENV``` - 会是 "development"、"production" 或 "test" 中的一个，具体取决于应用运行模式。
2. ```BASE_URL``` - 会和 vue.config.js 中的 publicPath 选项相符，即你的应用会部署到的基础路径。
