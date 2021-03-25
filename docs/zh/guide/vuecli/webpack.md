---
title: webpack相关
---
## 简单的配置方式
调整 webpack 配置最简单的方式就是在 vue.config.js 中的 configureWebpack 选项提供一个对象：
```js
// vue.config.js
module.exports = {
  configureWebpack: {
    plugins: [
      new MyAwesomeWebpackPlugin()
    ]
  }
}
```
该对象将会被 webpack-merge 合并入最终的 webpack 配置。

## 链式操作 (高级)
Vue CLI 内部的 webpack 配置是通过 webpack-chain 维护的。你需要熟悉 [webpack-chain](https://github.com/neutrinojs/webpack-chain#getting-started) 的 API 并阅读一些源码以便了解如何最大程度利用好这个选项。

## 审查 webpack 配置
因为 ```@vue/cli-service``` 对 webpack 配置进行了抽象，所以理解配置中包含的东西会比较困难。    
vue-cli-service 暴露了 inspect 命令用于审查解析好的 webpack 配置。    
那个全局的 ```vue``` 可执行程序同样提供了 inspect 命令，这个命令只是简单的把 vue-cli-service inspect 代理到了你的项目中。
```bash
vue inspect > output.js
```
## config的文件和位置
```
<projectRoot>/node_modules/@vue/cli-service/webpack.config.js
```
该文件会动态解析并输出 ```vue-cli-service``` 命令中使用的相同的 webpack 配置，包括那些来自插件甚至是你自定义的配置。