---
title: HTML和静态资源
sidebarDepth: 2
---
## HTML
### index文件
```public/index.html``` 文件是一个会被 ```html-webpack-plugin``` 处理的模板。在构建过程中，资源链接会被自动注入。另外，Vue CLI 也会自动注入 resource hint (preload/prefetch、manifest 和图标链接 (当用到 PWA 插件时) 以及构建过程中处理的 JavaScript 和 CSS 文件的资源链接。

### Preload
```<link rel="preload"> ```是一种 ```resource hint```，用来指定页面加载后很快会被用到的资源，所以在页面加载的过程中，我们希望在浏览器开始主体渲染之前尽早 ```preload```。

默认情况下，一个 Vue CLI 应用会为所有初始化渲染需要的文件自动生成 preload 提示。

这些提示会被 ```@vue/preload-webpack-plugin``` 注入，并且可以通过 ```chainWebpack``` 的 ```config.plugin('preload')``` 进行修改和删除。

### Prefetch
```<link rel="prefetch">``` 是一种 resource hint，用来告诉浏览器在页面加载完成后，利用空闲时间提前获取用户未来可能会访问的内容。

默认情况下，一个 Vue CLI 应用会为所有作为 ```async chunk``` 生成的 JavaScript 文件 (通过动态 ```import()``` 按需 ```code splitting``` 的产物) 自动生成 ```prefetch``` 提示。

这些提示会被 ```@vue/preload-webpack-plugin``` 注入，并且可以通过 ```chainWebpack``` 的 ```config.plugin('prefetch')``` 进行修改和删除。

示例：
```js
// vue.config.js
module.exports = {
  chainWebpack: config => {
    // 移除 prefetch 插件
    config.plugins.delete('prefetch')
    // 或者修改它的选项：
    config.plugin('prefetch').tap(options => {
      options[0].fileBlacklist = options[0].fileBlacklist || []
      options[0].fileBlacklist.push(/myasyncRoute(.)+?\.js$/)
      return options
    })
  }
}
```
当 ```prefetch``` 插件被禁用时，你可以通过webpack的内联注释手动选定要提前获取的代码区块。webpack的运行时会在父级区块被加载之后注入 prefetch 链接。
```js
import(/* webpackPrefetch: true */ './someAsyncComponent.vue')
```
::: danger 移动端，最好手动选择
Prefetch 链接将会消耗带宽。如果用户主要使用的是对带宽较敏感的**移动端**，你的应用很大且有很多 async chunk，那么你可能需要**关掉prefetch** 链接**并手动选择**要提前获取的代码区块。
:::

### 服务端渲染，不生成index.html
```js
// vue.config.js
module.exports = {
  // 去掉文件名中的 hash
  filenameHashing: false,
  // 删除 HTML 相关的 webpack 插件
  chainWebpack: config => {
    config.plugins.delete('html')
    config.plugins.delete('preload')
    config.plugins.delete('prefetch')
  }
}
```
```filenameHashing: false```，并不是很推荐，硬编码的文件名有诸多缺点。
### 构建多页面应用
Vue CLI 支持使用 ```vue.config.js``` 中的[pages选项](https://cli.vuejs.org/zh/config/#pages)构建一个多页面的应用。构建好的应用将会在不同的入口之间高效共享通用的 ```chunk``` 以获得最佳的加载性能。

## 处理静态资源
静态资源可以通过两种方式进行处理：   
1. 在 JavaScript 被导入或在 template/CSS 中通过相对路径被引用，会被 webpack 处理。
2. 放置在 public 目录下或通过绝对路径被引用，将会直接被拷贝，而不会经过 webpack 的处理。
### 从相对路径导入
```<img src="...">、background: url(...) 和 CSS @import```的资源URL都会被解析为一个模块依赖。    
如：```url(./image.png)``` 会被翻译为 ```require('./image.png')```   
如：```<img src="./image.png">``` 将被编译为 ```h('img', { attrs: { src: require('./image.png') }})```     
在其内部，我们通过```file-loader```用版本哈希值和正确的公共基础路径来决定最终的文件路径，再用```url-loader```将小于 **4kb** 的资源内联，以减少HTTP请求的数量。

你可以通过 chainWebpack 调整内联文件的大小限制。例如，下列代码会将其限制设置为 10kb：
```js {8}
// vue.config.js
module.exports = {
  chainWebpack: config => {
    config.module
      .rule('images')
        .use('url-loader')
          .loader('url-loader')
          .tap(options => Object.assign(options, { limit: 10240 }))
  }
}
```
### URL 转换规则
1. 如果 URL 是一个绝对路径 (例如 /images/foo.png)，它将会被保留不变。
2. 如果 URL 以 . 开头，它会作为一个相对模块请求被解释且基于你的文件系统中的目录结构进行解析。
3. 如果 URL 以 ~ 开头，其后的任何内容都会作为一个模块请求被解析。这意味着你甚至可以引用 Node 模块中的资源：
```<img src="~some-npm-package/foo.png">```
4. 如果 URL 以 @ 开头，它也会作为一个模块请求被解析。它的用处在于 Vue CLI 默认会设置一个指向 ```<projectRoot>/src 的别名 @```。(仅作用于模版中)

### public文件夹
任何放置在 public 文件夹的静态资源都会被简单的复制，而不经过 webpack。你需要通过绝对路径来引用它们。
::: tip 资源为依赖享webpack好处
我们推荐，将资源作为你的模块依赖图的一部分导入，这样才能获得使用 webpack 的好处。
:::