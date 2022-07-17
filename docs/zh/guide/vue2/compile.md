---
title: 编译
---
## 版本及源码路径
**Vue提供了2个版本**：```Runtime + Compiler```(包含编译代码)和```Runtime-only```。   
1、默认是Runtime-only，源码路径为```vue/dist/vue.runtime.esm.js```   
2、Runtime + Compiler模式，源码路径为```vue/dist/vue.esm.js```，同时需要配置```vue.config.js```文件，内容如下：[参考及缘由](https://www.jianshu.com/p/e8254007f6c4)
```js
module.exports = {
  runtimeCompiler: true,
}
```

## vm._render
vm._render 函数的作用是调用 vm.$options.render 函数并返回生成的虚拟节点(vnode)

## vm._update
vm._update 函数的作用是把 vm._render 函数生成的虚拟节点渲染成真正的 DOM

## 解析(parse)

## 优化(optimize)

## 生成代码(gene code)
