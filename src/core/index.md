---
title: JavaScript基础
---

# JavaScript 基础

## 模块化及规范

模块化，就是复杂程序按照规范拆分成相互独立的文件，同时对外暴露一些数据或方法与外部整合。模块化主要特点是：**可复用性、可组合性、独立性、中心化**

> 解决了哪些问题？  
> `解决了命名冲突`：因为每个模块是独立的，所以变量或函数名重名不会发生冲突  
> `提高可维护性`：因为每个文件的职责单一，有利于代码维护  
> `性能优化`：异步加载模块对页面性能会非常好  
> `模块的版本管理`：通过别名等配置，配合构建工具，可以实现模块的版本管理  
> `跨环境共享模块`：通过 Sea.js 的 NodeJS 版本，可以实现模块的跨服务器和浏览器共享

**主流标准有**：CommonJS、AMD、CMD、UMD、ES6 【[参考资料](https://juejin.cn/post/6996595779037036580#heading-0)】

#### CommonJS(cjs)

Node 用的就是 CommonJS 模块化规范。

#### AMD

CommonJS 规范加载模块是同步加载，只有加载完成，才能执行后面的操作，而 AMD 是异步加载模块，可以指定回调函数。该规范的实现就是 require.js。

#### CMD

CMD 规范整合了上面说的 CommonJS 规范和 AMD 规范的特点，CMD 规范的实现就是 sea.js。CMD 规范最大的特点就是**懒加载**，并且同时支持同步和异步加载模块。

#### UMD

UMD 没有专门的规范，而是集合了上面说的三个规范于一身，它可以让我们在合适的环境选择合适的模块规范。  
比如在 Node.js 环境中用 CommonJS 模块规范管理，在浏览器端支持 AMD 的话就采用 AMD 模块规范，不支持就导出为全局函数。

#### ES6 模块化(esm)

CommonJS 和 AMD 都是在运行时确定依赖关系，也就是运行时加载，CommonJS 加载的是拷贝，而 ES6 module 是在编译时就确定依赖关系，所有的加载都是引用，这样做的好处是可以执行静态分析和类型检查。

:::tip ES6 Module 和 CommonJS 的区别：

- ES6 Module 的 import 是静态引入，CommonJS 的 require 是动态引入
- Tree-Shaking 就是通过 ES6 Module 的 import 来进行静态分析，并且只支持 ES6 Module 模块的使用。Tree-Shaking 就是移除掉 JS 上下文中没有引用的代码，比如 import 导入模块没有返回值的情况下，webpack 在打包编译时 Tree-Shaking 会默认忽略掉此文件
- ES6 Module 是对模块的引用，输出的是值的引用，改变原来模块中的值引用的值也会改变；CommonJS 是对模块的拷贝，修改原来模块的值不会影响引用的值
- ES6 Module 里的 this 指向 undefined；CommonJS 里的 this 指向模块本身
- ES6 Module 是在编译时确定依赖关系，生成接口并对外输出；CommonJS 是在运行时加载模块
- ES6 Module 可以单独加载某个方法；CommonJS 是加载整个模块
- ES6 Module 不能被重新赋值，会报错；CommonJS 可以重新赋值(改变 this 指向)
  :::
