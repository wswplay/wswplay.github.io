---
title: 模块规范
---
## CommonJS
CommonJS 采用同步加载模块，主要运行于服务器端，Node.js 为主要实践者。
1. 该规范指出，一个单独的文件就是一个模块。
2. ```module.exports``` 命令用于规范模块的对外接口，输出的是一个**值的拷贝**，输出之后就不能改变了，会缓存起来。
3. ```require``` 命令用于输入其他模块提供的功能。

## AMD
AMD 是"Asynchronous Module Definition"的缩写，意思就是"异步模块定义"，它采用异步方式加载模块，RequireJS 是最佳实践者。
1. 模块功能主要的几个命令：define、require、return 和 define。
2. define 来定义模块，return 来输出接口， require 来加载模块，这是 AMD 官方推荐用法。

## CMD
CMD (Common Module Definition - 通用模块定义)规范，主要是 Sea.js 推广中形成。
1. 一个文件就是一个模块，可以像 Node.js 一般书写模块代码。主要在浏览器中运行，当然也可以在 Node.js 中运行。
2. 它与 AMD 很类似，不同点在于：AMD 推崇依赖前置、提前执行，CMD 推崇依赖就近、延迟执行。

## UMD
UMD (Universal Module Definition - 通用模块定义)模式，该模式主要用来解决 CommonJS 模式和 AMD 模式代码不能通用的问题，并同时还支持老式的全局变量规范。
1. 判断define为函数，并且是否存在define.amd，来判断是否为AMD规范,
2. 判断module是否为一个对象，并且是否存在module.exports来判断是否为CommonJS规范
3. 如果以上两种都没有，设定为原始的代码规范。

## ES Modules
ES modules(ESM)是 JavaScript 官方的标准化模块系统。
1. 它因为是标准，所以未来很多浏览器会支持，可以很方便的在浏览器中使用。(浏览器默认加载不能省略.js)
2. 它同时兼容在 node 环境下运行。
3. 模块的导入导出，通过 import 和 export 来确定。可以和 Commonjs 模块混合使用。
4. ES modules 输出的是值的引用，输出接口动态绑定，而 CommonJS 输出的是值的拷贝。
5. ES modules 模块编译时执行，而 CommonJS 模块总是在运行时加载。

## 总结
1. CommonJS 同步加载
2. AMD 异步加载
3. UMD = CommonJS + AMD
4. ES Module 是标准规范, 取代 UMD，是大势所趋