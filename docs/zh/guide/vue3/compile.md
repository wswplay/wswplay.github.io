---
title: 编译
---
## 本地怎么调试编译
:::tip
1、上Github荡下Vue3.0源码；[Vuejs/core](https://github.com/vuejs/core)   
2、在根目录下，```pnpm i```安装依赖(注意是 ```pnpm```)；      
3、进入```packages/template-explorer```，```npm i```安装依赖；   
4、根目录运行 ```npm run dev-compiler```，会自动进入编译，并完成编译；   
5、新开一个命令行窗口，执行```npm run open```，就会自动打开编译的html界面了；   
6、此时```packages/template-explorer```里面，应该有dist目录了，   
  在```template-explorer.global.js```里面打上debugger，即可调试。
:::

```js
function compile(template, options = {}) { 
  return baseCompile(template, extend({}, parserOptions, options, { 
    nodeTransforms: [...DOMNodeTransforms, ...(options.nodeTransforms || [])], 
    directiveTransforms: extend({}, DOMDirectiveTransforms, options.directiveTransforms || {}), 
    transformHoist:  null 
  })) 
}
function baseCompile(template,  options = {}) { 
  const prefixIdentifiers = false 
  // 解析 template 生成 AST 
  const ast = isString(template) ? baseParse(template, options) : template 
  const [nodeTransforms, directiveTransforms] = getBaseTransformPreset() 
  // AST 转换 
  transform(ast, extend({}, options, { 
    prefixIdentifiers, 
    nodeTransforms: [ 
      ...nodeTransforms, 
      ...(options.nodeTransforms || []) 
    ], 
    directiveTransforms: extend({}, directiveTransforms, options.directiveTransforms || {} 
    ) 
  })) 
  // 生成代码 
  return generate(ast, extend({}, options, { 
    prefixIdentifiers 
  })) 
}
```
baseCompile 函数主要做三件事情：**解析 template 生成 AST，AST 转换和生成代码**。

## 生成 AST 抽象语法树
```js
function baseParse(content, options = {}) {
    const context = createParserContext(content, options);
    const start = getCursor(context);
    return createRoot(parseChildren(context, 0 /* DATA */, []), getSelection(context, start));
}
```
baseParse 主要就做三件事情：**创建解析上下文，解析子节点，创建 AST 根节点**。

## Block
为了运行时的更新优化，Vue.js 3.0 设计了一个 Block tree 的概念。Block tree 是一个将模版基于动态节点指令切割的嵌套区块，每个区块只需要以一个 Array 来追踪自身包含的动态节点。借助 Block tree，Vue.js 将 vnode 更新性能由与模版整体大小相关提升为与动态内容的数量相关，极大优化了 diff 的效率，模板的动静比越大，这个优化就会越明显。

因此在编译阶段，我们需要找出哪些节点可以构成一个 Block，其中动态组件、svg、foreignObject 标签以及动态绑定的 prop 的节点都被视作一个 Block。

## 静态提升hoistStatic
因为这些静态节点不依赖动态数据，一旦创建了就不会改变，所以只有静态节点才能被提升到外部创建。所以，它可以创建在render函数之外。

如果说 parse 阶段是一个词法分析过程，构造基础的 AST 节点对象，那么 transform 节点就是语法分析阶段，把 AST 节点做一层转换，构造出语义化更强，信息更加丰富的 codegenCode，它在后续的代码生成阶段起着非常重要的作用。




