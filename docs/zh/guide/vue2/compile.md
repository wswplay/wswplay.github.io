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
## 流程
百转千回的绕，就是为了把```template```转化成```render```函数
```js
// 入口
var mount = Vue.prototype.$mount;
Vue.prototype.$mount = function() {
  var options = this.$options;
  if (!options.render){
    if (template){
      var ref = compileToFunctions(template, ...)
      var render = ref.render;
      var staticRenderFns = ref.staticRenderFns;
      options.render = render;
      options.staticRenderFns = staticRenderFns;
    }
  }
  return mount.call(this, el, hydrating)
}
// 中继
var compiled = baseCompile(template.trim(), finalOptions);
// 庐山真面目
var createCompiler = createCompilerCreator(function baseCompile (
  template,
  options
) {
  // 解析ast
  var ast = parse(template.trim(), options);
  if (options.optimize !== false) {
    // 静态标记(优化)
    optimize(ast, options);
  }
  // 生成代码
  var code = generate(ast, options);
  return {
    ast: ast,
    render: code.render,
    staticRenderFns: code.staticRenderFns
  }
});
// 上面生成代码的函数。这里有render函数
function generate (
  ast,
  options
) {
  var state = new CodegenState(options);
  var code = ast ? (ast.tag === 'script' ? 'null' : genElement(ast, state)) : '_c("div")';
  return {
    render: ("with(this){return " + code + "}"),
    staticRenderFns: state.staticRenderFns
  }
}
```
## vm._render
vm._render 函数的作用是调用 vm.$options.render 函数并返回生成的虚拟节点(vnode)

## vm._update
vm._update 函数的作用是把 vm._render 函数生成的虚拟节点渲染成真正的 DOM

## 解析(parse)

## 优化(optimize)

## 生成代码(gene code)
