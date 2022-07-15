---
title: 视野与格局
---
## 开发范式和套路
Vue总共12000+行代码，难道还看不完吗？

框架，就是通过循环、递归，来构建、解构树形结构的过程。

```Dep``` 和 ```Wather``` 就是相互添加吧。这是典型的```观察者模式```。

```Vnode``` 的树形结构，父子关系也是相互添加啊。

```js
var emptyObject = Object.freeze({});
var _toString = Object.prototype.toString;
var hasOwnProperty = Object.prototype.hasOwnProperty;

// can we use __proto__?
var hasProto = '__proto__' in {};
// Browser environment sniffing
var inBrowser = typeof window !== 'undefined';
var inWeex = typeof WXEnvironment !== 'undefined' && !!WXEnvironment.platform;
var weexPlatform = inWeex && WXEnvironment.platform.toLowerCase();
var UA = inBrowser && window.navigator.userAgent.toLowerCase();
var isIE = UA && /msie|trident/.test(UA);
var isIE9 = UA && UA.indexOf('msie 9.0') > 0;
var isEdge = UA && UA.indexOf('edge/') > 0;
var isAndroid = (UA && UA.indexOf('android') > 0) || (weexPlatform === 'android');
var isIOS = (UA && /iphone|ipad|ipod|ios/.test(UA)) || (weexPlatform === 'ios');
var isChrome = UA && /chrome\/\d+/.test(UA) && !isEdge;
var isPhantomJS = UA && /phantomjs/.test(UA);
var isFF = UA && UA.match(/firefox\/(\d+)/);

initMixin(Vue);
stateMixin(Vue);
eventsMixin(Vue);
lifecycleMixin(Vue);
renderMixin(Vue);

initGlobalAPI(Vue);

Vue.compile = compileToFunctions;
```

哪些不可名具的变量和属性
```js
// vm_render() 组件占位符vnode？？
// this allows render functions to have access to the data on the placeholder node.
vm.$vnode = _parentVnode;
// vm_render() set parent
vnode.parent = _parentVnode;
// vm本身元素vnode？？最后的实体元素vnode
vm._vnode = vnode;
// 这是什么操作
var componentVNodeHooks = {
  init: function init (vnode, hydrating) {},
  prepatch: function prepatch (oldVnode, vnode) {},
  insert: function insert (vnode) {},
  destroy: function destroy (vnode) {}
};
// 执行到这里，格局和视野
var child = vnode.componentInstance = createComponentInstanceForVnode(vnode,activeInstance);
child.$mount(hydrating ? vnode.elm : undefined, hydrating);
```
