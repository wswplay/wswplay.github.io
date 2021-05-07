---
title: 视野与格局
---
Vue总共12000+行代码，难道还看不完吗？

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
// 占位符vnode？？
vm._vnode = vnode; 
// 这是什么操作
var componentVNodeHooks = {
  init: function init (vnode, hydrating) {},
  prepatch: function prepatch (oldVnode, vnode) {},
  insert: function insert (vnode) {},
  destroy: function destroy (vnode) {}
};
```
