---
title: h函数：createElement
---
## h函数的真身
```js {2}
new Vue({
  render: h => h(App),
}).$mount('#app')
```
h，就是createElement函数。
```js
// ...
vnode = render.call(vm._renderProxy, vm.$createElement);
// ...
// 手写render函数时
vm.$createElement = function (a, b, c, d) { return createElement(vm, a, b, c, d, true); };
// 用于vue-loader编译生成的render函数时
vm._c = function (a, b, c, d) { return createElement(vm, a, b, c, d, false); };
// ...
```