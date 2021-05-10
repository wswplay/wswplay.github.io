---
title: 路由初始化
---
## 简约流程
```js
import VueRouter from 'vue-router'
Vue.use(VueRouter) -> install(Vue) {
  // 混入相关设置
  Vue.mixin({
    beforeCreate() {
      this._router.init(this) -> VueRouter.prototype.init = function() {
        
      }
    },
    destroyed() {}
  })
  // 定义原型属性
  Object.defineProperty(Vue.prototype, '$router', {
    get: function get () { return this._routerRoot._router }
  });
  Object.defineProperty(Vue.prototype, '$route', {
    get: function get () { return this._routerRoot._route }
  });
  // 注册路由全局组件
  Vue.component('RouterView', View);
  Vue.component('RouterLink', Link);
}

const routes = [/*你定义的那些路由*/]
// 创建router实例对象
const router = new VueRouter({
  // mode: 'history',
  base: process.env.BASE_URL,
  routes
}) -> function VueRouter() {
  this.matcher = createMatcher(options.routes || [], this)
}
```
## Vue.use
VueRouter做为一个Vue的插件调用，通常备有install方法。
```js
// vue.js
function initUse (Vue) {
  Vue.use = function (plugin) {
    var installedPlugins = (this._installedPlugins || (this._installedPlugins = []));
    if (installedPlugins.indexOf(plugin) > -1) {
      return this
    }
    // additional parameters
    debugger
    var args = toArray(arguments, 1);
    args.unshift(this);
    if (typeof plugin.install === 'function') {
      plugin.install.apply(plugin, args);
    } else if (typeof plugin === 'function') {
      plugin.apply(null, args);
    }
    installedPlugins.push(plugin);
    return this
  };
}
```
## new VueRouter

## Vue.mixin