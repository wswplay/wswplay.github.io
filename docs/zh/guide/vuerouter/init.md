---
title: 路由初始化
---

路由始终会维护当前的线路，路由切换的时候会把当前线路切换到目标线路，切换过程中会执行一系列的导航守卫钩子函数，会更改 url，同样也会渲染对应的组件，切换完毕后会把目标线路更新替换当前线路，这样就会作为下一次的路径切换的依据。

## 思路
new VueRouter 本质上就是往实例对象 this 上挂载属性。如最重要的matcher、history属性。
```js
VueRouter {
  matcher: {
    match: () => {}
  },
  history: {
    current: {}
  }
}
```
### transitionTo
history.transitionTo 改变、更新路由的方法。
```js
VueRouter.prototype.init
history.push
history.replace
VueRouter.prototype.addRoute
VueRouter.prototype.addRoutes
history.setupListeners() {
  let handleRoutingEvent = function() {}
  window.addEventListener('popstate', handleRoutingEvent)
}
```
```js
// 启动路由
this._router.init(this)
// 准备跳转
history.transitionTo
  // match出目标路径对象
  route = this.router.match(location, this.current)   
  var prev = this.current
// 确认跳转  
this.confirmTransition(route) 
  var current = this.current  
  this.pending = route
  var queue = [].concat(
    // in-component leave guards
    extractLeaveGuards(deactivated),
    // global before hooks
    this.router.beforeHooks,
    // in-component update hooks
    extractUpdateHooks(updated),
    // in-config enter guards
    activated.map(function (m) { return m.beforeEnter; }),
    // async components
    resolveAsyncComponents(activated)
  );
  // 迭代器
  var iterator = function (hook, next) {
    if (this$1.pending !== route) {
      return abort(createNavigationCancelledError(current, route))
    }
    try {
      hook(route, current, function (to) {
        if (to === false) {
          // next(false) -> abort navigation, ensure current URL
          this$1.ensureURL(true);
          abort(createNavigationAbortedError(current, route));
        } else if (isError(to)) {
          this$1.ensureURL(true);
          abort(to);
        } else if (
          typeof to === 'string' ||
          (typeof to === 'object' &&
            (typeof to.path === 'string' || typeof to.name === 'string'))
        ) {
          // next('/') or next({ path: '/' }) -> redirect
          abort(createNavigationRedirectedError(current, route));
          if (typeof to === 'object' && to.replace) {
            this$1.replace(to);
          } else {
            this$1.push(to);
          }
        } else {
          // confirm transition and pass on the value
          next(to);
        }
      });
    } catch (e) {
      abort(e);
    }
  };

  runQueue(queue, iterator, function () {
    // wait until async components are resolved before
    // extracting in-component enter guards
    var enterGuards = extractEnterGuards(activated);
    var queue = enterGuards.concat(this$1.router.resolveHooks);
    runQueue(queue, iterator, function () {
      if (this$1.pending !== route) {
        return abort(createNavigationCancelledError(current, route))
      }
      this$1.pending = null;
      onComplete(route);
      if (this$1.router.app) {
        this$1.router.app.$nextTick(function () {
          handleRouteEntered(route);
        });
      }
    });
  });
```    


## 简约流程
```js
import VueRouter from 'vue-router'
Vue.use(VueRouter) -> install(Vue) {
  // 混入相关设置
  Vue.mixin({
    beforeCreate() {
      this._router.init(this) -> VueRouter.prototype.init
      -> history.transitionTo() {
        // 计算匹配route
        var route = this.router.match(location, this.current)
        -> VueRouter.prototype.match -> this.matcher.match
        -> this.matcher = createMatcher() {
          // 根据传入参数创建映射
          var ref = createRouteMap(routes) {
            routes.forEach(function (route) {
              addRouteRecord(pathList, pathMap, nameMap, route) {
                var record = {
                  path: normalizedPath,
                  regex: compileRouteRegex(normalizedPath, pathToRegexpOptions),
                  components: route.components || { default: route.component },
                  instances: {},
                  name: name,
                  parent: parent,
                  matchAs: matchAs,
                  redirect: route.redirect,
                  beforeEnter: route.beforeEnter,
                  meta: route.meta || {},
                  props:
                    route.props == null
                      ? {}
                      : route.components
                        ? route.props
                        : { default: route.props }
                };
                if (!pathMap[record.path]) {
                  pathList.push(record.path);
                  pathMap[record.path] = record;
                }
              }
            });
            return { pathList,pathMap,nameMap }
          }
          // 创建动态添加路由 的功能方法
          function addRoutes (routes) {
            createRouteMap(routes, pathList, pathMap, nameMap);
          }
          // 这里这里this.router.match
          function match() {} -> _createRoute -> createRoute -> return Object.freeze(route);
          // 确认切换
          this.confirmTransition -> History.prototype.confirmTransition() {
            // 导航守卫
            var queue = [].concat(
              // in-component leave guards
              extractLeaveGuards(deactivated),
              // global before hooks
              this.router.beforeHooks,
              // in-component update hooks
              extractUpdateHooks(updated),
              // in-config enter guards
              activated.map(function (m) { return m.beforeEnter; }),
              // async components
              resolveAsyncComponents(activated)
            );
            var iterator = function (hook, next) {}
            runQueue(queue, iterator, fn) {
              // 那些被省略的代码...
              // 递归
              runQueue(queue, iterator, fn) {}
            }
          }
        }
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

## url
当我们点击 router-link 的时候，实际上最终会执行 router.push
```js
VueRouter.prototype.push -> this.history.push()
-> HashHistory.prototype.push() {
  this.transitionTo(location, function (route) {
      pushHash(route.fullPath) -> pushState() {
        replace ? history.replaceState : history.pushState
      }
      handleScroll(this$1.router, route, fromRoute, false);
      onComplete && onComplete(route);
    },
    onAbort
  );
}
```

## 组件
路由最终的渲染离不开组件，Vue-Router 内置了 ```<router-view>``` 组件

## 参考
#### Vue.use
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