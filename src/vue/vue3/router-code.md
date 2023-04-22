---
title: Vue-Router源码分析
---

# Vue-Router 源码摘要

## 初始化安装 install

```ts
const router: Router = {
  install(app: App) {
    const router = this
    // 注册全局组件
    app.component('RouterLink', RouterLink)
    app.component('RouterView', RouterView)
    app.config.globalProperties.$router = router
    Object.defineProperty(app.config.globalProperties, '$route', {
      enumerable: true,
      get: () => unref(currentRoute),
    })
    const reactiveRoute = {}
    for (const key in START_LOCATION_NORMALIZED) {
      reactiveRoute[key] = computed(() => currentRoute.value[key])
    }
    // 数据注入
    app.provide(..., ...)
    // 卸载
    const unmountApp = app.unmount
    installedApps.add(app)
    app.unmount = function () {
      installedApps.delete(app)
      unmountApp()
    }
  }
}
return router
```

## 创建路由 createRouter

```ts
 createRouter(options) {
  const matcher = createRouterMatcher(options.routes, options) {
    // 优化排序后的匹配项数组
    const matchers: RouteRecordMatcher[] = []
    const matcherMap = new Map()
    function getRecordMatcher(name: RouteRecordName) {
      return matcherMap.get(name)
    }
    function addRoute(...) { /* createRouterMatcher/addRoute */ }
    routes.forEach(route => addRoute(route))
    return { addRoute, resolve, removeRoute, getRoutes, getRecordMatcher }
  }
  const routerHistory = options.history
  const beforeGuards = useCallbacks<NavigationGuardWithThis<undefined>>()
  const beforeResolveGuards = useCallbacks<NavigationGuardWithThis<undefined>>()
  const afterGuards = useCallbacks<NavigationHookAfter>()
  const currentRoute = shallowRef<RouteLocationNormalizedLoaded>(
    START_LOCATION_NORMALIZED
  )
  // 改变路由
  function push(to: RouteLocationRaw) {
    return pushWithRedirect(to)
  }
  function pushWithRedirect(to, redirectedFrom) {
    const targetLocation: RouteLocation = (pendingLocation = resolve(to))
    const shouldRedirect = handleRedirectRecord(targetLocation)
    if (shouldRedirect) pushWithRedirect(...)
    let failure: NavigationFailure | void | undefined
    return (failure ? Promise.resolve(failure) : navigate(toLocation, from)).then(() => {
      // 执行 全局钩子 afterEach
      triggerAfterEach(to, from, failure) {
        for (const guard of afterGuards.list()) guard(to, from, failure)
      }
    })
  }
  const go = (delta: number) => routerHistory.go(delta)

  const installedApps = new Set<App>()
  const router: Router = {
    currentRoute,
    addRoute,
    beforeEach: beforeGuards.add,
    beforeResolve: beforeResolveGuards.add,
    afterEach: afterGuards.add,
    install(app: App) { /* 如下 */}
  }
  return router
}
```

## 添加匹配 createRouterMatcher/addRoute

```ts
function addRoute(
  record: RouteRecordRaw,
  parent?: RouteRecordMatcher,
  originalRecord?: RouteRecordMatcher
) {
  const mainNormalizedRecord = normalizeRouteRecord(record) {
    return {
      path: record.path,
      redirect: record.redirect,
      name: record.name,
      meta: record.meta || {},
      ...
    }
  }
  const normalizedRecords = [mainNormalizedRecord]
  let matcher: RouteRecordMatcher
  for (const normalizedRecord of normalizedRecords) {
    matcher = createRouteRecordMatcher(normalizedRecord, parent, options) {
      // normalizedRecord => record
      const parser = tokensToParser(tokenizePath(record.path), options)
      const matcher: RouteRecordMatcher = assign(parser, {
        record,
        parent,
        children: [],
        alias: [],
      })
      return matcher
    }
    if (mainNormalizedRecord.children) {
      const children = mainNormalizedRecord.children
      for (let i = 0; i < children.length; i++) {
        addRoute(
          children[i],
          matcher,
          originalRecord && originalRecord.children[i]
        )
      }
    }
    originalRecord = originalRecord || matcher
    if(...) {
      insertMatcher(matcher) {
        let i = 0
        while(i < matchers.length && ...) {
          i++
        }
        matchers.splice(i, 0, matcher)
        // 缓存记录
        if (matcher.record.name && !isAliasRecord(matcher)) {
          matcherMap.set(matcher.record.name, matcher)
        }
      }
    }
  }
}
```

## 触发导航、执行守卫钩子 navigate

`push` => `pushWithRedirect` => `navigate`

::: tip 钩子分为 3 种：

1. 全局钩子：beforeEach、beforeResolve、afterEach
2. 配置文件内钩子：beforeEnter
3. 组件内钩子：beforeRouteEnter、beforeRouteUpdate、beforeRouteLeave

:::
[官网：守卫钩子](https://router.vuejs.org/zh/guide/advanced/navigation-guards.html)

```ts
function navigate(to, from) {
  let guards: Lazy<any>[];
  const [leavingRecords, updatingRecords, enteringRecords] =
    extractChangingRecords(to, from);
  // 提取 组件内钩子 beforeRouteLeave
  guards = extractComponentsGuards(
    leavingRecords.reverse(),
    "beforeRouteLeave",
    to,
    from
  );
  // 提取 setup 状态下钩子 beforeRouteLeave
  for (const record of leavingRecords) {
    record.leaveGuards.forEach((guard) => {
      guards.push(guardToPromiseFn(guard, to, from));
    });
  }
  // 导航取消确认 机制
  const canceledNavigationCheck = checkCanceledNavigationAndReject.bind(
    null,
    to,
    from
  );
  guards.push(canceledNavigationCheck);
  return runGuardQueue(guards)
    .then(() => {
      // 解析、执行 全局钩子 beforeEach
      guards = [];
      for (const guard of beforeGuards.list()) {
        guards.push(guardToPromiseFn(guard, to, from));
      }
      guards.push(canceledNavigationCheck);
      return runGuardQueue(guards);
    })
    .then(() => {
      // 解析、执行重用 组件内钩子 beforeRouteUpdate
    })
    .then(() => {
      // 解析、执行 配置文件钩子 beforeEnter
      guards = [];
      for (const record of to.matched) {
        if (record.beforeEnter && !from.matched.includes(record)) {
          if (isArray(record.beforeEnter)) {
            for (const beforeEnter of record.beforeEnter)
              guards.push(guardToPromiseFn(beforeEnter, to, from));
          } else {
            guards.push(guardToPromiseFn(record.beforeEnter, to, from));
          }
        }
      }
      guards.push(canceledNavigationCheck);
      return runGuardQueue(guards);
    })
    .then(() => {
      // 解析、执行 组件内钩子 beforeRouteEnter
    })
    .then(() => {
      // 解析、执行 全局钩子 beforeResolve
    });
}
function runGuardQueue(guards: Lazy<any>[]): Promise<void> {
  return guards.reduce(
    (promise, guard) => promise.then(() => guard()),
    Promise.resolve()
  );
}
```

## 辅助信息集锦

```ts
export const START_LOCATION_NORMALIZED: RouteLocationNormalizedLoaded = {
  path: "/",
  name: undefined,
  params: {},
  query: {},
  hash: "",
  fullPath: "/",
  matched: [],
  meta: {},
  redirectedFrom: undefined,
};
```
