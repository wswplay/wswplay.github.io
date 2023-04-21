---
title: Vue-Router源码分析
---

# Vue-Router 源码摘要

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
  const go = (delta: number) => routerHistory.go(delta)

  const router: Router = {
    currentRoute,
    addRoute,
    install(app: App) {}
  }
  return router
}
```

## createRouterMatcher/addRoute

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
