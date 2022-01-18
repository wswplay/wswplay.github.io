---
title: 基础概念
---

## 代理：Proxy

Proxy 对象用于创建一个对象的代理，从而实现基本操作的拦截和自定义（如属性查找、赋值、枚举、函数调用等）。[MDN 定义](https://developer.mozilla.org/zh-CN/docs/Web/JavaScript/Reference/Global_Objects/Proxy)  
:::tip
`Proxy` 是一个对象，它包装了另一个对象，并允许你拦截对该对象的任何交互。

:::

## 拦截器：Reflect

Reflect 是一个内置的对象，它提供拦截 JavaScript 操作的方法。这些方法与 Proxy handlers 的方法相同。Reflect 的所有属性和方法都是静态的（就像 Math 对象）。[MDN](https://developer.mozilla.org/zh-CN/docs/Web/JavaScript/Reference/Global_Objects/Reflect)

```js {8,11,15}
const nanZhi = {
  id: "边城",
  address: "深圳",
};

const handler = {
  get(target, property) {
    return Reflect.get(...arguments);
  },
  set(target, property, value) {
    return Reflect.set(...arguments);
  },
};

const bianCheng = new Proxy(nanZhi, handler);

console.log(bianCheng.id); // 边城
console.log(bianCheng.nid); // undefined
bianCheng.id = "沈从文";
bianCheng.nid = "看过许多地方的云";
console.log(bianCheng.id); // 沈从文
console.log(bianCheng.nid); // 看过许多地方的云
```

## 创建 app 上下文 context

```js
function createAppContext() {
  return {
    app: null,
    config: {
      isNativeTag: NO,
      performance: false,
      globalProperties: {},
      optionMergeStrategies: {},
      errorHandler: undefined,
      warnHandler: undefined,
      compilerOptions: {},
    },
    mixins: [],
    components: {},
    directives: {},
    provides: Object.create(null),
    optionsCache: new WeakMap(),
    propsCache: new WeakMap(),
    emitsCache: new WeakMap(),
  };
}
```

## 创建 app 对象

:::warning
与大多数应用方法不同的是，mount 不返回应用本身。相反，它返回的是根组件实例。
:::

```js
const app = (context.app = {
  _uid: uid++,
  _component: rootComponent,
  _props: rootProps,
  _container: null,
  _context: context,
  _instance: null,
  version,
  get config() {
    return context.config;
  },
  set config(v) {
    if (process.env.NODE_ENV !== "production") {
      warn(`sth msg`);
    }
  },
  use(plugin, ...options) {
    // sth
    return app;
  },
  mixin(mixin) {
    // sth
    return app;
  },
  component(name, component) {
    // sth
    return app;
  },
  directive(name, directive) {
    // sth
    return app;
  },
  mount(rootContainer, isHydrate, isSVG) {
    if (!isMounted) {
      // sth
      return getExposeProxy(vnode.component) || vnode.component.proxy;
    } else if (process.env.NODE_ENV !== "production") {
      warn();
    }
  },
  unmount() {
    if (isMounted) {
      // sth
      delete app._container.__vue_app__;
    } else if (process.env.NODE_ENV !== "production") {
      warn(`Cannot unmount an app that is not mounted.`);
    }
  },
  provide(key, value) {
    // sth
    return app;
  },
});
return app;
```

## 创建 vnode

```js
function createBaseVNode(xxx) {
  const vnode = {
    __v_isVNode: true,
    __v_skip: true,
    type,
    props,
    key: props && normalizeKey(props),
    ref: props && normalizeRef(props),
    scopeId: currentScopeId,
    slotScopeIds: null,
    children,
    component: null,
    suspense: null,
    ssContent: null,
    ssFallback: null,
    dirs: null,
    transition: null,
    el: null,
    anchor: null,
    target: null,
    targetAnchor: null,
    staticCount: 0,
    shapeFlag,
    patchFlag,
    dynamicProps,
    dynamicChildren: null,
    appContext: null,
  };
  return vnode;
}
```

## 创建组件实例 instance

:::tip
每个组件将有自己的组件实例 vm。对于一些组件，如 TodoItem，在任何时候都可能有多个实例渲染。  
这个应用中的所有组件实例将`共享同一个应用实例`。
:::

```js
function createComponentInstance(vnode, parent, suspense) {
  const type = vnode.type;
  // inherit parent app context - or - if root, adopt from root vnode
  const appContext =
    (parent ? parent.appContext : vnode.appContext) || emptyAppContext;
  const instance = {
    uid: uid$1++,
    vnode,
    type,
    parent,
    appContext,
    root: null,
    next: null,
    subTree: null,
    effect: null,
    update: null,
    scope: new EffectScope(true /* detached */),
    render: null,
    proxy: null,
    exposed: null,
    exposeProxy: null,
    withProxy: null,
    provides: parent ? parent.provides : Object.create(appContext.provides),
    accessCache: null,
    renderCache: [],
    // local resovled assets
    components: null,
    directives: null,
    // resolved props and emits options
    propsOptions: normalizePropsOptions(type, appContext),
    emitsOptions: normalizeEmitsOptions(type, appContext),
    // emit
    emit: null,
    emitted: null,
    // props default value
    propsDefaults: EMPTY_OBJ,
    // inheritAttrs
    inheritAttrs: type.inheritAttrs,
    // state
    ctx: EMPTY_OBJ,
    data: EMPTY_OBJ,
    props: EMPTY_OBJ,
    attrs: EMPTY_OBJ,
    slots: EMPTY_OBJ,
    refs: EMPTY_OBJ,
    setupState: EMPTY_OBJ,
    setupContext: null,
    // suspense related
    suspense,
    suspenseId: suspense ? suspense.pendingId : 0,
    asyncDep: null,
    asyncResolved: false,
    // lifecycle hooks
    // not using enums here because it results in computed properties
    isMounted: false,
    isUnmounted: false,
    isDeactivated: false,
    bc: null,
    c: null,
    bm: null,
    m: null,
    bu: null,
    u: null,
    um: null,
    bum: null,
    da: null,
    a: null,
    rtg: null,
    rtc: null,
    ec: null,
    sp: null,
  };
  // sth
  return instance;
}
```

## 组件渲染 mountComponent

```js
const mountComponent = ( initialVNode, container, anchor, parentComponent,
  parentSuspense, isSVG, optimized
) => {
  const instance = (initialVNode.component = createComponentInstance(
    initialVNode, parentComponent, parentSuspense ));
  // ...
  setupComponent(instance);
  // ...
  setupRenderEffect( instance, initialVNode, container, anchor,
  parentSuspense, isSVG, optimized );
  // ...
};
```
