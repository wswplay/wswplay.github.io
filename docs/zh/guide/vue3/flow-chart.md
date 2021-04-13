---
title: 流程图
---
```js
render(_ctx, _cache) {
  return Object(vue__WEBPACK_IMPORTED_MODULE_0__["openBlock"])(), Object(vue__WEBPACK_IMPORTED_MODULE_0__["createBlock"])("div", _hoisted_1, [_hoisted_2]);
}
function createBlock(type, props, children, patchFlag, dynamicProps) {
  const vnode = createVNode(type, props, children, patchFlag, dynamicProps, true /* isBlock: prevent a block from tracking itself */);
  // save current block children on the block vnode
  vnode.dynamicChildren = currentBlock || EMPTY_ARR;
  // close block
  closeBlock();
  // a block is always going to be patched, so track it as a child of its
  // parent block
  if (shouldTrack > 0 && currentBlock) {
      currentBlock.push(vnode);
  }
  return vnode;
}
```
```js
// node_modules/@vue/runtime-dom/dist/runtime-dom.esm-bundler.js
const createApp = ((...args) => {
  const app = ensureRenderer().createApp(...args);
  app.mount = (containerOrSelector) => {}
  return app
}
function ensureRenderer() {
  return renderer || (renderer = createRenderer(rendererOptions));
}
```
```js
// node_modules/@vue/runtime-core/dist/runtime-core.esm-bundler.js
function createRenderer(options) {
  return baseCreateRenderer(options);
}
function baseCreateRenderer(options, createHydrationFns) {
  const render = (vnode, container, isSVG) => {}
  return {
    render,
    hydrate,
    createApp: createAppAPI(render, hydrate)
  };
}
function createAppAPI(render, hydrate) {
  return function createApp(rootComponent, rootProps = null) {
    const context = createAppContext();
    const installedPlugins = new Set();
    let isMounted = false;
    const app = (context.app = {
      _uid: uid++,
      _component: rootComponent,
      _props: rootProps,
      _container: null,
      _context: context,
      version,
      get config() {
        return context.config;
      },
      set config(v) {
        if ((process.env.NODE_ENV !== 'production')) {
          warn(`app.config cannot be replaced. Modify individual options instead.`);
        }
      },
      use(plugin, ...options) {},
      mixin(mixin) {},
      component(name, component) {},
      directive(name, directive) {},
      mount(rootContainer, isHydrate, isSVG) {},
      unmount() {},

    })
    return app
  }
}
```
app.use
```js
use(plugin, ...options) {
  if (installedPlugins.has(plugin)) {
    (process.env.NODE_ENV !== 'production') && warn(`Plugin has already been applied to target app.`);
  }
  else if (plugin && isFunction(plugin.install)) {
    installedPlugins.add(plugin);
    plugin.install(app, ...options);
  }
  else if (isFunction(plugin)) {
    installedPlugins.add(plugin);
    plugin(app, ...options);
  }
  else if ((process.env.NODE_ENV !== 'production')) {
    warn(`A plugin must either be a function or an object with an "install" ` +
          `function.`);
  }
  return app;
},
```
app.mount
```js
mount(rootContainer, isHydrate, isSVG) {
  if (!isMounted) {
    const vnode = createVNode(rootComponent, rootProps);
    // store app context on the root VNode.
    // this will be set on the root instance on initial mount.
    vnode.appContext = context;
    // HMR root reload
    if ((process.env.NODE_ENV !== 'production')) {
      context.reload = () => {
        render(cloneVNode(vnode), rootContainer, isSVG);
      };
    }
    if (isHydrate && hydrate) {
      hydrate(vnode, rootContainer);
    }
    else {
      render(vnode, rootContainer, isSVG);
    }
    isMounted = true;
    app._container = rootContainer;
    rootContainer.__vue_app__ = app;
    if ((process.env.NODE_ENV !== 'production') || __VUE_PROD_DEVTOOLS__) {
      devtoolsInitApp(app, version);
    }
    return vnode.component.proxy;
  }
  else if ((process.env.NODE_ENV !== 'production')) {
    warn(`App has already been mounted.\n` +
      `If you want to remount the same app, move your app creation logic ` +
      `into a factory function and create fresh app instances for each ` +
      `mount - e.g. \`const createMyApp = () => createApp(App)\``);
  }
},
const createVNode = ((process.env.NODE_ENV !== 'production')
    ? createVNodeWithArgsTransform
    : _createVNode);

const createVNodeWithArgsTransform = (...args) => {
    return _createVNode(...(vnodeArgsTransformer
        ? vnodeArgsTransformer(args, currentRenderingInstance)
        : args));
};
function _createVNode(type, props = null, children = null, patchFlag = 0, dynamicProps = null, isBlockNode = false) {
  const vnode = {}
  return vnode
}
```
render
```js
const render = (vnode, container, isSVG) => {
  if (vnode == null) {
    if (container._vnode) {
      unmount(container._vnode, null, null, true);
    }
  }
  else {
    patch(container._vnode || null, vnode, container, null, null, null, isSVG);
  }
  flushPostFlushCbs();
  container._vnode = vnode;
};
const patch = (n1, n2, container, anchor = null, parentComponent = null, parentSuspense = null, isSVG = false, slotScopeIds = null, optimized = false) => {

}
```
processComponent    
mountComponent

```js
mountComponent = (initialVNode, container, anchor, parentComponent, parentSuspense, isSVG, optimized) => {
  const instance = (initialVNode.component = createComponentInstance(initialVNode, parentComponent, parentSuspense));
  setupComponent(instance);
  setupRenderEffect(instance, initialVNode, container, anchor, parentSuspense, isSVG, optimized);
}
```

setupComponent    
setupRenderEffect    
effect  node_modules/@vue/reactivity/dist/reactivity.esm-bundler.js    
```js
function effect(fn, options = EMPTY_OBJ) {
  if (isEffect(fn)) {
    fn = fn.raw;
  }
  const effect = createReactiveEffect(fn, options);
  if (!options.lazy) {
    effect();
  }
  return effect;
}
function createReactiveEffect(fn, options) {
  const effect = function reactiveEffect() {
    if (!effect.active) {
      return options.scheduler ? undefined : fn();
    }
    if (!effectStack.includes(effect)) {
      cleanup(effect);
      try {
        enableTracking();
        effectStack.push(effect);
        activeEffect = effect;
        return fn();
      }
      finally {
        effectStack.pop();
        resetTracking();
        activeEffect = effectStack[effectStack.length - 1];
      }
    }
  };
  effect.id = uid++;
  effect.allowRecurse = !!options.allowRecurse;
  effect._isEffect = true;
  effect.active = true;
  effect.raw = fn;
  effect.deps = [];
  effect.options = options;
  return effect;
}
```
componentEffect    
renderComponentRoot    
render    


