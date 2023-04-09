---
title: Vue3.0源码分析
outline: deep
---

# Vue3.0 源码摘要

## 流程函数谱系集锦

### 创建 app：createApp()

```ts
createApp(...args) {
  // 创建app实例
  const app = ensureRenderer() {
    return renderer || renderer = createRenderer(rendererOptions) {
      return baseCreateRenderer(options) {
        const render = () => {}
        return {
          render,
          createApp: createAppAPI(render, hydrate) {
            return function createApp(rootComponent, rootProps = null) {
              const context = createAppContext() {
                return { app, config: {}, mixins = [], components: {}, ...}
              }
              let isMounted = false
              const app = context.app = {
                _uid: uid++,
                _component: rootComponent as ConcreteComponent,
                _props: rootProps,
                version,
                use(plugin: Plugin, ...options: any[]) {},
                mixin(mixin: ComponentOptions) {}
                component(name: string, component?: Component) {}
                directive(name: string, directive?: Directive) {}
                mount(container, false, container instanceof SVGElement) {
                  if (!isMounted) {
                    const vnode = createVNode(rootComponent as ConcreteComponent, rootProps) {}
                    createVNode = _createVNode(type, props, children) {
                      const shapeFlag = isString(type) ? ... : ...
                      return createBaseVNode(type, props, children, shapeFlag) {
                        const vnode = {
                          __v_isVNode: true,
                          __v_skip: true,
                          type,
                          props,
                          children,
                          shapeFlag,
                        }
                        if (needFullChildrenNormalization) {
                          normalizeChildren(vnode, children)
                        }
                        return vnode
                      }
                    }
                    vnode.appContext = context
                    if (isHydrate && hydrate) {
                      // sth
                    } else {
                      render(vnode, rootContainer = container, isSVG) {
                        if (vnode == null) {
                          if (container._vnode) {
                            unmount(container._vnode, null, null, true)
                          }
                        } else {
                          // 详情见下面的 patch 函数
                          patch(container._vnode || null, vnode, container, ...)
                        }
                        flushPreFlushCbs()
                        flushPostFlushCbs()
                        container._vnode = vnode
                      }
                    }
                    isMounted = true
                    app._container = rootContainer
                  }
                }
                unmount() {},
                provide(key, value) {}
              }
              return app
            }
          }
        }
      }
    }
  }.createApp(...args)
  // 缓存mount源函数
  const { mount } = app
  // 重写mount函数
  app.mount = (containerOrSelector: Element | ShadowRoot | string): any => {
    // 容器规范化
    const container = normalizeContainer(containerOrSelector)
    const component = app._component
    // 打扫清空容器内容
    container.innerHTML = ''
    const proxy = mount(container, false, container instanceof SVGElement)
    return proxy
  }

  return app
}.mount(container)
```

### 渲染 or 更新：patch()

```ts
patch(container._vnode || null, vnode, container, ...) {
  // patch(n1, n2, container)
  const { type, ref, shapeFlag } = n2
  switch (type) {
    case Text:
      processText(n1, n2, container, anchor) {
        if (n1 == null) {
          hostInsert((n2.el = hostCreateText(n2.children as string)), container, anchor)
        } else {
          const el = (n2.el = n1.el!)
          if (n2.children !== n1.children) {
            hostSetText(el, n2.children as string)
          }
        }
      }
      break
    ...
    default:
      if (shapeFlag & ShapeFlags.ELEMENT) {
        processElement(n1, n2, container) {
          if (n1 == null) {
            mountElement(n2, container) {
              // mountElement(vnode, container)
              let el: RendererElement
              const { type, props, shapeFlag, transition, dirs } = vnode
              el = vnode.el = hostCreateElement(vnode.type as string)
              if (shapeFlag & ShapeFlags.TEXT_CHILDREN) {
                hostSetElementText(el, vnode.children as string)
              } else if (shapeFlag & ShapeFlags.ARRAY_CHILDREN) {
                mountChildren(vnode.children, el, optimized) {
                  // mountChildren(children, container, optimized)
                  for (let i = start; i < children.length; i++) {
                    const child = (children[i] =
                      optimized ? cloneIfMounted(children[i]) : normalizeVNode(children[i]))
                    patch(null, child, container)
                  }
                }
              }
              hostInsert(el, container)
            }
          } else {
            patchElement(n1, n2)
          }
        }
      } else if(shapeFlag & ShapeFlags.COMPONENT) {
        processComponent(n1, n2, container) {
          if (n1 == null) {
            mountComponent(n2, container) {
              // mountComponent(initialVNode, container)
              // 创建组件实例
              const instance = initialVNode.component = createComponentInstance(initialVNode) {
                // createComponentInstance(vnode, parent)
                const type = vnode.type
                const instance = {
                  uid: uid++,
                  vnode,
                  type,
                  parent,
                  isMounted: false
                }
                if (__DEV__) {
                  instance.ctx = createDevRenderContext(instance)
                } else {
                  instance.ctx = { _: instance }
                }
                instance.root = parent ? parent.root : instance
                instance.emit = emit.bind(null, instance)
                return instance
              }
              // 执行、设置setup
              setupComponent(instance) {
                const { props, children } = instance.vnode
                const isStateful = isStatefulComponent(instance)
                initProps(instance, props, isStateful, isSSR)
                initSlots(instance, children)
                const setupResult = isStateful && setupStatefulComponent(instance, isSSR) {
                  const Component = instance.type as ComponentOptions
                  const { setup } = Component
                  if (setup) {
                    createSetupContext(instance)
                    setCurrentInstance(instance)
                  } else {
                    finishComponentSetup(instance, isSSR)
                  }
                }
                return setupResult
              }
              // 设置响应式渲染机制
              setupRenderEffect(instance, initialVNode, container) {
                const componentUpdateFn = () => {
                  if (!instance.isMounted) {
                    // beforeMount 钩子
                    if (bm) invokeArrayFns(bm)
                    if (el && hydrateNode) {
                      // sth
                    } else {
                      // 构建子vnode
                      const subTree = (instance.subTree = renderComponentRoot(instance)) {
                        const { type: Component, vnode, render } = instance
                        let result
                        // 设置当前渲染实例
                        const prev = setCurrentRenderingInstance(instance)
                        try {
                          if (vnode.shapeFlag & ShapeFlags.STATEFUL_COMPONENT) {
                            const proxyToUse = withProxy || proxy
                            result = normalizeVNode(render!.call(proxyToUse, ...)) {
                              // normalizeVNode(child: VNodeChild)
                              if (child == null || typeof child === 'boolean') {
                                // empty placeholder
                                return createVNode(Comment)
                              } else if (isArray(child)) {
                                // fragment
                                return createVNode(Fragment, null, child.slice()
                                )
                              } else if (typeof child === 'object') {
                                return cloneIfMounted(child)
                              } else {
                                // 字符串或数字
                                return createVNode(Text, null, String(child))
                              }
                            }
                          } else {
                            // functional 函数式组件
                          }
                        }
                        setCurrentRenderingInstance(prev)
                        return result
                      }
                      patch(null, subTree, container)
                      initialVNode.el = subTree.el
                    }
                    // 执行 mounted 钩子
                    if(m) queuePostRenderEffect(m, parentSuspense)
                    // 标记实例为已挂载
                    instance.isMounted = true
                  } else {
                  }
                }
                // 为渲染器创建响应式效应
                const effect = (instance.effect = new ReactiveEffect({
                  componentUpdateFn,
                  () => queueJob(update),
                  instance.scope
                }))
                const update: SchedulerJob = (instance.update = () => effect.run())
                update.id = instance.uid
                update()
              }
            }
          }
        }
      }
  }
}
```

### 响应式效应 class ReactiveEffect

```ts
export let activeEffect: ReactiveEffect | undefined;
export class ReactiveEffect<T = any> {
  active = true;
  deps: Dep[] = [];
  constructor(
    public fn: () => T,
    public scheduler: EffectScheduler | null = null,
    scope?: EffectScope
  ) {
    recordEffectScope(this, scope);
  }
  run() {
    if (!this.active) return this.fn();
    let parent: ReactiveEffect | undefined = activeEffect;
    try {
      this.parent = activeEffect;
      activeEffect = this;
      shouldTrack = true;
      return this.fn();
    } finally {
    }
  }
}
```

### 响应式 reactive

```ts
export const reactiveMap = new WeakMap<Target, any>()
export function reactive(target: object) {
  if (isReadonly(target)) return target
  return createReactiveObject(target, false, mutableHandlers, collectionHandlers, reactiveMap) {
    // createReactiveObject(target, false, baseHandlers, collectionHandlers, proxyMap)
    const existingProxy = proxyMap.get(target)
    if (existingProxy) { return existingProxy }
    const proxy = new Proxy(
      target,
      argetType === TargetType.COLLECTION ? collectionHandlers : baseHandlers
    )
    proxyMap.set(target, proxy)
    return proxy
  }
}
```

## 其他集锦

### 枚举 vnode 类型标识（[位运算](/core/javascript/operators.html)）

```ts
export const enum ShapeFlags {
  ELEMENT = 1,
  FUNCTIONAL_COMPONENT = 1 << 1,
  STATEFUL_COMPONENT = 1 << 2,
  TEXT_CHILDREN = 1 << 3,
  ARRAY_CHILDREN = 1 << 4,
  SLOTS_CHILDREN = 1 << 5,
  TELEPORT = 1 << 6,
  SUSPENSE = 1 << 7,
  COMPONENT_SHOULD_KEEP_ALIVE = 1 << 8,
  COMPONENT_KEPT_ALIVE = 1 << 9,
  COMPONENT = ShapeFlags.STATEFUL_COMPONENT | ShapeFlags.FUNCTIONAL_COMPONENT,
}
export const mutableHandlers: ProxyHandler<object> = {
  get,
  set,
  deleteProperty,
  has,
  ownKeys,
};
export const collectionHandlers: ProxyHandler<CollectionTypes> = {
  get: /*#__PURE__*/ createInstrumentationGetter(false, false),
};
```
