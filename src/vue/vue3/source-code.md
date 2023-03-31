---
title: Vue3.0源码分析
outline: deep
---

# Vue3.0 源码摘要

## 流程函数谱系集锦
### 创建app：createApp()

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
    // 格式化容器
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
### 补丁：patch()函数

```ts
patch(container._vnode || null, vnode, container, ...) {
  // patch(n1, n2, container)
  const { type, ref, shapeFlag } = n2
  switch (type) {
    case:
    ...
    default: 
      if(shapeFlag & ShapeFlags.COMPONENT) {
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
              setupRenderEffect(instance, initialVNode, container) {
                // 设置副作用函数
              }
            }
          }
        }
      }
  }
}
```
