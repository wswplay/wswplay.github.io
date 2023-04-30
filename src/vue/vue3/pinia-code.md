---
title: Pinia源码分析
---

# Pinia 源码摘要

`Pinia` (发音为 `/piːnjʌ/`，类似英文中的 “`peenya`”) 是最接近有效包名 `piña` (西班牙语中的 `pineapple`，即“菠萝”) 的词。 菠萝花实际上是一组各自独立的花朵，它们结合在一起，由此形成一个多重的水果。 它(菠萝)也是一种原产于南美洲的美味热带水果。

与 `Store` 类似，每一个都是**独立诞生**，但最终**又相互联系**。

## 创建 pinia 实例

```ts
export let activePinia: Pinia | undefined
export const setActivePinia: _SetActivePinia = (pinia) => (activePinia = pinia)
createPinia() {
  // 声明插件列表
  let _p: Pinia['_p'] = []
  let toBeInstalled: PiniaPlugin[] = []
  const pinia: Pinia = markRaw({
    // 将pinia安装到Vue实例
    install(app: App) {
      setActivePinia(pinia)
      if (!isVue2) {
        pinia._a = app
        app.provide(piniaSymbol, pinia)
        app.config.globalProperties.$pinia = pinia
        toBeInstalled.forEach((plugin) => _p.push(plugin))
        toBeInstalled = []
      }
    },
    // pinia插件安装
    use(plugin) {
      if (!this._a && !isVue2) {
        toBeInstalled.push(plugin)
      } else {
        _p.push(plugin)
      }
      return this
    },
    _p,
    state,
  })
  return pinia
}
```

## 定义 store

```ts
export function defineStore(
  // TODO: add proper types from above
  idOrOptions: any,
  setup?: any,
  setupOptions?: any
): StoreDefinition {
  let id: string;
  const isSetupStore = typeof setup === "function";
  if (typeof idOrOptions === "string") {
    id = idOrOptions;
    options = isSetupStore ? setupOptions : setup;
  } else {
    options = idOrOptions;
    id = idOrOptions.id;
  }
  function useStore(pinia?: Pinia | null, hot?: StoreGeneric): StoreGeneric {
    const currentInstance = getCurrentInstance();
    if (pinia) setActivePinia(pinia);
    pinia = activePinia!;
    if (!pinia._s.has(id)) {
      if (isSetupStore) {
        createSetupStore(id, setup, options, pinia) {
          // $id, setup, options, pinia, hot, isOptionsStore
          const initialState = pinia.state.value[$id] as UnwrapRef<S> | undefined
          const hotState = ref({} as S)
          const partialStore = {
            _p: pinia,
            $id,
            $patch,
            $reset,
            $subscribe,
            ...
          }
          const store: Store<Id, S, G, A> = reactive(partialStore)
          pinia._s.set($id, store)
          const setupStore = pinia._e.run(() => {
            scope = effectScope()
            return scope.run(() => setup())
          })!
          for (const key in setupStore) {}
          if (isVue2) {
            Object.keys(setupStore).forEach((key) => {
              set(store, key, setupStore[key])
            })
          } else {
            assign(store, setupStore)
            assign(toRaw(store), setupStore)
          }
          Object.defineProperty(store, '$state', {...})
          pinia._p.forEach()
          return store
        }
      } else {
        createOptionsStore(id, options as any, pinia) {
          const { state, actions, getters } = options
          const initialState: StateTree | undefined = pinia.state.value[id]
          let store: Store<Id, S, G, A>
          function setup() {
            if (!initialState && (!__DEV__ || !hot)) {
              if (isVue2) {
                set(pinia.state.value, id, state ? state() : {})
              } else {
                pinia.state.value[id] = state ? state() : {}
              }
            }
            const localState = toRefs(pinia.state.value[id])
            return assign(localState, actions, Object.keys(getters || {}).reduce(...))
          }
          store = createSetupStore(id, setup, options, pinia, hot, true)
          return store as any
        }
      }
    }
    const store: StoreGeneric = pinia._s.get(id)!;
    return store as any
  }
  useStore.$id = id;
  return useStore;
}
```

## 辅助信息集锦

```ts
const { assign } = Object;
```
