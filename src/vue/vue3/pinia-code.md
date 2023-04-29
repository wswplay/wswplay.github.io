---
title: Pinia源码分析
---

# Pinia 源码摘要

`Pinia` (发音为 `/piːnjʌ/`，类似英文中的 “`peenya`”) 是最接近有效包名 `piña` (西班牙语中的 `pineapple`，即“菠萝”) 的词。 菠萝花实际上是一组各自独立的花朵，它们结合在一起，由此形成一个多重的水果。 它(菠萝)也是一种原产于南美洲的美味热带水果。

与 `Store` 类似，每一个都是**独立诞生**，但最终**又相互联系**。

## 创建 pinia 实例

```ts
export let activePinia: Pinia | undefined
createPinia() {
  let _p: Pinia['_p'] = []
  let toBeInstalled: PiniaPlugin[] = []
  const pinia: Pinia = markRaw({
    // 将pinia安装到Vue实例
    install(app: App) {
      setActivePinia(pinia) {
        (pinia) => (activePinia = pinia)
      }
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
