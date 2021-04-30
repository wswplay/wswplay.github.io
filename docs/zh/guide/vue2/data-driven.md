---
title: 数据驱动
---
每个人都有他的脾气，每一行代码都有它存在的意义，每一种脾气都有意义。

## 灵魂问题
1. 如何定义非响应式数据？

:::tip 特点
与jQuery相比，Vue的最大特点就是：数据驱动、组件化。
:::
Vue是怎么实现数据驱动，并自动化的更新视图的呢？
## Object.defineProperty
```js {8,10,36}
var childOb = !shallow && observe(val);
Object.defineProperty(obj, key, {
  enumerable: true,
  configurable: true,
  get: function reactiveGetter () {
    var value = getter ? getter.call(obj) : val;
    if (Dep.target) {
      dep.depend(); // 收集依赖
      if (childOb) {
        childOb.dep.depend(); // 递归收集依赖
        if (Array.isArray(value)) {
          dependArray(value);
        }
      }
    }
    return value
  },
  set: function reactiveSetter (newVal) {
    var value = getter ? getter.call(obj) : val;
    /* eslint-disable no-self-compare */
    if (newVal === value || (newVal !== newVal && value !== value)) {
      return
    }
    /* eslint-enable no-self-compare */
    if (process.env.NODE_ENV !== 'production' && customSetter) {
      customSetter();
    }
    // #7981: for accessor properties without setter
    if (getter && !setter) { return }
    if (setter) {
      setter.call(obj, newVal);
    } else {
      val = newVal;
    }
    childOb = !shallow && observe(newVal);
    dep.notify(); // 派发更新
  }
});
```
通过```Object.defineProperty```把数据```data```转化成```getter```、```setter```。    
当取值```getter```时，收集依赖。也就是把这个值的观察者，收集到一个队列中(数组)。        
当赋值```setter```时，派发更新。通知数据的每一个观察者，去更新相关联的地方。

另外这里有一个优先级的关系：props优先级 > data优先级 > methods优先级。

## Proxy数据代理
```js
const sharedPropertyDefinition = {
  enumerable: true,
  configurable: true,
  get: noop,
  set: noop
}
export function proxy (target: Object, sourceKey: string, key: string) {
  sharedPropertyDefinition.get = function proxyGetter () {
    return this[sourceKey][key]
  }
  sharedPropertyDefinition.set = function proxySetter (val) {
    this[sourceKey][key] = val
  }
  Object.defineProperty(target, key, sharedPropertyDefinition)
}
```
