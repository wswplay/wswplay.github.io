---
title: 数据驱动
---
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
## Proxy
