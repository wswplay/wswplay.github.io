---
title: 计算属性
---

## 源码

```js
function computed(getterOrOptions, debugOptions) {
  debugger;
  let getter;
  let setter;
  const onlyGetter = isFunction(getterOrOptions);
  if (onlyGetter) {
    getter = getterOrOptions;
    setter =
      process.env.NODE_ENV !== "production"
        ? () => {
            console.warn("Write operation failed: computed value is readonly");
          }
        : NOOP;
  } else {
    getter = getterOrOptions.get;
    setter = getterOrOptions.set;
  }
  const cRef = new ComputedRefImpl(getter, setter, onlyGetter || !setter);
  if (process.env.NODE_ENV !== "production" && debugOptions) {
    cRef.effect.onTrack = debugOptions.onTrack;
    cRef.effect.onTrigger = debugOptions.onTrigger;
  }
  return cRef;
}
class ComputedRefImpl {
  constructor(getter, _setter, isReadonly) {
    this._setter = _setter;
    this.dep = undefined;
    this._dirty = true;
    this.__v_isRef = true;
    this.effect = new ReactiveEffect(getter, () => {
      if (!this._dirty) {
        this._dirty = true;
        triggerRefValue(this);
      }
    });
    this["__v_isReadonly" /* IS_READONLY */] = isReadonly;
  }
  get value() {
    // the computed ref may get wrapped by other proxies e.g. readonly() #3376
    const self = toRaw(this);
    trackRefValue(self);
    if (self._dirty) {
      self._dirty = false;
      self._value = self.effect.run();
    }
    return self._value;
  }
  set value(newValue) {
    this._setter(newValue);
  }
}
```

## 示例

```vue
<template>
  <div class="title" @click="changeTitle">{{ newTitle }}</div>
</template>

<script>
import { computed, ref } from "@vue/reactivity";

export default {
  name: "App",
  data() {
    return {
      miao: "喵~",
    };
  },
  setup() {
    // data
    let title = ref("独钓寒江雪");
    let newTitle = computed(() => title.value + 2);

    // function
    function changeTitle() {
      title.value = "岩上无心云相逐";
    }

    // result
    return {
      title,
      changeTitle,
      newTitle,
    };
  },
};
</script>
```
:::tip 步骤解析
**首次渲染 init：**   
1、执行setup函数，初始化ref和computed，得出setupState结果；   
2、执行渲染render函数，访问数据。首选访问newTitle，触发其getter，将模板render的effect push进dep;   
3、然后执行```self.effect.run()```,    
  ```effectStack.push((activeEffect = this))```,将activeEffect设置成计算属性的effect,   
  ```return this.fn();```,执行表达式```() => title.value + 2```,   
  触发title ref getter，**由于此时的activeEffect是newTitle的effect**，track将其加入到title的依赖dep中。   
4、```patch(null, subTree, ...）```进入、并完成渲染。 

**更新渲染 update**   
click，触发title setter -> triggerRefValue -> triggerEffects   
  title dep中的effect，现在只有newTitle的effect，   
  执行```effect.scheduler()```,即执行```triggerRefValue(this)```,   
  newTitle的effect dep中有模板render effect,   
  执行```effect.scheduler()```,即执行```queueJob(instance.update)```。  
:::