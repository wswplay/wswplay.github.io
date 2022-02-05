---
title: watch
---

## 源码摘要

```js
function watch(source, cb, options) {
  if (process.env.NODE_ENV !== "production" && !isFunction(cb)) {
    warn(
      `\`watch(fn, options?)\` signature has been moved to a separate API. ` +
        `Use \`watchEffect(fn, options?)\` instead. \`watch\` now only ` +
        `supports \`watch(source, cb, options?) signature.`
    );
  }
  return doWatch(source, cb, options);
}
function doWatch(
  source,
  cb,
  { immediate, deep, flush, onTrack, onTrigger } = EMPTY_OBJ
) {
  if (isRef(source)) {
    getter = () => source.value;
    forceTrigger = !!source._shallow;
  }
  const job = () => {
    if (!effect.active) {
      return;
    }
    if (cb) {
      // watch(source, cb)
      const newValue = effect.run();
      if (
        deep ||
        forceTrigger ||
        (isMultiSource
          ? newValue.some((v, i) => hasChanged(v, oldValue[i]))
          : hasChanged(newValue, oldValue)) ||
        false
      ) {
        // cleanup before running cb again
        if (cleanup) {
          cleanup();
        }
        callWithAsyncErrorHandling(cb, instance, 3 /* WATCH_CALLBACK */, [
          newValue,
          // pass undefined as the old value when it's changed for the first time
          oldValue === INITIAL_WATCHER_VALUE ? undefined : oldValue,
          onInvalidate,
        ]);
        oldValue = newValue;
      }
    } else {
      // watchEffect
      effect.run();
    }
  };
  if() {
  // sth
  } else {
    // default: 'pre'
    scheduler = () => {
      if (!instance || instance.isMounted) {
        queuePreFlushCb(job);
      }
      else {
        // with 'pre' option, the first call must happen before
        // the component is mounted so it is called synchronously.
        job();
      }
    };
  }
}
```

## 示例解析

```vue
<template>
  <div class="title" @click="changeTitle">{{ newTitle }}</div>
</template>

<script>
import { ref } from "@vue/reactivity";
import { watch } from "@vue/runtime-core";

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
    let newTitle = ref("000");

    watch(title, () => {
      newTitle.value = "999";
    });

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
1、执行setup函数。初始化ref，得到setupState结果；   
2、执行watch，创建watch effect、job、scheduler;
  运行```oldValue = effect.run()```求ref title初值，此时的fn为```() => source.value```，可将watch effect push到dep中；   
3、render时，ref newTitle getter将render effect加入到dep中；

**更新渲染 update**   
click事件，触发ref title setter，执行```effect.scheduler()```,   
  即执行watch effect scheduler ```queuePreFlushCb(job)```,      
  即执行```job```函数， 执行```newValue = effect.run()```计算新值,   
  然后执行watch的回调函数```callWithAsyncErrorHandling(cb, instance, ...args)```,   
  触发ref newTitle setter，执行dep中render ```effect.scheduler()```,   
最后，就是千篇一律的render了啦！
:::
