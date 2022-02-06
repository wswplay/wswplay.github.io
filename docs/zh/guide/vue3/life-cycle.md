---
title: 生命周期
---

## 源码摘要(例 onMounted)

```js
// 定义
const onMounted = createHook("m" /* MOUNTED */);
const createHook = (lifecycle)
      => (hook, target = currentInstance)
      => injectHook(lifecycle, hook, target);
function injectHook(type, hook, target = currentInstance, prepend = false) {
  if (target) {
    const hooks = target[type] || (target[type] = []);
    const wrappedHook =
      hook.__weh ||
      (hook.__weh = (...args) => {
        debugger;
        if (target.isUnmounted) {
          return;
        }
        pauseTracking();
        setCurrentInstance(target);
        const res = callWithAsyncErrorHandling(hook, target, type, args);
        unsetCurrentInstance();
        resetTracking();
        return res;
      });
    if (prepend) {
      hooks.unshift(wrappedHook);
    } else {
      hooks.push(wrappedHook);
    }
    return wrappedHook;
  }
}
// 加入队列
const setupRenderEffect = (...args) => {
  const componentUpdateFn = () => {
    // mounted hook
    if (m) {
      queuePostRenderEffect(m, parentSuspense);
    }
  }
}
// 触发
const render = (...args) => {
  if (vnode == null) {
    if (container._vnode) {
      unmount(container._vnode, null, null, true);
    }
  }
  else {
    patch(container._vnode || null, vnode, container, null, null, null, isSVG);
  }
  // 触发onMounted
  flushPostFlushCbs();
}
```

钩子函数，是一个数组，也就可以多次注册同一个钩子。
```js
// is ok
setup() {
  onMounted(() => {
    console.log("onMounted");
  });
  onMounted(() => {
    console.log("onMounted---2222");
  });
},
```
组件渲染结束，就触发 onMounted 钩子。

## Vue2 和 Vue3

```js
// Vue.js 2.x 定义生命周期钩子函数
export default {
  created() {
    // 做一些初始化工作
  },
  mounted() {
    // 可以拿到 DOM 节点
  },
  beforeDestroy() {
    // 做一些清理操作
  },
};
//  Vue.js 3.x 生命周期 API 改写上例
import { onMounted, onBeforeUnmount } from "vue";
export default {
  setup() {
    // 做一些初始化工作

    onMounted(() => {
      // 可以拿到 DOM 节点
    });
    onBeforeUnmount(() => {
      // 做一些清理操作
    });
  },
};
```

可以看到，在 Vue.js 3.0 中，setup 函数已经替代了 Vue.js 2.x 的 beforeCreate 和 created 钩子函数，我们可以在 setup 函数做一些初始化工作，比如发送一个异步 Ajax 请求获取数据。

我们用 onMounted API 替代了 Vue.js 2.x 的 mounted 钩子函数，用 onBeforeUnmount API 替代了 Vue.js 2.x 的 beforeDestroy 钩子函数。

其实，Vue.js 3.0 针对 Vue.js 2.x 的生命周期钩子函数做了全面替换，映射关系如下：

```js
beforeCreate -> 使用 setup()
created -> 使用 use setup()
beforeMount -> onBeforeMount
mounted -> onMounted
beforeUpdate -> onBeforeUpdate
updated -> onUpdated
beforeDestroy-> onBeforeUnmount
destroyed -> onUnmounted
activated -> onActivated
deactivated -> onDeactivated
errorCaptured -> onErrorCaptured
```

此外，Vue.js 3.0 还新增了两个用于调试的生命周期 API：onRenderTracked 和 onRenderTriggered。
