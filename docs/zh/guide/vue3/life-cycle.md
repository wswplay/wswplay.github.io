---
title: 生命周期
---
## Vue2和Vue3
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
  } 
} 
//  Vue.js 3.x 生命周期 API 改写上例 
import { onMounted, onBeforeUnmount } from 'vue' 
export default { 
  setup() { 
    // 做一些初始化工作 

    onMounted(() => { 
      // 可以拿到 DOM 节点 
    }) 
    onBeforeUnmount(()=>{ 
      // 做一些清理操作 
    }) 
  } 
}
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

