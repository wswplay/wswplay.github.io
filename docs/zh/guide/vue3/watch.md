---
title: watch
---
## watch 实现原理
```js
function watch(source, cb, options) { 
  if ((process.env.NODE_ENV !== 'production') && !isFunction(cb)) { 
    warn(`\`watch(fn, options?)\` signature has been moved to a separate API. ` + 
      `Use \`watchEffect(fn, options?)\` instead. \`watch\` now only ` + 
      `supports \`watch(source, cb, options?) signature.`) 
  } 
  return doWatch(source, cb, options) 
} 
function doWatch(source, cb, { immediate, deep, flush, onTrack, onTrigger } = EMPTY_OBJ) { 
  // 标准化 source 
  // 构造 applyCb 回调函数 
  // 创建 scheduler 时序执行函数 
  // 创建 effect 副作用函数 
  // 返回侦听器销毁函数 
}
```
### 标准化 source
### 构造回调函数
### 创建 scheduler
### 创建 effect
### 异步任务队列的设计
