---
title: 渲染器(render)
---
## 责任重大的Render
:::tip
所谓渲染器，简单的说就是将 Virtual DOM 渲染成特定平台下真实 DOM 的工具(就是一个函数，通常叫 render)。 
:::   
1. 控制部分组件生命周期钩子的调用
2. 多端渲染的桥梁
3. 与异步渲染有直接关系
4. 包含最核心的 ```Diff``` 算法    
Diff 算法是渲染器的核心特性之一，可以说正是 Diff 算法的存在才使得 Virtual DOM 如此成功。

### 渲染器的工作流程分为两个阶段：```mount``` 和 ```patch```。
1. 如果旧的 VNode 存在，则会使用新的 VNode 与旧的 VNode 进行对比，试图以最小的资源开销完成 DOM 的更新，这个过程就叫 patch，或“打补丁”。
2. 如果旧的 VNode 不存在，则直接将新的 VNode 挂载成全新的 DOM，这个过程叫做 mount。

### function render(vnode, container)
通常渲染器接收两个参数，第一个参数是将要被渲染的 VNode 对象，第二个参数是一个用来承载内容的容器(container)，通常也叫挂载点，如下代码所示：
```js {6,13,18}
function render(vnode, container) {
  const prevVNode = container.vnode
  if (prevVNode == null) {
    if (vnode) {
      // 没有旧的 VNode，只有新的 VNode。使用 `mount` 函数挂载全新的 VNode
      mount(vnode, container)
      // 将新的 VNode 添加到 container.vnode 属性下，这样下一次渲染时旧的 VNode 就存在了
      container.vnode = vnode
    }
  } else {
    if (vnode) {
      // 有旧的 VNode，也有新的 VNode。则调用 `patch` 函数打补丁
      patch(prevVNode, vnode, container)
      // 更新 container.vnode
      container.vnode = vnode
    } else {
      // 有旧的 VNode 但是没有新的 VNode，这说明应该移除 DOM，在浏览器中可以使用 removeChild 函数。
      container.removeChild(prevVNode.el)
      container.vnode = null
    }
  }
}
```
## 挂载mount
挂载，本质上就是将各种类型的 VNode 渲染成真实DOM的过程。
#### 挂载普通元素
巴拉巴拉小魔仙。。。

## 更新patch
渲染器除了将全新的 VNode 挂载```mount```成真实DOM之外，它的另外一个职责是负责对新旧 VNode 进行比对，并以合适的方式更新DOM，也就是我们常说的 patch。

更新的本质就是，对比新旧节点，即```diff```。对比首先就是类型对比，如果类型不同，直接替换。
