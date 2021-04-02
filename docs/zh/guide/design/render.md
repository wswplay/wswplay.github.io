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
```js
function mount(vnode, container) {
  const { flags } = vnode
  if (flags & VNodeFlags.ELEMENT) {
    // 挂载普通标签
    mountElement(vnode, container)
  } else if (flags & VNodeFlags.COMPONENT) {
    // 挂载组件
    mountComponent(vnode, container)
  } else if (flags & VNodeFlags.TEXT) {
    // 挂载纯文本
    mountText(vnode, container)
  } else if (flags & VNodeFlags.FRAGMENT) {
    // 挂载 Fragment
    mountFragment(vnode, container)
  } else if (flags & VNodeFlags.PORTAL) {
    // 挂载 Portal
    mountPortal(vnode, container)
  }
}
```
我们根据 VNode 的 flags 属性值能够区分一个 VNode 对象的类型，不同类型的 VNode 采用不同的挂载函数：
![](http://hcysun.me/vue-design/assets/img/flags-mount.4756a068.png)
### 挂载普通元素
巴拉巴拉小魔仙。。。
#### class的处理
:::danger
功能实现、应用层设计。很重要
:::
#### Attributes 和 DOM Properties
setAttribute设置非标准属性。
```js
const domPropsRE = /\[A-Z]|^(?:value|checked|selected|muted)$/
// ...
if (domPropsRE.test(key)) {
  // 当作 DOM Prop 处理
  el[key] = data[key]
} else {
  // 当作 Attr 处理
  el.setAttribute(key, data[key])
}
```
:::tip
一些特殊的 attribute，比如 checked/disabled 等，只要出现了，对应的 property 就会被初始化为 true，无论设置的值是什么,只有调用 removeAttribute 删除这个 attribute，对应的 property 才会变成 false。
:::
#### 事件处理
```html
<div @click="handler"></div>
```
```js
const elementVNode = h('div', {
  click: handler
})
// --------------------------------------
if (key[0] === 'o' && key[1] === 'n') {
  // 事件
  el.addEventListener(key.slice(2), data[key])
}
```
### 挂载文本节点
```js {11,20-24}
function mount(vnode, container, isSVG) {
  const { flags } = vnode
  if (flags & VNodeFlags.ELEMENT) {
    // 挂载普通标签
    mountElement(vnode, container, isSVG)
  } else if (flags & VNodeFlags.COMPONENT) {
    // 挂载组件
    mountComponent(vnode, container, isSVG)
  } else if (flags & VNodeFlags.TEXT) {
    // 挂载纯文本
    mountText(vnode, container)
  } else if (flags & VNodeFlags.FRAGMENT) {
    // 挂载 Fragment
    mountFragment(vnode, container, isSVG)
  } else if (flags & VNodeFlags.PORTAL) {
    // 挂载 Portal
    mountPortal(vnode, container)
  }
}
function mountText(vnode, container) {
  const el = document.createTextNode(vnode.children)
  vnode.el = el
  container.appendChild(el)
}
```
只需要调用 document.createTextNode 函数创建一个文本节点即可，然后将其添加到 container 中。

### 挂载 Fragment
```js {7,13,20}
function mountFragment(vnode, container, isSVG) {
  const { children, childFlags } = vnode
  switch (childFlags) {
    case ChildrenFlags.SINGLE_VNODE:
      mount(children, container, isSVG)
      // 单个子节点，就指向该节点
      vnode.el = children.el
      break
    case ChildrenFlags.NO_CHILDREN:
      const placeholder = createTextVNode('')
      mountText(placeholder, container)
      // 没有子节点指向占位的空文本节点
      vnode.el = placeholder.el
      break
    default:
      for (let i = 0; i < children.length; i++) {
        mount(children[i], container, isSVG)
      }
      // 多个子节点，指向第一个子节点
      vnode.el = children[0].el
  }
}
```

### 挂载 Portal


## 更新patch
渲染器除了将全新的 VNode 挂载```mount```成真实DOM之外，它的另外一个职责是负责对新旧 VNode 进行比对，并以合适的方式更新DOM，也就是我们常说的 patch。

更新的本质就是，对比新旧节点，即```diff```。对比首先就是类型对比，如果类型不同，直接替换。

## 核心 ```Diff``` 算法
1. 减小DOM操作的性能开销
2. 尽可能的复用 DOM 元素
#### 同层比较
当新旧 VNode 标签类型相同时，只需要更新 VNodeData 和 children 即可，不会“移除”和“新建”任何 DOM 元素的，而是复用已有 DOM 元素。
#### 有用的key
遍历比较(React)和双端比较(Vue2)

#### Vue3的 Diff 算法
在 Vue3 中将采用另外一种核心 Diff 算法，它借鉴于 ivi 和 inferno。