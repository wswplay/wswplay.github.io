---
title: 渲染器
---
## createApp()
```js {4,7,18}
// 初始化
import { createApp } from 'vue'
import App from './App.vue'
createApp(App).mount('#app')
```
```js
const createApp = ((...args) => {
  // 创建 app 对象
  const app = ensureRenderer().createApp(...args)
  const { mount } = app
  // 重写 mount 方法
  app.mount = (containerOrSelector) => {
    // ...
  }
  return app
})
```
createApp做了2件事情：
**1. 创建app对象**
**2. 重写 app.mount 方法**
## 创建app对象
```js
// 渲染相关的一些配置，比如更新属性的方法，操作 DOM 的方法
const rendererOptions = {
  patchProp,
  ...nodeOps
}
let renderer
// 延时创建渲染器，当用户只依赖响应式包的时候，可以通过 tree-shaking 移除核心渲染逻辑相关的代码
function ensureRenderer() {
  return renderer || (renderer = createRenderer(rendererOptions))
}
function createRenderer(options) {
  return baseCreateRenderer(options)
}
function baseCreateRenderer(options) {
  function render(vnode, container) {
    // 组件渲染的核心逻辑
  }
  return {
    render,
    createApp: createAppAPI(render)
  }
}
function createAppAPI(render) {
  // createApp createApp 方法接受的两个参数：根组件的对象和 prop
  return function createApp(rootComponent, rootProps = null) {
    const app = {
      _component: rootComponent,
      _props: rootProps,
      mount(rootContainer) {
        // 创建根组件的 vnode
        const vnode = createVNode(rootComponent, rootProps)
        // 利用渲染器渲染 vnode
        render(vnode, rootContainer)
        app._container = rootContainer
        return vnode.component.proxy
      }
    }
    return app
  }
}
```
## 重写 app.mount 方法
为什么要到外部重写？    
这是因为 Vue.js 不仅仅是为 Web 平台服务，它的目标是支持跨平台渲染，而 createApp 函数内部的 app.mount 方法是一个标准的可跨平台的组件渲染流程：
```js
mount(rootContainer) {
  // 创建根组件的 vnode
  const vnode = createVNode(rootComponent, rootProps)
  // 利用渲染器渲染 vnode
  render(vnode, rootContainer)
  app._container = rootContainer
  return vnode.component.proxy
}
```
接下来，我们再来看 app.mount 重写都做了哪些事情：
```js
app.mount = (containerOrSelector) => {
  // 标准化容器
  const container = normalizeContainer(containerOrSelector)
  if (!container)
    return
  const component = app._component
   // 如组件对象没有定义 render 函数和 template 模板，
   // 则取容器的 innerHTML 作为组件模板内容
  if (!isFunction(component) && !component.render && !component.template) {
    component.template = container.innerHTML
  }
  // 挂载前清空容器内容
  container.innerHTML = ''
  // 真正的挂载
  return mount(container)
}
```
## 核心渲染流程：创建 vnode 和渲染 vnode
### 创建 vnode
#### 普通元素
```html
<div class="bian" style="color:green">边城</div>
```
转化为vnode就是：
```js
const bianVNode = {
  type: 'div',
  props: {
    class: 'bian',
    style: {
      color: 'green'
    }
  },
  children: '边城'
}
```
children大部分情况是数组，由于字符串能直接表示，也可以为字符串。
#### 组件
```html
<bian-cheng title="边城" />
```
```js
const bianCom = {
  // 各种属性、方法等
}
const bianComVNode = {
  type: bianCom,  // 这里引用上面定义的实体
  props: {
    title: '边城'
  }
}
```
#### Vue怎么创建vnode
```js
const vnode = createVNode(rootComponent, rootProps)
```
比如，Vue通过createVNode创建了根组件的vnode。

### 渲染 vnode
```js
render(vnode, rootContainer)
const render = (vnode, container) => {
  if (vnode == null) {
    // 销毁组件
    if (container._vnode) {
      unmount(container._vnode, null, null, true)
    }
  } else {
    // 创建或者更新组件
    patch(container._vnode || null, vnode, container)
  }
  // 缓存 vnode 节点，表示已经渲染
  container._vnode = vnode
}
```
