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
:::tip
1. 组件通过h函数生成vnode，调用render函数，将vnode初次挂载(mount)真实的DOM。    
2. 借助响应式更新，当数据发生改变时，更新(patch)渲染DOM。
:::
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
...待续
### 挂载有状态组件和原理
挂载一个有状态组件只需要四步：
```js {19,27,40,47,54-63}
class MyComponent {
  render() {
    return h(
      'div',
      {
        style: {
          background: 'green'
        }
      },
      [
        h('span', null, '我是组件的标题1......'),
        h('span', null, '我是组件的标题2......')
      ]
    )
  }
}
//----------------------------------------------------------
const compVnode = h(MyComponent) // h 函数的第一个参数是组件类
render(compVnode, document.getElementById('app'))
//----------------------------------------------------------
function render(vnode, container) {
  const prevVNode = container.vnode
  if (prevVNode == null) {
    if (vnode) {
      // 没有旧的 VNode，只有新的 VNode。
      // 使用 `mount` 函数挂载全新的 VNode
      mount(vnode, container)
      // 将新的 VNode 添加到 container.vnode 属性下，
      // 这样下一次渲染时旧的 VNode 就存在了
      container.vnode = vnode
    }
  } else {
  // ...
}
//----------------------------------------------------------
function mount(vnode, container, isSVG) {
  // ...
  } else if (flags & VNodeFlags.COMPONENT) {
    // 挂载组件
    mountComponent(vnode, container, isSVG)
  } else if (flags & VNodeFlags.TEXT) {
  // ...
}
//----------------------------------------------------------
function mountComponent(vnode, container, isSVG) {
  if (vnode.flags & VNodeFlags.COMPONENT_STATEFUL) {
    mountStatefulComponent(vnode, container, isSVG)
  } else {
    mountFunctionalComponent(vnode, container, isSVG)
  }
}
//----------------------------------------------------------
// 挂载的4步
function mountStatefulComponent(vnode, container, isSVG) {
  // 创建组件实例
  const instance = new vnode.tag()
  // 渲染VNode
  instance.$vnode = instance.render()
  // 挂载
  mount(instance.$vnode, container, isSVG)
  // el 属性值 和 组件实例的 $el 属性都引用组件的根DOM元素
  instance.$el = vnode.el = instance.$vnode.el
}
```
如果组件的 render 返回的是一个片段(Fragment)，那么 instance.$el 和 vnode.el 引用的就是该片段的第一个DOM元素。

### 挂载函数式组件和原理
```js
function MyFunctionalComponent() {
  // 返回要渲染的内容描述，即 VNode
  return h(
    'div',
    {
      style: {
        background: 'green'
      }
    },
    [
      h('span', null, '我是组件的标题1......'),
      h('span', null, '我是组件的标题2......')
    ]
  )
}
// ...
function mountFunctionalComponent(vnode, container, isSVG) {
  // 获取 VNode
  const $vnode = vnode.tag()
  // 挂载
  mount($vnode, container, isSVG)
  // el 元素引用该组件的根元素
  vnode.el = $vnode.el
}
```

## 更新patch
渲染器除了将全新的 VNode 挂载```mount```成真实DOM之外，它的另外一个职责是负责对新旧 VNode 进行比对，并以合适的方式更新DOM，也就是我们常说的 patch。

更新的本质就是，对比新旧节点，即```diff```。对比首先就是类型对比，如果类型不同，直接替换。

### 基本原则和原理
组件的更新本质上还是对真实DOM的更新，或者说是对标签元素的更新，所以我们就优先来看一下如何更新一个标签元素。
```js
function patch(prevVNode, nextVNode, container) {
  // 分别拿到新旧 VNode 的类型，即 flags
  const nextFlags = nextVNode.flags
  const prevFlags = prevVNode.flags
  // 检查新旧 VNode 的类型是否相同，如果类型不同，则直接调用 replaceVNode 函数替换 VNode
  // 如果新旧 VNode 的类型相同，则根据不同的类型调用不同的比对函数
  if (prevFlags !== nextFlags) {
    replaceVNode(prevVNode, nextVNode, container)
  } else if (nextFlags & VNodeFlags.ELEMENT) {
    patchElement(prevVNode, nextVNode, container)
  } else if (nextFlags & VNodeFlags.COMPONENT) {
    patchComponent(prevVNode, nextVNode, container)
  } else if (nextFlags & VNodeFlags.TEXT) {
    patchText(prevVNode, nextVNode)
  } else if (nextFlags & VNodeFlags.FRAGMENT) {
    patchFragment(prevVNode, nextVNode, container)
  } else if (nextFlags & VNodeFlags.PORTAL) {
    patchPortal(prevVNode, nextVNode)
  }
```
如果类型不同，则直接调用 replaceVNode 函数使用新的 VNode 替换旧的 VNode，否则根据不同的类型调用与之相符的比对函数。
### 替换 VNode
```js
function replaceVNode(prevVNode, nextVNode, container) {
  // 将旧的 VNode 所渲染的 DOM 从容器中移除
  container.removeChild(prevVNode.el)
  // 再把新的 VNode 挂载到容器中
  mount(nextVNode, container)
}
```
### 更新标签元素
如果标签不同，那么直接替换。replaceVNode。    
如果标签相同，那两个 VNode 之间的差异就只会出现在 VNodeData 和 children 上了。所以，对比去掉旧的data和children，添加新的data和children就可以了。

#### 更新 VNodeData
。。。
#### 更新子节点
。。。
### 更新文本节点
```js
function patchText(prevVNode, nextVNode) {
  // 拿到文本元素 el，同时让 nextVNode.el 指向该文本元素
  const el = (nextVNode.el = prevVNode.el)
  // 只有当新旧文本内容不一致时才有必要更新
  if (nextVNode.children !== prevVNode.children) {
    el.nodeValue = nextVNode.children
  }
}
```
### 更新 Fragment
```js
function patchFragment(prevVNode, nextVNode, container) {
  // 直接调用 patchChildren 函数更新 新旧片段的子节点即可
  patchChildren(
    prevVNode.childFlags, // 旧片段的子节点类型
    nextVNode.childFlags, // 新片段的子节点类型
    prevVNode.children,   // 旧片段的子节点
    nextVNode.children,   // 新片段的子节点
    container
  )
  switch (nextVNode.childFlags) {
    case ChildrenFlags.SINGLE_VNODE:
      nextVNode.el = nextVNode.children.el
      break
    case ChildrenFlags.NO_CHILDREN:
      nextVNode.el = prevVNode.el
      break
    default:
      nextVNode.el = nextVNode.children[0].el
  }
}
```
### 更新 Portal
挂载目标不同，就需要搬运el。

### 更新有状态组件
#### 主动更新
所谓主动更新指的是组件自身的状态发生变化所导致的更新，例如组件的 data 数据发生了变化就必然需要重渲染。
#### 被动更新
除了自身状态之外，很可能还包含从父组件传递进来的外部状态(props)，像这种就叫做被动更新。

### 更新函数式组件
。。。概念有点绕啊

## 核心 Diff 算法
那什么才是核心的 Diff 算法呢？如图：
![](http://hcysun.me/vue-design/assets/img/patch-children-3.06453ea2.png)
### 1. 减小DOM操作的性能开销
我们不应该总是遍历旧的 children，而是应该遍历新旧 children 中长度较短的那一个，这样我们能够做到尽可能多的应用 patch 函数进行更新。

### 2. 尽可能的复用 DOM 元素
在上一小节中，我们通过减少 DOM 操作的次数使得更新的性能得到了提升，但它仍然存在可优化的空间。
#### key 的作用
如果移动可以达成目的，那么应该怎么移动呢？那就需要有唯一值的映射关系。    
没有 key 的情况下，我们是没办法知道新 children 中的节点是否可以在旧 children 中找到可复用的节点的。
#### 找到需要移动的节点
如果在寻找的过程中遇到的索引呈现递增趋势，则说明新旧 children 中节点顺序相同，不需要移动操作。相反的，如果在寻找的过程中遇到的索引值不呈现递增趋势，则说明需要移动操作。

寻找过程中在旧 children 中所遇到的最大索引值。如果在后续寻找的过程中发现存在索引值比最大索引值小的节点，意味着该节点需要被移动。实际上，这就是 React 所使用的算法。
#### 移动节点
:::danger
移动的是真实 DOM ，而非 VNode。
:::
![](http://hcysun.me/vue-design/assets/img/diff-react-2.e6cef98d.png)
新 children 中的第一个节点是 li-c，它在旧 children 中的索引为 2，由于 li-c 是新 children 中的**第一个节点，所以它始终都是不需要移动的**，只需要调用 patch 函数更新即可。
#### 添加新元素
```js {22-25}
let lastIndex = 0
for (let i = 0; i < nextChildren.length; i++) {
  const nextVNode = nextChildren[i]
  let j = 0,
    find = false
  for (j; j < prevChildren.length; j++) {
    const prevVNode = prevChildren[j]
    if (nextVNode.key === prevVNode.key) {
      find = true
      patch(prevVNode, nextVNode, container)
      if (j < lastIndex) {
        // 需要移动
        const refNode = nextChildren[i - 1].el.nextSibling
        container.insertBefore(prevVNode.el, refNode)
        break
      } else {
        // 更新 lastIndex
        lastIndex = j
      }
    }
  }
  if (!find) {
    // 挂载新节点
    mount(nextVNode, container, false)
  }
}
```
#### 移除不存在的元素
```js {14-26}
let lastIndex = 0
for (let i = 0; i < nextChildren.length; i++) {
  const nextVNode = nextChildren[i]
  let j = 0,
    find = false
  for (j; j < prevChildren.length; j++) {
    // 省略...
  }
  if (!find) {
    // 挂载新节点
    // 省略...
  }
}
// 移除已经不存在的节点
// 遍历旧的节点
for (let i = 0; i < prevChildren.length; i++) {
  const prevVNode = prevChildren[i]
  // 拿着旧 VNode 去新 children 中寻找相同的节点
  const has = nextChildren.find(
    nextVNode => nextVNode.key === prevVNode.key
  )
  if (!has) {
    // 如果没有找到相同的节点，则移除
    container.removeChild(prevVNode.el)
  }
}
```
:::tip
以上就是 React 所采用的 Diff 算法。但该算法仍然存在可优化的空间。
:::

### 另一个思路 - 双端比较
#### 双端比较的原理
![](http://hcysun.me/vue-design/assets/img/diff-vue2-1.216b174f.png)
React 的 diff 需要移动2次。而我们肉眼观察，实际上只需要移动1次，就能达到目的。

**所谓双端比较，就是同时从新旧 children 的两端开始进行比较的一种方式**。所以我们需要四个索引值，分别指向新旧 children 的两端，如下图所示：
![](http://hcysun.me/vue-design/assets/img/diff-vue2-3.933b8708.png)
#### 双端比较的优势
双端比较在移动 DOM 方面更具有普适性，不会因为 DOM 结构的差异而产生影响。
#### 非理想情况的处理方式
```js {10-15}
while (oldStartIdx <= oldEndIdx && newStartIdx <= newEndIdx) {
  if (oldStartVNode.key === newStartVNode.key) {
    // 省略...
  } else if (oldEndVNode.key === newEndVNode.key) {
    // 省略...
  } else if (oldStartVNode.key === newEndVNode.key) {
    // 省略...
  } else if (oldEndVNode.key === newStartVNode.key) {
    // 省略...
  } else {
    // 遍历旧 children，试图寻找与 newStartVNode 拥有相同 key 值的元素
    const idxInOld = prevChildren.findIndex(
      node => node.key === newStartVNode.key
    )
  }
}
```
#### 添加新元素
#### 移除不存在的元素
:::tip Vue2的diff算法
以上就是相对完整的双端比较算法的实现，这是 Vue2 所采用的算法，借鉴于开源项目：snabbdom，但最早采用双端比较算法的库是 citojs。
:::

#### 同层比较
当新旧 VNode 标签类型相同时，只需要更新 VNodeData 和 children 即可，不会“移除”和“新建”任何 DOM 元素的，而是复用已有 DOM 元素。
#### 有用的key
遍历比较(React)和双端比较(Vue2)

### Vue3的 Diff 算法 —— inferno
在 Vue3 中将采用另外一种核心 Diff 算法，它借鉴于 [ivi](https://github.com/localvoid/ivi) 和 [inferno](https://github.com/infernojs/inferno)。

但总体上的性能表现并不是单纯的由核心 Diff 算法来决定的，我们在前面章节的讲解中已经了解到的了一些优化手段，例如在**创建 VNode 时就确定其类型，以及在 mount/patch 的过程中采用位运算来判断一个 VNode 的类型，在这个基础之上再配合核心的 Diff 算法**，才使得性能上产生一定的优势，这也是 Vue3 接纳这种算法的原因之一。

#### 相同的前置和后置元素
在diff之前，会有一些预处理的过程，以避免diff算法低效频繁的执行。
预处理的类型：
1. 相等比较
2. 相同的前缀和后缀
那么是否可以把前缀和后缀的方法，也借鉴到diff算法中呢？可以。
![](http://hcysun.me/vue-design/assets/img/diff7.df9450ee.png)
```js
if (j > prevEnd && j <= nextEnd) {
  // j -> nextEnd 之间的节点应该被添加
  const nextPos = nextEnd + 1
  const refNode =
    nextPos < nextChildren.length ? nextChildren[nextPos].el : null
  while (j <= nextEnd) {
    mount(nextChildren[j++], container, false, refNode)
  }
} else if (j > nextEnd) {
  // j -> prevEnd 之间的节点应该被移除
  while (j <= prevEnd) {
    container.removeChild(prevChildren[j++].el)
  }
}
```
