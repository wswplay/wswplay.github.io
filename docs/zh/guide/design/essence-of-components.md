---
title: 组件的本质
---
## 一个组件就是一个函数
Vue 来说，一个组件最核心的东西是 render 函数，剩余的其他内容，如 data、compouted、props 等都是为 render 函数提供数据来源服务的。

## 组件的输出就是：```Virtual DOM```
:::tip
在后续行文时，将统一使用 VNode 来简称 Virtual DOM 。    
在Vue.js 中虚拟 DOM 的 JavaScript 对象就是 VNode。
:::
**为什么不直接输出```html```，而是```Virtual DOM```呢？**    
其原因是 Virtual DOM 带来了 分层设计，它对渲染过程的抽象，使得框架可以渲染到 **web(浏览器) 以外的平台，以及能够实现 SSR 等**。    
>至于 Virtual DOM 相比原生 DOM 操作的性能，这并非 Virtual DOM 的目标，确切地说，如果要比较二者的性能是要“控制变量”的，例如：页面的大小、数据变化量等。

## 组件的 ```VNode``` 如何表示
可以让 VNode 的 tag 属性指向组件本身，从而使用 VNode 来描述组件。
```js
export interface VNode {
  // _isVNode 属性在上文中没有提到，它是一个始终为 true 的值
  // 有了它，我们就可以判断一个对象是否是 VNode 对象
  _isVNode: true
  // el：当一个 VNode 被渲染为真实 DOM 后，el 属性的值会引用该真实DOM
  el: Element | null
  flags: VNodeFlags
  tag: string | FunctionalComponent | ComponentClass | null
  data: VNodeData | null
  children: VNodeChildren
  childFlags: ChildrenFlags
}
```

## VNode的属性称为VNodeData
1. 假如一个 VNode 的类型是 html 标签，则 VNodeData 中可以包含 class、style 以及一些事件。    
2. 如果 VNode 的类型是组件，那么我们同样可以用 VNodeData 来描述组件，比如组件的事件、组件的 props 等等。例如：
```html
<MyComponent @some-event="handler" prop-a="1" />
```
则其对应的 VNodeData 应为：
```js {4-9}
{
  flags: VNodeFlags.COMPONENT_STATEFUL,
  tag: 'div',
  data: {
    on: {
      'some-event': handler
    },
    propA: '1'
    // 其他数据...
  }
}
```

## 组件的种类
第一种方式是使用一个普通的函数：
```js
function MyComponent(props) {}
```
第二种方式是使用一个类：
```js
class MyComponent {}
```
实际上它们分别代表两类组件：    
函数式组件(```Functional component```) 和 有状态组件(```Stateful component```)。

它们的区别如下：    
#### 函数式组件：
1. 是一个纯函数
2. 没有自身状态，只接收外部数据
3. 产出 VNode 的方式：单纯的函数调用

#### 有状态组件：
1. 是一个类，可实例化
2. 可以有自身状态
3. 产出 VNode 的方式：需要实例化，然后调用其 render 函数

## VNode 的种类
1. 比如一个 VNode 对象是 html 标签的描述，那么其 tag 属性值就是一个字符串，即标签的名字；
2. 如果是组件的描述，那么其 tag 属性值则引用组件类(或函数)本身；
3. 如果是文本节点的描述，那么其 tag 属性值为 null。

总的来说，我们可以把 VNode 分成五类，分别是：    
**html/svg 元素**、**组件**、**纯文本**、**Fragment** 以及 **Portal**：
![](http://hcysun.me/vue-design/assets/img/vnode-types.7d99313d.png)
