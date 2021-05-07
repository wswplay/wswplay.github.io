---
title: 数据驱动
---
每个人都有他的脾气，每一行代码都有它存在的意义，每一种脾气都有意义。

## 灵魂问题
1. 如何定义非响应式数据？
2. 根数据data为什么不是响应式？
3. 当然了前提是回调函数不能是箭头函数，其实在平时的使用中，回调函数使用箭头函数也没关系，只要你能够达到你的目的即可。
4. Vue实现的nextTick是宏任务，还是微任务？
## 其他问题
1. 文件加载是什么顺序，比如main.js和router/index.js？

:::tip 特点
与jQuery相比，Vue的最大特点就是：数据驱动、组件化。
:::
Vue是怎么实现数据驱动，并自动化的更新视图的呢？
## Object.defineProperty
将数据对象的**数据属性转换为访问器属性**，即为数据对象的属性设置一对 ```getter/setter``` 。
```js {8,10,36}
var childOb = !shallow && observe(val);
Object.defineProperty(obj, key, {
  enumerable: true,
  configurable: true,
  get: function reactiveGetter () {
    var value = getter ? getter.call(obj) : val;
    if (Dep.target) {
      dep.depend(); // 收集依赖
      if (childOb) {
        childOb.dep.depend(); // 递归收集依赖
        if (Array.isArray(value)) {
          dependArray(value);
        }
      }
    }
    return value
  },
  set: function reactiveSetter (newVal) {
    var value = getter ? getter.call(obj) : val;
    /* eslint-disable no-self-compare */
    if (newVal === value || (newVal !== newVal && value !== value)) {
      return
    }
    /* eslint-enable no-self-compare */
    if (process.env.NODE_ENV !== 'production' && customSetter) {
      customSetter();
    }
    // #7981: for accessor properties without setter
    if (getter && !setter) { return }
    if (setter) {
      setter.call(obj, newVal);
    } else {
      val = newVal;
    }
    childOb = !shallow && observe(newVal);
    dep.notify(); // 派发更新
  }
});
```
通过```Object.defineProperty```把数据```data```转化成```getter```、```setter```。    
当取值```getter```时，收集依赖。也就是把这个值的观察者，收集到一个队列中(数组)。        
当赋值```setter```时，派发更新。通知数据的每一个观察者，去更新相关联的地方。

另外这里有一个优先级的关系：props优先级 > data优先级 > methods优先级。

### Object.defineProperty.get
```get``` 主要完成了两部分重要的工作，一个是返回正确的属性值，另一个是收集依赖。

### 闭包引用dep
这里大家要明确一件事情，即**每一个数据字段都通过闭包引用着属于自己的 dep 常量**。

### childOb和__ob__属性

只有对象或者数组才是```observe```的目标，才有```__ob__```属性，有具体值的 key字段 没有这个属性。    
所以 ```__ob__``` 属性以及 ```__ob__```.dep 的主要作用是为了添加、删除属性时有能力触发依赖，而这就是 Vue.set 或 Vue.delete 的原理。

**解析**：
1. 当一个key的值为基础类型时，它的值操作只有一种方式，那就是用**等号重新赋值**。
2. 当一个key的值为引用类型(数组或对象)，值操作方式有两种：等号重新赋值 和 添加属性或元素。

### Object.defineProperty.set
```set``` 函数也要完成两个重要的事情，第一正确地为属性设置新值，第二是能够触发相应的依赖。
```js
if (newVal === value || (newVal !== newVal && value !== value)) {
  return
}
```
newVal !== newVal 说明新值与新值自身都不全等，同时旧值与旧值自身也不全等，大家想一下在 js 中什么时候会出现一个值与自身都不全等的？答案就是 NaN。

### 响应式数组的处理
那么为什么数组需要这样处理，而纯对象不需要呢？那是因为 数组的索引是非响应式的。现在我们已经知道了数据响应系统对纯对象和数组的处理方式是不同，对于纯对象只需要逐个将对象的属性重新定义为访问器属性，并且当属性的值同样为纯对象时进行递归定义即可，而对于数组的处理则是通过拦截数组变异方法的方式。

因为对于数组来讲，其索引并不是“访问器属性”。

### Vue.set/$set 和 Vue.del/$del
每个实例根数据(data)对象不是响应式的，因为模板里面不会出现以下用法：
```vue {2}
<template>
  <div class="warpper">{{根data}}</div>
</template>
```
因此没机会添加```watcher```，无法添加和触发新增属性。这也是data各属性需要**提前声明**的原因。

如果目标为数组，用的都是 ```splice``` 方法
```js
target.splice(key, 1)
```

### watch
```js
watch: {
  name: 'handleNameChange'
},
methods: {
  handleNameChange () {
    console.log('name change')
  }
}
```
上面的代码中我们在 watch 选项中观察了 name 属性，但是我们没有指定回调函数，而是指定了一个字符串 handleNameChange，这等价于指定了 methods 选项中同名函数作为回调函数。

### 编译器
Vue 的编译器也不例外，大致也分为三个阶段，即：词法分析 -> 句法分析 -> 代码生成。在词法分析阶段 Vue 会把字符串模板解析成一个个的令牌(token)，该令牌将用于句法分析阶段，在句法分析阶段会根据令牌生成一棵 AST，最后再根据该 AST 生成最终的渲染函数，这样就完成了代码的生成。


## Proxy数据代理
```js
const sharedPropertyDefinition = {
  enumerable: true,
  configurable: true,
  get: noop,
  set: noop
}
export function proxy (target: Object, sourceKey: string, key: string) {
  sharedPropertyDefinition.get = function proxyGetter () {
    return this[sourceKey][key]
  }
  sharedPropertyDefinition.set = function proxySetter (val) {
    this[sourceKey][key] = val
  }
  Object.defineProperty(target, key, sharedPropertyDefinition)
}
```
