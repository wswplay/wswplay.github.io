---
title: 组件
---
## 子孙组件怎么共享数据？
provide 和 inject。
```js
// 提供者
import { provide, ref } from 'vue' 
export default { 
  setup() { 
    const theme = ref('dark') 
    provide('theme', theme) 
  } 
}
// 后代组件
import { inject } from 'vue' 
export default { 
  setup() { 
    const theme = inject('theme', 'light') 
    return { 
      theme 
    } 
  } 
}
```
实际上，你可以把依赖注入看作一部分“大范围有效的 prop”。而且它的规则更加宽松：    
**祖先组件不需要知道哪些后代组件在使用它提供的数据，后代组件也不需要知道注入的数据来自哪里**。

## provide API
```js
function provide(key, value) { 
  let provides = currentInstance.provides 
  const parentProvides = currentInstance.parent && currentInstance.parent.provides 
  if (parentProvides === provides) { 
    provides = currentInstance.provides = Object.create(parentProvides) 
  } 
  provides[key] = value 
}
```
所以在默认情况下，组件实例的 provides 继承它的父组件，但是当组件实例需要提供自己的值的时候，它使用父级提供的对象创建自己的 provides 的对象原型。通过这种方式，在 inject 阶段，我们可以非常容易通过原型链查找来自直接父级提供的数据。

另外，如果组件实例提供和父级 provides 中有相同 key 的数据，是可以覆盖父级提供的数据。

## inject API
```js
function inject(key, defaultValue) { 
  const instance = currentInstance || currentRenderingInstance 
  if (instance) { 
    const provides = instance.provides 
    if (key in provides) { 
      return provides[key] 
    } 
    else if (arguments.length > 1) { 
      return defaultValue 
    } 
    else if ((process.env.NODE_ENV !== 'production')) { 
      warn(`injection "${String(key)}" not found.`) 
    } 
  } 
}
```

## 对比模块化共享数据的方式
为何要用 provide/inject ？直接 export/import 数据行吗？    

## 依赖注入的缺陷和应用场景
正因为依赖注入是上下文相关的，所以它会将你应用程序中的组件与它们当前的组织方式耦合起来，这使得重构变得困难。    
所以，我**并不推荐在普通应用程序代码中使用依赖注入**。但是我推荐你在组件库的开发中使用，因为对于一个特定组件，它和其嵌套的子组件上下文联系很紧密。这里来举一个 Element-UI 组件库 Select 组件的例子。

