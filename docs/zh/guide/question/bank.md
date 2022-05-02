---
title: Vue
---
## mutation怎么调用另一个mutation
待解。

## 解构会破坏响应式吗
待解。

## mutation的返回值是什么
待解。

## ref和$refs
:::tip
ref应用在一个普通元素上(如div)，```this.$refs.xxx```取到的是Dom元素。    
ref应用到组建上，```this.$refs.xxx```获取的是组件实例。```this.$refs.xxx.$el```是Dom元素。
ref应用在循环中，```this.$refs.xxx[0]```才能取到值。注意，是```[0]```。
:::

## textContent和innerText、innerHTML的区别
1、textContent VS innerText：不会触发回流(reflow)，且能获取不可见元素。   
2、textContent VS innerHTML：性能更好，还可以防止XSS攻击。