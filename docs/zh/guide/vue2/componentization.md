---
title: 组件化
---
## 组件类型
### 对象式组件
敬请期待。。。
### 函数式组件
敬请期待。。。

## 职责分离原则

## 功能和逻辑合并

## 父子组件怎么通信
### 1、props & $emit
### 2、ref
#### ref 写在组件上，```$refs.xxx```获取的，是这个组件的```vm实例```。
ref如果放在for循环里面，且取值不是变量，那```$refs.xxx```是一个同名的数组，可以通过下标取到相应的组件实例。可调用实例的方法和属性。
#### ref 写在div上，```$refs.xxx```获取的DOM的原生对象，可以调用原生方法和属性等。如```innerText```。
### 3、Vuex
### 4、mixin
### 5、依赖注入(provide/inject)

## 父子组件生命周期函数执行顺序
### 加载渲染过程
父beforeCreate->父created->父beforeMount->子beforeCreate->子created->子beforeMount->子mounted->父mounted
### 更新过程
父beforeUpdate->子beforeUpdate->子updated->父updated
### 销毁过程
父beforeDestroy->子beforeDestroy->子destroyed->父destroyed

## 兄弟组件怎么通信
### 1、依赖注入(provide/inject)，尤其是非SPA，更加适合；
### 2、Vuex；