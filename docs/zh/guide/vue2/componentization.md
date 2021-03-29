---
title: 组件化
---
## 职责分离原则

## 功能和逻辑合并

## 父子组件怎么通信
#### 1、props & $emit
#### 2、ref
#### 3、Vuex

## 父子组件生命周期函数执行顺序
#### 加载渲染过程
父beforeCreate->父created->父beforeMount->子beforeCreate->子created->子beforeMount->子mounted->父mounted
#### 更新过程
父beforeUpdate->子beforeUpdate->子updated->父updated
#### 销毁过程
父beforeDestroy->子beforeDestroy->子destroyed->父destroyed