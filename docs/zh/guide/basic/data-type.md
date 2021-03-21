---
title: 数据类型
---
## 一切皆对象
::: tip 
js中，所有类型，都是对象。
:::
## 基础类型(值传递) typeof
>字符串(String)、数字(Number)、布尔(Boolean)、对空(Null)、未定义(Undefined)、Symbol
```js
// 类型判断方法：typeof
typeof '边城' //"string"
typeof 1902 //"number"
typeof true //"boolean"
typeof null //"object"
typeof undefined //"undefined"
```
## 引用类型(地址传递) instanceof
>对象(Object)、数组(Array)、函数(Function)
```js
// 数组对象typeof判断不准确
const person = {name: '沈从文'}
const works = ['湘行散记', '中国服饰研究']
function Xiao() { 
  console.log(this)
}
typeof person //"object"
typeof works //"object"
typeof Xiao //"function"
// 类型判断方法：instanceof
person instanceof Object //true
person instanceof Array //false

works instanceof Object //true
works instanceof Array //true

Xiao instanceof Object //true
Xiao instanceof Function //true
Xiao instanceof Array //false
```