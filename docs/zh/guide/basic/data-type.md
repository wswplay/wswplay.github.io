---
title: 数据类型
---
## 一切皆对象
```js
const story = '边城'
typeof story //"string"
// 擦，instanceof是不能用来判断基础类型滴
story instanceof Object //false
story.length //2
```
比如：字符串，和对象一样有属性和方法。怎么解释和理解呢？   
>当读取它属性时，js把```string```通过```new String()```方式创建一个字符串临时对象(学术名叫包装对象)。对象就有了属性。但这个对象只是临时的，一旦引用结束，对象就被销毁了。
::: tip 《JavaScript权威指南》：
其实（包装对象）在实现上并不一定创建或销毁这个临时对象，只是整个过程看起来像而已。
:::
## 基础类型(值传递) typeof
[MDN](https://developer.mozilla.org/zh-CN/docs/Web/JavaScript/Reference/Operators/typeof)：```typeof```操作符返回一个字符串，表示未经计算的操作数的类型。
>字符串(String)、数字(Number)、布尔(Boolean)、对空(Null)、未定义(Undefined)、符号(Symbol)
```js
// 类型判断方法：typeof
typeof '边城' //"string"
typeof 1902 //"number"
typeof true //"boolean"
typeof null //"object"
typeof undefined //"undefined"
```
## 引用类型(地址传递) instanceof
[MDN](https://developer.mozilla.org/zh-CN/docs/Web/JavaScript/Reference/Operators/instanceof)：```instanceof```运算符用于检测构造函数的 prototype 属性是否出现在某个实例对象的原型链上。
>对象(Object)、数组(Array)、函数(Function)
```js
// 数组对象typeof判断不准确
const person = {name: '沈从文'}
const works = ['湘行散记', '中国服饰研究']
function Xiao(params) {
  params.id = 520
  if(!(this instanceof Xiao)) console.log('请使用new关键字')
}
typeof person //"object"
typeof works //"object"
typeof Xiao //"function"
// 类型判断方法：instanceof
person instanceof Object //true
person instanceof Array //false

works instanceof Array //true
works instanceof Object //true

Xiao instanceof Function //true
Xiao instanceof Array //false
Xiao instanceof Object //true

// 地址传递
Xiao(person)
console.log(person) // {name: "沈从文", id: 520}
```
## Vue怎么判断类型
待续。。。
