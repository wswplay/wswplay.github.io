---
title: 数据结构
---
数据结构即数据元素相互之间存在的一种和多种特定的关系集合。一般你可以从两个维度来理解它，逻辑结构和存储结构。
## Object

## Map(键值对)
[Map MDN](https://developer.mozilla.org/zh-CN/docs/Web/JavaScript/Reference/Global_Objects/Map)
初始化Map需要一个```二维数组```，或者直接初始化一个空Map
```js
let myMap = new Map([['Michael', 95], ['Bob', 75], ['Tracy', 85]]);
// or
let youMap = new Map();
let nameStr = "xiao"
youMap.set(100, '一百')
youMap.set(nameStr, 'nanzhi')

let newMap = new Map([...youMap, [true, "真的"]])

console.log(newMap) // Map(3) { 100 => '一百', 'xiao' => 'nanzhi', true => '真的' }
console.log(youMap.size, newMap.size) // 2 3
console.log(youMap.get(nameStr)) // nanzhi
console.log(newMap.get(nameStr)) // nanzhi
// 使用 for..of 方法迭代 Map
for(let [key, val] of newMap) {
  console.log(`${key}===${val}`)
}
// 100===一百
// xiao===nanzhi
// true===真的
```
:::tip
1. Map“键”的范围不限于字符串，各种类型的值（包括对象）都可以当作键。Object结构提供了“字符串—值”的对应，Map 结构提供了“值—值”的对应，是一种更完善的 Hash 结构实现。快速查找，毫无压力。
2. Map 元素的顺序遵循插入的顺序，而 Object 的则没有这一特性。
3. Map 存取都有原生方法可用。has什么的。
4. Map有size属性，但Object没有。
5. Map自身支持迭代，但Object只能for...in。
:::
[Map和WeakMap的区别](https://zhuanlan.zhihu.com/p/366505417)，Weak可以被GC回收。

## WeakMap
[WeakMap](https://developer.mozilla.org/zh-CN/docs/Web/JavaScript/Reference/Global_Objects/WeakMap) 对象是一组键/值对的集合，其中的键是弱引用的。其键必须是对象，而值可以是任意的。
```js
// Vue3.0
function createAppContext() {
  return {
    app: null,
    config: {
      isNativeTag: NO,
      performance: false,
      globalProperties: {},
      optionMergeStrategies: {},
      errorHandler: undefined,
      warnHandler: undefined,
      compilerOptions: {}
    },
    mixins: [],
    components: {},
    directives: {},
    provides: Object.create(null),
    optionsCache: new WeakMap(),
    propsCache: new WeakMap(),
    emitsCache: new WeakMap()
  };
}
```
## Set(key的集合)
[Set MDN](https://developer.mozilla.org/zh-CN/docs/Web/JavaScript/Reference/Global_Objects/Set)    
Set 对象允许你存储任何类型的**唯一值**，无论是原始值或者是对象引用。Set中的元素只会出现一次。    
Set对象是值的集合，你可以按照插入的**顺序**迭代它的元素。   
要创建一个Set，需要提供一个Array作为输入，或者直接创建一个空Set：
```js
let youSet = new Set();
youSet.add(2022)
youSet.add('虎')
youSet.add('喵')
console.log(youSet, youSet.size) // Set(3) { 2022, '虎', '喵' } 3

youSet.delete(2022)
console.log(youSet, youSet.size) // Set(2) { '虎', '喵' } 2

console.log([...youSet]) // [ '虎', '喵' ]

// for...of迭代
for(let item of youSet) {
  console.log(item)
}
for(let [key, val] of youSet.entries()) {
  // key和val相同
  console.log(`${key}===${val}`)
}
for(let [key, val] of [...youSet].entries()) {
  // key和val不不不不不同
  console.log(`${key}===${val}`)
}
```

#### 参考
[MDN数据结构](https://developer.mozilla.org/zh-CN/docs/Web/JavaScript/Data_structures)