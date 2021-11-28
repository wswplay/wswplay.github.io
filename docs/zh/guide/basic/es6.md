---
title: ECMAScript 6.0
---
## some
## every
## reduce
:::tip
1、previousValue （上一次调用回调返回的值，或者是提供的初始值）  
2、currentValue （数组中当前被处理的元素）  
3、index （当前元素在数组中的索引）  
4、array （调用 reduce 的数组）  
:::
将多维数组 转为 一维数组
```js
const arr = [[0, 1], [2, 3], [4,[5,6,7]]]
const multDimenToOne = (arr) => {
   return arr.reduce((pre, cur) => {
     return pre.concat(Array.isArray(cur) ? multDimenToOne(cur) : cur);
   }, []);
}
console.log(multDimenToOne(arr)); //[0, 1, 2, 3, 4, 5, 6, 7]
```
将一维数组 转为 二维数组
```js
// const originList = Array.apple(null, {length: 10}).map((item, index) => index)
const originList = Array.from({length: 10}).map((item, index) => index)
const step = 4
let targetList = []
function oneDimen2Two(data) {
  if(data.length > step) {
    let rest = data.splice(step)
    targetList.push(data)
    oneDimen2Two(rest)
  } else {
    targetList.push(data)
  }
}
oneDimen2Two(originList)
console.log(targetList) // [ [ 0, 1, 2, 3 ], [ 4, 5, 6, 7 ], [ 8, 9 ] ]
```