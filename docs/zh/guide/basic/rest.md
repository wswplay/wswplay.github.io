---
title: rest扩展运算符
---
就是那个三个点 ... 啦

```js
// 合并数组
const one = [11, 111]
const two = [10001, 22, 222]
let three = [...one, ...two] // [11, 111, 10001, 22, 222]

// 求数组内的最大值
Math.max(...three) // 10001

// 将类数组转化为真数组
function miao(...arg) {
  console.log('arg', arg)
}
miao(6, 66, 699) // arg [6, 66, 699]
```