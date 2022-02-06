---
title: 柯里化(Currying)
---
## 定义与理解
:::tip 理解
将，**一个多入参函数，转化成，单一入参函数**，的过程。   

这玩意儿这么玩，就跟闭包搅在一起，不可分割了啊。   
1、返回一个函数；2、此函数，访问外部变量；
:::
## 作用
1、持久持有变量，固化通用参数。
## Vue中怎么用
```js
// 源码定义
const createHook = (lifecycle)
      => (hook, target = currentInstance)
      => injectHook(lifecycle, hook, target);
const onMounted = createHook("m" /* MOUNTED */);
// 业务适用
onMounted(() => {
  console.log("柯里化")
})
```
## 简单demo
```js
const createHello = (nation, province) => `我来自${nation}-${province}`;
const geneNation = (nation) => (province) => createHello(nation, province);
// function geneNation(nation) {
//   return function(province) {
//     return `我来自${nation}-${province}`;
//   }
// }

const fromChina = geneNation("中国");
const fromUSA = geneNation("美国")

console.log(fromChina("湖南"))
console.log(fromChina("广东"))

console.log(fromUSA("洛杉矶"))

console.log(geneNation("银河系")("地球"))
console.log(geneNation("三体")("黑暗森林"))

// 我来自中国-湖南
// 我来自中国-广东
// 我来自美国-洛杉矶
// 我来自银河系-地球
// 我来自三体-黑暗森林
```
