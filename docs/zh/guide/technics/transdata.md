---
title: 数据转化
---
## 怎么跳出多重循环
用```for```，然后用```label```即可。```youOuter```就是你自定义的label啦。
```js
const demoList = [
  { name: 'beijing', address: '010'},
  { name: '', address: '0755'},
  { name: 'dongguan', address: ''},
]
youOuter:for(let item of demoList) {
  for(let key in item) {
    if(!item[key]) {
      console.log('for-of', key);
      break youOuter;
    }
  }
}
```

## 数组怎么按给定的order排序
```js
const demoList = [
  { name: 'beijing', address: '010' },
  { name: 'shenzhen', address: '0755' },
  { name: 'shanghai', address: '021' },
  { name: 'guangzhou', address: '020' },
];
const orderList = ["beijing", "shanghai", "guangzhou", "shenzhen"];
demoList.sort((a, b) => {
  return orderList.indexOf(a.name) - orderList.indexOf(b.name)
});
```

## 深度复制
```js
JSON.parse(JSON.stringify(youData))
```